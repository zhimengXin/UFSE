import torch
import logging
import numpy as np
from torch import nn
from typing import Dict
from detectron2.layers import ShapeSpec
from detectron2.utils.registry import Registry
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.utils.events import get_event_storage
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from .box_head import build_box_head
from .fast_rcnn import ROI_BOX_OUTPUT_LAYERS_REGISTRY, FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.modeling.roi_heads.roi_heads import (
    ROI_HEADS_REGISTRY, StandardROIHeads)
from typing import Dict, List
from .fast_rcnn import build_roi_box_output_layers
from PIL import Image

from food.modeling.roi_heads.VAE import VAE
# from transformers import CLIPModel, CLIPProcessor
import clip
from detectron2.layers import cat

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


@ROI_HEADS_REGISTRY.register()
class OpenSetStandardROIHeads(StandardROIHeads):

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals: List[Instances], targets: List[Instances]) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes
            # NOTE: add iou of each proposal
            ious, _ = match_quality_matrix.max(dim=0)
            proposals_per_image.iou = ious[sampled_idxs]

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets])

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels,
                  height=pooler_resolution, width=pooler_resolution)
        )
        # register output layers
        box_predictor = build_roi_box_output_layers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

def select_foreground_proposals(proposals, bg_label):
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.test_score_thresh        = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(
            weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS
        )

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes,
            self.batch_size_per_image,
            self.positive_sample_fraction,
            self.num_classes,
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns `self.batch_size_per_image` random samples from proposals and groundtruth boxes,
        with a fraction of positives that is no larger than `self.positive_sample_fraction.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:
                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                   then the ground-truth box is random)
                Other fields such as "gt_classes" that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(
                match_quality_matrix
            )
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs] # type: ignore
            proposals_per_image.gt_classes = gt_classes

            # NOTE: add iou of each proposal
            ious, _ = match_quality_matrix.max(dim=0)
            proposals_per_image.iou = ious[sampled_idxs]

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # will filter the proposals again (by foreground/background,
                # etc), so we essentially index the data twice.
                for (
                    trg_name,
                    trg_value,
                ) in targets_per_image.get_fields().items():
                    if trg_name.startswith(
                        "gt_"
                    ) and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(
                            trg_name, trg_value[sampled_targets]
                        )
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros(
                        (len(sampled_idxs), 4)
                    )
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append(
                (gt_classes == self.num_classes).sum().item()
            )
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, targets=None):
        """
        Args:
            images (ImageList):
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`s. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:
                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].

        Returns:
            results (list[Instances]): length `N` list of `Instances`s containing the
                detected instances. Returned during inference only; may be []
                during training.
            losses (dict[str: Tensor]): mapping from a named loss to a tensor
                storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where the heads share the
    cropping and the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device)
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / self.feature_strides[self.in_features[0]], ) # type: ignore
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)

        # self.box_head = build_box_head(
        #     cfg, ShapeSpec(channels=out_channels,
        #                    height=pooler_resolution, width=pooler_resolution)
        # )
        output_layer = cfg.MODEL.ROI_BOX_HEAD.OUTPUT_LAYERS
        # self.box_predictor = ROI_BOX_OUTPUT_LAYERS_REGISTRY.get(output_layer)(
        #     cfg, out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        # )

        # unknown
        self.box_predictor = ROI_BOX_OUTPUT_LAYERS_REGISTRY.get(output_layer)(
            cfg, out_channels
        )
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device) 
        self.withvae = cfg.VAE
        self.with_attach_feature = False

        if self.withvae:
            input_dim = 32768
            hidden_dim = 20
            # latent_dim = 20
            latent_dim = 20
            self.vae = VAE(input_dim, hidden_dim, latent_dim)  
        
        self.with_text_clip = cfg.text_clip
        self.with_image_clip = cfg.image_clip
        if self.with_text_clip or self.with_image_clip:
            # self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base32")
            # self.clip_processor = CLIPProcessor.from_pretrained("/home/user/xinzhimeng/YOLO-World/clip-vit-base-patch32/")  
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

            for param in self.clip_model.parameters():
                param.requires_grad = True


            nDim = 4096
            self.linear = nn.Sequential(nn.Linear(512, nDim), nn.ReLU(), 
                                        nn.Linear(nDim, nDim),nn.ReLU(), nn.Linear(nDim, 2048))
            #INIT self.linear
            for m in self.linear:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on
        if 'swin' in cfg.MODEL.BACKBONE.NAME:
            blocks = make_stage(
                BottleneckBlock,
                3,
                stride_per_block=[2, 1, 1],
                in_channels=cfg.MODEL.SWINT.IN_CHANNELS,
                bottleneck_channels=bottleneck_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                norm=norm,
                stride_in_1x1=stride_in_1x1,
            )
        else:
            blocks = make_stage(
                BottleneckBlock,
                3,
                stride_per_block=[2, 1, 1],
                in_channels=out_channels // 2,
                bottleneck_channels=bottleneck_channels,
                out_channels=out_channels,
                num_groups=num_groups,
                norm=norm,
                stride_in_1x1=stride_in_1x1,
            )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        # print('pooler:', x.size())
        x = self.res5(x) # 16384*2048*4*4
        # print('res5:', x.size())
        return x

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        labels_names_list = ["aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat","chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor", "unknown", 'background']
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
            # gt_classes_list = []
            # for target in targets: # type: ignore
            #     gt_classes_list.append(target.gt_classes[0].cpu().item())        
            # gt_labels = [labels_names_list[index - 1] for index in gt_classes_list] 
        del targets
        # 
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1 4096*2048 16384*2048
       #  feature_pooled = self.box_head(box_features)

        if self.with_attach_feature and self.training:
            feature_clone = feature_pooled.clone()
            pred_class_logits, pred_proposal_deltas, s_, _ = self.box_predictor( feature_clone)
            # predictions = self.box_predictor( feature_clone )

            # losses = self.box_predictor.losses(predictions, proposals)
            outputs = FastRCNNOutputs(
                            self.box2box_transform,
                            pred_class_logits,
                            pred_proposal_deltas,
                            proposals,
                            self.smooth_l1_beta,
                        )
            #remove None element of predictions
            # predictions = (x for x in predictions if x is not None)
            # predictions = (x for x in predictions if x is not None)
            # losses_cur = outputs.losses()['loss_box_reg'] * 0.1
            losses_cur = outputs.losses()['loss_box_reg'] * 0.2 + outputs.losses()['loss_cls'] * 0.05 # + outputs.losses()['loss_cls_unknown_2'] 

            feature_pooled_grad = torch.autograd.grad(losses_cur, feature_pooled, retain_graph=True)[0]
            x = 0.1 * torch.sign(feature_pooled_grad)

            feature_pooled = feature_pooled + x
        
        # pred_class_logits, pred_proposal_deltas = self.box_predictor(
        #     feature_pooled
        # )
        # unknown
        pred_class_logits, pred_proposal_deltas, s_, _ = self.box_predictor(
            feature_pooled
        )
        predictions = self.box_predictor(
            feature_pooled
        )
        # del feature_pooled

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        
        
        if self.withvae:
            # feature_pooled as VAE input
            
            B,C,_,_ = box_features.shape 
            vae_features = box_features.view(B, -1)  
            x_recon, mu, logvar = self.vae(vae_features)
            vae_features = vae_features.view(box_features.shape[0], box_features.shape[1], box_features.shape[2],-1 ).mean(dim=[2,3])            
            x_recon_mean = x_recon.view(box_features.shape[0], box_features.shape[1], box_features.shape[2],-1 ).mean(dim=[2,3])
            vae_loss_value = self.calculate_vae_loss(x_recon_mean, vae_features, mu, logvar)
            if self.training:                
                predictions = self.box_predictor(feature_pooled, x_recon_mean)
            else: 
                predictions = self.box_predictor(feature_pooled)
            
            if self.training:
                losses = self.box_predictor.losses(predictions, proposals)
                losses['vae_loss'] = vae_loss_value 
                return [], losses
            else:
                pred_instances, _ = outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )                
                return pred_instances, {'vae_loss': vae_loss_value}
                        
        if self.with_image_clip and self.training:
            # 获取 gt_classes 等于 20 的索引
            gt_classes_21_indices = []
            for i, proposal in enumerate(proposals):
                gt_classes = proposal.gt_classes
                gt_classes_21_indices.extend((gt_classes == 81).nonzero(as_tuple=True)[0] + i * len(gt_classes))
           
            if isinstance(gt_classes_21_indices, list):
                gt_classes_21_indices = list(set(gt_classes_21_indices))
            elif isinstance(gt_classes_21_indices, torch.Tensor):
                gt_classes_21_indices = torch.unique(gt_classes_21_indices)
           
            selected_proposal_boxes = []
            mask = torch.zeros(box_features.shape[0], dtype=torch.bool).cuda()
            valid_indices = []
            for index in gt_classes_21_indices:
                if index < box_features.shape[0]:  # 检查索引是否在合法范围内
                    valid_indices.append(index)
                else:
                    print(f"索引 {index} 超出 box_features 维度范围，已忽略")
            if valid_indices:
                valid_indices_tensor = torch.tensor(valid_indices).to(device)
                mask[valid_indices_tensor] = True
                selected_proposal_boxes = box_features[mask].to(device)
            else:
                print("没有合法的索引，selected_proposal_boxes 为空")
                # 这里也可以根据业务逻辑赋予 selected_proposal_boxes 一个合适的默认值，比如全零张量等
                selected_proposal_boxes = torch.zeros_like(box_features[0:0]).to(device)
            
            if len(selected_proposal_boxes) > 0:
                bg_features = self.clip_model.encode_text(selected_proposal_boxes.view(selected_proposal_boxes.size(0),-1)[:,:77].long()).float() # type: ignore
                #image_features = self.clip_model.encode_image(torch.stack(selected_proposal_boxes)).float()
                bg_features = self.linear(bg_features)
                if bg_features.shape[0]<feature_pooled.shape[0]:
                    padding_size = feature_pooled.shape[0] - bg_features.shape[0]
                    padding_tensor = torch.zeros(padding_size, bg_features.shape[1])                        
                    bg_features = torch.cat((padding_tensor.to(device), bg_features), dim=0)
                    bg_features = bg_features[:4096,:]
                
                feature_pooledclip = feature_pooled + bg_features                    
            if self.training:
                predictions = self.box_predictor(feature_pooledclip)
            else:
                predictions = self.box_predictor(feature_pooled)
            if self.training:
                # del features
                # losses = outputs.losses()
                # unknown
                losses = self.box_predictor.losses(predictions, proposals)
                return [], losses
            else:
                pred_instances, _ = outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
                return pred_instances, {}

        if self.withvae:
            # feature_pooled as VAE input
            
            B,C,_,_ = box_features.shape 
            vae_features = box_features.view(B, -1)  
            x_recon, mu, logvar = self.vae(vae_features)
            vae_features = vae_features.view(box_features.shape[0], box_features.shape[1], box_features.shape[2],-1 ).mean(dim=[2,3])            
            x_recon_mean = x_recon.view(box_features.shape[0], box_features.shape[1], box_features.shape[2],-1 ).mean(dim=[2,3])
            
            
            if self.with_text_clip and self.training:
                gt_classes_new = ( cat([(p.gt_classes ) for p in proposals], dim=0) if len( proposals) else torch.empty(0) )
                gt_boxes = []
                gt_image_idx = []
                for idx, p in enumerate(proposals):
                    gt_boxes.extend(p.proposal_boxes)
                    gt_image_idx.extend([idx for box in p.proposal_boxes])
                if self.with_text_clip:
                    # 将图像数据格式调整为适合CLIP输入的格式，并放到对应的设备上（和box_features一致的设备）
                    
                    # with torch.no_grad():
                    # text_tokens = clip.tokenize(labels_names_list).to(vae_features.device) # type: ignore
                    # text_features = self.clip_model.encode_text(text_tokens).float() # type: ignore
                    # text_features = self.linear(text_features)
                    # text_selected = torch.index_select(text_features, 0, gt_classes_new)                    
                    text_features_2 = self.clip_model.encode_text(feature_pooled[:,:77].long()).float()
                    text_features_2 = self.linear(text_features_2)
                    # text_features_fuse = text_features
                    
                    
                       
            if self.with_image_clip and self.training:
                # 获取 gt_classes 等于 20 的索引
                gt_classes_21_indices = []
                for i, proposal in enumerate(proposals):
                    gt_classes = proposal.gt_classes
                    gt_classes_21_indices.extend((gt_classes == 81).nonzero(as_tuple=True)[0] + i * len(gt_classes))
                # 从 box_features 中提取 gt_classes 等于 21 的特征
                # if isinstance(gt_classes_21_indices, list):
                #     gt_classes_21_indices = sorted(gt_classes_21_indices)
                # elif isinstance(gt_classes_21_indices, torch.Tensor):
                #     gt_classes_21_indices = gt_classes_21_indices.sort().values

                if isinstance(gt_classes_21_indices, list):
                    gt_classes_21_indices = list(set(gt_classes_21_indices))
                elif isinstance(gt_classes_21_indices, torch.Tensor):
                    gt_classes_21_indices = torch.unique(gt_classes_21_indices)
                
                # selected_proposal_boxes = []
                # mask = torch.zeros(box_features.shape[0], dtype=torch.bool).cuda()
                # mask[torch.tensor(gt_classes_21_indices).to(device)] = True
                # selected_proposal_boxes = box_features[mask].to(device)

                selected_proposal_boxes = []
                mask = torch.zeros(box_features.shape[0], dtype=torch.bool).cuda()
                valid_indices = []
                for index in gt_classes_21_indices:
                    if index < box_features.shape[0]:  # 检查索引是否在合法范围内
                        valid_indices.append(index)
                    # else:
                    #     print(f"索引 {index} 超出 box_features 维度范围，已忽略")
                if valid_indices:
                    valid_indices_tensor = torch.tensor(valid_indices).to(device)
                    mask[valid_indices_tensor] = True
                    selected_proposal_boxes = box_features[mask].to(device)
                else:
                    # print("没有合法的索引，selected_proposal_boxes 为空")
                    # 这里也可以根据业务逻辑赋予 selected_proposal_boxes 一个合适的默认值，比如全零张量等
                    selected_proposal_boxes = torch.zeros_like(box_features[0:0]).to(device)

                if len(selected_proposal_boxes) > 0:

                    bg_features = self.clip_model.encode_text(selected_proposal_boxes.view(selected_proposal_boxes.size(0),-1)[:,:77].long()).float() # type: ignore
                    #image_features = self.clip_model.encode_image(torch.stack(selected_proposal_boxes)).float()
                    bg_features = self.linear(bg_features)
                    if bg_features.shape[0]<x_recon_mean.shape[0]:
                        padding_size = x_recon_mean.shape[0] - bg_features.shape[0]
                        padding_tensor = torch.zeros(padding_size, bg_features.shape[1])                        
                        bg_features = torch.cat((padding_tensor.to(device), bg_features), dim=0)
                        bg_features = bg_features[:4096,:]
                    x_recon_mean = x_recon_mean + bg_features * 0.2
                                     
            vae_loss_value = self.calculate_vae_loss(x_recon_mean, vae_features, mu, logvar)
            
            # VAE for cls
            # predictions = self.box_predictor(feature_pooled, x_recon_mean)

            if self.training:
                predictions = self.box_predictor(feature_pooled, x_recon_mean)
            else: 
                predictions = self.box_predictor(feature_pooled)
            
            if self.training:
                losses = self.box_predictor.losses(predictions, proposals)
                losses['vae_loss'] = vae_loss_value 
                return [], losses
            else:
                pred_instances, _ = outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
                return pred_instances, {'vae_loss': vae_loss_value}    
                
        else:
            if self.training:
                # del features
                # losses = outputs.losses()
                # unknown
                losses = self.box_predictor.losses(predictions, proposals)
                return [], losses
            else:
                pred_instances, _ = outputs.inference(
                    self.test_score_thresh,
                    self.test_nms_thresh,
                    self.test_detections_per_img,
                )
                return pred_instances, {}
    
    def calculate_vae_loss(self, x_recon, x, mu, logvar):
        import torch.nn.functional as F
        x = torch.clamp(x, 0, 1)
        #recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        recon_loss = F.mse_loss(x_recon, x)
        #kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)

        loss = recon_loss + 0.00025 * kld_loss
        return loss

    

@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg)

    def _init_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / self.feature_strides[k] for k in self.in_features) # type: ignore
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        self.cls_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels,
                height=pooler_resolution,
                width=pooler_resolution,
            ),
        )

        output_layer = cfg.MODEL.ROI_HEADS.OUTPUT_LAYER
        self.box_predictor = ROI_BOX_OUTPUT_LAYERS_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

        self.cls_predictor = ROI_BOX_OUTPUT_LAYERS_REGISTRY.get(output_layer)(
            cfg,
            self.box_head.output_size,
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals)
            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals)
            return pred_instances, {}

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.

        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )

        cls_features = self.cls_head(box_features)
        pred_class_logits, _ = self.cls_predictor(
            cls_features
        )

        box_features = self.box_head(box_features)
        _, pred_proposal_deltas = self.box_predictor(
            box_features
        )
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh,
                self.test_nms_thresh,
                self.test_detections_per_img,
            )
            return pred_instances



