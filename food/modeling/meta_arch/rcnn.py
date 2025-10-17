import torch
import logging
from torch import nn
from detectron2.structures import ImageList
from detectron2.utils.logger import log_first_n
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .build import META_ARCH_REGISTRY
from .gdl import decouple_layer, AffineLayer
from food.modeling.roi_heads import build_roi_heads
import cv2
import numpy as np
__all__ = ["GeneralizedRCNN"]

from food.modeling.roi_heads.VAE import VAE

@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self._SHAPE_ = self.backbone.output_shape()
        self.proposal_generator = build_proposal_generator(cfg, self._SHAPE_)
        self.roi_heads = build_roi_heads(cfg, self._SHAPE_)
        self.normalizer = self.normalize_fn()
        if 'swin' in cfg.MODEL.BACKBONE.NAME:
            self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['stage5'].channels, bias=True)
            self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['stage5'].channels, bias=True)
        else:
            self.affine_rpn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
            self.affine_rcnn = AffineLayer(num_channels=self._SHAPE_['res4'].channels, bias=True)
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("froze backbone parameters")

        if cfg.MODEL.RPN.FREEZE:
            for p in self.proposal_generator.parameters(): # type: ignore
                p.requires_grad = False
            print("froze proposal generator parameters")

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.res5.parameters():
                p.requires_grad = False
            print("froze roi_box_head parameters")

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device) 
        self.withvae = False
        if self.withvae:
            input_dim = 1024
            hidden_dim = 2048
            # latent_dim = 20
            latent_dim = 2048
            self.vae = VAE(input_dim, hidden_dim, latent_dim)

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        

        if self.withvae:       
            proposal_losses, detector_losses, _, _, lossvae = self._forward_once_(batched_inputs, gt_instances) # type: ignore
            losses = {}
            losses['vae_loss'] = lossvae
            losses.update(detector_losses)
            losses.update(proposal_losses)
            
        else:
            proposal_losses, detector_losses, _, _ = self._forward_once_(batched_inputs, gt_instances) # type: ignore
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
        
        return losses

    def inference(self, batched_inputs):
        assert not self.training
        if self.withvae:
            _, _, results, image_sizes, _ = self._forward_once_(batched_inputs, None) # type: ignore
        else:
            _, _, results, image_sizes = self._forward_once_(batched_inputs, None) # type: ignore # type: ignor
        processed_results = []
        for r, input, image_size in zip(results, batched_inputs, image_sizes):
            height = input.get("height", image_size[0])
            width = input.get("width", image_size[1])
            r = detector_postprocess(r, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def _forward_once_(self, batched_inputs, gt_instances=None):

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if self.cfg.TEST.SAVE_FEATURE_MAP == True:
            x = features['res4']
            x_size = x.size()
            x_np = x.cpu().numpy()
            x_np = x_np.reshape(1024, x_size[2], x_size[3])
            for i in range(1024):
                each_f = x_np[i, :, :]
                # plt.imshow(each_f)
                path_save = "/home/subinyi/Users/DeFRCN-main/paper_image/feature_map/"
                # plt.savefig(path_save + 'fpn'+str(j)+'_' + str(i)+'.jpg')
                ret = each_f.reshape(x_size[2], x_size[3])
                ret = (ret-ret.min())/(ret.max()-ret.min())*256
                ret = ret.astype(np.uint8)
                gray = ret[:, :, None]
                ret = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                ret = ret.astype(np.uint8)
                cv2.imwrite(path_save + 'map'+'_' + str(i)+'.png', ret)
            quit()

        if self.withvae:            
            
           # B,C,W,H = features["res4"].shape 
            
            vae_features = features["res4"].permute(0, 2, 3, 1)
             
            # vae_features = features["res4"]       
                                          
            x_recon, mu, logvar = self.vae(vae_features)
            # 计算 VAE 损失并添加到 losses 中
            vae_loss_value = self.calculate_vae_loss(x_recon, vae_features, mu, logvar)

            combined_features = 0.9 * vae_features + 0.1 * x_recon #.view(vae_features.shape[0], vae_features.shape[1], vae_features.shape[2],-1 ).mean(dim=[2,3])
            features["res4"] = combined_features.permute(0, 3, 1, 2) 
            
        features_de_rpn = features
        if self.cfg.MODEL.RPN.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.RPN.BACKWARD_SCALE
            features_de_rpn = {k: self.affine_rpn(decouple_layer(features[k], scale)) for k in features}
        proposals, proposal_losses = self.proposal_generator(images, features_de_rpn, gt_instances) # type: ignore

        features_de_rcnn = features
        if self.cfg.MODEL.ROI_HEADS.ENABLE_DECOUPLE:
            scale = self.cfg.MODEL.ROI_HEADS.BACKWARD_SCALE
            features_de_rcnn = {k: self.affine_rcnn(decouple_layer(features[k], scale)) for k in features}
        results, detector_losses = self.roi_heads(images, features_de_rcnn, proposals, gt_instances)


        if self.withvae:
            # losses = {}
            # # losses.update(detector_losses)
            # # losses.update(proposal_losses)
            # losses['vae_loss'] = vae_loss_value.mean()  
            # # detector_losses['vae_loss'] = vae_loss_value.mean()
            # # losses.update(detector_losses)         
        
            return proposal_losses, detector_losses, results, images.image_sizes,vae_loss_value.mean()
        else:
        
            return proposal_losses, detector_losses, results, images.image_sizes
    
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

    def preprocess_image(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    def normalize_fn(self):
        assert len(self.cfg.MODEL.PIXEL_MEAN) == len(self.cfg.MODEL.PIXEL_STD)
        num_channels = len(self.cfg.MODEL.PIXEL_MEAN)
        pixel_mean = (torch.Tensor(
            self.cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1))
        pixel_std = (torch.Tensor(
            self.cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1))
        return lambda x: (x - pixel_mean) / pixel_std
