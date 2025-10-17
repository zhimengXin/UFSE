import matplotlib
import matplotlib.pyplot as plt
from detectron2_gradcam import Detectron2GradCAM

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from food.engine import DefaultTrainer, default_argument_parser, default_setup
from gradcam import GradCAM, GradCamPlusPlus
import os
from torchcam.utils import overlay_mask
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
#
from food.evaluation import DatasetEvaluators, verify_results
import torch
import numpy as np
import random

plt.rcParams["figure.figsize"] = (30,10)

img_path = "/home/xzm/xzm/FOOD/food-2025/datasets/VOC2007/JPEGImages/"
config_file = "configs/voc10-5-5/food_gfsod_r50_novel1_10shot_seed1.yaml"
model_file = "/home/xzm/xzm/FOOD/food-2025/output/r50/CLIP+bgloss+VAE/10shot/model_final.pth"

# config_list = [
# "MODEL.ROI_HEADS.SCORE_THRESH_TEST", "0.5",
# "MODEL.ROI_HEADS.NUM_CLASSES", "20",
# "MODEL.WEIGHTS", model_file
# ]

# layer_name = "ext_conv_0"
# layer_name = "mamba_block_0"
# layer_name = "lateral_conv_out"
layer_name = "backbone.res4.22.conv3"
# instance = 8 #CAM is generated per object instance, not per class!
instance = 1


class Trainer(DefaultTrainer):

    @classmethod   
            
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type == "coco":
            from food.evaluation import COCOEvaluator
            evaluator_list.append(COCOEvaluator(dataset_name, True, output_folder))
        if evaluator_type == "pascal_voc":
            from food.evaluation import PascalVOCDetectionEvaluator
            return PascalVOCDetectionEvaluator(dataset_name)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
from tqdm import tqdm
def main():
    seed = 1037
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cam_extractor = Detectron2GradCAM(config_file, None, img_path=img_path)
    grad_cam = GradCAM
    cfg = cam_extractor.cfg
    for idx, dataset_name in enumerate(cfg.DATASETS.TRAIN):
        data_loader = Trainer.build_test_loader(cfg, dataset_name)
        for idx_data, inputs in tqdm(enumerate(data_loader)):
            
            image_dict, cam_orig,results = cam_extractor.get_cam(inputs,target_instance=instance, layer_name=layer_name, grad_cam_instance=grad_cam)

            # v = Visualizer(image_dict["image"], MetadataCatalog.get(cam_extractor.cfg.DATASETS.TRAIN[0]), scale=1.0)
            # out = v.draw_instance_predictions(image_dict["output"]["instances"][instance].to("cpu"))

            file_name = inputs[0]['file_name']
            output_full = os.path.join(cfg.OUTPUT_DIR, os.path.dirname(file_name))
            img_name = os.path.basename(file_name).split(".")[0]
            if not os.path.exists(output_full):
                os.makedirs(output_full)
            img_org = to_pil_image(image_dict["image"])
            overlay_img = overlay_mask(img_org, to_pil_image(image_dict["cam"], mode='F'), alpha=0.5)
            
                #draw box to overlay_img
                
                
            org_size = ( inputs[0]['width'] , inputs[0]['height'])
            #resize overlay_img to original size
            overlay_img = overlay_img.resize(org_size)
            if 0:
                draw = ImageDraw.Draw(overlay_img)
                for idx_key in results.keys():
                    box = results[idx_key][1].tensor[0].detach().cpu().numpy()
                    label = results[idx_key][0]
                    label_name= MetadataCatalog.get(cam_extractor.cfg.DATASETS.TRAIN[0]).thing_classes[label]
                    draw.rectangle([int(box[0]), int(box[1]), int(box[2]), int(box[3])], outline="red", width=2)
                    draw.text((box[0], box[1]), label_name, fill="red")

            overlay_img.save(os.path.join(output_full, f"{img_name}_cam.jpg"))

            
            # plt.imshow(out.get_image(), interpolation='none')
            # plt.imshow(image_dict["cam"], cmap='jet', alpha=0.5)
            # plt.title(f"CAM for Instance {instance} (class {image_dict['label']})")
            # plt.savefig(os.path.join(output_full, f"{img_name}_instance_{instance}_cam.jpg"), dpi=100)
            # plt.show()
            if idx_data  >= 100:
                break
            
            # break


if __name__ == "__main__":
    main()