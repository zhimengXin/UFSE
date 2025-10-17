import cv2
from food.config import get_cfg
from food.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import os
import torch

# 检查CUDA是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Running on CPU.")

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat","chair", "cow",
        "diningtable", "dog", "horse", "motorbike", "person",
         "pottedplant", "sheep", "sofa", "train", "tvmonitor", "unknown"]

# "pottedplant", "sheep", "sofa", "train", "tvmonitor", "unknown",

# yaml= "/home/user/xinzhimeng/FSOD/defrcn/checkpoints/voc/defrcn_r101_pam_all_layer4/config.yaml"
# # pth = "/home/user/xinzhimeng/desk1/mamba/ECEASENet/base1/5shotgfsod/model_0001999.pth"
# pth = "/home/user/xinzhimeng/FSOD/defrcn/checkpoints/voc/defrcn_r101_pam_all_layer4/model_0024999.pth"

yaml="/media/xzm2/a4a1dba5-468a-4b2e-9e30-47d18517ce18/food-2025/output/r50-res/config.yaml"

pth="/media/xzm2/a4a1dba5-468a-4b2e-9e30-47d18517ce18/food-2025/output/r50-res/model_final.pth"




def detect_objects(image_path,filename):
    # 加载配置文件
    cfg = get_cfg()
    cfg.merge_from_file(yaml)  # 根据您的配置文件路径进行修改
    cfg.MODEL.WEIGHTS = pth  # 根据您的权重文件路径进行修改
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # 设置置信度阈值

    predictor = DefaultPredictor(cfg)

    # 读取图像
    image = cv2.imread(image_path)

    # predictor.model.to(device)
    # image = torch.from_numpy(image.transpose((2, 0, 1))).float().unsqueeze(0).to(device)
    # 进行预测
    outputs = predictor(image)
    
    # model = predictor.model
    # se_module = model.backbone.bottom_up.stem.conv1.se_module
    # features = se_module.get_features()
   #exit()
    # 获取类别元数据
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    # 可视化结果
    v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    # print(outputs["instances"].to("cpu"))
    # exit()
    # # 在图像上显示类名
    instances = outputs["instances"].to("cpu")
    
    
    img = out.get_image()[:, :, ::-1]
    img_putlabel = img.copy()
    if len(instances.pred_classes) > 0:
        class_id = instances.pred_classes[0].item()
    # print(len(instances))        
        for i in range(len(instances)):
            class_id = instances.pred_classes[i].item()
            class_name = classes[class_id]
            #class_name = metadata.thing_classes[class_id]
            bbox = instances.pred_boxes.tensor[i].tolist()
            x, y, w, h = bbox

            #cv2.putText(img_putlabel, class_name, ((int)(x), (int)(y)+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

        # img = cv2.putText(out.get_image()[:, :, ::-1], class_name, (int(x), int(y) - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
    
    # if len(instances):
    #     print("Have instances")
    #     cv2.imwrite("/home/user/xinzhimeng/FSOD/demo-visualization/test.jpg", img)
    # else:
    #     print("No instances")
    #     cv2.imwrite("/home/user/xinzhimeng/FSOD/demo-visualization/test.jpg", img_putlabel)

    # 显示检测结果
    cv2.imwrite("/media/xzm2/a4a1dba5-468a-4b2e-9e30-47d18517ce18/food-2025/output/vis-res/"+ filename, img_putlabel)
    # cv2.imshow("Detection Results", img_putlabel[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def process_images_in_folder(folder_path, test_file):
    # 读取测试文件中的图像文件名
    with open(test_file, "r") as f:
        test_images = f.read().splitlines()

    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        # 检查文件类型，例如图片
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 检查文件名是否在测试文件中
            if filename.split(".")[0] in test_images:
                # 构建图像文件的完整路径
                image_path = os.path.join(folder_path, filename)
                
                # 进行目标检测并输出结果
                detect_objects(image_path, filename)
        else:
            # 跳过不感兴趣的文件
            print("Skipping file:", filename)

# 图像文件夹路径
image_folder = "/media/xzm2/a4a1dba5-468a-4b2e-9e30-47d18517ce18/food-2025/datasets/VOC2007/TEST/VOC2007/JPEGImages"  # 根据您的图像文件夹路径进行修改

# 测试文件路径
test_file = "/media/xzm2/a4a1dba5-468a-4b2e-9e30-47d18517ce18/food-2025/datasets/VOC2007/TEST/VOC2007/ImageSets/test.txt"  # 根据您的测试文件路径进行修改

# 处理图像文件夹中的测试图像
process_images_in_folder(image_folder, test_file)


# detect_objects("/home/user/xinzhimeng/FSOD/defrcn/visualization/test-image/004993.jpg","xx")