# Language-Inspired Unknown-Space Expansion for Few-Shot Open-Set Detection






## Setup

The code is based on detectron2 v0.3

- ### **Installation**

```
conda create -n ufse python=3.8 -y
conda activate ufse
```

- **Prepare datasets**

You should download：

- train and val set of COCO2017

- trainval and test set of VOC2007、VOC2012

following the structure described below：

```
datasets/
  coco/
  VOC20{07,12}/
```

In coco：

```
coco/
  annotations/
    instances_{train,val}2017.json
    person_keypoints_{train,val}2017.json
  {train,val}2017/
```

In  VOC20{07,12}：

```
VOC20{07,12}/
  Annotations/
  ImageSets/
    Main/
      trainval.txt
      test.txt
  JPEGImages/
```

Then we generate all datasets for FOOD:

```
bash prepare_food_voc_coco.sh
```

## Training and Evaluation

#### VOC-COCO dataset settings:

```
bash run_voc_coco_AR.sh
```

#### VOC10-5-5 dataset settings:

```
bash run_voc_AR.sh
```

### Citation

If you find this repo useful, please consider citing our paper:

```
@article{UFSE,
  title={Language-Inspired Unknown-Space Expansion for Few-Shot Open-Set Detection},
  author={Xin, Zhimeng and Wu, Tianxu and Zou, Yixiong and Chen, Shiming and Peng, Qinmu and You, Xinge},
  journal={IEEE Transactions on Multimedia},
  year={2026},
  publisher={IEEE}
}
```




