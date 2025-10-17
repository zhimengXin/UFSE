gpu_id=$3
echo "gpu_id: ${gpu_id}"
for shot in 1 5 10 30
do 
    config_file="configs/voc_coco/food_gfsod_r50_novel_${shot}shot_seedx.yaml" 
    CUDA_VISIBLE_DEVICES=0 python main.py --config-file ${config_file} --num-gpus 1 \
        --opts \
        OUTPUT_DIR "/home/xzm/xzm/FOOD/food-2025/output/r50/voc-coco/CLIP=unfreeze/${shot}shot" \
        MODEL.WEIGHTS "pretrained/voccoco/r50/model_reset_remove.pth" \
        SOLVER.IMS_PER_BATCH 1 \
        SOLVER.BASE_LR 0.01 \
        MODEL.RESNETS.DEPTH 50 \
        _ALPHA 1.0 \
        cons_loss False \
        VAE False \
        background_loss False \
        image_clip True \
        UPLOSS.ENABLE_UPLOSS True \
        UPLOSS.TOPK 6 \
        UPLOSS.SAMPLING_METRIC "random" \
        # edl_dirichlet  random
        # --resume \
        # MODEL.WEIGHTS "/home/user/xinzhimeng/desk1/FOOD/voc_r101/model_reset_remove.pth" \
        # MODEL.BACKBONE.WITHSCSM True \
        # MODEL.BACKBONE.WITHSFA = False
        # MODEL.BACKBONE.FREEZE_SFA = False
        # MODEL.BACKBONE.WITHMAMBA = False
        # MODEL.BACKBONE.WITHSCSM = False
        # MODEL.BACKBONE.WITHSFASA = False
        # MODEL.BACKBONE.WITHSFASENet = False
        # MODEL.BACKBONE.CNN = False
done
