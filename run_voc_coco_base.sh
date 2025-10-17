gpu_id=$3
echo "gpu_id: ${gpu_id}"
for shot in 2 3 5 10
do 
    config_file="configs/voc10-5-5/food_gfsod_r50_novel1_${shot}shot_seed1.yaml"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config-file ${config_file} --num-gpus 4\
        --opts \
        OUTPUT_DIR "/home/user/xinzhimeng/desk1/FOOD/r50_maxunknonw/vae0.9/${shot}shot" \
        MODEL.WEIGHTS "/home/user/xinzhimeng/desk1/FOOD/voc_r50/voc10-5-5/food_r50_voc_base/model_reset_remove.pth" \
        SOLVER.IMS_PER_BATCH 16 \
        SOLVER.BASE_LR 0.02 \
        MODEL.RESNETS.DEPTH 50 \
        cons_loss True \
        _ALPHA 0.95 \
        UPLOSS.ENABLE_UPLOSS True \
        UPLOSS.SAMPLING_METRIC "random" \
        #--resume \
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


