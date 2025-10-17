gpu_id=$3
echo "gpu_id: ${gpu_id}"
for shot in 1 3 5 10
do 
    config_file="configs/voc10-5-5/food_gfsod_r50_novel1_${shot}shot_seed1.yaml"
    CUDA_VISIBLE_DEVICES=0 python main.py --config-file ${config_file} --num-gpus 1 \
        --opts \
        OUTPUT_DIR "/home/xzm/xzm/FOOD/food-2025/output/r50/distacebaseline/${shot}shot" \
        MODEL.WEIGHTS "pretrained/r50/model_reset_remove.pth" \
        SOLVER.IMS_PER_BATCH 8 \
        SOLVER.BASE_LR 0.02 \
        MODEL.RESNETS.DEPTH 50 \
        _ALPHA 1.0 \
        cons_loss False \
        VAE True \
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

# for shot in 1 3 5 10 # if final, 10 -> 1 2 3 5 10 30
#     do
        
#         CONFIG_PATH=configs/voc10-5-5/food_gfsod_r50_novel1_${shot}shot_seed1.yaml
#         OUTPUT_DIR=/home/user/xinzhimeng/desk1/FOOD/voc_r101/${shot}shotgfsod
#         BASE_WEIGHT=/home/user/xinzhimeng/desk1/FOOD/voc_r101/model_reset_surgery.pth
#         CUDA_VISIBLE_DEVICES=2,3 python main.py --num-gpus 2 --config-file ${CONFIG_PATH}                            \
#             --opts MODEL.WEIGHTS ${BASE_WEIGHT} OUTPUT_DIR ${OUTPUT_DIR}                          \
#                    $TEST.PCB_MODELPATH ${IMAGENET_PRETRAIN_TORCH}                                  \
#                    #SOLVER.IMS_PER_BATCH 16
#         # rm ${CONFIG_PATH}
#         # rm ${OUTPUT_DIR}/model_final.pth
# done
