gpu_id=$3
echo "gpu_id: ${gpu_id}"
for shot in 1
do 
    config_file="configs/voc10-5-5/food_gfsod_r50_novel1_${shot}shot_seed1.yaml"
    CUDA_VISIBLE_DEVICES=0 python main.py --config-file ${config_file} --num-gpus 1 --resume \
        --opts \
        OUTPUT_DIR "/home/xzm/xzm/FOOD/food-2025/output/r50/clip/${shot}shot" \
        MODEL.WEIGHTS "pretrained/r50/model_reset_remove.pth" \
        SOLVER.IMS_PER_BATCH 8 \
        SOLVER.BASE_LR 0.02 \
        SOLVER.MAX_ITER 2000 \
        SOLVER.CHECKPOINT_PERIOD 500 \
        TEST.EVAL_PERIOD 501 \
        MODEL.RESNETS.DEPTH 50 \
        UPLOSS.ENABLE_UPLOSS True \
        UPLOSS.SAMPLING_METRIC "random" \
        cons_loss False \
        _ALPHA 1.0 \
        background_loss False \
        VAE True \
        image_clip True \
        text_clip True \
        BERT False \
        
done


