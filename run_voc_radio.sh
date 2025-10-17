gpu_id=$1
echo "gpu_id: ${gpu_id}"
for shot in 5
do
    for radio in 0.2 0.3 #0.4 0.5 0.6 0.7 0.8
    do 
        config_file="configs/voc10-5-5/food_gfsod_r50_novel1_${shot}shot_seed1.yaml"
        CUDA_VISIBLE_DEVICES=0 python3 main.py --config-file ${config_file} --num-gpus 1 --resume \
            --opts \
            OUTPUT_DIR "/media/xzm2/mntt/xzm/food/${shot}shot/${radio}" \
            MODEL.WEIGHTS "/media/xzm2/a4a1dba5-468a-4b2e-9e30-47d18517ce18/food-2025/pretrained/r50/model_reset_remove.pth" \
            SOLVER.IMS_PER_BATCH 6 \
            SOLVER.BASE_LR 0.01 \
            MODEL.RESNETS.DEPTH 50 \
            _ALPHA 0.95 \
            cons_loss True \
            VAE True \
            background_loss True \
            image_clip True \
            text_clip True \
            UPLOSS.ENABLE_UPLOSS True \
            UPLOSS.TOPK 6 \
            UPLOSS.SAMPLING_METRIC "random" \
            Radio ${radio} \
            # --resume \
    done
done