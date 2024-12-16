#!/bin/bash
#SBATCH --job-name=dpo # Job name
#SBATCH --output=logs/dpo_pref_vicuna7b.txt # Standard output and error.
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --mem=200G # Total RAM to be used
#SBATCH --cpus-per-task=64 # Number of CPU cores
#SBATCH --gres=gpu:4 # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p # Use the gpu partition
#SBATCH --time=12:00:00 # Specify the time needed for you job
#SBATCH -q cscc-gpu-qos # To enable the use of up to 8 GPUs


deepspeed --master_port=16547 --include=localhost:0,1,2,3 llava/train/train_dpo.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed scripts/zero2.json \
    --model_name_or_path liuhaotian/llava-v1.6-vicuna-7b \
    --version v1 \
    --data_path /l/users/yongxin.wang/code/STIC/pref/pref_vicuna.json \
    --image_folder /l/users/yongxin.wang/data/train2014 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./ckpt/llava-v1.6-vicuna-7b-stage1-judge-new_lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 1e-7 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to wandb 