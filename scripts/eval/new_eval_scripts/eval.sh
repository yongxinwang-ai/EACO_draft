#!/bin/bash

python3 -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/panwen.hu/workspace/yongxin.wang/STIC/ckpt/whole_model/llava-v1.6-mistral-7b-STIC_lora" \
    --tasks mme \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme_mmbenchen \
    --output_path ./logs/