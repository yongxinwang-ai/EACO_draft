#!/bin/bash

python ./stic/generate_des_stage2.py \
    --model-path liuhaotian/llava-v1.6-vicuna-7b \
    --image-dir /l/users/yongxin.wang/data \
    --save-dir des/image_description_new_vicuna.jsonl \
    --adapter-path /l/users/yongxin.wang/code/STIC/ckpt/llava-v1.6-vicuna-7b-stage1-judge-new_lora
