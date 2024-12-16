#!/bin/bash

python ./stic/generate_pref.py \
    --model-path /fsx/homes/Yongxin.Wang@mbzuai.ac.ae/code/STIC/ckpt/llava-v1.6-mistral-7b-STIC-Iter1_lora \
    --model-base liuhaotian/llava-v1.6-mistral-7b \
    --image-dir /fsx/homes/Yongxin.Wang@mbzuai.ac.ae/data/train2014/ \
    --save-dir pref/pref_data_mscoco_iteration2_2k_5.jsonl \
