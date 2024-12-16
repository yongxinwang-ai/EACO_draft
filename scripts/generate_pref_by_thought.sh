#!/bin/bash

python ./stic/generate_by_thought.py \
    --model-path liuhaotian/llava-v1.6-mistral-7b \
    --image-dir /l/users/yongxin.wang/data/MathV360K/data_images/ \
    --save-dir pref/pref_cot_math.jsonl \