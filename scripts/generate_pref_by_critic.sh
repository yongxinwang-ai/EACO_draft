#!/bin/bash

python ./stic/generate_pref_by_critic.py \
    --model-path liuhaotian/llava-v1.6-mistral-7b \
    --critic-path path/to/critic \
    --image-dir /l/users/yongxin.wang/data/train2014/ \
    --save-dir pref/pref_output_all.jsonl \