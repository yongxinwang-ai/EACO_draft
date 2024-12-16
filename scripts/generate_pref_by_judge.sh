#!/bin/bash

python ./stic/generate_pref_by_judge.py \
    --model-path liuhaotian/llava-v1.6-mistral-7b \
    --judge-path path/to/judge \
    --image-dir /l/users/yongxin.wang/data/train2014/ \
    --save-dir pref/pref_rating.jsonl \