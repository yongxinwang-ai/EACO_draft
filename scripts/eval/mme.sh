#!/bin/bash

# /fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Haokun.Lin@mbzuai.ac.ae/checkpoint/llava-v1.5-7b-mix_arxiv_screen_chart_doc
ANSWER="llava-v1.5-7b-temp1"
python -m llava.eval.model_vqa_loader \
    --model-path /home/panwen.hu/workspace/haokun.lin/checkpoints/$ANSWER \
    --question-file /home/panwen.hu/workspace/haokun.lin/benchmark/MME/MME/llava_mme.jsonl \
    --image-folder /home/panwen.hu/workspace/haokun.lin/benchmark/MME/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/MME/$ANSWER.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    # --model-base lmsys/vicuna-7b-v1.5 \

cd /home/panwen.hu/workspace/haokun.lin/benchmark/MME/MME

python convert_answer_to_mme.py --experiment $ANSWER

cd eval_tool

python calculation.py --results_dir answers/$ANSWER
