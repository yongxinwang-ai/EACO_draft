#!/bin/bash
#SBATCH --job-name=merge_lora # Job name
#SBATCH --output=logs/merge_lora.txt # Standard output and error.
#SBATCH --nodes=1 # Run all processes on a single node
#SBATCH --ntasks=1 # Run on a single CPU
#SBATCH --mem=200G # Total RAM to be used
#SBATCH --cpus-per-task=64 # Number of CPU cores
#SBATCH --gres=gpu:1 # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p # Use the gpu partition
#SBATCH --time=12:00:00 # Specify the time needed for you job
#SBATCH -q cscc-gpu-qos # To enable the use of up to 8 GPUs

python scripts/merge_lora_weights.py \
    --model-path /l/users/yongxin.wang/code/STIC/ckpt/llava-v1.6-vicuna-7b-stage1-judge-new_lora \
    --model-base liuhaotian/llava-v1.6-vicuna-7b \
    --save-model-path /l/users/yongxin.wang/code/STIC/ckpt/whole_model/llava-v1.6-vicuna-7b-stage1-judge-new