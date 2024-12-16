CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc-per-node=2 run.py --data MME ScienceQA_TEST SEEDBench_IMG POPE  --model llava_next_mistral_7b_judge_pros --verbose
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node=4 run.py --data MME ScienceQA_TEST SEEDBench_IMG POPE  --model llava_next_mistral_7b_judge_pros_new --verbose
