#!/usr/bin/env bash
JSON="/home/yz2337/project/multi_fmri/code/json_files/pretrained/abide_group_all.json"
VERSION=v4_mlp_maskroi
BATCH_SIZE=64
MODEL=simple-transformer
SAVEPATH="./runs/abide_all/${VERSION}"

CUDA_LAUNCH_BLOCKING=1 python3 main_train.py --lr=1e-4 --savepath=${SAVEPATH} --json_file=${JSON} \
   --bs=${BATCH_SIZE} --opt=adamw --model_arch=${MODEL} --num_workers=4 \
   --eval_num=50000 --num_steps=50000  --warmup_steps=0 --lrdecay -wd=1e-5