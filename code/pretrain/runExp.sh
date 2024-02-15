#!/usr/bin/env bash

JSON="/home/yz2337/project/multi_fmri/code/json_files/pretrained/abide.json"
VERSION=v0
BATCH_SIZE=32
MODEL=simple-transformer
SAVEPATH="./runs/${VERSION}"

CUDA_LAUNCH_BLOCKING=1 python3 main_train.py --lr=1e-4 --savepath=${SAVEPATH} --json_file=${JSON} \
   --bs=${BATCH_SIZE} --opt=adamw --model_arch=${MODEL} --num_workers=4 \
   --eval_num=100 --num_steps=5000  --lrdecay

