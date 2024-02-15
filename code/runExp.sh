#!/usr/bin/env bash

FOLD=1
JSON="./json_files/fmri_only/fold_${FOLD}_points.json"
VERSION=v0
BATCH_SIZE=32
MODEL=simple-transformer
SAVEPATH="../runs/fold${FOLD}.${VERSION}"

CUDA_LAUNCH_BLOCKING=1 python3 main_train.py --lr=1e-5 --savepath=${SAVEPATH} --json_file=${JSON} \
   --bs=${BATCH_SIZE} --opt=adamw --model_arch=${MODEL} --num_workers=4 \
   --eval_num=50 --num_steps=5000  --lrdecay

