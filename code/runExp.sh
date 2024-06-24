#!/usr/bin/env bash

FOLD=1
PERC=20
# JSON="/home/yz2337/project/multi_fmri/code/json_files/pretrained/abide_group.json"
JSON="/home/yz2337/project/multi_fmri/code/json_files/abide_percent_cv/pretrain_fold5/${PERC}/fold_${FOLD}.json"
# JSON="/home/yz2337/project/multi_fmri/code/json_files/fmri_only/fold_${FOLD}_points.json"
VERSION=v0_nopretrain
BATCH_SIZE=16
MODEL='simple-transformer-classify'
# SAVEPATH="../runs/biopoint/${VERSION}/fold${FOLD}.${VERSION}"
SAVEPATH="../runs/abide_percent_cv/pretrain_fold5/${PERC}/${VERSION}/fold${FOLD}"
# SAVEPATH="../runs/${VERSION}"
 
 

CUDA_LAUNCH_BLOCKING=1 python3 main_train.py --lr=1e-4 --savepath=${SAVEPATH} --json_file=${JSON} \
   --bs=${BATCH_SIZE} --opt=adamw --model_arch=${MODEL} --num_workers=4 \
   --eval_num=200 --num_steps=5000 --warmup_steps=50 --lrdecay -wd=1e-3 \
   # --use_pretrained 