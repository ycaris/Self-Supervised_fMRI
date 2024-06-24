
#!/usr/bin/env bash

fold=1
PERC=100
JSON="/home/yz2337/project/multi_fmri/code/json_files/ace/srs/fold_${fold}.json"
VERSION=v5_pretrain_masktime;
BATCH_SIZE=16 
MODEL='simple-transformer'
# SAVEPATH="../runs/biopoint/${VERSION}/fold${FOLD}.${VERSION}"
SAVEPATH="../run_ace/abide_all_srs/ace_fold${fold}/${VERSION}"
 
 

CUDA_LAUNCH_BLOCKING=1 python3 main_train.py --lr=1e-3 --savepath=${SAVEPATH} --json_file=${JSON} \
   --bs=${BATCH_SIZE} --opt=adamw --model_arch=${MODEL} --num_workers=4 \
   --eval_num=100 --num_steps=5000  --warmup_steps=50 --lrdecay -wd=1e-3 \
   --use_pretrained 