# Self-supervised transformer for time-series fMRI in autism detection
This is the code for [Self-supervised transformer for time-series fMRI in autism detection](https://arxiv.org/pdf/2409.12304).

## Directory
- Pretrain: the directory contains the code for pretraining of the code using different random masking strategies. 
- Code: the directory contains the code for downstream ASD classification task

## Pretraining
- Three random masking tasks classes are in the python script: pretrain/utils/transform.py, and each is named as RandomMask, RandomMaskTime, RandomMaskROI
- To change different masking task, change line 132 in pretrain/utils/data_util.py to the class you want. The current implementation is RandomMaskROI
- To run the pretraining process, change the parser arguments in pretrain/runExp.sh, or pretrain/utils/parser_util.py, then run the following command: 
```
bash pretrain/runExp.sh 
```
