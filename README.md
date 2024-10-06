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

## ASD Classification
- Change the parser arguments in code/runExp.sh or code/utils/parser_util.py to determine the file path, save path, and whether to utilize pretrained model or not
- ASD classification can be performed on ACE or ABIDE by running the following command for model training:
```
bash code/runExp.sh
bash code/runExpAce.sh
```
- the testing script is as follow:
```
bash code/runExp.sh
bash code/runExpAce.sh
```
