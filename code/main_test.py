import numpy as np
import torch
import os
import argparse
from utils.transform import *
from utils.data_util import get_test_loader, get_val_loader
from utils import parser_util
from models.simple_transformer import SimpleTransformerClassification
from models.whole_transformer import WholeTransformerClassification
from models.lstm import LSTMClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv


parser = parser_util.prepare_parser()

# # testing parameter biopoint
# fold = 1
# version = 'v1_pretrain_64_pos2_pool2'
# parser.add_argument('--version', default=f'fold{fold}.{version}', type=str)
# parser.add_argument('--checkpoint_path',
#                     default=f'/home/yz2337/project/multi_fmri/runs/biopoint/{version}/fold{fold}.{version}/model.pt', type=str)
# parser.add_argument(
#     '--save_dir', default=f'/home/yz2337/project/multi_fmri/results/biopoint', type=str)

# abide
fold = 1
pretrain_fold = 1
percent = 100
version = 'v4_pretrain_maskroi'
parser.add_argument('--version', default=f'{version}_fold{fold}', type=str)
parser.add_argument('--checkpoint_path',
                    default=f'/home/yz2337/project/multi_fmri/runs/abide_percent_cv/pretrain_fold{pretrain_fold}/{percent}/{version}/fold{fold}/model.pt', type=str)
parser.add_argument(
    '--save_dir', default=f'/home/yz2337/project/multi_fmri/results/percent/{percent}', type=str)


args = parser.parse_args()
torch.backends.cudnn.benchmark = True

# args.json_file = f'/home/yz2337/project/multi_fmri/code/json_files/fmri_only/fold_{fold}_points.json'
# args.json_file = '/home/yz2337/project/multi_fmri/code/json_files/pretrained/abide_group.json'
args.json_file = f'/home/yz2337/project/multi_fmri/code/json_files/abide_percent_cv/pretrain_fold{pretrain_fold}/100/fold_1.json'


# specify the GPU id's, GPU id's start from 0.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------

# set the output saving folder
results_dir = os.path.join(args.save_dir, args.version)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# get the test data loader
test_loader = get_test_loader(args)

# set the saved model path
checkpoint_dir = args.checkpoint_path

# load model
model = SimpleTransformerClassification(feature_size=args.feature_size,
                                        emb_dim=args.emb_dim,
                                        num_layers=args.n_layers,
                                        dropout=args.dropout,
                                        ).to(device)
# model = WholeTransformer(feature_size=args.feature_size,
#                          emb_dim=args.emb_dim,
#                          nhead=args.nhead,
#                          num_layers=args.n_layers,
#                          dropout=args.dropout,
#                          time_period=args.time_period
#                          ).to(device)

# model = LSTMClassifier(feature_size=args.feature_size,
#                        emb_dim=args.emb_dim,
#                        num_layers=args.n_layers,
#                        dropout=args.dropout,
#                        ).to(device)

ckpt = torch.load(checkpoint_dir, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=True)
model.to(device)
model.eval()

# csv writer
filename = os.path.join(results_dir, f'{args.version}.csv')
f = open(filename, 'w')
writer = csv.writer(f)

header = ['Subject ID', 'True Group', 'Pred Group']
writer.writerow(header)

correct_sub = 0
seq_acc_sum = 0
total = 0

pred_list = []
target_list = []

with torch.no_grad():
    for step, batch in enumerate(test_loader):
        # get model output
        ids, x, y = (batch['id'], batch['x'].to(
            device), batch['y'].to(device))

        x = x.squeeze(0)  # put slided time windows as batch dimension
        y_true = y.item()
        y = y.expand(x.size(0))

        logit_map = model(x, device).squeeze(1)

        # calculate sequence accuracy
        seq_prob = torch.sigmoid(logit_map)
        seq_pred = torch.round(seq_prob)
        seq_acc = torch.eq(seq_pred, y).float().mean().item()
        seq_acc_sum += seq_acc

        # calculate subject accuracy
        seq_mean = seq_pred.mean().item()
        sub_pred = 1 if seq_mean >= 0.5 else 0
        correct_sub += (sub_pred == y_true)

        row = [ids[0], int(y_true), int(sub_pred)]
        writer.writerow(row)

        total += 1

        # append pred and target
        pred_list.append(seq_mean)
        # pred_list.append(seq_prob.mean().item())
        target_list.append(y_true)


# plot roc curve
fpr, tpr, threshold = metrics.roc_curve(target_list, pred_list)
roc_auc = metrics.auc(fpr, tpr)


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig(os.path.join(results_dir, f'{args.version}.png'))
print(f'AUC is {roc_auc}')


print(f'{correct_sub} is correct, Subject Accuracy: {correct_sub/total} Sequence Accuracy {seq_acc_sum/total}')
f.close()
