import numpy as np
import torch
import os
import argparse
from utils.transform import *
from utils.data_util import get_test_loader, get_val_loader
from utils import parser_util
from models.simple_transformer import SimpleTransformerClassification, SimpleTransformer
from models.whole_transformer import WholeTransformerClassification
from models.lstm import LSTMClassifier
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv


# parser = parser_util.prepare_parser()

# ace_fold = 3
# percent = 100
# version = 'v5_pretrain_masktime2'
# parser.add_argument(
#     '--version', default=f'{version}', type=str)
# parser.add_argument('--checkpoint_path',
#                     default=f'/home/yz2337/project/multi_fmri/run_ace/abide_all/ace_fold{ace_fold}/{version}/model.pt', type=str)
# parser.add_argument(
#     '--save_dir', default=f'/home/yz2337/project/multi_fmri/results_ace/finetune/abide_all', type=str)
# # parser.add_argument('--version', default=f'{version}_ace_fold{fold}', type=str)
# # parser.add_argument('--checkpoint_path',
# #                     default=f'/home/yz2337/project/multi_fmri/runs/abide_percent_cv/pretrain_fold{pretrain_fold}/{version}/fold{fold}/model.pt', type=str)
# # parser.add_argument(
# #     '--save_dir', default=f'/home/yz2337/project/multi_fmri/code/run_ace/results/transfer/{percent}/fold{fold}', type=str)


# args = parser.parse_args()
# torch.backends.cudnn.benchmark = True

# args.json_file = f'/home/yz2337/project/multi_fmri/code/json_files/ace/cls/fold_{ace_fold}.json'


# # specify the GPU id's, GPU id's start from 0.
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# # -----------------------------------------------------------------

# # set the output saving folder
# results_dir = os.path.join(args.save_dir, str(ace_fold), args.version)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)


# # get the test data loader
# test_loader = get_test_loader(args)

# # set the saved model path
# checkpoint_dir = args.checkpoint_path

# # load model
# model = SimpleTransformerClassification(feature_size=args.feature_size,
#                                         emb_dim=args.emb_dim,
#                                         num_layers=args.n_layers,
#                                         dropout=args.dropout,
#                                         ).to(device)
# # model = WholeTransformer(feature_size=args.feature_size,
# #                          emb_dim=args.emb_dim,
# #                          nhead=args.nhead,
# #                          num_layers=args.n_layers,
# #                          dropout=args.dropout,
# #                          time_period=args.time_period
# #                          ).to(device)

# # model = LSTMClassifier(feature_size=args.feature_size,
# #                        emb_dim=args.emb_dim,
# #                        num_layers=args.n_layers,
# #                        dropout=args.dropout,
# #                        ).to(device)

# ckpt = torch.load(checkpoint_dir, map_location='cpu')
# model.load_state_dict(ckpt['state_dict'], strict=True)
# model.to(device)
# model.eval()

# # csv writer
# filename = os.path.join(results_dir, f'{args.version}.csv')
# f = open(filename, 'w')
# writer = csv.writer(f)

# header = ['Subject ID', 'True Group', 'Pred Group']
# writer.writerow(header)

# tp, tn, fn, fp = 0, 0, 0, 0
# seq_acc_sum = 0
# total = 0

# pred_list = []
# target_list = []

# with torch.no_grad():
#     for step, batch in enumerate(test_loader):
#         # get model output

#         ids, x, y = (batch['id'], batch['x'].to(
#             device), batch['y'].to(device))

#         x = x.squeeze(0)  # put slided time windows as batch dimension
#         y_true = y.item()
#         y = y.expand(x.size(0))

#         logit_map = model(x, device).squeeze(1)

#         # calculate sequence accuracy
#         seq_prob = torch.sigmoid(logit_map)
#         seq_pred = torch.round(seq_prob)
#         seq_acc = torch.eq(seq_pred, y).float().mean().item()
#         seq_acc_sum += seq_acc

#         # calculate subject accuracy
#         seq_mean = seq_pred.mean().item()
#         sub_pred = 1 if seq_mean >= 0.5 else 0

#         # calculate specifity and sensitivity
#         if sub_pred == 1 and y_true == 1:
#             tp += 1
#         elif sub_pred == 1 and y_true == 0:
#             fp += 1
#         elif sub_pred == 0 and y_true == 0:
#             tn += 1
#         elif sub_pred == 0 and y_true == 1:
#             fn += 1

#         row = [ids[0], int(y_true), int(sub_pred)]
#         writer.writerow(row)

#         total += 1

#         # append pred and target
#         pred_list.append(seq_mean)
#         # pred_list.append(seq_prob.mean().item())
#         target_list.append(y_true)


# # plot roc curve
# fpr, tpr, threshold = metrics.roc_curve(target_list, pred_list)
# roc_auc = metrics.auc(fpr, tpr)

# # Create a DataFrame to hold FPR, TPR, and Threshold
# data = {'FPR': fpr, 'TPR': tpr, 'Threshold': threshold}


# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
# plt.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.savefig(os.path.join(results_dir, f'{args.version}.png'))
# print(f'AUC is {roc_auc}, Sub ACC is {(tn+tp)/total}, Sensitvity is {tp/(tp+fn)}, Specificity is {tn/(tn+fp)}')
# print(f'Sequence Accuracy {seq_acc_sum/total}')

# # print(f'{correct_sub} is correct, Subject Accuracy: {correct_sub/total} Sequence Accuracy {seq_acc_sum/total}')
# f.close()


parser = parser_util.prepare_parser()

ace_fold = 1
percent = 100
version = 'v5_pretrain_masktime'
parser.add_argument(
    '--version', default=f'{version}', type=str)
parser.add_argument('--checkpoint_path',
                    default=f'/home/yz2337/project/multi_fmri/run_ace/abide_all_srs/ace_fold{ace_fold}/{version}/model.pt', type=str)
parser.add_argument(
    '--save_dir', default=f'/home/yz2337/project/multi_fmri/results_ace/finetune/abide_all_srs', type=str)
# parser.add_argument('--version', default=f'{version}_ace_fold{fold}', type=str)
# parser.add_argument('--checkpoint_path',
#                     default=f'/home/yz2337/project/multi_fmri/runs/abide_percent_cv/pretrain_fold{pretrain_fold}/{version}/fold{fold}/model.pt', type=str)
# parser.add_argument(
#     '--save_dir', default=f'/home/yz2337/project/multi_fmri/code/run_ace/results/transfer/{percent}/fold{fold}', type=str)


args = parser.parse_args()
torch.backends.cudnn.benchmark = True

args.json_file = f'/home/yz2337/project/multi_fmri/code/json_files/ace/srs/fold_{ace_fold}.json'


# specify the GPU id's, GPU id's start from 0.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------

# set the output saving folder
results_dir = os.path.join(args.save_dir, f'fold{ace_fold}', args.version)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)


# get the test data loader
test_loader = get_test_loader(args)

# set the saved model path
checkpoint_dir = args.checkpoint_path

# load model
model = SimpleTransformer(feature_size=args.feature_size,
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

header = ['Subject ID', 'True SRS', 'Pred SRS']
writer.writerow(header)

mse_sum = 0
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

        # calculate subject accuracy
        final_srs = logit_map.mean().item()

        row = [ids[0], y_true, final_srs]
        writer.writerow(row)

        total += 1
        mse_sum += (y_true-final_srs)**2

print(f'SRS MSE is {mse_sum/total}')
f.close()
