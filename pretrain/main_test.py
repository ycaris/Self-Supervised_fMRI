import numpy as np
import scipy.io
import torch
import os
import argparse
from utils.transform import *
from utils.data_util import get_test_loader, get_val_loader
from utils import parser_util
from models.simple_transformer import SimpleTransformer
from models.whole_transformer import WholeTransformer
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import torch.nn.functional as F


# testing parameter
version = 'v3_mlp_whole_64_sm'
parser = parser_util.prepare_parser()
parser_util.set_seed(0)
parser.add_argument('--version', default=f'{version}', type=str)
parser.add_argument('--checkpoint_path',
                    default=f'/home/yz2337/project/multi_fmri/pretrain/runs/percentage/v3_mlp_whole_64_sm/model.pt', type=str)
args = parser.parse_args()
torch.backends.cudnn.benchmark = True
args.json_file = f'/home/yz2337/project/multi_fmri/code/json_files/pretrained/abide_group.json'

# specify the GPU id's, GPU id's start from 0.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------------------------

# set the output saving folder
save_dir = f'/home/yz2337/project/multi_fmri/pretrain/output/{version}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# get the test data loader
test_loader = get_val_loader(args)

# set the saved model path
checkpoint_dir = args.checkpoint_path

# load model
model = SimpleTransformer(feature_size=args.feature_size,
                          emb_dim=args.emb_dim,
                          nhead=args.nhead,
                          num_layers=args.n_layers,
                          dropout=args.dropout,
                          time_period=args.time_period
                          ).to(device)

# model = WholeTransformer(feature_size=args.feature_size,
#                          emb_dim=args.emb_dim,
#                          nhead=args.nhead,
#                          num_layers=args.n_layers,
#                          dropout=args.dropout,
#                          time_period=args.time_period
#                          ).to(device)

ckpt = torch.load(checkpoint_dir, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=True)
model.to(device)
model.eval()

# # csv writer
# filename = os.path.join(results_dir, f'{args.version}.csv')
# f = open(filename, 'w')
# writer = csv.writer(f)

# header = ['Subject ID', 'True Group', 'Pred Group']
# writer.writerow(header)

total = 0

pred_list = []
target_list = []

with torch.no_grad():
    total = 0
    total_loss = 0

    for step, batch in enumerate(test_loader):

        # get model output
        ids, x, y = (batch['id'], batch['x'].to(
            device), batch['y'].to(device))

        logit_map = model(x, device)

        total += y.size(0)

        diff = F.mse_loss(logit_map, y)
        total_loss += diff

        # save the output of time periods every 50 subjects
        if step % 50 == 0:
            out = logit_map.squeeze().cpu().detach().numpy()
            scipy.io.savemat(os.path.join(
                save_dir, f'{ids}_{step}.mat'), {'mydata': out})
            true = y.squeeze().cpu().detach().numpy()
            scipy.io.savemat(os.path.join(
                save_dir, f'{ids}_{step}_true.mat'), {'mydata': true})


print(total_loss/total)
