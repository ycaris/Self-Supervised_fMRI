import numpy as np
import torch
import os
import argparse
from utils.transform import *
from utils.data_util import get_test_loader
from utils import parser_util
from models.simple_transformer import SimpleTransformer
from tqdm import tqdm
import csv


# testing parameter
parser = parser_util.prepare_parser()
parser.add_argument('--version', default='fold1.v0', type=str)
parser.add_argument('--checkpoint_path',
                    default='/home/yz2337/project/multi_fmri/runs/fold1.v0/model.pt', type=str)
parser.add_argument(
    '--save_dir', default='/home/yz2337/project/multi_fmri/results', type=str)
args = parser.parse_args()
torch.backends.cudnn.benchmark = True
args.json_file = '/home/yz2337/project/multi_fmri/code/json_files/fold_1_points.json'

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
model = SimpleTransformer(feature_size=args.feature_size,
                          emb_dim=args.emb_dim,
                          num_layers=args.n_layers,
                          dropout=args.dropout,
                          ).to(device)

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

correct = 0
total = 0

with torch.no_grad():
    for step, batch in enumerate(test_loader):
        # get model output
        ids, x, y = (batch['id'], batch['x'].to(
            device), batch['y'].to(device))

        logit_map = model(x, device).squeeze()
        y_pred = torch.round(torch.sigmoid(logit_map))

        row = [ids[0], int(y.item()), int(y_pred.item())]
        writer.writerow(row)

        # calculate the accuracy
        total += y.size(0)
        correct += (y_pred == y).sum().item()


print(f'{correct} is correct, Accuracy: {correct/total}')
f.close()
