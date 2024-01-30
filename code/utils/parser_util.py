import argparse
import random
import numpy as np
import torch


def set_seed(seed=0):
	"""
	Sets all random seeds.
	:param seed: int
		Seed value.
	:return: None
	"""
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)


MODEL_ARCH = ['resnet18']
DATASET_NAME = ['CIFAR10', 'CIFAR100']
RULE = ['iid', 'Dirichlet']


def prepare_parser():
	usage = 'Parser for all scripts.'
	parser = argparse.ArgumentParser(description=usage)
	# Basic parameters
	# --seed    --model_arch    --method    --dataset   --exp_name  --n_class   --save  --savepath
	# --print_freq --save_period
	parser.add_argument('--seed',
						default=0,
						type=int,
						help='random seed')
	parser.add_argument('--device',
						default='cuda:0')
	parser.add_argument('--model_arch',
						default="resnet18", choices=MODEL_ARCH,
						type=str,
						help='which model architecture is utilized to train')
	parser.add_argument('--dataset', '-d',
						default="CIFAR10", choices=DATASET_NAME,
						type=str,
						help='which dataset is utilized to train')
	parser.add_argument('--exp_name', '-n',
						default="Federated",
						type=str,
						help='experiment name, used for saving results')
	parser.add_argument('--save', '-s',
						action='store_true',
						help='whether save the training results, default is False')
	parser.add_argument('--savepath',
						default='results/',
						type=str,
						help='directory to save exp results')
	parser.add_argument('--save_period',
						default=200,
						type=int,
						help='the frequency of saving the checkpoint')

	# Training parameters
	# --localE    --comm_amount   --active_frac   --bs    --n_minibatch    --optimizer --lr
	# --momentum  --weight_decay    --lr_decay    --coef_alpha    --mu    --sch_step  --sch_gamma
	parser.add_argument('--bs',
						default=50,
						type=int,
						help='batch size on each client')
	parser.add_argument('--n_minibatch',
						default=50,
						type=int,
						help='the number of minibatch size in SCAFFOLD')
	parser.add_argument('--optimizer',
						default='SGD',
						type=str,
						help='optimizer name')
	parser.add_argument('--lr', 
                     	default=0.1,
                      	type=float, 
                       	help='client learning rate')
	parser.add_argument('--momentum',
						default=0.0,
						type=float,
						help='local (client) momentum factor')
	parser.add_argument('--weight_decay', '-wd',
						default=1e-3,
						type=float,
						help='local (client) weight decay factor')
	parser.add_argument('--lr_decay', '-ld', default=0.998,
						type=float,
						help='local (client) learning rate decay factor')
	parser.add_argument('--sch_step',
						default=1,
						type=int,
						help='The learning rate scheduler step')
	parser.add_argument('--sch_gamma',
						default=1.0,
						type=float,
						help='The learning rate scheduler gamma')
	parser.add_argument('--pretrain',
						action='store_true')
	parser.add_argument('--use_pretrained',
						action='store_true')
	parser.add_argument('--pretrained_dir',
                     	default=None,
                      	type=str)

	# parser.add_argument('--logdir', default=None, type=str)
	# parser.add_argument('--pos_embedd', default='perceptron', type=str)
	# parser.add_argument('--norm_name', default='instance', type=str)
	# parser.add_argument('--gpu', default='0', type=str)
	# parser.add_argument('--num_steps', default=20000, type=int)
	# parser.add_argument('--eval_num', default=500, type=int)
	# parser.add_argument('--warmup_steps', default=0, type=int)
	# parser.add_argument('--num_heads', default=16, type=int)
	# parser.add_argument('--mlp_dim', default=3072, type=int)
	# parser.add_argument('--hidden_size', default=768, type=int)
	# parser.add_argument('--feature_size', default=16, type=int)
	# parser.add_argument('--in_channels', default=1, type=int)
	# parser.add_argument('--out_channels', default=2, type=int)
	# parser.add_argument('--num_classes', default=2, type=int)
	# parser.add_argument('--res_block', action='store_true')
	# parser.add_argument('--conv_block', action='store_true')
	# parser.add_argument('--featResBlock', action='store_true')
	# parser.add_argument('--roi_x', default=48, type=int)
	# parser.add_argument('--roi_y', default=48, type=int)
	# parser.add_argument('--roi_z', default=48, type=int)
	# parser.add_argument('--batch_size', default=1, type=int)
	# parser.add_argument('--dropout_rate', default=0.0, type=float)
	# parser.add_argument('--fold', default=0, type=int)
	# parser.add_argument('--sw_batch_size', default=4, type=int)
	# parser.add_argument('--lr', default=1e-4, type=float)
	# parser.add_argument('--decay', default=1e-5, type=float)
	# parser.add_argument('--momentum', default=0.9, type=float)
	# parser.add_argument('--lrdecay', action='store_true')
	# parser.add_argument('--clara_split', action='store_true')
	# parser.add_argument('--amp', action='store_true')
	# parser.add_argument('--amp_scale', action='store_true')
	# parser.add_argument('--max_grad_norm', default=1.0, type=float)
	# parser.add_argument('--val', action='store_true')
	# parser.add_argument('--model_type', default='nest_unetr', type=str)
	# parser.add_argument('--opt_level', default='O2', type=str)
	# parser.add_argument('--loss_type', default='dice_ce', type=str)
	# parser.add_argument('--opt', default='adamw', type=str)

	return parser
