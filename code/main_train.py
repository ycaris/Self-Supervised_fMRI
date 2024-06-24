import torch
import os
from torch import nn
from apex import amp
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from optimizer.lr_scheduler import WarmupCosineSchedule
from models.swin_transformer import SwinTransformerV2
from models.simple_transformer import SimpleTransformerClassification, SimpleTransformer
from models.whole_transformer import WholeTransformerClassification
from models.lstm import LSTMClassifier
from utils import data_util, parser_util
from train import train, save_ckp


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


def main(args):

    # Defining cuda device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    if torch.cuda.device_count() > 1 and 'cuda' in device.type and args['multi gpu']:
        model = nn.DataParallel(model)
        print('Using multiple GPUs')

    # Defining model:
    if args.model_arch == 'swin-transformer':
        model = SwinTransformerV2(img_size=224, patch_size=4, in_chans=1, num_classes=1,
                                  embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                                  window_size=7, mlp_ratio=4., qkv_bias=True,
                                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                                  use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]).to(device)
    elif args.model_arch == 'simple-transformer-classify':
        model = SimpleTransformerClassification(feature_size=args.feature_size,
                                                emb_dim=args.emb_dim,
                                                nhead=args.nhead,
                                                num_layers=args.n_layers,
                                                dropout=args.dropout,
                                                ).to(device)
    elif args.model_arch == 'whole-transformer-classify':
        model = WholeTransformerClassification(feature_size=args.feature_size,
                                               emb_dim=args.emb_dim,
                                               nhead=args.nhead,
                                               num_layers=args.n_layers,
                                               dropout=args.dropout,
                                               ).to(device)
    elif args.model_arch == 'lstm':
        model = LSTMClassifier(feature_size=args.feature_size,
                               emb_dim=args.emb_dim,
                               num_layers=args.n_layers,
                               dropout=args.dropout,
                               ).to(device)
    elif args.model_arch == 'simple-transformer':
        model = SimpleTransformer(feature_size=args.feature_size,
                                  emb_dim=args.emb_dim,
                                  nhead=args.nhead,
                                  num_layers=args.n_layers,
                                  dropout=args.dropout,
                                  time_period=args.time_period
                                  ).to(device)
    else:
        raise RuntimeError('Wrong dataset or model_arch parameter setting.')

    # load pretrained weights
    if args.use_pretrained:
        args.pretrained_dir = '/home/yz2337/project/multi_fmri/pretrain/runs/abide_all/v5_mlp_masktime/model_50000.pt'
        model.load_from(torch.load(args.pretrained_dir)['state_dict'])
        print('Use pretrained weights')

        # freeze the parameters

        for param in model.parameters():
            param.requires_grad = False  # Freeze all parameters first

        for param in model.decoder.parameters():
            param.requires_grad = True  # Ensure the decoder is trainable

    model.to(device)
    num_params = count_parameters(model)
    print("Total Parameter: \t%2.1fM" % num_params)

    # define optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(
            params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(params=model.parameters(
        ), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.amp:
        model, optimizer = amp.initialize(
            models=model, optimizers=optimizer, opt_level=args.opt_level)
        if args.amp_scale:
            amp._amp_state._scalers[0]._loss_scale = 2 ** 20

    if args.lrdecay:
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=100, gamma=0.8)

    # define tensorboard writer
    writer = SummaryWriter(logdir=args.savepath)

    # train_loader = data_util.get_train_loader(args)
    val_loader = data_util.get_val_loader(args)
    # print(f'{len(train_loader)} subjects for training, {len(val_loader)} subjects for testing')

    # loss_function = torch.nn.BCEWithLogitsLoss()  # L1 loss
    # loss_function = torch.nn.MSELoss()
    loss_function = torch.nn.MSELoss()  #

    global_step = 1

    # global_step, val_best = train(model, train_loader, val_loader, loss_function,
    #   scheduler, optimizer, device, writer, global_step, args)
    global_step, val_best = train(model, val_loader, loss_function,
                                  scheduler, optimizer, device, writer, global_step, args)

    checkpoint = {'global_step': global_step, 'state_dict': model.state_dict(
    ), 'optimizer': optimizer.state_dict()}

    save_ckp(checkpoint, args.savepath+'/model_final_epoch.pt')


if __name__ == '__main__':
    # argument parser
    parser = parser_util.prepare_parser()
    args = parser.parse_args()

    # set random seed
    parser_util.set_seed(args.seed)

    print(args)
    main(args)
