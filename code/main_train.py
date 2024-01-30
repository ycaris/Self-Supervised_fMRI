import torch
import torch.optim as optim
import os
from torch import nn
from apex import amp
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from model import RDUNet
from transforms import AdditiveWhiteGaussianNoise, RandomHorizontalFlip, RandomVerticalFlip, RandomRot90
from utils import data_util, parser_util, data_monai
from train import train, save_ckp

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def main(args):

    # Defining cuda deice
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))
    if torch.cuda.device_count() > 1 and 'cuda' in device.type and args['multi gpu']:
        model = nn.DataParallel(model)
        print('Using multiple GPUs')


    # Defining model:
    if args.model_arch == 'UNETR':
        model = UNETR(in_channel=1,
                    out_channel=2).to(device)
    elif args.model_arch == 'resnet50':
        model = resnet50(n_input_channels=2,
                num_classes=6,
                pretrained=False,
            ).to(device) 
    elif args.model_arch == 'unetr':
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(48,48,48),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embedd,
            norm_name=args.norm_name,
            conv_block=args.conv_block,
            res_block=args.res_block,
            dropout_rate=0.0).to(device)
    else:
        raise RuntimeError('Wrong dataset or model_arch parameter setting.')

    # load pretrained weights
    if args.pretrain:
        args.pretrained_dir = './pretrained_models/'+args.model_type+'.npz'
        model.load_from(weights=torch.load(args.pretrained_dir))
        print('Use pretrained weights')
    model.to(device)    
    num_params = count_parameters(model)
    print("Total Parameter: \t%2.1fM" % num_params)
    
    # define optimizer
    if args.opt == "adam":
        optimizer = torch.optim.Adam(params = model.parameters(), lr=args.lr,weight_decay= args.decay)

    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(params = model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.amp:
        model, optimizer = amp.initialize(models=model,optimizers=optimizer,opt_level=args.opt_level)
        if args.amp_scale:
            amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    if args.lrdecay:
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)
    
    
    # define tensorboard writer
    writer = SummaryWriter(logdir=args.savepath)
    
    train_loader, val_loader = data_util.get_loader(args)
    
    loss_function = torch.nn.L1Loss() #L1 loss
    
    global_step, val_best = train(model, train_loader, val_loader, loss_function, 
                                  scheduler, optimizer, device, writer, args)

        
    checkpoint = {'global_step': global_step,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
    
    save_ckp(checkpoint, args.logdir+'/model_final_epoch.pt')


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    # argument parser 
    parser = parser_util.prepare_parser()
    args = parser.parse_args()

    # set random seed
    parser_util.set_seed(args['seed'])

    print(args)
    main(args)