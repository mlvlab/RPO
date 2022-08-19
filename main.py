from model import PromptOptim
from config import cfg
import argparse
import torch
import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True, type=str, help = 'cpu, mps, cuda...')
    parser.add_argument('--dataset', required=True, help='dataset name', type=str)
    parser.add_argument('--type', required=True, default='text', type=str, help = 'one of text | text+vision | text+vision_metanet')
    parser.add_argument('--layer', default = None, type=int, help = 'layer to feed in visual prompt')
    parser.add_argument('--kshot', required=True, type=int, help = '# of shots for few-shot setting')
    parser.add_argument('--start_epoch', required=True, type=int)
    parser.add_argument('--division', required=True, default='base', type=str, help = 'one of entire | base')
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--train_textprompt', required=True, type=str)
    parser.add_argument('--regularize_vprompt', required=True, type=str)
    args = parser.parse_args()

    if args.train_textprompt == 'y':
        cfg.train.train_textprompt = True
    else:
        cfg.train.train_textprompt = False
    
    if args.regularize_vprompt == 'y':
        cfg.train.visualreg = True
    else:
        cfg.train.visualreg = False

    if (cfg.model.prefix is not None) & (not cfg.train.train_textprompt):
        if args.dataset == 'eurosat':
            cfg.model.prefix = 'a centered satellite photo of _'
        elif args.dataset == 'caltech101':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'oxfordpets':
            cfg.model.prefix = 'a photo of a _, a type of pet'
        elif args.dataset == 'stanfordcars':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'imagenet':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'flowers102':
            cfg.model.prefix = 'a photo of a _, a type of flower'
        elif args.dataset == 'food101':
            cfg.model.prefix = 'a photo of a _, a type of food'
        elif args.dataset == 'fgvcaircraft':
            cfg.model.prefix = 'a photo of a _, a type of aircraft'
        elif args.dataset == 'sun397':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'dtd':
            cfg.model.prefix = '_ texture'
        elif args.dataset == 'ucf101':
            cfg.model.prefix = 'a photo of a person doing _'

    print(cfg)

    # cfg.model.prefix = 'a photo of a _'
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    if args.division == 'base':
        # train with only base classes
        proptim = PromptOptim(cfg, device, args.layer, args.dataset, args.kshot, args.type , args.start_epoch, only_base=True, seed=args.seed)
    else:
        # train with entire classes
        proptim = PromptOptim(cfg, device, args.layer,  args.dataset, args.kshot, args.type , args.start_epoch, only_base=False, seed=args.seed)
    proptim.train()