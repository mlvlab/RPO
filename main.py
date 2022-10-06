from trainer import PromptOptim
from config import cfg
import argparse
import random
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

    if args.type == 'coop':
        cfg.train.n_epochs = 200
        cfg.train.batch_size = 32
    else:
        cfg.train.n_epochs = 10
        cfg.train.batch_size = 1

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
        elif args.dataset == 'oxfordpet':
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
    '''
    if (cfg.model.prefix is not None) & (not cfg.train.train_textprompt):
        if args.dataset == 'eurosat':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'caltech101':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'oxfordpet':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'stanfordcars':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'imagenet':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'flowers102':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'food101':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'fgvcaircraft':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'sun397':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'dtd':
            cfg.model.prefix = 'a photo of a _'
        elif args.dataset == 'ucf101':
            cfg.model.prefix = 'a photo of a _'
    '''

    cfg.seed = args.seed
    cfg.device = args.device
    cfg.model.type = args.type
    cfg.train.start_epoch = args.start_epoch
    cfg.dataset.name = args.dataset
    cfg.dataset.kshot = args.kshot
    cfg.dataset.division = args.division

    print(cfg)

    # cfg.model.prefix = 'a photo of a _'
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(cfg.device)

    if cfg.dataset.division == 'base':
        # train with only base classes
        proptim = PromptOptim(cfg, device, only_base=True)
    else:
        # train with entire classes
        proptim = PromptOptim(cfg, device, only_base=False)
    proptim.train()