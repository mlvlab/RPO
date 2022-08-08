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
    args = parser.parse_args()

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