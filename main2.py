from trainer import IncPromptOptim
from config import cfg
import argparse
import random
import torch
import numpy as np
import pandas as pd

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', required=True, type=str, help = 'cpu, mps, cuda...')
    parser.add_argument('--layer', default = None, type=int, help = 'layer to feed in visual prompt')
    parser.add_argument('--start_epoch', required=True, type=int)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--train_textprompt', required=True, type=str)
    args = parser.parse_args()

    if args.train_textprompt == 'y':
        cfg.train.train_textprompt = True
    else:
        cfg.train.train_textprompt = False
    

    cfg.seed = args.seed
    cfg.device = args.device
    cfg.train.start_epoch = args.start_epoch

    print(cfg)

    # cfg.model.prefix = 'a photo of a _'
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(cfg.device)

    proptim = IncPromptOptim(cfg, device, n_tasks=10)
    proptim.train()