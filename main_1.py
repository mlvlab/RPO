from model import PromptOptim
from config import cfg
import argparse
import torch

if __name__ == '__main__':
    torch.manual_seed(2022)

    parser = argparse.ArgumentParser(description = 'training settings')
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--type', required=True, default='text')
    parser.add_argument('--kshot', required=True)
    parser.add_argument('--start_epoch', required=True)
    args = parser.parse_args()

    proptim = PromptOptim(cfg, args.dataset, args.kshot, args.type , args.start_epoch)
    proptim.train()