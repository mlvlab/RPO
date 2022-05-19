from model import PromptOptim
from config import cfg
import argparse
import torch

if __name__ == '__main__':
    torch.manual_seed(2022)
    device = torch.device('mps')
    parser = argparse.ArgumentParser(description = 'training settings')
    parser.add_argument('--dataset', required=True, help='dataset name', type=str)
    parser.add_argument('--type', required=True, default='text', type=str)
    parser.add_argument('--kshot', required=True, type=int)
    parser.add_argument('--start_epoch', required=True, type=int)
    args = parser.parse_args()

    proptim = PromptOptim(cfg, device, args.dataset, args.kshot, args.type , args.start_epoch)
    proptim.train()