from model import PromptOptim
from config import cfg
import argparse

if __name__ == '__main__':
    proptim = PromptOptim(cfg,  type = 'text', start_epoch=0)
    proptim.train()