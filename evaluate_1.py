import torch
import torch.utils.data.DataLoader as DataLoader
import argparse

from model import PromptLRN, VCPromptLRN
from dataset import Dataset
from config import cfg

def top_k_acc(pred, y, top_k=1):
    if top_k == 1:
        acc = (pred.reshape(-1,) == y).sum() / y.shape[0] * 100
        return acc
    else:
        corr = 0
        for p, t in zip(pred, y):
            if t in p:
                corr += 1
        acc = corr / y.shape[0] * 100
        return acc 


if __name__ == '__main__':

    # parsing
    parser = argparse.ArgumentParser(description = 'evaluation settings')
    parser.add_argument('--dataset', required=True, help='dataset name')
    parser.add_argument('--epoch', required = True, default=100)
    parser.add_argument('--type', required=True, default='text')
    parser.add_argument('--kshot', required=True)
    parser.add_argument('--topk', required=True, default=1)
    args = parser.parse_args()

    # set model and evaluation dataloader 
    testset = Dataset(args.dataset, args.k_shot, train=False)
    testloader = DataLoader(testset, batch_size=100)
    model = PromptLRN(testset.labels, cfg)
    # load trained 
    model.load_state_dict(torch.load('./ckpt/promptlearn_{}/{}_shot/model_epoch{}.pt'.format(args.type, args.kshot, args.epoch)))
    model.eval()
    ys = torch.tensor(testset.df.labels.values)
    preds = torch.tensor([])
    # evaluation iteration
    with torch.no_grad():
        for step, pixel in enumerate(testloader):
            logits = model(pixel)
            pred = torch.topk(logits, topk=args.topk, dim=1).indices
            preds = torch.cat([preds, pred], dim=0)
        acc = top_k_acc(preds, ys, top_k = args.topk)
    
    print('top {} Accuracy on {} dataset with {} shot setting : {}%'.format(args.topk, args.dataset, args.kshot, acc))