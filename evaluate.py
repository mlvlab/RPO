import torch
from torch.utils.data import DataLoader
import argparse

from model import PromptLRN, VCPromptLRN, VCMetaPromptLRN
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


def val_acc(model, device, dataset, topk):
    # set device
    device = torch.device(device)

    # set model and evaluation dataloader 
    valset = Dataset(dataset, train='val')
    valloader = DataLoader(valset, batch_size=100)
    model.eval().to(device)
    ys = torch.tensor(valset.df.labels.values)
    preds = torch.tensor([])
    # evaluation iteration
    with torch.no_grad():
        for step, pixel in enumerate(valloader):
            logits = model(pixel.to(device))
            pred = torch.topk(logits, k=topk, dim=1).indices
            preds = torch.cat([preds, pred], dim=0)
        acc = top_k_acc(preds, ys, top_k = topk)
    
    print('top {} Accuracy on validation : {}%'.format(topk, acc))


if __name__ == '__main__':

    # parsing
    parser = argparse.ArgumentParser(description = 'evaluation settings')
    parser.add_argument('--device', type=str, required = True, default='cpu')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--epoch', type=int, required = True, default=100)
    parser.add_argument('--type', type=str, required=True, default='text')
    parser.add_argument('--kshot', type=int, required=True)
    parser.add_argument('--topk', type=int, required=True, default=1)
    args = parser.parse_args()

    # set device
    device = torch.device(args.device)

    # set model and evaluation dataloader 
    testset = Dataset(args.dataset, args.kshot, train=False)
    testloader = DataLoader(testset, batch_size=100)
    if args.type == 'text':
        model = PromptLRN(testset.labels, cfg, args.device)
    elif args.type == 'text+vision':
        model = VCPromptLRN(testset.labels, cfg, args.device)
    elif args.type == 'text+vision_metanet':
        model = VCMetaPromptLRN(testset.labels, cfg, args.device)
    # load trained 
    state_dict = torch.load('./ckpt/{}_promptlearn_{}/{}_shot/model_epoch{}.pt'.format(args.dataset, args.type, args.kshot, args.epoch))
    model.load_state_dict(state_dict())
    model.eval().to(device)
    ys = torch.tensor(testset.df.labels.values)
    preds = torch.tensor([])
    # evaluation iteration
    with torch.no_grad():
        for step, pixel in enumerate(testloader):
            logits = model(pixel.to(device))
            pred = torch.topk(logits, k=args.topk, dim=1).indices
            preds = torch.cat([preds, pred], dim=0)
            if (step+1) % 10:
                print('{} images evaluated'.format(step * len(testloader)))
        acc = top_k_acc(preds, ys, top_k = args.topk)
    
    print('top {} Accuracy on {} dataset with {} shot setting : {}%'.format(args.topk, args.dataset, args.kshot, acc))