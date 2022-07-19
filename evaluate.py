import torch
from torch.utils.data import DataLoader
import argparse

from model import PromptLRN, VTPromptLRN, VTMetaPromptLRN
from dataset import UnseenDataset
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
    parser.add_argument('--division', type=str, required=True, default='entire')
    parser.add_argument('--kshot', type=int, required=True)
    parser.add_argument('--topk', type=int, required=True, default=1)
    args = parser.parse_args()

    # set device
    device = torch.device(args.device)

    # set evaluation dataloader 
    if args.division == 'entire':
        testset = UnseenDataset(args.dataset, args.kshot, 'test', cfg.train.base_label_ratio, test_time='entire')
    elif args.division == 'base':
        testset = UnseenDataset(args.dataset, args.kshot, 'test', cfg.train.base_label_ratio, test_time='base')
    elif args.division == 'novel':
        testset = UnseenDataset(args.dataset, args.kshot, 'test', cfg.train.base_label_ratio, test_time='novel')
    testloader = DataLoader(testset, batch_size=100)
    
    # set model 
    # evaluate with novel classes
    if args.division == 'novel':
        if args.type == 'text':
            model = PromptLRN(testset.novel_labels, cfg, device)
        elif args.type == 'text+vision':
            model = VTPromptLRN(testset.novel_labels, cfg, device)
        elif args.type == 'text+vision_metanet':
            model = VTMetaPromptLRN(testset.novel_labels, cfg, device)
    
    # evaluate with base classes(classes used for training)
    elif args.division == 'base':
        if args.type == 'text':
            model = PromptLRN(testset.base_labels, cfg, device)
        elif args.type == 'text+vision':
            model = VTPromptLRN(testset.base_labels, cfg, device)
        elif args.type == 'text+vision_metanet':
            model = VTMetaPromptLRN(testset.base_labels, cfg, device)

    # evaluate with entire classes(trained with entire classes)
    elif args.division == 'entire':
        if args.type == 'text':
            model = PromptLRN(testset.labels, cfg, device)
        elif args.type == 'text+vision':
            model = VTPromptLRN(testset.labels, cfg, device)
        elif args.type == 'text+vision_metanet':
            model = VTMetaPromptLRN(testset.labels, cfg, device)
    
    # load trained 
    state_dict = torch.load('./ckpt/{}_promptlearn_{}/{}_shot/model_epoch{}.pt'.format(args.dataset, args.type, args.kshot, args.epoch),
                            map_location=device)
    model.load_state_dict(state_dict())
    if device == torch.device('cpu'):
        model = model.type(torch.float32)
    model.to(device)
    if args.division == 'entire':
        ys = torch.tensor(testset.df.labels.values)
    elif args.division == 'base':
        ys = torch.tensor(testset.base_df.labels.values)
    if args.division == 'novel':
        ys = torch.tensor(testset.novel_df.labels.values)
        ys = ys - torch.min(ys)
    preds = torch.tensor([])
    
    # evaluation iteration
    with torch.no_grad():
        print(len(testloader))
        for step, pixel in enumerate(testloader):
            logits = model(pixel.type(torch.float32))
            pred = torch.topk(logits, k=args.topk, dim=1).indices
            preds = torch.cat([preds, pred], dim=0)
            if (step+1) % 10:
                print('{} images evaluated'.format(step * testloader.batch_size))
        acc = top_k_acc(preds, ys, top_k = args.topk)
    
    print('top {} Accuracy on {} dataset with {} shot setting ({} classes): {}%'.format(args.topk, args.dataset, args.kshot, args.division, acc))