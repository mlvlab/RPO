import torch
from torch.utils.data import DataLoader
import argparse

from model import CoOp, CoCoOp, VisualCoOp, VisualCoCoOpv2, VisualCoCoOpv1, ZSCLIP
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
    parser.add_argument('--layer', default = None, type=int, help = 'layer to feed in visual prompt')
    parser.add_argument('--type', type=str, required=True, default='text')
    parser.add_argument('--division', type=str, required=True, default='entire')
    parser.add_argument('--kshot', type=int, required=True)
    parser.add_argument('--topk', type=int, required=True, default=1)
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

    # set device
    device = torch.device(args.device)

    # set evaluation dataloader 
    if args.division == 'entire':
        testset = UnseenDataset(args.dataset, args.kshot, 'test', cfg.train.base_label_ratio, test_time='entire')
    elif args.division == 'base':
        testset = UnseenDataset(args.dataset, args.kshot, 'test', cfg.train.base_label_ratio, test_time='base')
    elif args.division == 'novel':
        testset = UnseenDataset(args.dataset, args.kshot, 'test', cfg.train.base_label_ratio, test_time='novel')
    
    if args.type == 'cocoop':
        testloader = DataLoader(testset, batch_size=1)
    else:
        testloader = DataLoader(testset, batch_size=100)
    # set model 
    # evaluate with novel classes
    if args.division == 'novel':
        if args.type == 'zsclip':
            model = ZSCLIP(testset.novel_labels, cfg, device, prefix = cfg.model.prefix)
        elif args.type == 'coop':
            model = CoOp(testset.novel_labels, cfg, device)
        elif args.type == 'cocoop':
            model = CoCoOp(testset.novel_labels, cfg, device, prefix=cfg.model.prefix)
        elif args.type == 'visualcoop':
            model = VisualCoOp(testset.novel_labels, cfg, device, args.layer)
        elif args.type == 'visualcocoopv1':
            model = VisualCoCoOpv1(testset.novel_labels, cfg, device, args.layer, prefix=cfg.model.prefix)
        elif args.type == 'visualcocoopv2':
            model = VisualCoCoOpv2(testset.novel_labels, cfg, device, args.layer, prefix =cfg.model.prefix)
    
    # evaluate with base classes(classes used for training)
    elif args.division == 'base':
        if args.type == 'zsclip':
            model = ZSCLIP(testset.base_labels, cfg, device, prefix = cfg.model.prefix)
        elif args.type == 'coop':
            model = CoOp(testset.base_labels, cfg, device)
        elif args.type == 'cocoop':
            model = CoCoOp(testset.base_labels, cfg, device, prefix=cfg.model.prefix)
        elif args.type == 'visualcoop':
            model = VisualCoOp(testset.base_labels, cfg, device, args.layer)
        elif args.type == 'visualcocoopv1':
            model = VisualCoCoOpv1(testset.base_labels, cfg, device, args.layer, prefix=cfg.model.prefix)
        elif args.type == 'visualcocoopv2':
            model = VisualCoCoOpv2(testset.base_labels, cfg, device, args.layer, prefix=cfg.model.prefix)

    # evaluate with entire classes(trained with entire classes)
    elif args.division == 'entire':
        if args.type == 'zsclip':
            model = ZSCLIP(testset.labels, cfg, device, prefix = cfg.model.prefix)
        elif args.type == 'coop':
            model = CoOp(testset.labels, cfg, device)
        elif args.type == 'cocoop':
            model = CoCoOp(testset.labels, cfg, device, prefix=cfg.model.prefix)
        elif args.type == 'visualcoop':
            model = VisualCoOp(testset.labels, cfg, device, args.layer)
        elif args.type == 'visualcocoopv1':
            model = VisualCoCoOpv1(testset.labels, cfg, device, args.layer, prefix=cfg.model.prefix)
        elif args.type == 'visualcocoopv2':
            model = VisualCoCoOpv2(testset.labels, cfg, device, args.layer, prefix=cfg.model.prefix)
    
    # load trained ckpt
    if args.type != 'zsclip':
        state_dict = torch.load('./ckpt/{}_promptlearn_{}/{}_shot/model_epoch{}_traintext{}_visualreg{}_seed{}.pt'.format(args.dataset, args.type, args.kshot, args.epoch, cfg.train.train_textprompt,  cfg.train.visualreg, args.seed),
                                map_location=device)
        model.load_state_dict(state_dict())

    if device == torch.device('cpu'):
        model = model.type(torch.float32)
    model.to(device)

    model.eval()

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
        for step, pixel in enumerate(testloader):
            logits = model(pixel.type(torch.float32))
            logits = logits.to(torch.device('cpu'))
            pred = torch.topk(logits, k=args.topk, dim=1).indices
            preds = torch.cat([preds, pred], dim=0)
            #if (step+1) % 10:
            #    print('{} images evaluated'.format(step * testloader.batch_size))
        acc = top_k_acc(preds, ys, top_k = args.topk)
    
    print('top {} Accuracy on {} dataset with {} shot setting ({} classes, model type:{}, train textprompt:{}, reg_visualprompt:{}, seed :{}): {}%'.format(
        args.topk, args.dataset, args.kshot, args.division, args.type, cfg.train.train_textprompt, cfg.train.visualreg, args.seed, acc))