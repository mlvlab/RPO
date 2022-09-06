import os
import collections
from collections import OrderedDict

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR, _LRScheduler

import clip

from dataset import UnseenDataset
from lr_scheduler import ConstantWarmupScheduler
from model import VisualCoCoOpv1, VisualCoCoOpv2, VisualCoCoOpv3, CoCoOp, CoOp, CoOpv2, VisualCoCoOpv4

# Prompt Optmizer : Trainer
class PromptOptim(object):
    def __init__(self, cfg, device, L=None, dataset = None, kshot = None, type = 'coop', start_epoch = 0, val = False, only_base = True, seed=2022):
        super(PromptOptim, self).__init__()
        # set configuration
        self.cfg = cfg
        self.type = type
        self.kshot = kshot
        self.dataset = dataset
        self.start_epoch = start_epoch
        self.device = device
        self.val = val
        self.type = type
        self.seed = seed

        # set batch size
        batch_size = cfg.train.cocoop_batch if type == 'cocoop' else cfg.train.batch_size
        self.n_epochs = cfg.train.cocoop_epochs if type == 'cocoop' else cfg.train.n_epochs

        # set dataloader
        if only_base:
            self.dataloader = torch.utils.data.DataLoader(UnseenDataset(dataset=dataset,
                                                                        base_label_ratio=self.cfg.train.base_label_ratio,
                                                                        k_shot=kshot,
                                                                        mode='train',
                                                                        train_time='base',
                                                                        device = self.device),
                                                                batch_size = batch_size,
                                                                shuffle = True)
        else:
            self.dataloader = torch.utils.data.DataLoader(UnseenDataset(dataset=dataset,
                                                                    base_label_ratio=self.cfg.train.base_label_ratio,
                                                                    k_shot=kshot,
                                                                    mode='train',
                                                                    train_time='entire',
                                                                    device = self.device),
                                                                batch_size = batch_size, 
                                                                shuffle = True)
    
        # define model
        # if want to train with only base classes
        if only_base:
            if type == 'coop':
                self.model = CoOp(self.dataloader.dataset.base_labels, cfg, device)
            elif type == 'cocoop':
                self.model = CoCoOp(self.dataloader.dataset.base_labels, cfg, device, prefix=self.cfg.model.prefix)
            elif type == 'visualcocoopv1':
                self.model = VisualCoCoOpv1(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif type == 'visualcocoopv2':
                self.model = VisualCoCoOpv2(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif type == 'visualcocoopv3':
                self.model = VisualCoCoOpv3(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif type == 'visualcocoopv4':
                self.model = VisualCoCoOpv4(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif type == 'coopv2':
                self.model = CoOpv2(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix, alpha=self.cfg.model.alpha)
        # if want to train with entire classes
        else:
            if type == 'coop':
                self.model = CoOp(self.dataloader.dataset.labels, cfg, device)
            elif type == 'cocoop':
                self.model = CoCoOp(self.dataloader.dataset.labels, cfg, device, prefix = self.cfg.model.prefix)
            elif type == 'visualcocoopv1':
                self.model = VisualCoCoOpv2(self.dataloader.dataset.labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif type == 'visualcocoopv2':
                self.model = VisualCoCoOpv2(self.dataloader.dataset.labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif type == 'visualcocoopv3':
                self.model = VisualCoCoOpv3(self.dataloader.dataset.labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif type == 'visualcocoopv4':
                self.model = VisualCoCoOpv4(self.dataloader.dataset.labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif type == 'coopv2':
                self.model = CoOpv2(self.dataloader.dataset.labels, cfg, device, L, prefix = self.cfg.model.prefix, alpha=self.cfg.model.alpha)
        self.model.to(device)

        #if self.device == torch.device('cpu'):
        self.model = self.model.type(torch.float32)
        # freeze weight
        for n, param in self.model.named_parameters():
            if ('meta_net.meta_linear' not in n) and ('prompt' not in n):
                param.requires_grad = False

        # set optimizer & lr scheduler
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.train.max_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.cfg.train.n_epochs)
        self.lr_sched = ConstantWarmupScheduler(self.optimizer, scheduler, self.cfg.train.warmup_epoch, self.cfg.train.base_lr)
        
        # load pretrained model / optimizer / lr_scheduler
        if start_epoch > 5:
            self.model.load_state_dict(torch.load('ckpt/{}_promptlearn_{}/{}_shot/model_epoch{}.pt'.format(self.dataset, self.type, self.kshot, self.start_epoch))())
            # self.optimizer.load_state_dict(torch.load('ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, self.start_epoch)))
            self.lr_sched.load_state_dict(torch.load('ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, self.start_epoch))())

        # set loss function
        self.criterion = nn.CrossEntropyLoss()

        # check trainable parameters
        trainable = []
        for n, param in self.model.named_parameters():
            if param.requires_grad:
                trainable.append(n)
        print('trainable params : {}'.format(trainable))
        print('# labels for training : {}'.format(len(self.model.labels)))

    def train(self):
        history = []
        for epoch in range(self.start_epoch, self.n_epochs):
            print('-'*10 + 'Epoch {}'.format(epoch+1)+'-'*10)
            epoch_loss = 0
            print('current lr : {}'.format(self.lr_sched.get_lr()[0]))
            for step, (img, label) in enumerate(self.dataloader):
                logits = self.model(img) # (batch_size, n_cls)

                loss = self.criterion(logits, label.to(self.device))
                # l2 regularization on visual prompt
                if self.cfg.train.visualreg & (self.type in ['visualcocoopv1', 'visualcocoopv2', 'visualcocoopv3', 'visualcocoopv4']):
                     loss = loss + self.model.v_prompt_emb.norm()# + self.model.meta_net.meta_linear1.weight.norm() + self.model.meta_net.meta_linear2.weight.norm()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.lr_sched.step()
                epoch_loss += loss.item()
                history.append(epoch_loss / (step+1))
                # verbosity
                #if ((step+1) % 10) == 0:
                print('| {} / {} | train loss : {}'.format(step+1, len(self.dataloader), epoch_loss/(step+1)))
    
            # save checkpoint
            #if epoch == (self.n_epochs - 1):
            if (epoch+1) % 10 == 0:
                if self.val:
                    val_acc(self.model, self.device, self.dataset, 1)
                if not os.path.exists('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot)):
                    os.makedirs('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot))
                torch.save(self.model.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/model_epoch{}_traintext{}_visualreg{}_seed{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1, self.cfg.train.train_textprompt, self.cfg.train.visualreg, self.seed))
                #torch.save(self.optimizer.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1))
                #torch.save(self.lr_sched.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/lrsched_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1)) 
                print('checkpoint saved')