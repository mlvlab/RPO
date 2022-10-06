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

from dataset import CILImageNet100, UnseenDataset
from lr_scheduler import ConstantWarmupScheduler
from model import VisualCoCoOpv1, VisualCoCoOpv2, VisualCoCoOpv3, CoCoOp, CoOp, VisualCoCoOpv3_inc

# Prompt Optmizer : Trainer
class PromptOptim(object):
    def __init__(self, cfg, device, L=None, val = False, only_base = True):
        super(PromptOptim, self).__init__()
        # set configuration
        self.cfg = cfg
        self.type = cfg.model.type
        self.kshot = cfg.dataset.kshot
        self.dataset = cfg.dataset.name
        self.start_epoch = cfg.train.start_epoch
        self.seed = cfg.seed
        self.device = device
        self.val = val

        # set batch size
        batch_size = cfg.train.cocoop_batch if type == 'cocoop' else cfg.train.batch_size
        self.n_epochs = cfg.train.cocoop_epochs if type == 'cocoop' else cfg.train.n_epochs

        # set dataloader
        if only_base:
            self.dataloader = torch.utils.data.DataLoader(UnseenDataset(dataset=self.dataset,
                                                                        base_label_ratio=self.cfg.train.base_label_ratio,
                                                                        k_shot=self.kshot,
                                                                        mode='train',
                                                                        train_time='base',
                                                                        device = self.device),
                                                                batch_size = batch_size,
                                                                shuffle = True)
        else:
            self.dataloader = torch.utils.data.DataLoader(UnseenDataset(dataset=self.dataset,
                                                                    base_label_ratio=self.cfg.train.base_label_ratio,
                                                                    k_shot=self.kshot,
                                                                    mode='train',
                                                                    train_time='entire',
                                                                    device = self.device),
                                                                batch_size = batch_size, 
                                                                shuffle = True)
    
        # define model
        # if want to train with only base classes
        if only_base:
            if self.type == 'coop':
                self.model = CoOp(self.dataloader.dataset.base_labels, cfg, device)
            elif self.type == 'cocoop':
                self.model = CoCoOp(self.dataloader.dataset.base_labels, cfg, device, prefix=self.cfg.model.prefix)
            elif self.type == 'visualcocoopv1':
                self.model = VisualCoCoOpv1(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif self.type == 'visualcocoopv2':
                self.model = VisualCoCoOpv2(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif self.type == 'visualcocoopv3':
                self.model = VisualCoCoOpv3(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix)
            
            
        # if want to train with entire classes
        else:
            if self.type == 'coop':
                self.model = CoOp(self.dataloader.dataset.labels, cfg, device)
            elif self.type == 'cocoop':
                self.model = CoCoOp(self.dataloader.dataset.labels, cfg, device, prefix = self.cfg.model.prefix)
            elif self.type == 'visualcocoopv1':
                self.model = VisualCoCoOpv2(self.dataloader.dataset.labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif self.type == 'visualcocoopv2':
                self.model = VisualCoCoOpv2(self.dataloader.dataset.labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif self.type == 'visualcocoopv3':
                self.model = VisualCoCoOpv3(self.dataloader.dataset.labels, cfg, device, L, prefix = self.cfg.model.prefix)
            
            
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
        if self.start_epoch > 5:
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
                print('Dataset : {} |Epoch {} | {} / {} | train loss : {}'.format(self.dataset, epoch+1, step+1, len(self.dataloader), epoch_loss/(step+1)))
    
            # save checkpoint
            #if epoch == (self.n_epochs - 1):
            if self.type != 'coop':
                if (epoch+1) % 10 == 0:
                    if self.val:
                        val_acc(self.model, self.device, self.dataset, 1)
                    if not os.path.exists('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot)):
                        os.makedirs('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot))
                    torch.save(self.model.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/{}_model_epoch{}_traintext{}_visualreg{}_seed{}.pt'.format(self.dataset, self.type, self.kshot, self.cfg.model.backbone.replace('/','-'), epoch+1, self.cfg.train.train_textprompt, self.cfg.train.visualreg, self.seed))
                    #torch.save(self.optimizer.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1))
                    #torch.save(self.lr_sched.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/lrsched_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1)) 
                    print('checkpoint saved')
            else:
                if (epoch+1) % 200 == 0:
                    if self.val:
                        val_acc(self.model, self.device, self.dataset, 1)
                    if not os.path.exists('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot)):
                        os.makedirs('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot))
                    torch.save(self.model.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/{}_model_epoch{}_traintext{}_visualreg{}_seed{}.pt'.format(self.dataset, self.type, self.kshot, self.cfg.model.backbone.replace('/','-'), epoch+1, self.cfg.train.train_textprompt, self.cfg.train.visualreg, self.seed))
                    #torch.save(self.optimizer.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1))
                    #torch.save(self.lr_sched.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/lrsched_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1)) 
                    print('checkpoint saved')


class IncPromptOptim(object):
    def __init__(self, cfg, device, L=None, val = False, n_tasks=10):
        super(IncPromptOptim, self).__init__()
        # set configuration
        self.cfg = cfg
        self.type = cfg.model.type
        self.dataset = cfg.dataset.name
        self.start_epoch = cfg.train.start_epoch
        self.seed = cfg.seed
        self.device = device
        self.val = val
        self.n_tasks = n_tasks

        # set batch size
        batch_size = cfg.train.batch_size
        self.n_epochs = cfg.train.n_epochs

        # extract entire labels
        labels = CILImageNet100(mode='train', n_tasks=10).labels
        # define model
        self.model = VisualCoCoOpv3_inc(labels, cfg, device, prefix=self.cfg.model.prefix, mode='train', n_tasks=n_tasks)
        self.model.to(device)

        #if self.device == torch.device('cpu'):
        self.model = self.model.type(torch.float32)

        # freeze weight
        for n, param in self.model.named_parameters():
            if ('linears' not in n) and ('prompt' not in n):
                param.requires_grad = False

        # set optimizer & lr scheduler
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.train.max_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.cfg.train.n_epochs)
        self.lr_sched = ConstantWarmupScheduler(self.optimizer, scheduler, self.cfg.train.warmup_epoch, self.cfg.train.base_lr)
        
        # load pretrained model / optimizer / lr_scheduler
        if self.start_epoch > 5:
            self.model.load_state_dict(torch.load('ckpt/{}_promptlearn_{}/{}_shot/model_epoch{}.pt'.format(self.dataset, self.type, self.kshot, self.start_epoch))())
            # self.optimizer.load_state_dict(torch.load('ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, self.start_epoch)))
            self.lr_sched.load_state_dict(torch.load('ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, self.start_epoch))())

        # set loss function
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        for cur_task_idx in range(self.n_tasks):
            print('Training {}th task'.format(cur_task_idx+1))
            # define dataloader for i th task
            self.dataloader = torch.utils.data.DataLoader(CILImageNet100(mode='train',
                                                                         cur_step=cur_task_idx, 
                                                                         n_tasks=self.n_tasks, 
                                                                         device=self.device),
                                                            batch_size=self.cfg.train.batch_size)
            # define optimizer & lr scheduler
            self.optimizer = Adam(self.model.parameters(), lr=self.cfg.train.max_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.cfg.train.n_epochs)
            self.lr_sched = ConstantWarmupScheduler(self.optimizer, scheduler, self.cfg.train.warmup_epoch, self.cfg.train.base_lr)
            # dynamically expand parameter of the model
            self.model.expand_parameter()

            # check parameter freeze
            # check trainable parameters
            trainable = []
            for n, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable.append(n)
            print('trainable params : {}'.format(trainable))
            print('# labels for training : {}'.format(len(self.model.labels)))

            # train each task
            for epoch in range(self.start_epoch, self.n_epochs):
                print('-'*10 + 'Epoch {}'.format(epoch+1)+'-'*10)
                epoch_loss = 0
                print('current lr : {}'.format(self.lr_sched.get_lr()[0]))
                for step, (img, label) in enumerate(self.dataloader):
                    logits = self.model(img, cur_task_idx) # (batch_size, n_cls)
                    loss = self.criterion(logits, label.to(self.device))
                    # l2 regularization on visual prompt
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.lr_sched.step()
                    epoch_loss += loss.item()
                    # verbosity
                    #if ((step+1) % 10) == 0:
                    print('Task : {} |Epoch {} | {} / {} | train loss : {}'.format(cur_task_idx+1, epoch+1, step+1, len(self.dataloader), epoch_loss/(step+1)))
        
            # save checkpoint
            if self.val:
                val_acc(self.model, self.device, self.dataset, 1)
            if not os.path.exists('./ckpt/incremental_learning/'):
                os.makedirs('./ckpt/incremental_learning/')
            torch.save(self.model.state_dict, './ckpt/incremental_learning/task_{}_seed{}.pt'.format(cur_task_idx+1, self.seed))
            #torch.save(self.optimizer.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1))
            #torch.save(self.lr_sched.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/lrsched_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1)) 
            print('checkpoint saved')
            