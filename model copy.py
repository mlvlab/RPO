import os
import collections
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR, CyclicLR, _LRScheduler

import clip

from dataset import Dataset
from lr_scheduler import ConstantWarmupScheduler




############### for evaluation ###############################
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

###############################################################


# zero-shot classifier baseline
class Zeroshot_CLIP(nn.Module):
    def __init__(self, labels, prompt = 'A photo of _'):
        '''
        labels : [List] containing entire categories
        prompt : manual prompt for context
        '''
        super().__init__()
        self.labels = labels
        self.prompt = prompt
        self.clip = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        self.preprocessor = transformers.CLIPFeatureExtractor.from_pretrained('openai/clip-vit-base-patch32')
        self.compute_label_emb()
    
    @torch.no_grad()
    def compute_label_emb(self):
        prompts = []
        for label in self.labels:
            prompts.append(self.prompt.replace('_', label))
        token_dict = self.tokenizer.batch_encode_plus(prompts, max_length = 15, return_tensors='pt', padding='max_length')
        self.label_emb = self.clip.get_text_features(**token_dict)

    @torch.no_grad()
    def classifiy_single_image(self, image):
        pixels = self.preprocessor(image)['pixel_values'][0]
        image_emb = self.clip.get_image_features(torch.tensor(pixels).unsqueeze(0))
        sim_score = torch.nn.functional.cosine_similarity(image_emb, self.label_emb)
        argmax_idx = torch.argmax(sim_score)
        return self.labels[argmax_idx], sim_score

    @torch.no_grad()
    def evaluate_clf(self, dataloader, top_k=5):
        preds = []
        ans = []
        for i, (X,y) in enumerate(dataloader):
            image_embs = self.clip.get_image_features(X)
            sim_mat = torch.matmul(image_embs , self.label_emb.t())
            top_k_pred = torch.topk(sim_mat, k=top_k, dim=1).indices
            preds.append(top_k_pred)
            ans.append(y)
            if (i+1) % 10 == 0:
                print('{} photos classified'.format(dataloader.batch_size*(i+1)))
        preds = torch.cat(preds, dim=0)
        ans = torch.cat(ans, dim=0)
        print(preds)
        print(ans)
        # compute accuracy
        N = preds.shape[0]
        correct = 0
        for i in range(N):
            if ans[i] in preds[i]:
                correct += 1
        acc = correct / N
        return acc 

# CLIP+Linear Probe
'''
class CLIPLinear(nn.Module):
    def __init__(self, cfg, dataset):
        super(CLIPLinear, self).__init__()
        self.clipmodel = transformers.CLIPModel.from_pretrained(cfg.model.backbone)
        for param in self.clipmodel.parameters():
            param.requires_grad = False
        self.linear_probe = nn.Linear(cfg.model.h_dim, len(dataset.labels))
    
    def forward(self, pixel_values):
        x = self.clipmodel.get_image_features(pixel_values)
        x = self.linear_probe(x)
        return x

class LinearProbe(object):
    def __init__(self, cfg, dataset, kshot, start_epoch):
        self.cfg = cfg
        self.start_epoch = start_epoch
        self.dataloader = torch.utils.data.DataLoader(Dataset(dataset=dataset,
                                                            k_shot=kshot),
                                                    batch_size = self.cfg.train.batch_size, 
                                                    shuffle = True)
        
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.train.max_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.cfg.train.n_epochs)
        self.lr_sched = ConstantWarmupScheduler(self.optimizer, scheduler, self.cfg.train.warmup_epoch, self.cfg.train.base_lr)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self):
        for epoch in range(self.start_epoch, self.cfg.train.n_epochs):
            print('-'*10 + 'Epoch {}'.format(epoch+1)+'-'*10)
            epoch_loss = 0
            print('current lr : {}'.format(self.lr_sched.get_lr()[0]))
            for step, (pixel_values, label) in enumerate(self.dataloader):
                logits = self.model(pixel_values.to(self.device)) # (batch_size, n_cls)
                loss = self.criterion(logits, label.to(self.device))
                loss.backward()
                self.optimizer.step()
                self.lr_sched.step()
                epoch_loss += loss.item()
                # verbosity
                #if ((step+1) % 10) == 0:
                print('| {} / {} | train loss : {}'.format(step+1, len(self.dataloader), epoch_loss/(step+1)))
    
            # save checkpoint
            if (epoch+1)%50 == 0:
                if self.val:
                    val_acc(self.model, self.device, self.dataset, 1)
                if not os.path.exists('./ckpt/{}_linearprobe_{}/{}_shot/'.format(self.dataset, self.type, self.kshot)):
                    os.makedirs('./ckpt/{}_linearprobe_{}/{}_shot/'.format(self.dataset, self.type, self.kshot))
                torch.save(self.model.state_dict, './ckpt/{}_linearprobe_{}/{}_shot/model_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1))
                torch.save(self.optimizer.state_dict, './ckpt/{}_linearprobe_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1))
                torch.save(self.lr_sched.state_dict, './ckpt/{}_linearprobe_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1)) 
                print('checkpoint saved')

'''
# text prompt learning
class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super(TextEncoder, self).__init__()
        clipmodel = transformers.CLIPModel.from_pretrained(cfg.model.backbone)
        self.pos_embedding = clipmodel.text_model.embeddings.position_embedding.weight
        self.transformers = clipmodel.text_model.encoder
        self.fin_layer_norm = clipmodel.text_model.final_layer_norm
        self.text_proj = clipmodel.text_projection
        self.dtype = clipmodel.dtype
    
    def forward(self, prompt):
        '''
        prompt : torch.FloatTensor shape of NxLxD
        prompt_tokenized : torch.LongTensor shape of NxL
        '''
        x = prompt + self.pos_embedding
        x = self.transformers(x).last_hidden_state[:,0,:]
        x = self.fin_layer_norm(x)
        x = self.text_proj(x)
        return x

class VisualEncoder(nn.Module):
    def __init__(self, cfg):
        super(VisualEncoder, self).__init__()
        clipmodel = transformers.CLIPModel.from_pretrained(cfg.model.backbone)
        self.visual = clipmodel.vision_model
        self.vision_proj = clipmodel.visual_projection
    
    def forward(self, pixel_values):
        x = self.visual(pixel_values).last_hidden_state[:,0,:]
        x = self.vision_proj(x)
        return x

class PromptLRN(nn.Module):
    def __init__(self, labels, cfg, device):
        super(PromptLRN, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        clipmodel = transformers.CLIPModel.from_pretrained(self.cfg.model.backbone)
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(self.cfg.model.backbone)
        self.token_embedding = clipmodel.text_model.embeddings.token_embedding
        self.img_enc = VisualEncoder(cfg)
        self.text_enc = TextEncoder(cfg)
        self.logit_scale = clipmodel.logit_scale
        self.construct_prompt()
        del clipmodel

    def construct_prompt(self):
        ctx_len = self.cfg.model.ctx_len

        # initialize prompt embedding
        prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.t_h_dim, dtype=self.text_enc.dtype)
        nn.init.normal_(prompt_vec, std=0.02)
        self.prompt_emb = nn.Parameter(prompt_vec)

        # tokenize "prompt_prefix + [class]"
        prompt_prefix = " ".join(['V']*ctx_len)
        classnames = [name.replace("_", " ") for name in self.labels]
        name_lens = [len(self.tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts_tokenized = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding='max_length')
        with torch.no_grad():
            embedding = self.token_embedding(prompts_tokenized['input_ids']).type(self.text_enc.dtype)
        
        # extract [SOS] word embedding & [CLASS],[EOS] word embedding
        self.sos_emb = embedding[:,0,:].unsqueeze(1) # n_cls x 1 x h_dim
        self.class_emb = embedding[:, 1+ctx_len:, :] # n_cls x * x h_dim

    def forward(self, pixel_values):
        context = self.prompt_emb.repeat(self.n_cls, 1,1)
        prefix = self.sos_emb
        suffix = self.class_emb
        # create continuous prompt
        prompt = torch.cat([prefix.to(self.device), context.to(self.device), suffix.to(self.device)], dim=1) # n_cls x 77 x h_dim
        img_f = self.img_enc(pixel_values)
        text_f = self.text_enc(prompt)
        logits = self.logit_scale.exp() * torch.matmul(img_f, text_f.t()) # batch_size, n_cls
        return logits

# visual prompt learning
class VisualEncoder2(nn.Module):
    def __init__(self, cfg):
        super(VisualEncoder2, self).__init__()
        clipmodel = transformers.CLIPModel.from_pretrained(cfg.model.backbone)
        self.pre_ln = clipmodel.vision_model.pre_layrnorm
        self.visual = clipmodel.vision_model.encoder
        self.post_ln = clipmodel.vision_model.post_layernorm
        self.vision_proj = clipmodel.visual_projection
    
    def forward(self, prompt):
        '''
        prompt : torch.FloatTensor shape of (N, 50+n_ctx, 512)
        '''
        x = self.pre_ln(prompt)
        x = self.visual(x).last_hidden_state[:,0,:]
        x = self.post_ln(x)
        x = self.vision_proj(x)
        return x

class VPromptLRN(nn.Module):
    def __init__(self, labels, cfg, device, dataset):
        super(VPromptLRN, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        clipmodel = transformers.CLIPModel.from_pretrained(self.cfg.model.backbone)
        # text encoder
        self.clipmodel = clipmodel
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(self.cfg.model.backbone)

        # vision encoder
        self.patch_embedding = clipmodel.vision_model.embeddings.patch_embedding.to(self.device)
        self.pos_embedding = clipmodel.vision_model.embeddings.position_embedding.weight.to(self.device)
        self.cls_embedding = clipmodel.vision_model.embeddings.class_embedding.to(self.device)
        self.img_enc = VisualEncoder2(cfg)

        self.logit_scale = clipmodel.logit_scale
        if dataset == 'eurosat':
            self.construct_prompt('A centered satellite photo of _')
        elif dataset == 'fgvcaircraft':
            self.construct_prompt('A photo of a _, a type of aircraft')
        elif dataset == 'sun397':
            self.construct_prompt('A photo of a _')
        del clipmodel

    def construct_prompt(self, text_prompt):
        classnames = [name.replace("_", " ") for name in self.labels]
        prompt = list(map(lambda x: text_prompt.replace('_', x), classnames))
        prompts_tokenized = self.tokenizer.batch_encode_plus(prompt, return_tensors='pt', padding=True, max_length=76)
        with torch.no_grad():
            self.text_f = self.clipmodel.get_text_features(**prompts_tokenized)
        
        # visual prompt embedding
        ## initialize visual prompt embedding
        self.v_ctx_len = self.cfg.model.v_ctx_len
        v_prompt_vec = torch.empty(self.v_ctx_len, self.cfg.model.v_h_dim)
        nn.init.normal_(v_prompt_vec, std=0.02)
        self.v_prompt_emb = nn.Parameter(v_prompt_vec, requires_grad=True).to(self.device) ######################

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        # forward propagate image features
        x = self.patch_embedding(pixel_values) # (batch_size, h_dim, 7, 7)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0,2,1) # (batch_size, 49, h_dim)
        x = torch.cat([self.cls_embedding.repeat(batch_size,1,1), x], dim=1) # (batch_size, 50, h_dim)
        x = x + self.pos_embedding # (N,L,D)
        
        v_prompt = torch.cat([x[:,:1,:], self.v_prompt_emb.repeat(batch_size,1,1), x[:,1:,:]], dim=1) 
        img_f = self.img_enc(v_prompt)
        logits = self.logit_scale.exp() * torch.matmul(img_f, self.text_f.t()) 
        return logits

# text prompt + visual prompt learning
class VisualEncoder2(nn.Module):
    def __init__(self, cfg):
        super(VisualEncoder2, self).__init__()
        clipmodel = transformers.CLIPModel.from_pretrained(cfg.model.backbone)
        self.pre_ln = clipmodel.vision_model.pre_layrnorm
        self.visual = clipmodel.vision_model.encoder
        self.post_ln = clipmodel.vision_model.post_layernorm
        self.vision_proj = clipmodel.visual_projection
    
    def forward(self, prompt):
        '''
        prompt : torch.FloatTensor shape of (N, 50+n_ctx, 512)
        '''
        x = self.pre_ln(prompt)
        x = self.visual(x).last_hidden_state[:,0,:]
        x = self.post_ln(x)
        x = self.vision_proj(x)
        return x

class VCPromptLRN(nn.Module):
    def __init__(self, labels, cfg, device):
        super(VCPromptLRN, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        clipmodel = transformers.CLIPModel.from_pretrained(self.cfg.model.backbone)
        # text encoder
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(self.cfg.model.backbone)
        self.token_embedding = clipmodel.text_model.embeddings.token_embedding
        self.text_enc = TextEncoder(cfg)
    

        # vision encoder
        self.patch_embedding = clipmodel.vision_model.embeddings.patch_embedding.to(self.device)
        self.pos_embedding = clipmodel.vision_model.embeddings.position_embedding.weight.to(self.device)
        self.cls_embedding = clipmodel.vision_model.embeddings.class_embedding.to(self.device)
        self.img_enc = VisualEncoder2(cfg)

        self.logit_scale = clipmodel.logit_scale
        self.construct_prompt()
        del clipmodel

    def construct_prompt(self):
        self.ctx_len = self.cfg.model.ctx_len
        self.v_ctx_len = self.cfg.model.v_ctx_len

        # text prompt embedding
        ## initialize prompt embedding
        prompt_vec = torch.empty(self.ctx_len, self.cfg.model.t_h_dim, dtype=self.text_enc.dtype)
        nn.init.normal_(prompt_vec, std=0.02)
        self.prompt_emb = nn.Parameter(prompt_vec, requires_grad=True).to(self.device) ##################

        ## tokenize "prompt_prefix + [class]"
        prompt_prefix = " ".join(['V']*self.ctx_len)
        classnames = [name.replace("_", " ") for name in self.labels]
        name_lens = [len(self.tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts_tokenized = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding='max_length')
        with torch.no_grad():
            embedding = self.token_embedding(prompts_tokenized['input_ids']).type(self.text_enc.dtype)
        
        ## extract [SOS] word embedding & [CLASS],[EOS] word embedding
        self.sos_emb = embedding[:,:1,:] # n_cls x 1 x h_dim
        self.class_emb = embedding[:, 1+self.ctx_len:, :] # n_cls x * x h_dim


        # visual prompt embedding
        ## initialize visual prompt embedding
        v_prompt_vec = torch.empty(self.v_ctx_len, self.cfg.model.v_h_dim, dtype=self.text_enc.dtype)
        nn.init.normal_(v_prompt_vec, std=0.02)
        self.v_prompt_emb = nn.Parameter(v_prompt_vec, requires_grad=True).to(self.device) ######################

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        # forward propagate class features
        context = self.prompt_emb.repeat(self.n_cls, 1,1)
        prefix = self.sos_emb
        suffix = self.class_emb
        prompt = torch.cat([prefix.to(self.device), context.to(self.device), suffix.to(self.device)], dim=1)
        text_f = self.text_enc(prompt)

        # forward propagate image features
        x = self.patch_embedding(pixel_values) # (batch_size, h_dim, 7, 7)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0,2,1) # (batch_size, 49, h_dim)
        x = torch.cat([self.cls_embedding.repeat(batch_size,1,1), x], dim=1) # (batch_size, 50, h_dim)
        x = x + self.pos_embedding # (N,L,D)
        
        v_prompt = torch.cat([x[:,:1,:], self.v_prompt_emb.repeat(batch_size,1,1), x[:,1:,:]], dim=1) 
        img_f = self.img_enc(v_prompt)
        logits = self.logit_scale.exp() * torch.matmul(img_f, text_f.t()) 
        return logits


# Visual Prompt Generating Meta network
class VCMetaPromptLRN(nn.Module):
    def __init__(self, labels, cfg, device):
        super(VCMetaPromptLRN, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        clipmodel = transformers.CLIPModel.from_pretrained(self.cfg.model.backbone)
        self.clipmodel = clipmodel
        # text encoder
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained(self.cfg.model.backbone)
        self.token_embedding = clipmodel.text_model.embeddings.token_embedding
        self.text_enc = TextEncoder(cfg)
    

        # vision encoder
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(cfg.model.h_dim, cfg.model.h_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(cfg.model.h_dim // 16, cfg.model.v_h_dim))
        ]))
        self.patch_embedding = clipmodel.vision_model.embeddings.patch_embedding.to(self.device)
        self.pos_embedding = clipmodel.vision_model.embeddings.position_embedding.weight.to(self.device)
        self.cls_embedding = clipmodel.vision_model.embeddings.class_embedding.to(self.device)
        self.img_enc = VisualEncoder2(cfg)

        self.logit_scale = clipmodel.logit_scale
        self.construct_prompt()
        del clipmodel

    def construct_prompt(self):
        self.ctx_len = self.cfg.model.ctx_len
        self.v_ctx_len = self.cfg.model.v_ctx_len

        # text prompt embedding
        ## initialize prompt embedding
        prompt_vec = torch.empty(self.ctx_len, self.cfg.model.t_h_dim, dtype=self.text_enc.dtype)
        nn.init.normal_(prompt_vec, std=0.02)
        self.prompt_emb = nn.Parameter(prompt_vec, requires_grad=True).to(self.device) ##################

        ## tokenize "prompt_prefix + [class]"
        prompt_prefix = " ".join(['V']*self.ctx_len)
        classnames = [name.replace("_", " ") for name in self.labels]
        name_lens = [len(self.tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts_tokenized = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding='max_length')
        with torch.no_grad():
            embedding = self.token_embedding(prompts_tokenized['input_ids']).type(self.text_enc.dtype)
        
        ## extract [SOS] word embedding & [CLASS],[EOS] word embedding
        self.sos_emb = embedding[:,:1,:] # n_cls x 1 x h_dim
        self.class_emb = embedding[:, 1+self.ctx_len:, :] # n_cls x * x h_dim
        ######################

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        # forward propagate through meta-net for visual prompt generation
        with torch.no_grad():
            img_f = self.clipmodel.get_image_features(pixel_values.to(self.device))
        v_prompt_emb = self.meta_net(img_f) # (batch_size, v_h_dim)

        # forward propagate class features
        context = self.prompt_emb.repeat(self.n_cls, 1,1)
        prefix = self.sos_emb
        suffix = self.class_emb
        prompt = torch.cat([prefix.to(self.device), context.to(self.device), suffix.to(self.device)], dim=1)
        text_f = self.text_enc(prompt)

        # forward propagate image features
        x = self.patch_embedding(pixel_values) # (batch_size, h_dim, 7, 7)
        x = x.view(x.shape[0], x.shape[1], -1).permute(0,2,1) # (batch_size, 49, h_dim)
        x = torch.cat([self.cls_embedding.repeat(batch_size,1,1), x], dim=1) # (batch_size, 50, h_dim)
        x = x + self.pos_embedding # (N,L,D)
        
        v_prompt = torch.cat([x[:,:1,:], v_prompt_emb.unsqueeze(1), x[:,1:,:]], dim=1) 
        img_f = self.img_enc(v_prompt)
        logits = self.logit_scale.exp() * torch.matmul(img_f, text_f.t()) 
        return logits

# Prompt Optmizer : Trainer
class PromptOptim(object):
    def __init__(self, cfg, device, dataset = None, kshot = None, type = 'text', start_epoch = 0, val = False):
        super(PromptOptim, self).__init__()
        
        # set configuration
        self.cfg = cfg
        self.type = type
        self.kshot = kshot
        self.dataset = dataset
        self.start_epoch = start_epoch
        self.device = device
        self.val = val
        # set dataloader
        self.dataloader = torch.utils.data.DataLoader(Dataset(dataset=dataset,
                                                            k_shot=kshot),
                                                    batch_size = self.cfg.train.batch_size, 
                                                    shuffle = True)
    
        # define model
        if type == 'text':
            self.model = PromptLRN(self.dataloader.dataset.labels, cfg, device)
        elif type == 'vision':
            self.model = VPromptLRN(self.dataloader.dataset.labels, cfg, device, dataset)
        elif type == 'text+vision':
            self.model = VCPromptLRN(self.dataloader.dataset.labels, cfg, device)
        elif type == 'text+vision_metanet':
            self.model = VCMetaPromptLRN(self.dataloader.dataset.labels, cfg, device)
        self.model.to(device)
        # freeze weight
        for n, param in self.model.named_parameters():
            if ('meta' not in n) and ('prompt' not in n):
                param.requires_grad = False

        # set optimizer & lr scheduler
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.train.max_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.cfg.train.n_epochs)
        self.lr_sched = ConstantWarmupScheduler(self.optimizer, scheduler, self.cfg.train.warmup_epoch, self.cfg.train.base_lr)
        # load pretrained model / optimizer / lr_scheduler
        if start_epoch > 5:
            self.model.load_state_dict(torch.load('ckpt/promptlearn_{}/{}_shot/model_epoch{}.pt'.format(self.type, self.kshot, self.start_epoch)))
            self.optimizer.load_state_dict(torch.load('ckpt/promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.type, self.kshot, self.start_epoch)))
            self.lr_sched.load_state_dict(torch.load('ckpt/promptlearn_{}/{}_shot/lr_sched_epoch{}.pt'.format(self.type, self.kshot, self.start_epoch)))

        # set loss function
        self.criterion = nn.CrossEntropyLoss()

        # check trainable parameters
        trainable = []
        for n, param in self.model.named_parameters():
            if param.requires_grad:
                trainable.append(n)
        print('trainable params : {}'.format(trainable))

    def train(self):
        history = []
        for epoch in range(self.start_epoch, self.cfg.train.n_epochs):
            print('-'*10 + 'Epoch {}'.format(epoch+1)+'-'*10)
            epoch_loss = 0
            print('current lr : {}'.format(self.lr_sched.get_lr()[0]))
            for step, (pixel_values, label) in enumerate(self.dataloader):
                logits = self.model(pixel_values.to(self.device)) # (batch_size, n_cls)
                loss = self.criterion(logits, label.to(self.device))
                loss.backward()
                self.optimizer.step()
                self.lr_sched.step()
                epoch_loss += loss.item()
                history.append(epoch_loss / (step+1))
                # verbosity
                #if ((step+1) % 10) == 0:
                print('| {} / {} | train loss : {}'.format(step+1, len(self.dataloader), epoch_loss/(step+1)))
    
            # save checkpoint
            if (epoch+1)%50 == 0:
                if self.val:
                    val_acc(self.model, self.device, self.dataset, 1)
                if not os.path.exists('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot)):
                    os.makedirs('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot))
                torch.save(self.model.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/model_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1))
                torch.save(self.optimizer.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1))
                torch.save(self.lr_sched.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1)) 
                print('checkpoint saved')



# Prompt Optmizer : Trainer
class CrossPromptOptim(object):
    def __init__(self, cfg, device, dataset = None, kshot = None, type = 'text', start_epoch = 0):
        super(CrossPromptOptim, self).__init__()
        
        # set configuration
        self.cfg = cfg
        self.type = type
        self.kshot = kshot
        self.dataset = dataset
        self.start_epoch = start_epoch
        self.device = device
        # set dataloader
        self.dataloader = torch.utils.data.DataLoader(Dataset(dataset=dataset,
                                                            k_shot=kshot),
                                                    batch_size = self.cfg.train.batch_size, 
                                                    shuffle = True)
    
        # define model
        if type == 'text':
            self.model = PromptLRN(self.dataloader.dataset.labels, cfg, device)
        else:
            self.model = VCPromptLRN(self.dataloader.dataset.labels, cfg, device)
        self.model.to(device)
        # freeze weight
        for n, param in self.model.named_parameters():
            if 'prompt' not in n:
                param.requires_grad = False

        # set optimizer & lr scheduler
        self.optimizer = Adam(self.model.parameters(), lr=self.cfg.train.max_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, self.cfg.train.n_epochs)
        self.lr_sched = ConstantWarmupScheduler(self.optimizer, scheduler, self.cfg.train.warmup_epoch, self.cfg.train.base_lr)
        # load pretrained model / optimizer / lr_scheduler
        if start_epoch > 5:
            self.model.load_state_dict(torch.load('ckpt/promptlearn_{}/{}_shot/model_epoch{}.pt'.format(self.type, self.kshot, self.start_epoch)))
            self.optimizer.load_state_dict(torch.load('ckpt/promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.type, self.kshot, self.start_epoch)))
            self.lr_sched.load_state_dict(torch.load('ckpt/promptlearn_{}/{}_shot/lr_sched_epoch{}.pt'.format(self.type, self.kshot, self.start_epoch)))

        # set loss function
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        history = []
        for epoch in range(self.start_epoch, self.cfg.train.n_epochs):
            if epoch % 2 == 0:
                # train only visual prompt
                self.model.v_prompt_emb.requires_grad = True
                self.model.prompt_emb.requires_grad = False
            else:
                # train only text prompt
                self.model.v_prompt_emb.requires_grad = False
                self.model.prompt_emb.requires_grad = True
            
            print('-'*10 + 'Epoch {}'.format(epoch+1)+'-'*10)
            epoch_loss = 0
            for step, (pixel_values, label) in enumerate(self.dataloader):
                logits = self.model(pixel_values.to(self.device)) # (batch_size, n_cls)
                loss = self.criterion(logits, label.to(self.device))
                loss.backward()
                self.optimizer.step()
                self.lr_sched.step()
                epoch_loss += loss.item()
                history.append(epoch_loss / (step+1))
                # verbosity
                #if ((step+1) % 10) == 0:
                print('| {} / {} | train loss : {}'.format(step+1, len(self.dataloader), epoch_loss/(step+1)))
            # save checkpoint
            if (epoch+1)%20 == 0:
                if not os.path.exists('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot)):
                    os.makedirs('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot))
                torch.save(self.model.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/model_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1))
                torch.save(self.optimizer.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1))
                torch.save(self.lr_sched.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1)) 