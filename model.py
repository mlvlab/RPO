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
    valset = UnseenDataset(dataset, train='val')
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


# text prompt learning
class TextEncoder(nn.Module):
    def __init__(self, cfg, device):
        super(TextEncoder, self).__init__()
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)
        self.pos_embedding = clipmodel.positional_embedding
        self.transformers = clipmodel.transformer
        self.ln_final = clipmodel.ln_final
        self.text_proj = clipmodel.text_projection
        self.device = device

        # set dtype
        #if device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #    self.dtype = torch.float16
    
    def forward(self, prompt, token_id):
        '''
        prompt : torch.FloatTensor shape of NxLxD
        prompt_tokenized : torch.LongTensor shape of NxL
        '''
        x = prompt + self.pos_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformers(x.contiguous())
        x = x.permute(1, 0, 2) # LND -> NLD
        x = x.contiguous() ################ 수정
        x = self.ln_final(x).type(self.dtype)

        ######## 수정
        #x = x.to(torch.device('cpu'))
        #token_id = token_id.to(torch.device('cpu'))
        x = x[torch.arange(x.shape[0]), token_id.argmax(dim=-1)]
        #x = x.to(self.device)
        #########
        x = x @ self.text_proj
        return x


class CoOp(nn.Module):
    def __init__(self, labels, cfg, device, L = None):
        super(CoOp, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        # transformation pipeline
        self.transforms_clip = T.Compose([
                                     T.Resize((224,224)),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)
        self.dtype = torch.float32
        self.token_embedding = clipmodel.token_embedding
        self.img_enc = clipmodel.visual
        self.text_enc = TextEncoder(cfg, device)
        self.logit_scale = clipmodel.logit_scale
        self.construct_prompt()
        # set device
        #if self.device == torch.device('cpu'):
        #    self.dtype = torch.float32
        #else:
        #    self.dtype = torch.float32
        del clipmodel

    def construct_prompt(self):
        ctx_len = self.cfg.model.ctx_len

        # initialize prompt embedding
        prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.t_h_dim, dtype=self.dtype)
        nn.init.normal_(prompt_vec, std=0.02)
        self.prompt_emb = nn.Parameter(prompt_vec)

        # tokenize "prompt_prefix + [class]"
        prompt_prefix = " ".join(['V']*ctx_len)
        classnames = [name.replace("_", " ") for name in self.labels]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype)
        
        # extract [SOS] word embedding & [CLASS],[EOS] word embedding
        self.sos_emb = embedding[:,:1,:] # n_cls x 1 x h_dim
        self.class_emb = embedding[:, 1+ctx_len:, :] # n_cls x * x h_dim

    def forward(self, img):
        pixel_values = self.transforms_clip(img).to(self.device)
        context = self.prompt_emb.repeat(self.n_cls, 1,1)
        prefix = self.sos_emb
        suffix = self.class_emb
        
        # create continuous prompt
        prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) # n_cls x 77 x h_dim
        img_f = self.img_enc(pixel_values.type(self.dtype).contiguous())
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * torch.matmul(img_f, text_f.t()) # batch_size, n_cls
        return logits


class CoCoOp(nn.Module):
    def __init__(self, labels, cfg, device, L = None, prefix=None):
        super(CoCoOp, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        self.prefix = prefix
        # transformation pipeline
        self.transforms_clip = T.Compose([
                                     T.Resize((224,224)),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)
        self.dtype = torch.float32
        self.token_embedding = clipmodel.token_embedding
        self.img_enc = clipmodel.visual
        self.text_enc = TextEncoder(cfg, device)
        self.meta_net = nn.Sequential(OrderedDict([
            ("meta_linear1", nn.Linear(cfg.model.h_dim, cfg.model.h_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("meta_linear2", nn.Linear(cfg.model.h_dim // 16, cfg.model.h_dim))
        ]))
        self.logit_scale = clipmodel.logit_scale
        self.construct_prompt()
        # set device
        #if self.device == torch.device('cpu'):
        #    self.dtype = torch.float32
        #else:
        #    self.dtype = torch.float32
        del clipmodel

    def construct_prompt(self):
        ctx_len = self.cfg.model.ctx_len

        # initialize text prompt
        if self.prefix is None:
            prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.t_h_dim, dtype=self.dtype)
            nn.init.normal_(prompt_vec, std=0.02)
            self.prompt_emb = nn.Parameter(prompt_vec)
            prompt_prefix = " ".join(['V']*ctx_len)
        else:
            # tokenize "prompt_prefix"
            ctx_len = len(self.prefix.split(' '))
            prompt = clip.tokenize(self.prefix).to(self.device)
            with torch.no_grad():
                embedding = self.token_embedding(prompt).type(self.dtype)
            self.prompt_emb = nn.Parameter(embedding[0, 1:1+ctx_len, :])
            prompt_prefix = self.prefix
        
        classnames = [name.replace("_", " ") for name in self.labels]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype)
        
        # extract [SOS] word embedding & [CLASS],[EOS] word embedding
        self.sos_emb = embedding[:,:1,:] # n_cls x 1 x h_dim
        self.class_emb = embedding[:, 1+ctx_len:, :] # n_cls x * x h_dim

    def forward(self, img):
        pixel_values = self.transforms_clip(img).to(self.device)
        batch_size = pixel_values.shape[0]
        context = self.prompt_emb # (ctx_len, h_dim)
        prefix = self.sos_emb # (n_cls, 1, h_dim)
        suffix = self.class_emb # (n_cls, *, h_dim)

        img_f = self.img_enc(pixel_values.type(self.dtype).contiguous())
        img_f = (img_f / img_f.norm(dim=-1, keepdim=True)) # (batch_size == 1, h_dim)
        bias = self.meta_net(img_f).unsqueeze(1) # (batch_size == 1, 1, h_dim)
        ctx = context.unsqueeze(0) # (1, ctx_len, h_dim)
        ctx_shifted = ctx + bias # (1, ctx_len, h_dim)

        # create continuous prompt
        cond_prompt = torch.cat([prefix, ctx_shifted.repeat(self.n_cls,1,1).to(self.device), suffix], dim=1) # n_cls x 77 x h_dim
        text_f = self.text_enc(cond_prompt.type(self.dtype), self.prompts_tokenized).unsqueeze(0).repeat(batch_size, 1,1) # (batch_size, n_cls, h_dim)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        logits = (self.logit_scale.exp() * torch.bmm(text_f, img_f.unsqueeze(1).permute(0,2,1))).squeeze(-1) # batch_size, n_cls
        return logits



# text prompt + visual prompt learning
class VisualEncoder(nn.Module):
    def __init__(self, cfg, device):
        super(VisualEncoder, self).__init__()
        self.device = device
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)
        self.pre_ln = clipmodel.visual.ln_pre
        self.transformer = clipmodel.visual.transformer
        self.post_ln = clipmodel.visual.ln_post
        self.vision_proj = clipmodel.visual.proj

        #if device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #   self.dtype = torch.float16
    def forward(self, prompt):
        '''
        prompt : torch.FloatTensor shape of (N, 50+n_ctx, 512)
        '''
        x = self.pre_ln(prompt)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.post_ln(x[:, 0, :]).type(self.dtype) # 16
        x = x @ self.vision_proj
        return x

class VisualEncoder_int(nn.Module):
    def __init__(self, L, cfg, device):
        super(VisualEncoder_int, self).__init__()
        self.device = device
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)
        self.pre_ln = clipmodel.visual.ln_pre
        self.transformer_1 = clipmodel.visual.transformer.resblocks[:L]
        self.transformer_2 = clipmodel.visual.transformer.resblocks[L:]
        self.post_ln = clipmodel.visual.ln_post
        self.vision_proj = clipmodel.visual.proj

        #if device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #   self.dtype = torch.float16
    def forward(self, x, prompt):
        '''
        prompt : torch.FloatTensor shape of (N, 50+n_ctx, 512)
        '''
        x = self.pre_ln(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_1(x)
        x = x.permute(1,0,2)
        # insert visual prompt from L-th layer
        x = torch.cat([x[:,:1,:], prompt.to(self.device), x[:,1:,:]], dim=1)
        x = x.permute(1, 0, 2)
        x = self.transformer_2(x)
        x = x.permute(1, 0, 2)
        x = self.post_ln(x[:, 0, :]).type(self.dtype) # 16
        x = x @ self.vision_proj
        return x


class VisualCoOp(nn.Module):
    def __init__(self, labels, cfg, device, L = None):
        super(VisualCoOp, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        # transformation pipeline
        self.transforms_clip = T.Compose([
                                     T.Resize((224,224)),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)

        # set device
        #if self.device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #    self.dtype = torch.float16

        # text encoder
        self.token_embedding = clipmodel.token_embedding
        self.text_enc = TextEncoder(cfg, device)
    
        # vision encoder
        self.patch_embedding = clipmodel.visual.conv1
        self.pos_embedding = clipmodel.visual.positional_embedding
        self.cls_embedding = clipmodel.visual.class_embedding

        if L is None:
            self.img_enc = VisualEncoder(cfg, device)
        else:
            self.img_enc = VisualEncoder_int(L, cfg, device)

        self.logit_scale = clipmodel.logit_scale
        self.construct_prompt()
        del clipmodel

    def construct_prompt(self):
        self.ctx_len = self.cfg.model.ctx_len
        self.v_ctx_len = self.cfg.model.v_ctx_len

        # text prompt embedding
        ## initialize prompt embedding
        prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.t_h_dim, dtype=self.dtype)
        nn.init.normal_(prompt_vec, std=0.02)
        self.prompt_emb = nn.Parameter(prompt_vec)

        ## tokenize "prompt_prefix + [class]"
        prompt_prefix = " ".join(['V']*self.ctx_len)
        classnames = [name.replace("_", " ") for name in self.labels]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.prompts_tokenized = clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype)
        
        ## extract [SOS] word embedding & [CLASS],[EOS] word embedding
        self.sos_emb = embedding[:,:1,:].to(self.device) # n_cls x 1 x h_dim
        self.class_emb = embedding[:, 1+self.ctx_len:, :].to(self.device) # n_cls x * x h_dim


        # visual prompt embedding
        ## initialize visual prompt embedding
        v_prompt_vec = torch.empty(self.v_ctx_len, self.cfg.model.v_h_dim, dtype=self.dtype)
        nn.init.normal_(v_prompt_vec, std=0.02)
        self.v_prompt_emb = nn.Parameter(v_prompt_vec, requires_grad=True) ######################

    def forward(self, img):
        pixel_values = self.transforms_clip(img).to(self.device)
        batch_size = pixel_values.shape[0]
        # forward propagate class features
        context = self.prompt_emb.repeat(self.n_cls, 1,1)
        prefix = self.sos_emb
        suffix = self.class_emb
        prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1)       
        text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)

        # forward propagate image features
        x = self.patch_embedding(pixel_values) # (batch_size, h_dim, 7, 7)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1) # (batch_size, 49, h_dim)
        x = torch.cat([self.cls_embedding.repeat(batch_size,1,1).type(self.dtype), x], dim=1) # 16 (batch_size, 50, h_dim)
        x = x + self.pos_embedding.type(self.dtype) # (N,L,D) 
        if self.L is None:
            # visual prompt in input layer
            x = torch.cat([x[:,:1,:], self.v_prompt_emb.repeat(batch_size,1,1).to(self.device), x[:,1:,:]], dim=1) 
            img_f = self.img_enc(x)
        else:
            # visual prompt in intermediate layer
            v_prompt = self.v_prompt_emb.repeat(batch_size, 1,1).to(self.device)
            img_f = self.img_enc(x, v_prompt)

        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * torch.matmul(img_f, text_f.t()) 
        return logits


'''
# text prompt + visual prompt generating metanet learning
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class MetaNet(nn.Module):
    def __init__(self, cfg):
        super(MetaNet, self).__init__()
        self.feature_extractor = torchvision.models.inception_v3(pretrained=True)
        self.feature_extractor.dropout = Identity()
        self.feature_extractor.fc = Identity()
        self.feature_extractor.eval()
        self.meta_linear_1 = nn.Linear(2048, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.meta_linear_2 = nn.Linear(1024, cfg.model.v_h_dim)
    
    def forward(self, pixel_values):
        with torch.no_grad():
            x = self.feature_extractor(pixel_values)
            #x = (x-x.mean(dim=1).unsqueeze(1))/x.std(dim=1).unsqueeze(1) * 0.02
        x = self.meta_linear_1(x)
        x = self.relu(x)
        x = self.meta_linear_2(x)
        return x

class VisualCoCoOp(nn.Module):
    def __init__(self, labels, cfg, device, L=None):
        super(VisualCoCoOp, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        # transformation pipeline
        self.transforms_clip = T.Compose([
                                     T.Resize((224,224)),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        self.transforms_meta = T.Compose([
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)

        # set device
        #if self.device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #    self.dtype = torch.float16

        # meta network for visual prompt generation
        self.meta_net = MetaNet(cfg).to(self.device) ############### 수정

        # text encoder
        self.token_embedding = clipmodel.token_embedding
        self.text_enc = TextEncoder(cfg, device)
    
        # vision encoder
        self.patch_embedding = clipmodel.visual.conv1  ######## 수정 
        self.pos_embedding = clipmodel.visual.positional_embedding ####### 수정
        self.cls_embedding = clipmodel.visual.class_embedding ######### 수정
        if self.L is None:
            self.img_enc = VisualEncoder(cfg, device)
        else:
            self.img_enc = VisualEncoder_int(L, cfg, device)

        self.logit_scale = clipmodel.logit_scale
        self.construct_prompt()
        del clipmodel

    def construct_prompt(self):
        self.ctx_len = self.cfg.model.ctx_len
        self.v_ctx_len = self.cfg.model.v_ctx_len

        # text prompt embedding
        ## initialize prompt embedding
        prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.t_h_dim, dtype=self.dtype, device=self.device)
        nn.init.normal_(prompt_vec, std=0.02)
        self.prompt_emb = nn.Parameter(prompt_vec)

        ## tokenize "prompt_prefix + [class]"
        prompt_prefix = " ".join(['V']*self.ctx_len)
        classnames = [name.replace("_", " ") for name in self.labels]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype)
        
        ## extract [SOS] word embedding & [CLASS],[EOS] word embedding
        self.sos_emb = embedding[:,:1,:].to(self.device) # n_cls x 1 x h_dim
        self.class_emb = embedding[:, 1+self.ctx_len:, :].to(self.device) # n_cls x * x h_dim

    def forward(self, img):
        pixel_values = self.transforms_clip(img).to(self.device)
        pixel_values_meta = self.transforms_meta(img).to(self.device)
        batch_size = pixel_values.shape[0]

        # forward propagate class features
        context = self.prompt_emb.repeat(self.n_cls, 1,1)
        prefix = self.sos_emb
        suffix = self.class_emb
        prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) #### 수정       
        text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
        
        # extract visual prompt using meta network
        v_prompt = self.meta_net(pixel_values_meta)
        v_prompt = v_prompt.unsqueeze(1) # (*, 1, v_h_dim)

        # forward propagate image features
        x = self.patch_embedding(pixel_values.type(self.dtype)) # (batch_size, h_dim, 7, 7)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1) # (batch_size, 49, h_dim)
        
        # concatenating visual prompt / adding visual prompt
        x = torch.cat([self.cls_embedding.repeat(batch_size,1,1).type(self.dtype), x], dim=1) # 16 (batch_size, 50, h_dim)
        x = x + self.pos_embedding.type(self.dtype) # (N,L,D) 
        
        if self.L is None:
            #visual_prompt = x + v_prompt
            x = torch.cat([x[:,:1,:], v_prompt, x[:,1:,:]], dim=1)
            img_f = self.img_enc(x)
        else:
            img_f = self.img_enc(x, v_prompt)

        # normalize features 
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * torch.matmul(img_f, text_f.t()) 
        return logits
'''

class VisualEncoder2(nn.Module):
    def __init__(self, cfg, device):
        super(VisualEncoder2, self).__init__()
        self.device = device
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)
        self.pre_ln = clipmodel.visual.ln_pre
        self.transformer = clipmodel.visual.transformer

        #if device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #   self.dtype = torch.float16
    def forward(self, prompt):
        '''
        prompt : torch.FloatTensor shape of (N, 50+n_ctx, 512)
        '''
        x = self.pre_ln(prompt)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        return x

class VisualEncoder_int2(nn.Module):
    def __init__(self, L, cfg, device):
        super(VisualEncoder_int2, self).__init__()
        self.device = device
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)
        self.pre_ln = clipmodel.visual.ln_pre
        self.transformer_1 = clipmodel.visual.transformer.resblocks[:L]
        self.transformer_2 = clipmodel.visual.transformer.resblocks[L:]

        #if device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #   self.dtype = torch.float16
    def forward(self, x, prompt):
        '''
        prompt : torch.FloatTensor shape of (N, 50+n_ctx, 512)
        '''
        x = self.pre_ln(x)
        x = x.permute(1, 0, 2)
        x = self.transformer_1(x)
        x = x.permute(1,0,2)
        # insert visual prompt from L-th layer
        x = torch.cat([x[:,:1,:], prompt.to(self.device), x[:,1:,:]], dim=1)
        x = x.permute(1, 0, 2)
        x = self.transformer_2(x)
        x = x.permute(1, 0, 2)

        return x


class VisualCoCoOpv1(nn.Module):
    def __init__(self, labels, cfg, device, L=None, prefix=None):
        super(VisualCoCoOpv1, self).__init__()
        self.cfg = cfg
        self.ctx_len = self.cfg.model.ctx_len
        self.v_ctx_len = self.cfg.model.v_ctx_len
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        self.prefix = prefix
        # transformation pipeline
        self.transforms_clip = T.Compose([
                                     T.Resize((224,224)),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        self.transforms_meta = T.Compose([
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)

        # set device
        #if self.device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #    self.dtype = torch.float16

        # meta network for visual prompt generation
        '''
        self.meta_net = nn.Sequential(OrderedDict([
            ("meta_linear1", nn.Linear(cfg.model.h_dim, cfg.model.h_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("meta_linear2", nn.Linear(cfg.model.h_dim // 16, cfg.model.h_dim)),
        ]))
        '''
        # text encoder
        self.token_embedding = clipmodel.token_embedding
        self.text_enc = TextEncoder(cfg, device)
    
        # vision encoder
        self.patch_embedding = clipmodel.visual.conv1  ######## 수정 
        self.pos_embedding = clipmodel.visual.positional_embedding ####### 수정
        self.cls_embedding = clipmodel.visual.class_embedding ######### 수정
        self.post_ln = clipmodel.visual.ln_post
        self.vision_proj = clipmodel.visual.proj
        if self.L is None:
            self.img_enc = VisualEncoder2(cfg, device)
        else:
            self.img_enc = VisualEncoder_int2(L, cfg, device)

        self.logit_scale = clipmodel.logit_scale
        self.construct_prompt()
        del clipmodel

    def construct_prompt(self):
        ctx_len = self.ctx_len
        v_ctx_len = self.v_ctx_len

        # initialize text prompt
        if self.prefix is None:
            prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.t_h_dim, dtype=self.dtype, device=self.device)
            nn.init.normal_(prompt_vec, std=0.02)
            self.prompt_emb = nn.Parameter(prompt_vec)
            prompt_prefix = " ".join(['V']*ctx_len)
        else:
            # tokenize "prompt_prefix"
            ctx_len = len(self.prefix.split(' '))
            prompt = clip.tokenize(self.prefix).to(self.device)
            with torch.no_grad():
                embedding = self.token_embedding(prompt).type(self.dtype)
            self.prompt_emb = nn.Parameter(embedding[0, 1:1+ctx_len, :])
            prompt_prefix = self.prefix
        
        classnames = [name.replace("_", " ") for name in self.labels]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype)
        
        # extract [SOS] word embedding & [CLASS],[EOS] word embedding
        self.sos_emb = embedding[:,:1,:] # (n_cls x 1 x h_dim)
        self.class_emb = embedding[:, 1+ctx_len:, :] # (n_cls x * x h_dim)

        # visual prompt embedding
        ## initialize visual prompt embedding
        v_prompt_vec = torch.empty(self.v_ctx_len, self.cfg.model.v_h_dim, dtype=self.dtype)
        nn.init.normal_(v_prompt_vec, std=0.02)
        self.v_prompt_emb = nn.Parameter(v_prompt_vec, requires_grad=True) ######################
    
    def forward(self, img):
        pixel_values = self.transforms_clip(img).to(self.device)
        pixel_values_meta = self.transforms_meta(img).to(self.device)
        batch_size = pixel_values.shape[0]

        # forward propagate class features
        context = self.prompt_emb.repeat(self.n_cls, 1,1)
        prefix = self.sos_emb
        suffix = self.class_emb
        prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) #### 수정       
        text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized) # (n_cls, h_dim)
        '''
        text_f = self.text_enc(self.embedding, self.prompts_tokenized)
        '''
        # forward propagate image features
        x = self.patch_embedding(pixel_values.type(self.dtype)) # (batch_size, h_dim, 7, 7)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1) # (batch_size, 49, h_dim)
        
        # concatenating visual prompt / adding visual prompt
        x = torch.cat([self.cls_embedding.repeat(batch_size,1,1).type(self.dtype), x], dim=1) # 16 (batch_size, 50, h_dim)
        x = x + self.pos_embedding.type(self.dtype) # (N,L,D) 
    
        if self.L is None:
            x = torch.cat([x[:,:1,:], self.v_prompt_emb.repeat(batch_size,1,1).to(self.device), x[:,1+self.v_ctx_len:,:]], dim=1)
            img_fm = self.img_enc(x)
        else:
            v_prompt = self.v_prompt_emb.repeat(batch_size, 1,1).to(self.device)
            img_fm = self.img_enc(x, v_prompt)

        img_fm = self.img_enc(x)
        img_f = (self.post_ln(img_fm[:,0,:])@self.vision_proj).unsqueeze(1) # (batch_size, 1, h_dim)
        
        # processing instance feature
        instance_f = self.post_ln(img_fm[:,1:1+self.cfg.model.v_ctx_len,:]) # (batch_size, v_ctx_len, v_h_dim)
        instance_f = torch.bmm(instance_f, self.vision_proj.repeat(batch_size, 1, 1)) # (batch_size, v_ctx_len, h_dim)
        instance_f = instance_f.mean(dim=1, keepdim=True) # (batch_size, 1, h_dim)
        # instance_f = self.meta_net(instance_f.squeeze(1)).unsqueeze(1) # (batch_size, h_dim)

        # add instance feature to class embeddings
        text_f = text_f.unsqueeze(0).repeat(batch_size, 1, 1) + instance_f # (batch_size, n_cls, h_dim)
        # normalize features 
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        logits = (self.logit_scale.exp() * torch.bmm(text_f, img_f.permute(0,2,1))).squeeze(-1)
        return logits # (batch_size, n_cls)


class VisualCoCoOpv2(nn.Module):
    def __init__(self, labels, cfg, device, L=None, prefix=None):
        super(VisualCoCoOpv2, self).__init__()
        self.cfg = cfg
        self.ctx_len = self.cfg.model.ctx_len
        self.v_ctx_len = self.cfg.model.v_ctx_len
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        self.prefix = prefix
        # transformation pipeline
        self.transforms_clip = T.Compose([
                                     T.Resize((224,224)),
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        self.transforms_meta = T.Compose([
                                     T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                    ])
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)

        # set device
        #if self.device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #    self.dtype = torch.float16

        # meta network for visual prompt generation
        self.meta_net = nn.Sequential(OrderedDict([
            ("meta_linear1", nn.Linear(cfg.model.h_dim, cfg.model.h_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("meta_linear2", nn.Linear(cfg.model.h_dim // 16, cfg.model.h_dim)),
        ]))
        
        # text encoder
        self.token_embedding = clipmodel.token_embedding
        self.text_enc = TextEncoder(cfg, device)
    
        # vision encoder
        self.patch_embedding = clipmodel.visual.conv1  ######## 수정 
        self.pos_embedding = clipmodel.visual.positional_embedding ####### 수정
        self.cls_embedding = clipmodel.visual.class_embedding ######### 수정
        self.post_ln = clipmodel.visual.ln_post
        self.vision_proj = clipmodel.visual.proj
        if self.L is None:
            self.img_enc = VisualEncoder2(cfg, device)
        else:
            self.img_enc = VisualEncoder_int2(L, cfg, device)

        self.logit_scale = clipmodel.logit_scale
        self.construct_prompt()
        del clipmodel

    def construct_prompt(self):
        ctx_len = self.ctx_len
        v_ctx_len = self.v_ctx_len

        # initialize text prompt
        if self.prefix is None:
            prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.t_h_dim, dtype=self.dtype)
            nn.init.normal_(prompt_vec, std=0.02)
            self.prompt_emb = nn.Parameter(prompt_vec)
            prompt_prefix = " ".join(['V']*ctx_len)
        else:
            # tokenize "prompt_prefix"
            ctx_len = len(self.prefix.split(' '))
            prompt = clip.tokenize(self.prefix).to(self.device)
            with torch.no_grad():
                embedding = self.token_embedding(prompt).type(self.dtype)
            self.prompt_emb = nn.Parameter(embedding[0, 1:1+ctx_len, :])
            prompt_prefix = self.prefix
        
        classnames = [name.replace("_", " ") for name in self.labels]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype)
        
        # extract [SOS] word embedding & [CLASS],[EOS] word embedding
        self.sos_emb = embedding[:,:1,:] # (n_cls x 1 x h_dim)
        self.class_emb = embedding[:, 1+ctx_len:, :] # (n_cls x * x h_dim)

        # visual prompt embedding
        ## initialize visual prompt embedding
        v_prompt_vec = torch.empty(self.v_ctx_len, self.cfg.model.v_h_dim, dtype=self.dtype)
        nn.init.normal_(v_prompt_vec, std=0.02)
        self.v_prompt_emb = nn.Parameter(v_prompt_vec, requires_grad=True) ######################
    
    def forward(self, img):
        pixel_values = self.transforms_clip(img).to(self.device)
        pixel_values_meta = self.transforms_meta(img).to(self.device)
        batch_size = pixel_values.shape[0]

        # forward propagate class features
        context = self.prompt_emb.repeat(self.n_cls, 1,1)
        prefix = self.sos_emb
        suffix = self.class_emb
        prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) #### 수정       
        text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized) # (n_cls, h_dim)
        '''
        text_f = self.text_enc(self.embedding, self.prompts_tokenized)
        '''
        # forward propagate image features
        x = self.patch_embedding(pixel_values.type(self.dtype)) # (batch_size, h_dim, 7, 7)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1) # (batch_size, 49, h_dim)
        
        # concatenating visual prompt / adding visual prompt
        x = torch.cat([self.cls_embedding.repeat(batch_size,1,1).type(self.dtype), x], dim=1) # 16 (batch_size, 50, h_dim)
        x = x + self.pos_embedding.type(self.dtype) # (N,L,D) 
    
        if self.L is None:
            x = torch.cat([x[:,:1,:], self.v_prompt_emb.repeat(batch_size,1,1).to(self.device), x[:,1+self.v_ctx_len:,:]], dim=1)
            img_fm = self.img_enc(x)
        else:
            v_prompt = self.v_prompt_emb.repeat(batch_size, 1,1).to(self.device)
            img_fm = self.img_enc(x, v_prompt)

        img_fm = self.img_enc(x)
        img_f = (self.post_ln(img_fm[:,0,:])@self.vision_proj).unsqueeze(1) # (batch_size, 1, h_dim)
        
        # processing instance feature
        instance_f = self.post_ln(img_fm[:,1:1+self.cfg.model.v_ctx_len,:]) # (batch_size, v_ctx_len, v_h_dim)
        instance_f = torch.bmm(instance_f, self.vision_proj.repeat(batch_size, 1, 1)) # (batch_size, v_ctx_len, h_dim)
        instance_f = instance_f.mean(dim=1, keepdim=True) # (batch_size, 1, h_dim)
        # instance_f = self.meta_net(instance_f.squeeze(1)).unsqueeze(1) # (batch_size, h_dim)

        # add instance feature to class embeddings
        text_f = text_f.unsqueeze(0).repeat(batch_size, 1, 1) + instance_f # (batch_size, n_cls, h_dim)
        # normalize features 
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        logits = (self.logit_scale.exp() * torch.bmm(text_f, img_f.permute(0,2,1))).squeeze(-1)
        return logits # (batch_size, n_cls)




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
            elif type == 'visualcoop':
                self.model = VisualCoOp(self.dataloader.dataset.base_labels, cfg, device, L)
            elif type == 'visualcocoopv1':
                self.model = VisualCoCoOpv2(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif type == 'visualcocoopv2':
                self.model = VisualCoCoOpv2(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix)
        # if want to train with entire classes
        else:
            if type == 'coop':
                self.model = CoOp(self.dataloader.dataset.labels, cfg, device)
            elif type == 'cocoop':
                self.model = CoCoOp(self.dataloader.dataset.labels, cfg, device, prefix = self.cfg.model.prefix)
            elif type == 'visualcoop':
                self.model = VisualCoOp(self.dataloader.dataset.labels, cfg, device, L)
            elif type == 'visualcocoopv1':
                self.model = VisualCoCoOpv2(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix)
            elif type == 'visualcocoopv2':
                self.model = VisualCoCoOpv2(self.dataloader.dataset.base_labels, cfg, device, L, prefix = self.cfg.model.prefix)
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
                if self.type == 'visualcoop' or self.type == 'visualcocoop':
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
            if epoch == (self.n_epochs - 1):
                if self.val:
                    val_acc(self.model, self.device, self.dataset, 1)
                if not os.path.exists('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot)):
                    os.makedirs('./ckpt/{}_promptlearn_{}/{}_shot/'.format(self.dataset, self.type, self.kshot))
                torch.save(self.model.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/model_epoch{}_layer{}_seed{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1, self.model.L, self.seed))
                #torch.save(self.optimizer.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/optimizer_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1))
                #torch.save(self.lr_sched.state_dict, './ckpt/{}_promptlearn_{}/{}_shot/lrsched_epoch{}.pt'.format(self.dataset, self.type, self.kshot, epoch+1)) 
                print('checkpoint saved')