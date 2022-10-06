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



class ZSCLIP(nn.Module):
    def __init__(self, labels, cfg, device, L = None, prefix=None):
        super(ZSCLIP, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        self.prefix = prefix
        # transformation pipeline
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

        # initialize text prompt
        prompt_prefix = self.prefix
        classnames = [name.replace("_", " ") for name in self.labels]
        prompts = [prompt_prefix.replace('_', c)+'.' for c in classnames]
        self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)
            text_f = self.text_enc(self.embedding.type(self.dtype), self.prompts_tokenized)
            self.text_f = text_f / text_f.norm(dim=-1, keepdim=True)
    
    def forward(self, img):
        pixel_values = img.to(self.device)
        # create continuous prompt
        img_f = self.img_enc(pixel_values.type(self.dtype).contiguous())
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * torch.matmul(img_f, self.text_f.t()) # batch_size, n_cls
        return logits



class CoOp(nn.Module):
    def __init__(self, labels, cfg, device, L = None, prefix=None):
        super(CoOp, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        self.prefix = prefix
        # transformation pipeline
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

    def forward(self, img):
        pixel_values = img.to(self.device)
        context = self.prompt_emb.repeat(self.n_cls, 1,1)
        prefix = self.sos_emb
        suffix = self.class_emb
        
        # create continuous prompt
        img_f = self.img_enc(pixel_values.type(self.dtype).contiguous())
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) # n_cls x 77 x h_dim
        if self.n_cls < 100:
            text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
        # if too many n_cls
        else:
            text_f = []
            for pr, tok in zip(prompt.chunk(self.n_cls//10 + 1), self.prompts_tokenized.chunk(self.n_cls//10 + 1)):
                t_f = self.text_enc(pr.type(self.dtype), tok)
                text_f.append(t_f)
            text_f = torch.vstack(text_f)
        
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        logits = self.logit_scale.exp() * torch.matmul(img_f, text_f.t()) # batch_size, n_cls
        return logits


class CoOpv2(nn.Module):
    def __init__(self, labels, cfg, device, L = None, prefix=None, inference=False, alpha=0.5):
        super(CoOpv2, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        self.prefix = prefix
        self.inference = inference
        self.alpha = alpha
        # transformation pipeline
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

        if self.inference == True:
            # fixed text prompt
            prompt_prefix = self.cfg.model.prefix
            classnames = [name.replace("_", " ") for name in self.labels]
            prompts = [prompt_prefix.replace('_', c)+'.' for c in classnames]
            self.zs_prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            with torch.no_grad():
                self.zs_embedding = self.token_embedding(self.zs_prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)

        # learnable text prompt
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

    def forward(self, img):
        pixel_values = img.to(self.device)
        context = self.prompt_emb.repeat(self.n_cls, 1,1)
        prefix = self.sos_emb
        suffix = self.class_emb
        
        # create continuous prompt
        prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) # n_cls x 77 x h_dim
        img_f = self.img_enc(pixel_values.type(self.dtype).contiguous())
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        
        if self.inference == True:
            zs_text_f = self.text_enc(self.zs_embedding.type(self.dtype), self.zs_prompts_tokenized)
            zs_text_f = zs_text_f / zs_text_f.norm(dim=-1, keepdim=True)
            text_f = text_f * self.alpha + zs_text_f * (1-self.alpha)
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
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)
        self.dtype = torch.float32
        self.token_embedding = clipmodel.token_embedding
        self.img_enc = clipmodel.visual
        self.text_enc = TextEncoder(cfg, device)
        self.meta_net = nn.Sequential(OrderedDict([
            ("meta_linear1", nn.Linear(cfg.model.h_dim, cfg.model.h_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("meta_linear2", nn.Linear(cfg.model.h_dim // 16, clipmodel.ln_final.weight.shape[0]))
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
        pixel_values = img.to(self.device)
        batch_size = pixel_values.shape[0]
        context = self.prompt_emb # (ctx_len, h_dim)
        prefix = self.sos_emb # (n_cls, 1, h_dim)
        suffix = self.class_emb # (n_cls, *, h_dim)

        with torch.no_grad():
            img_f = self.img_enc(pixel_values.type(self.dtype).contiguous())
            img_f = (img_f / img_f.norm(dim=-1, keepdim=True)) # (batch_size == 1, h_dim)
        bias = self.meta_net(img_f).unsqueeze(1) # (batch_size == 1, 1, h_dim)
        ctx = context.unsqueeze(0) # (1, ctx_len, h_dim)
        ctx_shifted = ctx + bias # (1, ctx_len, h_dim)

        # create continuous prompt
        cond_prompt = torch.cat([prefix, ctx_shifted.repeat(self.n_cls,1,1).to(self.device), suffix], dim=1) # n_cls x 77 x h_dim

        if self.n_cls < 100:
            text_f = self.text_enc(cond_prompt.type(self.dtype), self.prompts_tokenized).unsqueeze(0).repeat(batch_size, 1, 1)
        # if too many n_cls
        else:
            text_f = []
            for pr, tok in zip(cond_prompt.chunk(15), self.prompts_tokenized.chunk(15)):
                t_f = self.text_enc(pr.type(self.dtype), tok)
                text_f.append(t_f)
            text_f = torch.vstack(text_f)
            text_f = text_f.unsqueeze(0).repeat(batch_size, 1,1)
        
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        logits = (self.logit_scale.exp() * torch.bmm(img_f.unsqueeze(1), text_f.permute(0, 2, 1))).squeeze(1) # batch_size, n_cls
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
    
    def forward(self, prompt, attn_mask=None):
        '''
        prompt : torch.FloatTensor shape of (N, 50+n_ctx, 512)
        '''
        x = self.pre_ln(prompt)
        x = x.permute(1, 0, 2)
        x = self.transformer(x, attn_mask)
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
    def __init__(self, labels, cfg, device, L=None, prefix=None, mode = 'train'):
        super(VisualCoCoOpv1, self).__init__()
        self.cfg = cfg
        self.ctx_len = self.cfg.model.ctx_len
        self.v_ctx_len = self.cfg.model.v_ctx_len
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        self.prefix = prefix
        self.mode = mode
        # transformation pipeline
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

        # initialize randomly
        if self.prefix is None:
            prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.t_h_dim, dtype=self.dtype, device=self.device)
            nn.init.normal_(prompt_vec, std=0.02)
            self.prompt_emb = nn.Parameter(prompt_vec)
            prompt_prefix = " ".join(['V']*ctx_len)
            classnames = [name.replace("_", " ") for name in self.labels]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            with torch.no_grad():
                self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)
            # extract [SOS] word embedding & [CLASS],[EOS] word embedding
            self.sos_emb = self.embedding[:,:1,:] # (n_cls x 1 x h_dim)
            self.class_emb = self.embedding[:, 1+ctx_len:, :] # (n_cls x * x h_dim)
        
        # initialize with predefined prefix (i.e. A photo of a)
        else:
            # initialize with 'a photo of a'
            if self.cfg.train.train_textprompt:
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
                    self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)
                # extract [SOS] word embedding & [CLASS],[EOS] word embedding
                self.sos_emb = self.embedding[:,:1,:] # (n_cls x 1 x h_dim)
                self.class_emb = self.embedding[:, 1+ctx_len:, :] # (n_cls x * x h_dim)
            
            # initialize with manual prompt (do not train text prompt)
            else:
                prompt_prefix = self.prefix
                classnames = [name.replace("_", " ") for name in self.labels]
                prompts = [prompt_prefix.replace('_', c)+'.' for c in classnames]
                self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
                with torch.no_grad():
                    self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)
       
        # visual prompt embedding
        ## initialize visual prompt embedding
        v_prompt_vec = torch.empty(self.v_ctx_len, self.cfg.model.v_h_dim, dtype=self.dtype)
        nn.init.normal_(v_prompt_vec, std=0.02)
        self.v_prompt_emb = nn.Parameter(v_prompt_vec, requires_grad=True) ######################
        
        # if cfg.train.traintextprompt is False : pre-compute text features
        if (self.mode =='train') and (not self.cfg.train.train_textprompt):
            prompt = self.embedding
            with torch.no_grad():
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                self.text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)
        
        # for inference (pre-compute label embeddings)
        if self.mode != 'train':
            if self.cfg.train.train_textprompt:
            # forward propagate class features
                context = self.prompt_emb.repeat(self.n_cls, 1,1)
                prefix = self.sos_emb
                suffix = self.class_emb
                prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) #### (n_cls, 77, h_dim)    
            else:
                prompt = self.embedding
            with torch.no_grad():
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                self.text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)
    
    def forward(self, img):
        pixel_values = img.to(self.device)
        batch_size = pixel_values.shape[0]

        if self.mode == 'train':
            if self.cfg.train.train_textprompt:
            # forward propagate class features
                context = self.prompt_emb.repeat(self.n_cls, 1,1)
                prefix = self.sos_emb
                suffix = self.class_emb
                prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) #### (n_cls, 77, h_dim)
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)    
            else:
                text_f = self.text_f
        else:
            text_f = self.text_f

        # forward propagate image features
        x = self.patch_embedding(pixel_values.type(self.dtype)) # (batch_size, h_dim, 7, 7)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1) # (batch_size, 49, h_dim)
        
        # concatenating visual prompt / adding visual prompt
        x = torch.cat([self.cls_embedding.repeat(batch_size,1,1).type(self.dtype), x], dim=1) # 16 (batch_size, 50, h_dim)
        x = x + self.pos_embedding.type(self.dtype) # (N,L,D) 
    
        if self.L is None:
            x = torch.cat([x[:,:1,:], self.v_prompt_emb.repeat(batch_size,1,1).to(self.device), x[:,1:,:]], dim=1)
            img_fm = self.img_enc(x)
        else:
            v_prompt = self.v_prompt_emb.repeat(batch_size, 1,1).to(self.device)
            img_fm = self.img_enc(x, v_prompt)

        img_f = (self.post_ln(img_fm[:,0,:])@self.vision_proj).unsqueeze(1) # (batch_size, 1, h_dim)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        # processing instance feature
        instance_f = self.post_ln(img_fm[:,1:1+self.cfg.model.v_ctx_len,:]) # (batch_size, v_ctx_len, v_h_dim)
        instance_f = torch.bmm(instance_f, self.vision_proj.repeat(batch_size, 1, 1)) # (batch_size, v_ctx_len, h_dim)
        instance_f = instance_f.sum(dim=1, keepdim=True) # (batch_size, 1, h_dim)
        instance_f = instance_f / instance_f.norm(dim=-1, keepdim=True)
        # instance_f = self.meta_net(instance_f.squeeze(1)).unsqueeze(1) # (batch_size, h_dim)

        # add instance feature to class embeddings
        text_f = text_f.unsqueeze(0).repeat(batch_size, 1, 1) + instance_f #(batch_size, n_cls, h_dim)
        # normalize features 
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        logits = (self.logit_scale.exp() * torch.bmm(text_f, img_f.permute(0,2,1))).squeeze(-1)
        return logits # (batch_size, n_cls)



class VisualCoCoOpv2(nn.Module):
    def __init__(self, labels, cfg, device, L=None, prefix=None, mode='train'):
        super(VisualCoCoOpv2, self).__init__()
        self.cfg = cfg
        self.ctx_len = self.cfg.model.ctx_len
        self.v_ctx_len = self.cfg.model.v_ctx_len
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        self.prefix = prefix
        self.mode = mode
        # transformation pipeline
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)

        # set device
        #if self.device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #    self.dtype = torch.float16

        # meta network for visual prompt generation
        
        self.meta_net = nn.Sequential(OrderedDict([
            ("meta_linear1", nn.Linear(cfg.model.h_dim*2, cfg.model.h_dim)),
            ("relu", nn.ReLU(inplace=True)),
            ("meta_linear2", nn.Linear(cfg.model.h_dim, cfg.model.h_dim))
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
        
        # initialize randomly
        if self.prefix is None:
            prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.t_h_dim, dtype=self.dtype, device=self.device)
            nn.init.normal_(prompt_vec, std=0.02)
            self.prompt_emb = nn.Parameter(prompt_vec)
            prompt_prefix = " ".join(['V']*ctx_len)
            classnames = [name.replace("_", " ") for name in self.labels]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            with torch.no_grad():
                self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)
            # extract [SOS] word embedding & [CLASS],[EOS] word embedding
            self.sos_emb = self.embedding[:,:1,:] # (n_cls x 1 x h_dim)
            self.class_emb = self.embedding[:, 1+ctx_len:, :] # (n_cls x * x h_dim)
        
        # initialize with predefined prefix (i.e. A photo of a)
        else:
            # initialize with 'a photo of a'
            if self.cfg.train.train_textprompt:
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
                    self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)
                # extract [SOS] word embedding & [CLASS],[EOS] word embedding
                self.sos_emb = self.embedding[:,:1,:] # (n_cls x 1 x h_dim)
                self.class_emb = self.embedding[:, 1+ctx_len:, :] # (n_cls x * x h_dim)
            
            # initialize with manual prompt (do not train text prompt)
            else:
                prompt_prefix = self.prefix
                classnames = [name.replace("_", " ") for name in self.labels]
                prompts = [prompt_prefix.replace('_', c)+'.' for c in classnames]
                self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
                with torch.no_grad():
                    self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)

        # visual prompt embedding
        ## initialize visual prompt embedding
        v_prompt_vec = torch.empty(self.v_ctx_len, self.cfg.model.v_h_dim, dtype=self.dtype)
        nn.init.normal_(v_prompt_vec, std=0.02)
        self.v_prompt_emb = nn.Parameter(v_prompt_vec, requires_grad=True) ######################

        # if cfg.train.traintextprompt is False : pre-compute text features
        if (self.mode =='train') and (not self.cfg.train.train_textprompt):
            prompt = self.embedding
            with torch.no_grad():
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                self.text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)
        
        # for inference (pre-compute label embeddings)
        if self.mode != 'train':
            if self.cfg.train.train_textprompt:
            # forward propagate class features
                context = self.prompt_emb.repeat(self.n_cls, 1,1)
                prefix = self.sos_emb
                suffix = self.class_emb
                prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) #### (n_cls, 77, h_dim)    
            else:
                prompt = self.embedding
            with torch.no_grad():
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                self.text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)
    
    def forward(self, img):
        pixel_values = img.to(self.device)
        batch_size = pixel_values.shape[0]

        if self.mode == 'train':
            if self.cfg.train.train_textprompt:
            # forward propagate class features
                context = self.prompt_emb.repeat(self.n_cls, 1,1)
                prefix = self.sos_emb
                suffix = self.class_emb
                prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) #### (n_cls, 77, h_dim)
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)    
            else:
                text_f = self.text_f
        else:
            text_f = self.text_f
        
        # forward propagate image features
        x = self.patch_embedding(pixel_values.type(self.dtype)) # (batch_size, h_dim, 7, 7)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1) # (batch_size, 49, h_dim)
        
        # concatenating visual prompt / adding visual prompt
        x = torch.cat([self.cls_embedding.repeat(batch_size,1,1).type(self.dtype), x], dim=1) # 16 (batch_size, 50, h_dim)
        x = x + self.pos_embedding.type(self.dtype) # (N,L,D) 
    
        if self.L is None:
            x = torch.cat([x[:,:1,:], self.v_prompt_emb.repeat(batch_size,1,1).to(self.device), x[:,1:,:]], dim=1)
            img_fm = self.img_enc(x)
        else:
            v_prompt = self.v_prompt_emb.repeat(batch_size, 1,1).to(self.device)
            img_fm = self.img_enc(x, v_prompt)

        img_f = (self.post_ln(img_fm[:,0,:])@self.vision_proj).unsqueeze(1) # (batch_size, 1, h_dim)
        
        # processing instance feature
        instance_f = self.post_ln(img_fm[:,1:1+self.cfg.model.v_ctx_len,:]) # (batch_size, v_ctx_len, v_h_dim)
        instance_f = torch.bmm(instance_f, self.vision_proj.repeat(batch_size, 1, 1)) # (batch_size, v_ctx_len, h_dim)
        instance_f = instance_f.sum(dim=1, keepdim=True) # (batch_size, 1, h_dim)
        instance_f = instance_f / instance_f.norm(dim=-1, keepdim=True) #(batch_size, 1, h_dim)


        # concat instance feature to class embeddings
        text_f = torch.cat([text_f.unsqueeze(0).repeat(batch_size, 1, 1), instance_f.repeat(1, self.n_cls, 1)], dim=2) # (batch_size, n_cls, 2*h_dim)
        text_f = self.meta_net(text_f) #(batch_size, n_cls, h_dim)
        # normalize features 
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        logits = (self.logit_scale.exp() * torch.bmm(text_f, img_f.permute(0,2,1))).squeeze(-1)
        return logits # (batch_size, n_cls)


class VisualCoCoOpv3(nn.Module):
    def __init__(self, labels, cfg, device, L=None, prefix=None, mode='train'):
        super(VisualCoCoOpv3, self).__init__()
        self.cfg = cfg
        self.ctx_len = self.cfg.model.ctx_len
        self.v_ctx_len = self.cfg.model.v_ctx_len
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        self.prefix = prefix
        self.mode = mode
        # transformation pipeline
        
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)

        # set device
        #if self.device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #    self.dtype = torch.float16

        # meta network for visual prompt generation
        
        self.meta_net = nn.Sequential(OrderedDict([
            ("meta_linear1", nn.Linear(cfg.model.v_h_dim, cfg.model.h_dim))
            #("relu", nn.ReLU(inplace=True)),
            #("meta_linear2", nn.Linear(cfg.model.h_dim, cfg.model.h_dim))
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
        
        # initialize randomly
        if self.prefix is None:
            prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.t_h_dim, dtype=self.dtype, device=self.device)
            nn.init.normal_(prompt_vec, std=0.02)
            self.prompt_emb = nn.Parameter(prompt_vec)
            prompt_prefix = " ".join(['V']*ctx_len)
            classnames = [name.replace("_", " ") for name in self.labels]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            with torch.no_grad():
                self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)
            # extract [SOS] word embedding & [CLASS],[EOS] word embedding
            self.sos_emb = self.embedding[:,:1,:] # (n_cls x 1 x h_dim)
            self.class_emb = self.embedding[:, 1+ctx_len:, :] # (n_cls x * x h_dim)
        
        # initialize with predefined prefix (i.e. A photo of a)
        else:
            # initialize with 'a photo of a'
            if self.cfg.train.train_textprompt:
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
                    self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)
                # extract [SOS] word embedding & [CLASS],[EOS] word embedding
                self.sos_emb = self.embedding[:,:1,:] # (n_cls x 1 x h_dim)
                self.class_emb = self.embedding[:, 1+ctx_len:, :] # (n_cls x * x h_dim)
            
            # initialize with manual prompt (do not train text prompt)
            else:
                prompt_prefix = self.prefix
                classnames = [name.replace("_", " ") for name in self.labels]
                prompts = [prompt_prefix.replace('_', c)+'.' for c in classnames]
                self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
                with torch.no_grad():
                    self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)

        # visual prompt embedding
        ## initialize visual prompt embedding
        v_prompt_vec = torch.empty(self.v_ctx_len, self.cfg.model.v_h_dim, dtype=self.dtype)
        nn.init.normal_(v_prompt_vec, std=0.02)
        self.v_prompt_emb = nn.Parameter(v_prompt_vec, requires_grad=True) ######################

        # if cfg.train.traintextprompt is False : pre-compute text features
        if (self.mode =='train') and (not self.cfg.train.train_textprompt):
            prompt = self.embedding
            with torch.no_grad():
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                self.text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)
        
        # for inference (pre-compute label embeddings)
        if self.mode != 'train':
            if self.cfg.train.train_textprompt:
            # forward propagate class features
                context = self.prompt_emb.repeat(self.n_cls, 1,1)
                prefix = self.sos_emb
                suffix = self.class_emb
                prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) #### (n_cls, 77, h_dim)    
            else:
                prompt = self.embedding
            with torch.no_grad():
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                self.text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)
    
    def forward(self, img):
        pixel_values = img.to(self.device)
        batch_size = pixel_values.shape[0]

        if self.mode == 'train':
            if self.cfg.train.train_textprompt:
            # forward propagate class features
                context = self.prompt_emb.repeat(self.n_cls, 1,1)
                prefix = self.sos_emb
                suffix = self.class_emb
                prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) #### (n_cls, 77, h_dim)
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)    
            else:
                text_f = self.text_f
        else:
            text_f = self.text_f
        
        # forward propagate image features
        x = self.patch_embedding(pixel_values.type(self.dtype)) # (batch_size, h_dim, 7, 7)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1) # (batch_size, 49, h_dim)
        
        # concatenating visual prompt / adding visual prompt
        x = torch.cat([self.cls_embedding.repeat(batch_size,1,1).type(self.dtype), x], dim=1) # 16 (batch_size, 50, h_dim)
        x = x + self.pos_embedding.type(self.dtype) # (N,L,D) 
    
        if self.L is None:
            x = torch.cat([x[:,:1,:], self.v_prompt_emb.repeat(batch_size,1,1).to(self.device), x[:,1:,:]], dim=1)
            img_fm = self.img_enc(x)
        else:
            v_prompt = self.v_prompt_emb.repeat(batch_size, 1,1).to(self.device)
            img_fm = self.img_enc(x, v_prompt)

        img_f = (self.post_ln(img_fm[:,0,:])@self.vision_proj).unsqueeze(1) # (batch_size, 1, h_dim)
        #img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        # processing instance feature
        instance_f = self.post_ln(img_fm[:,1:1+self.cfg.model.v_ctx_len,:]) # (batch_size, v_ctx_len, v_h_dim)
        # instance_f = torch.bmm(instance_f, self.vision_proj.repeat(batch_size, 1, 1)) # (batch_size, v_ctx_len, h_dim)
        instance_f = instance_f.sum(dim=1, keepdim=True) # (batch_size, 1, h_dim)
        instance_f = self.meta_net(instance_f.squeeze(1)).unsqueeze(1) # (batch_size, 1, h_dim)
        instance_f = instance_f / instance_f.norm(dim=-1, keepdim=True)

        # add instance feature to class embeddings
        if self.mode=='train':
            img_f = img_f + instance_f # (batch_size, 1, h_dim)
        else:
            img_f = img_f + instance_f
        text_f = text_f.unsqueeze(0).repeat(batch_size, 1, 1)
        # normalize features 
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        logits = (self.logit_scale.exp() * torch.bmm(text_f, img_f.permute(0,2,1))).squeeze(-1)
        return logits # (batch_size, n_cls)


class VisualCoCoOpv3_inc(nn.Module):
    def __init__(self, labels, cfg, device, L=None, prefix=None, mode='train', n_tasks=10):
        super(VisualCoCoOpv3_inc, self).__init__()
        self.cfg = cfg
        self.ctx_len = self.cfg.model.ctx_len
        self.v_ctx_len = self.cfg.model.v_ctx_len
        self.labels = labels
        self.device = device
        self.n_cls = len(labels)
        self.L = L
        self.prefix = prefix
        self.mode = mode
        self.n_tasks = n_tasks
        # transformation pipeline
        
        clipmodel, _ = clip.load(cfg.model.backbone, device=device)

        # set device
        #if self.device == torch.device('cpu'):
        self.dtype = torch.float32
        #else:
        #    self.dtype = torch.float16

        # meta network for visual prompt generation
        self.linears = nn.ModuleDict()
        for i in range(n_tasks):
            self.linears['linear{}'.format(i)] = nn.Linear(cfg.model.v_h_dim, cfg.model.h_dim)
        
        
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
        
        # initialize randomly
        if self.prefix is None:
            prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.t_h_dim, dtype=self.dtype, device=self.device)
            nn.init.normal_(prompt_vec, std=0.02)
            self.prompt_emb = nn.Parameter(prompt_vec)
            prompt_prefix = " ".join(['V']*ctx_len)
            classnames = [name.replace("_", " ") for name in self.labels]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
            self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            with torch.no_grad():
                self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)
            # extract [SOS] word embedding & [CLASS],[EOS] word embedding
            self.sos_emb = self.embedding[:,:1,:] # (n_cls x 1 x h_dim)
            self.class_emb = self.embedding[:, 1+ctx_len:, :] # (n_cls x * x h_dim)
        
        # initialize with predefined prefix (i.e. A photo of a)
        else:
            # initialize with 'a photo of a'
            if self.cfg.train.train_textprompt:
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
                    self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)
                # extract [SOS] word embedding & [CLASS],[EOS] word embedding
                self.sos_emb = self.embedding[:,:1,:] # (n_cls x 1 x h_dim)
                self.class_emb = self.embedding[:, 1+ctx_len:, :] # (n_cls x * x h_dim)
            
            # initialize with manual prompt (do not train text prompt)
            else:
                prompt_prefix = self.prefix
                classnames = [name.replace("_", " ") for name in self.labels]
                prompts = [prompt_prefix.replace('_', c)+'.' for c in classnames]
                self.prompts_tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
                with torch.no_grad():
                    self.embedding = self.token_embedding(self.prompts_tokenized).type(self.dtype) # (n_cls, 77, h_dim)

        # visual prompt embedding
        self.v_prompt_emb = nn.ParameterList() ######################

        # if cfg.train.traintextprompt is False : pre-compute text features
        if (self.mode =='train') and (not self.cfg.train.train_textprompt):
            prompt = self.embedding
            with torch.no_grad():
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)
        
        # for inference (pre-compute label embeddings)
        if self.mode != 'train':
            if self.cfg.train.train_textprompt:
            # forward propagate class features
                context = self.prompt_emb.repeat(self.n_cls, 1,1)
                prefix = self.sos_emb
                suffix = self.class_emb
                prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) #### (n_cls, 77, h_dim)    
            else:
                prompt = self.embedding
            with torch.no_grad():
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)
        
        # chunk text features for continual learning evaluation / training
        self.text_fs = text_f.chunk(self.n_tasks) 
    
    def expand_parameter(self):
        # expand visual prompt
        v_prompt_vec = torch.empty(1, self.cfg.model.v_h_dim, dtype=self.dtype)
        nn.init.normal_(v_prompt_vec, std=0.02)
        self.v_prompt_emb.append(nn.Parameter(v_prompt_vec, requires_grad=True))
        for i, param in enumerate(self.v_prompt_emb):
            if i < len(self.v_prompt_emb)-1:
                param.requires_grad = False
        print('Current number of visual token : {}'.format(len(self.v_prompt_emb)))

    def forward(self, img, cur_task_idx):
        pixel_values = img.to(self.device)
        batch_size = pixel_values.shape[0]

        if self.mode == 'train':
            if self.cfg.train.train_textprompt:
            # forward propagate class features
                context = self.prompt_emb.repeat(self.n_cls, 1,1)
                prefix = self.sos_emb
                suffix = self.class_emb
                prompt = torch.cat([prefix, context.to(self.device), suffix], dim=1) #### (n_cls, 77, h_dim)
                text_f = self.text_enc(prompt.type(self.dtype), self.prompts_tokenized)
                text_f = text_f / text_f.norm(dim=-1, keepdim=True) # (n_cls, h_dim)    
            else:
                text_f = self.text_fs[cur_task_idx]
        else:
            text_f = self.text_fs[cur_task_idx]
        
        # forward propagate image features
        x = self.patch_embedding(pixel_values.type(self.dtype)) # (batch_size, h_dim, 7, 7)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0,2,1) # (batch_size, 49, h_dim)
        
        # concatenating visual prompt / adding visual prompt
        x = torch.cat([self.cls_embedding.repeat(batch_size,1,1).type(self.dtype), x], dim=1) # 16 (batch_size, 50, h_dim)
        x = x + self.pos_embedding.type(self.dtype) # (N,L,D) 
    
        if self.L is None:
            v_prompt = torch.cat([p for p in self.v_prompt_emb], dim=0)
            x = torch.cat([x[:,:1,:], v_prompt.repeat(batch_size,1,1).to(self.device), x[:,1:,:]], dim=1)
            img_fm = self.img_enc(x)

        else:
            v_prompt = torch.cat([p for p in self.v_prompt_emb], dim=0).repeat(batch_size, 1,1)
            img_fm = self.img_enc(x, v_prompt)

        img_f = (self.post_ln(img_fm[:,0,:])@self.vision_proj).unsqueeze(1) # (batch_size, 1, h_dim)
        #img_f = img_f / img_f.norm(dim=-1, keepdim=True)

        # processing instance feature
        instance_f = self.post_ln(img_fm[:,cur_task_idx+1,:]).unsqueeze(1) # (batch_size, 1, v_h_dim)
        instance_f = self.linears['linear{}'.format(cur_task_idx)](instance_f) # (batch_size, 1, h_dim)
        instance_f = instance_f / instance_f.norm(dim=-1, keepdim=True)

        # add instance feature to class embeddings
        img_f = img_f + instance_f # (batch_size, 1, h_dim)
        text_f = text_f.unsqueeze(0).repeat(batch_size, 1, 1)
        # normalize features 
        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        logits = (self.logit_scale.exp() * torch.bmm(text_f, img_f.permute(0,2,1))).squeeze(-1)
        return logits # (batch_size, n_cls)