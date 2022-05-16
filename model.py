import os
import torch
import torch.nn as nn
from torch.nn import functional as F

import transformers
from transformers import CLIPModel, CLIPTokenizer, CLIPFeatureExtractor


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


# text prompt learning
# text prompt learning
class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super(TextEncoder, self).__init__()
        clipmodel = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
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
        clipmodel = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.visual = clipmodel.vision_model
        self.vision_proj = clipmodel.visual_projection
    
    def forward(self, pixel_values):
        x = self.visual(pixel_values).last_hidden_state[:,0,:]
        x = self.vision_proj(x)
        return x

class PromptLRN(nn.Module):
    def __init__(self, labels, cfg):
        super(PromptLRN, self).__init__()
        self.cfg = cfg
        self.labels = labels
        self.n_cls = len(labels)
        clipmodel = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        self.token_embedding = clipmodel.text_model.embeddings.token_embedding
        self.img_enc = VisualEncoder(cfg)
        self.text_enc = TextEncoder(cfg)
        self.logit_scale = clipmodel.logit_scale
        self.construct_prompt()
        del clipmodel

    def construct_prompt(self):
        ctx_len = self.cfg.model.ctx_len

        # initialize prompt embedding
        prompt_vec = torch.empty(self.cfg.model.ctx_len, self.cfg.model.h_dim, dtype=self.text_enc.dtype)
        nn.init.normal_(prompt_vec, std=0.01)
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
        prompt = torch.cat([prefix, context, suffix], dim=1) # n_cls x 77 x h_dim
        text_f = self.text_enc(prompt)
        img_f = self.img_enc(pixel_values)
        logits = self.logit_scale.exp() * torch.matmul(img_f, text_f.t())
        return logits



# text prompt + visual prompt learning
class VisualEncoder2(nn.Module):
    def __init__(self, cfg):
        super(VisualEncoder, self).__init__()
        clipmodel = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.pre_ln = clipmodel.vision_model.pre_layernorm
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
    def __init__(self, labels, cfg):
        super().__init__()
        self.cfg = cfg
        self.labels = labels
        clipmodel = transformers.CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        # text encoder
        self.tokenizer = transformers.CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        self.token_embedding = clipmodel.text_model.embeddings.token_embedding
        self.text_enc = TextEncoder(cfg)

        # vision encoder
        self.patch_embedding = clipmodel.vision_model.embeddings.patch_embedding
        self.pos_embedding = clipmodel.vision_model.embeddings.positional_embedding.weight
        self.img_enc = VisualEncoder2(cfg)

        self.logit_scale = clipmodel.logit_scale
        self.construct_prompt()
        del clipmodel

    def construct_prompt(self):
        self.ctx_len = self.cfg.model.ctx_len
        self.v_ctx_len = self.cfg.model.v_ctx_len

        # text prompt embedding
        ## initialize prompt embedding
        prompt_vec = torch.empty(self.ctx_len, self.cfg.model.h_dim, dtype=self.text_enc.dtype)
        nn.init.normal_(prompt_vec, std=0.01)
        self.prompt_emb = nn.Parameter(prompt_vec)

        ## tokenize "prompt_prefix + [class]"
        prompt_prefix = " ".join(['V']*self.ctx_len)
        classnames = [name.replace("_", " ") for name in self.labels]
        name_lens = [len(self.tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        prompts_tokenized = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding='max_length')
        with torch.no_grad():
            embedding = self.token_embedding(prompts_tokenized['input_ids']).type(self.text_enc.dtype)
        
        ## extract [SOS] word embedding & [CLASS],[EOS] word embedding
        self.sos_emb = embedding[:,0,:].unsqueeze(1) # n_cls x 1 x h_dim
        self.class_emb = embedding[:, 1+self.ctx_len:, :] # n_cls x * x h_dim


        # visual prompt embedding
        ## initialize visual prompt embedding
        v_prompt_vec = torch.empty(self.v_ctx_len, self.cfg.model.h_dim, dtype=self.text_enc.dtype)
        nn.init.normal_(v_prompt_vec, std=0.01)
        self.v_prompt_emb = nn.Parameter(v_prompt_vec)

    def forward(self, pixel_values):
        batch_size = pixel_values.shape[0]
        # forward propagate class features
        context = self.prompt_emb
        prefix = self.sos_emb
        suffix = self.class_emb
        prompt = torch.cat([prefix, context, suffix], dim=1)
        text_f = self.text_enc(prompt)

        # forward propagate image features
        x = self.patch_embedding(pixel_values)
        x = x + self.pos_embedding # (N,L,D)
        v_prompt = torch.cat([x[:,:1,:], self.v_prompt_emb.repeat(batch_size,1,1), x[:,1+self.v_ctx_len]], dim=1)
        img_f = self.img_enc(v_prompt)
        logits = self.logit_scale.exp() * torch.matmul(img_f, text_f.t())
        return logits


class ContextOptim(nn.Module):
    def __init__(self, cfg, dataloader, text_only = True):
        super(ContextOptim, self).__init__()
        # define model
        if text_only:
            self.model = PromptLRN(dataloader.dataset.labels, cfg)
        else:
            self.model = VCPromptLRN(dataloader.dataset.labels, cfg)
        
        # freeze weight
        for n, param in self.model.parameters():
            if 'prompt' not in n:
                param.requires_grad = False
        
        self.dataloader = dataloader
        self.cfg = cfg
    
    def train(self):
        pass
    def save(self):
        pass 