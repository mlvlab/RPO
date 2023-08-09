## RPO 


import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        positional_embedding = clip_model.positional_embedding

        # Make sure K >= 1
        assert cfg.TRAINER.RPO.K >= 1, "K should be bigger than 0"

        self.K = cfg.TRAINER.RPO.K # the number of prompt pair
        self.dtype = clip_model.dtype
        self.d_t = clip_model.ln_final.weight.shape[0] #512
        self.d_v = 768

        clip_imsize = clip_model.visual.input_resolution # 224
        cfg_imsize = cfg.INPUT.SIZE[0] # (224, 224)[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        self.initialization_token(clip_model)
        
    def initialization_token(self, clip_model):
        #### text token initialization #####
        
        text_token = clip_model.token_embedding(torch.tensor([49407]))
        text_token = text_token.repeat(self.K, 1)
        text_noise = torch.randn(self.K, self.d_t)
        text_noise = text_noise / text_noise.norm(dim=-1, keepdim=True)
        text_token += 0.1 * text_noise
        text_token = text_token.type(self.dtype)
        self.text_prompt = nn.Parameter(text_token)
        '''
        t_prompt_vec = torch.empty(self.K, self.d_t, dtype=self.dtype)
        nn.init.normal_(t_prompt_vec, std=0.02)
        self.text_prompt = nn.Parameter(t_prompt_vec, requires_grad=True)
        '''
        #### visual token initialization ####
        
        visual_token = clip_model.visual.class_embedding
        visual_token = visual_token.repeat(self.K, 1)
        visual_noise = torch.randn(self.K, self.d_v)
        visual_noise = visual_noise / visual_noise.norm(dim=-1, keepdim=True)
        visual_token += 0.1 * visual_noise
        visual_token = visual_token.type(self.dtype)
        self.img_prompt = nn.Parameter(visual_token)
        '''
        v_prompt_vec = torch.empty(self.K, self.d_v, dtype=self.dtype)
        nn.init.normal_(v_prompt_vec, std=0.02)
        self.img_prompt = nn.Parameter(v_prompt_vec, requires_grad=True)
        '''
    def forward(self):
        return self.text_prompt, self.img_prompt


class CustomCLIP(nn.Module):
    '''
    cfg : model parameters
    device : model device
    layer : # of query generate FFN layers
    '''
    def __init__(self, cfg, classnames, prompt, clipmodel):
        super().__init__()
        self.cfg = cfg

        # text encoder
        self.token_embedding = clipmodel.token_embedding
        self.text_pos_embedding = clipmodel.positional_embedding
        self.text_transformers = clipmodel.transformer
        self.text_ln_final = clipmodel.ln_final
        self.text_proj = clipmodel.text_projection

        # vision encoder
        self.img_patch_embedding = clipmodel.visual.conv1
        self.img_cls_embedding = clipmodel.visual.class_embedding
        self.img_pos_embedding = clipmodel.visual.positional_embedding
        self.img_pre_ln = clipmodel.visual.ln_pre
        self.img_transformer = clipmodel.visual.transformer
        self.img_post_ln = clipmodel.visual.ln_post
        self.img_proj = clipmodel.visual.proj

        # logit
        self.logit_scale = clipmodel.logit_scale
        
        # initialization token
        self.prompt_learner = PromptLearner(self.cfg, clipmodel)

        #
        self.dtype = clipmodel.dtype
        self.prompts = self.make_prompts(classnames, prompt) # ["a photo of a dog.", ".."]

        # define mask
        self.define_mask()

    def make_prompts(self, classnames, prompt):
        prompts = [prompt.replace('_', c) for c in classnames]
        with torch.no_grad():
            self.text_tokenized = torch.cat([clip.tokenize(p) for p in prompts])
            self.text_x = self.token_embedding(self.text_tokenized).type(self.dtype) + self.text_pos_embedding.type(self.dtype)
            self.len_prompts = self.text_tokenized.argmax(dim=-1) + 1
        return prompts

    def define_mask(self):
        len_max = 77
        attn_head = 8

        text_mask = torch.empty(0, len_max, len_max)
        for idx in self.len_prompts:
            mask = torch.empty(len_max, len_max)
            mask.fill_(float("-inf"))
            mask.triu_(1)  # zero out the lower diagonal
            mask[:, idx:].fill_(float("-inf"))
            text_mask = torch.cat([text_mask, mask.repeat(attn_head, 1, 1)])
        self.text_mask = text_mask

        # image encoder mask
        att_size = 1 + 14 * 14 # + self.cfg.TRAINER.RPO.K
        visual_mask = torch.zeros((att_size, att_size), dtype=self.dtype, requires_grad=False)
        visual_mask[:, -1 * self.cfg.TRAINER.RPO.K:] = float("-inf")
        #####

        self.visual_mask = visual_mask

    def forward(self, image, label=None):
        device = image.device

        # load mask from predefined masks
        text_mask = self.text_mask
        visual_mask = self.visual_mask
        K = self.cfg.TRAINER.RPO.K

        # load prompts from prompt learner
        text_prompt, image_prompt = self.prompt_learner()

        ####################### text ###########################        
        text_x = self.text_x
        text_x = text_x.to(device)
        
        for i in range(K):
            text_x[torch.arange(text_x.shape[0]), self.len_prompts+i, :] = text_prompt[i, :].repeat(text_x.shape[0], 1)
        

        text_x = text_x.permute(1, 0, 2)  # NLD -> LND
        text_x = self.text_transformers(text_x, text_mask)
        text_x = text_x.permute(1, 0, 2)
        text_x = self.text_ln_final(text_x).type(self.dtype)

        text_f = torch.empty(text_x.shape[0], 0, 512, device=device, dtype=self.dtype)
        for i in range(K):
            idx = self.len_prompts + i
            x = text_x[torch.arange(text_x.shape[0]), idx]
            text_f = torch.cat([text_f, x[:, None, :]], dim=1)

        text_f = text_f @ self.text_proj
        t_f = text_x[torch.arange(text_x.shape[0]), self.text_tokenized.argmax(dim=-1)] @ self.text_proj
        
        ####################### img ###########################
        batch_size = image.shape[0]
        
        # forward propagate image features with token concatenation
        image_embedding = self.img_patch_embedding(image.type(self.dtype)) # (batch_size, h_dim, 7, 7)
        image_embedding = image_embedding.reshape(batch_size, image_embedding.shape[1], -1)
        image_embedding = image_embedding.permute(0,2,1) # (batch_size, 49, h_dim)
        image_embedding = torch.cat([self.img_cls_embedding.repeat(batch_size,1,1).type(self.dtype), image_embedding], dim=1) # 16 (batch_size, 50, h_dim)
        img_x = image_embedding + self.img_pos_embedding.type(self.dtype) # (N,L,D)
        # concatenation the token on visual encoder
        img_x = torch.cat([img_x, image_prompt.repeat(batch_size, 1, 1)], dim=1)
        # image encoder
        img_x = self.img_pre_ln(img_x)
        img_x = img_x.permute(1, 0, 2)
        img_x = self.img_transformer(img_x, visual_mask)
        img_x = img_x.permute(1, 0, 2)
        img_f = self.img_post_ln(img_x[:, -1 * K:, :]) @ self.img_proj
        i_f = self.img_post_ln(img_x[:, 0, :]) @ self.img_proj
        ####################### logit ###########################
        # logit

        text_f = text_f / text_f.norm(dim=-1, keepdim=True)
        t_f = t_f / t_f.norm(dim=-1, keepdim=True)

        img_f = img_f / img_f.norm(dim=-1, keepdim=True)
        i_f = i_f / i_f.norm(dim=-1, keepdim=True)

        logits = torch.zeros(img_f.shape[0], text_f.shape[0], device=device)
        for i in range(K):
            i_img_f = img_f[:,i,:]
            i_text_f = text_f[:,i,:]
            logit = self.logit_scale.exp() * i_img_f @ i_text_f.t()
            logits += logit
        logits /= K

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)
        
        return logits


@TRAINER_REGISTRY.register()
class RPO(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.RPO.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.RPO.PREC == "fp32" or cfg.TRAINER.RPO.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        prompt = cfg.DATASET.PROMPT
        ############################################# 통일 #####

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, prompt, clip_model)
        
        # parameter freeze
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)
        
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.RPO.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

        # nan detector
        torch.autograd.set_detect_anomaly(True)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.RPO.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)












  