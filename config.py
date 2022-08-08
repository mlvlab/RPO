import easydict
from easydict import EasyDict as edict

cfg = edict()

cfg.model = edict()
cfg.model.backbone = 'ViT-B/32'
cfg.model.prefix = 'A photo of a'
cfg.model.ctx_len = 4
cfg.model.v_ctx_len = 1
cfg.model.h_dim = 512
cfg.model.t_h_dim = 512
cfg.model.v_h_dim = 768
cfg.model.prompt_layer = 0

cfg.train = edict()
cfg.train.device = 'cuda:0'
cfg.train.base_label_ratio = 0.5
cfg.train.n_epochs = 200
cfg.train.cocoop_epochs = 10
cfg.train.batch_size = 32
cfg.train.cocoop_batch = 1
cfg.train.k_shot = 16
cfg.train.base_lr = 0.00001
cfg.train.max_lr = 0.002
cfg.train.warmup_epoch = 1

cfg.eval = edict()

