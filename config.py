import easydict
from easydict import EasyDict as edict

cfg = edict()

cfg.model = edict()
cfg.model.backbone = 'openai/clip-vit-base-patch32'
cfg.model.ctx_len = 16
cfg.model.v_ctx_len = 3
cfg.model.h_dim = 512
cfg.model.t_h_dim = 512
cfg.model.v_h_dim = 768

cfg.train = edict()
cfg.train.n_epochs = 200
cfg.train.batch_size = 32
cfg.train.k_shot = 16
cfg.train.base_lr = 0.00001
cfg.train.max_lr = 0.002
cfg.train.warmup_epoch = 1

cfg.eval = edict()
