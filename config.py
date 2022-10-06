import easydict
from easydict import EasyDict as edict

cfg = edict()

cfg.seed = None

cfg.model = edict()
cfg.model.backbone = 'ViT-B/16'
cfg.model.prefix = 'A photo of a _'
cfg.model.ctx_len = 4
cfg.model.v_ctx_len = 1
cfg.model.h_dim = 512
cfg.model.t_h_dim = 512
cfg.model.v_h_dim = 768
cfg.model.prompt_layer = 0
cfg.model.type = 'visualcocoopv3'

cfg.train = edict()
cfg.train.train_textprompt = False
cfg.train.visualreg = False
cfg.train.base_label_ratio = 0.5
cfg.train.n_epochs = 50
cfg.train.cocoop_epochs = 10
cfg.train.batch_size = 64
cfg.train.cocoop_batch = 1
cfg.train.k_shot = 16
cfg.train.base_lr = 0.00001
cfg.train.max_lr = 0.002
cfg.train.warmup_epoch = 1

cfg.dataset = edict()
cfg.dataset.name = None
cfg.dataset.kshot = 16
cfg.dataset.division = 'base'

cfg.eval = edict()

