import easydict
from easydict import EasyDict as edict

cfg = edict()

cfg.model = edict()
cfg.model.ctx_len = 16
cfg.model.v_ctx_len = 10
cfg.model.h_dim = 512

cfg.train = edict()
cfg.train.dataset = 'sun397'
cfg.train.n_epochs = 100
cfg.train.batch_size = 32
cfg.train.k_shot = 16

cfg.eval = edict()
