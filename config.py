import easydict
from easydict import EasyDict as edict

cfg = edict()

cfg.model = edict()
cfg.model.ctx_len = 16
cfg.model.v_ctx_len = 10
cfg.model.t_h_dim = 512
cfg.model.v_h_dim = 768

cfg.train = edict()
cfg.train.dataset = 'sun397'
cfg.train.n_epochs = 100
cfg.train.batch_size = 32
cfg.train.k_shot = 16
cfg.train.max_lr = 0.001
cfg.train.pct_start = 0.01

cfg.eval = edict()
