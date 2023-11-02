from yacs.config import CfgNode

gt = CfgNode()

gt.factor_vocab_size = 128
gt.num_factors = 32
gt.emb_size = 192
gt.max_len = 64
gt.num_layers = 4
gt.num_heads = 4
gt.dropout = 0.1

gt.__representation_size__ = gt.emb_size
