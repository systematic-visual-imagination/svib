from yacs.config import CfgNode

vit_configs = CfgNode()

vit_configs.encoder_d_model = 192
vit_configs.encoder_patch_size = 16
vit_configs.num_encoder_blocks = 8
vit_configs.num_encoder_heads = 8
vit_configs.encoder_dropout = 0.1

vit_configs.__representation_size__ = vit_configs.encoder_d_model
