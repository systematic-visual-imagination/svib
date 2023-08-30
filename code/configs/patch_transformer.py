from yacs.config import CfgNode

patch_transformer_configs = CfgNode()

patch_transformer_configs.num_decoder_blocks = 8
patch_transformer_configs.num_decoder_heads = 4
patch_transformer_configs.d_model = 192
patch_transformer_configs.dropout = 0.1
patch_transformer_configs.patch_size = 4
