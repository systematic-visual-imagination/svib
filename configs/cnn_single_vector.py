from yacs.config import CfgNode

cnn_single_vector_configs = CfgNode()

cnn_single_vector_configs.cnn_hidden_size = 64
cnn_single_vector_configs.out_size = 64

cnn_single_vector_configs.__representation_size__ = cnn_single_vector_configs.out_size
