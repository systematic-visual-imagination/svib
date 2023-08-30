from yacs.config import CfgNode

dynamic_slot_configs = CfgNode()

dynamic_slot_configs.num_slots = 4
dynamic_slot_configs.slot_size = 64
dynamic_slot_configs.num_iterations = 3
dynamic_slot_configs.cnn_hidden_size = 64
dynamic_slot_configs.mlp_hidden_size = 192
dynamic_slot_configs.input_size = 192

dynamic_slot_configs.num_dynamics_blocks = 4
dynamic_slot_configs.num_dynamics_heads = 4

dynamic_slot_configs.__representation_size__ = dynamic_slot_configs.slot_size
