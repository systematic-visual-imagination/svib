from yacs.config import CfgNode

dynamic_vae_configs = CfgNode()

dynamic_vae_configs.cnn_hidden_size = 64
dynamic_vae_configs.latent_size = 64
dynamic_vae_configs.num_dynamics_blocks = 4
dynamic_vae_configs.sigma = 0.01
dynamic_vae_configs.beta = 1.0

dynamic_vae_configs.__representation_size__ = dynamic_vae_configs.latent_size
