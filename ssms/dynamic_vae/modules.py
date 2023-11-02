from .utils import *
from .transformer import TransformerEncoder


class DynamicVAE(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.image_channels = args.image_channels
        self.image_size = args.image_size
        self.latent_size = args.latent_size
        self.cnn_hidden_size = args.cnn_hidden_size

        self.sigma = args.sigma
        self.beta = args.beta

        self.enc_cnn = nn.Sequential(
            Conv2dBlock(args.image_channels, args.cnn_hidden_size, 5, 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2),
        )
        
        self.enc_mlp = nn.Sequential(
            linear(args.cnn_hidden_size * (args.image_size // 16) * (args.image_size // 16), 4 * args.latent_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * args.latent_size, 2 * args.latent_size)
        )

        self.dec_mlp = nn.Sequential(
            linear(args.latent_size, 4 * args.latent_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(4 * args.latent_size, args.cnn_hidden_size * (args.image_size // 16) * (args.image_size // 16))
        )

        self.dec_cnn = nn.Sequential(
            nn.ConvTranspose2d(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(args.cnn_hidden_size, args.image_channels, 5, 2, 2, 1)
        )

        self.dynamics = TransformerEncoder(args.num_dynamics_blocks, args.latent_size, 1, dropout=0.)

    def forward(self, video):
        B, T, C, H, W = video.shape

        video_flat = video.flatten(end_dim=1)  # BT, C, H, W

        # VAE
        emb = self.enc_cnn(video_flat)  # BT, cnn_hidden_size, H_enc, W_enc
        _, _, H_enc, W_enc = emb.shape

        emb = self.enc_mlp(emb.flatten(start_dim=1))  # BT, 2 * latent_size
        q_mu, q_logvar = emb.split([self.latent_size, self.latent_size], dim=-1)  # (BT, latent_size), (BT, latent_size)
        p_mu, p_logvar = torch.zeros_like(q_mu), torch.zeros_like(q_logvar)  # (BT, latent_size), (BT, latent_size)

        kl = normal_kl(q_mu, q_logvar, p_mu, p_logvar).mean()  # 1

        eps = torch.randn_like(q_mu)  # BT, latent_size
        z = q_mu + (q_logvar.exp() ** 0.5) * eps  # BT, latent_size

        emb = self.dec_mlp(z)  # BT, cnn_hidden_size * H_enc * W_enc
        emb = emb.reshape(B * T, self.cnn_hidden_size, H_enc, W_enc)  # BT, cnn_hidden_size, H_enc, W_enc
        recon = self.dec_cnn(emb)  # BT, C, H, W

        nll = normal_nll(recon, torch.full_like(recon, math.log(self.sigma ** 2)), video_flat).mean()  # 1

        recon = recon.reshape(B, T, C, H, W)  # B, T, C, H, W

        # dynamics
        q_mu = q_mu.reshape(B, T, 1, -1)  # B, T, N, latent_size
        z_pred = self.dynamics(q_mu[:, :-1].flatten(end_dim=1).detach())  # B(T - 1), N, latent_size
        z_pred = z_pred.reshape(B, T - 1, 1, -1)  # B, T - 1, N, latent_size
        dynamics_mse = ((z_pred - q_mu[:, 1:].detach()) ** 2).mean()  # 1

        loss = nll + self.beta * kl + dynamics_mse  # 1

        return (
            loss,
            recon.clamp(0., 1.),
        )

    def generate(self, image, T=1):
        B, C, H, W = image.shape

        # encode
        emb = self.enc_cnn(image)  # B, cnn_hidden_size, H_enc, W_enc
        _, _, H_enc, W_enc = emb.shape
        emb = self.enc_mlp(emb.flatten(start_dim=1))  # B, 2 * latent_size
        z, _ = emb.split([self.latent_size, self.latent_size], dim=-1)  # B, latent_size
        z = z.unsqueeze(1)  # B, 1, latent_size

        # predict
        zs_pred = []
        for t in range(T):
            z = self.dynamics(z)
            zs_pred += [z]
        zs_pred = torch.stack(zs_pred, 1)  # B, T, N, latent_size

        # render
        emb = self.dec_mlp(zs_pred.flatten(end_dim=1)[:, 0])  # BT, cnn_hidden_size * H_enc * W_enc
        emb = emb.reshape(B * T, self.cnn_hidden_size, H_enc, W_enc)  # BT, cnn_hidden_size, H_enc, W_enc
        video_pred = self.dec_cnn(emb)  # BT, C, H, W
        video_pred = video_pred.reshape(B, T, C, H, W)

        return video_pred.clamp(0., 1.)


class NextFrameLatentPredictorVAE(DynamicVAE):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, image):
        emb = self.enc_cnn(image)  # B, cnn_hidden_size, H_enc, W_enc
        emb = self.enc_mlp(emb.flatten(start_dim=1))  # B, 2 * latent_size
        z, _ = emb.split([self.latent_size, self.latent_size], dim=-1)  # B, latent_size
        z_pred = self.dynamics(z.unsqueeze(1))

        return z_pred
