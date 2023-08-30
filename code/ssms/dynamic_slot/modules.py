from .utils import *
from .transformer import TransformerEncoder


class SAVi(nn.Module):

    def __init__(self, num_iterations, num_slots,
                 input_size, slot_size, mlp_hidden_size,
                 num_predictor_blocks=1,
                 num_predictor_heads=4,
                 dropout=0.0,
                 epsilon=1e-8):
        super().__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.input_size = input_size
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.epsilon = epsilon

        # parameters for Gaussian initialization (shared by all slots).
        self.slot_mu = nn.Parameter(torch.Tensor(1, 1, slot_size))
        self.slot_log_sigma = nn.Parameter(torch.Tensor(1, 1, slot_size))
        nn.init.xavier_uniform_(self.slot_mu)
        nn.init.xavier_uniform_(self.slot_log_sigma)

        # norms
        self.norm_inputs = nn.LayerNorm(input_size)
        self.norm_slots = nn.LayerNorm(slot_size)
        self.norm_mlp = nn.LayerNorm(slot_size)

        # linear maps for the attention module.
        self.project_q = linear(slot_size, slot_size, bias=False)
        self.project_k = linear(input_size, slot_size, bias=False)
        self.project_v = linear(input_size, slot_size, bias=False)

        # slot update functions.
        self.gru = gru_cell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            linear(slot_size, mlp_hidden_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(mlp_hidden_size, slot_size))

        self.predictor = TransformerEncoder(num_predictor_blocks, slot_size, num_predictor_heads, dropout)

    def forward(self, inputs):
        B, T, num_inputs, input_size = inputs.size()

        # initialize slots
        slots = inputs.new_empty(B, self.num_slots, self.slot_size).normal_()
        slots = self.slot_mu + torch.exp(self.slot_log_sigma) * slots

        # setup key and value
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        v = self.project_v(inputs)  # Shape: [batch_size, T, num_inputs, slot_size].
        k = (self.slot_size ** (-0.5)) * k

        # loop over frames
        attns_collect = []
        slots_collect = []
        for t in range(T):
            # corrector iterations
            for i in range(self.num_iterations):
                slots_prev = slots
                slots = self.norm_slots(slots)

                # Attention.
                q = self.project_q(slots)  # Shape: [batch_size, num_slots, slot_size].
                attn_logits = torch.bmm(k[:, t], q.transpose(-1, -2))
                attn_vis = F.softmax(attn_logits, dim=-1)
                # `attn_vis` has shape: [batch_size, num_inputs, num_slots].

                # Weighted mean.
                attn = attn_vis + self.epsilon
                attn = attn / torch.sum(attn, dim=-2, keepdim=True)
                updates = torch.bmm(attn.transpose(-1, -2), v[:, t])
                # `updates` has shape: [batch_size, num_slots, slot_size].

                # Slot update.
                slots = self.gru(updates.view(-1, self.slot_size),
                                 slots_prev.view(-1, self.slot_size))
                slots = slots.view(-1, self.num_slots, self.slot_size)
                slots = slots + self.mlp(self.norm_mlp(slots))

            # collect
            attns_collect += [attn_vis]
            slots_collect += [slots]

            # predictor
            slots = self.predictor(slots)

        attns_collect = torch.stack(attns_collect, dim=1)  # B, T, num_inputs, num_slots
        slots_collect = torch.stack(slots_collect, dim=1)  # B, T, num_slots, slot_size

        return slots_collect, attns_collect


class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, slot_size, output_size, broadcast_size=8):
        super().__init__()
        self.broadcast_size = broadcast_size

        self.decoder_pos = CartesianPositionalEmbedding(slot_size, broadcast_size)
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(slot_size, 64, 5, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 5, 2, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_size, 5, 2, 2, 1)
        )

    def forward(self, slots):
        '''

        :param slots: B, D
        :return:
        '''
        B, D = slots.size()
        x = slots[:, :, None, None].expand(B, D, self.broadcast_size, self.broadcast_size)  # B, D, H, W
        x = self.decoder_pos(x)  # B, D, H, W
        x = self.decoder_cnn(x)  # B, out, H, W

        return x


class DynamicSlot(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.image_channels = args.image_channels
        self.image_size = args.image_size

        self.num_slots = args.num_slots
        self.slot_size = args.slot_size
        self.num_iterations = args.num_iterations
        self.cnn_hidden_size = args.cnn_hidden_size
        self.mlp_hidden_size = args.mlp_hidden_size
        self.input_size = args.input_size

        # encoder networks
        self.cnn = nn.Sequential(
            Conv2dBlock(args.image_channels, args.cnn_hidden_size, 5, 1 if args.image_size == 64 else 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 1, 2),
            conv2d(args.cnn_hidden_size, args.input_size, 5, 1, 2),
        )

        self.pos = CartesianPositionalEmbedding(args.input_size,
                                                args.image_size if args.image_size == 64 else args.image_size // 2)

        self.layer_norm = nn.LayerNorm(args.input_size)

        self.mlp = nn.Sequential(
            linear(args.input_size, args.input_size, weight_init='kaiming'),
            nn.ReLU(),
            linear(args.input_size, args.input_size))

        self.savi = SAVi(
            args.num_iterations, args.num_slots,
            args.input_size, args.slot_size, args.mlp_hidden_size)

        self.decoder = SpatialBroadcastDecoder(args.slot_size, args.image_channels + 1)

        self.dynamics = TransformerEncoder(args.num_dynamics_blocks, args.slot_size, args.num_dynamics_heads, dropout=0.)

    def forward(self, video):
        B, T, C, H, W = video.shape

        emb = self.cnn(video.flatten(end_dim=1))  # B * T, input_size, H, W
        _, _, H_enc, W_enc = emb.shape

        emb = self.pos(emb)  # B * T, input_size, H, W

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B * T, H * W, input_size
        emb_set = self.mlp(self.layer_norm(emb_set))  # B * T, H * W, input_size
        emb_set = emb_set.reshape(B, T, H_enc * W_enc, self.input_size)  # B, T, H * W, input_size

        slots, _ = self.savi(emb_set)  # B, T, N, slot_size

        x = self.decoder(slots.flatten(end_dim=2))  # B * T * N, C + 1, H, W
        x = x.reshape(B, T, self.num_slots, self.image_channels + 1, H, W)  # B, T, N, C + 1, H, W

        recons, masks = x.split([self.image_channels, 1], dim=3)  # recons: B, T, N, C, H, W
                                                                  # masks: B, T, N, 1, H, W

        mask_probs = F.softmax(masks, dim=2)  # B, T, N, 1, H, W
        recons_masked = mask_probs * recons  # B, T, N, C, H, W
        recon_combined = torch.sum(recons_masked, dim=2)  # B, T, C, H, W

        autoencoding_mse = ((video - recon_combined) ** 2).sum() / (B * T)  # 1

        # dynamics
        z_pred = self.dynamics(slots[:, :-1].flatten(end_dim=1).detach())  # B(T - 1), N, latent_size
        z_pred = z_pred.reshape(B, T - 1, self.num_slots, -1)  # B, T - 1, N, latent_size
        dynamics_mse = ((z_pred - slots[:, 1:].detach()) ** 2).mean()  # 1

        loss = autoencoding_mse + dynamics_mse  # 1

        return (
            loss,
            recon_combined.clamp(0., 1.),
            recons_masked.clamp(0., 1.),
        )

    def generate(self, image, T=1):
        B, C, H, W = image.shape

        # encode
        emb = self.cnn(image)  # B, input_size, H, W
        _, _, H_enc, W_enc = emb.shape

        emb = self.pos(emb)  # B, input_size, H, W

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H * W, input_size
        emb_set = self.mlp(self.layer_norm(emb_set))  # B, H * W, input_size
        emb_set = emb_set.reshape(B, 1, H_enc * W_enc, self.input_size)  # B, 1, H * W, input_size

        slots, _ = self.savi(emb_set)  # slots: B, 1, N, slot_size
        slots = slots[:, 0]

        # predict
        slots_pred = []
        for t in range(T):
            slots = self.dynamics(slots)
            slots_pred += [slots]
        slots_pred = torch.stack(slots_pred, 1)  # B, T, N, latent_size

        # render
        x = self.decoder(slots_pred.flatten(end_dim=2))  # B * T * N, C + 1, H, W
        x = x.reshape(B, T, self.num_slots, self.image_channels + 1, H, W)  # B, T, N, C + 1, H, W

        gens, masks = x.split([self.image_channels, 1], dim=3)  # recons: B, T, N, C, H, W
                                                                # masks: B, T, N, 1, H, W

        mask_probs = F.softmax(masks, dim=2)  # B, T, N, 1, H, W
        gens_masked = mask_probs * gens  # B, T, N, C, H, W
        gen_combined = torch.sum(gens_masked, dim=2)  # B, T, C, H, W

        return (
            gen_combined.clamp(0., 1.),
            gens_masked.clamp(0., 1.)
        )


class NextFrameLatentPredictorSlot(DynamicSlot):
    def __init__(self, args):
        super().__init__(args)

    def forward(self, image):
        B, C, H, W = image.shape

        # encode
        emb = self.cnn(image)  # B, input_size, H, W
        _, _, H_enc, W_enc = emb.shape

        emb = self.pos(emb)  # B, input_size, H, W

        emb_set = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)  # B, H * W, input_size
        emb_set = self.mlp(self.layer_norm(emb_set))  # B, H * W, input_size
        emb_set = emb_set.reshape(B, 1, H_enc * W_enc, self.input_size)  # B, 1, H * W, input_size

        slots, _ = self.savi(emb_set)  # slots: B, 1, N, slot_size

        slots = self.dynamics(slots[:, 0])

        return slots
