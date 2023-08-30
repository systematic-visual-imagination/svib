from .utils import *
from .transformer import TransformerDecoder


class PatchTransformer(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.image_channels = args.image_channels
        self.image_size = args.image_size

        self.d_model = args.d_model
        self.patch_size = args.patch_size

        # networks
        self.input_proj = linear(args.__representation_size__, args.d_model, bias=False)

        self.pos = SinCosPositionalEmbedding2D(args.image_size // args.patch_size, args.image_size // args.patch_size, args.d_model)

        self.tf = TransformerDecoder(args.num_decoder_blocks, (args.image_size // args.patch_size) ** 2, args.d_model,
                                     args.num_decoder_heads, dropout=args.dropout, causal=False)

        self.head = linear(args.d_model, (args.patch_size ** 2) * args.image_channels, bias=True)

    def forward(self, inputs, image):
        B, C, H, W = image.size()

        L = (self.image_size // self.patch_size) ** 2

        slots = self.input_proj(inputs)  # B, num_slots, d_model

        tf_inputs = torch.zeros(B, L, self.d_model).to(inputs.device)  # BT, L, d_model
        tf_inputs = self.pos(tf_inputs)  # BT, L, d_model

        # pred
        pred = self.tf(tf_inputs, slots)  # B, L, d_model
        pred = self.head(pred)  # BT, L, patch_size * patch_size * C

        # target
        target = image.reshape(B, C, H // self.patch_size, self.patch_size, W // self.patch_size, self.patch_size)  # B, H_grid, patch_size, W_grid, patch_size, C
        target = target.permute(0, 2, 4, 3, 5, 1)  # B, H_grid, W_grid, patch_size, patch_size, C
        target = target.reshape(B, L, -1)  # B, L, patch_size * patch_size * C

        # loss
        loss = (pred - target) ** 2  # B, L, patch_size * patch_size * C
        loss = loss.mean()  # B, L

        return loss

    def decode(self, inputs):
        B, _, _ = inputs.size()

        L = (self.image_size // self.patch_size) ** 2

        slots = self.input_proj(inputs)  # B, num_slots, d_model

        tf_inputs = torch.zeros(B, L, self.d_model).to(inputs.device)  # BT, L, d_model
        tf_inputs = self.pos(tf_inputs)  # BT, L, d_model

        # pred
        pred = self.tf(tf_inputs, slots)  # B, L, d_model
        pred = self.head(pred)  # BT, L, patch_size * patch_size * C

        # recon
        recon = pred.reshape(B, self.image_size // self.patch_size, self.image_size // self.patch_size, self.patch_size, self.patch_size, self.image_channels)
        recon = recon.permute(0, 5, 1, 3, 2, 4)  # B, C, H_grid, patch_size, W_grid, patch_size
        recon = recon.reshape(B, self.image_channels, self.image_size, self.image_size)

        return recon.clamp(0., 1.)
