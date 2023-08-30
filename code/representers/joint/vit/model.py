from .utils import *
from .transformer import TransformerEncoder


class ViTEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.patchify = nn.Conv2d(
            args.image_channels, args.encoder_d_model, kernel_size=args.encoder_patch_size, stride=args.encoder_patch_size)

        self.pos = SinCosPositionalEmbedding2D(
            args.image_size // args.encoder_patch_size, args.image_size // args.encoder_patch_size, args.encoder_d_model)

        self.tf = TransformerEncoder(args.num_encoder_blocks, args.encoder_d_model,
                                     args.num_encoder_heads, dropout=args.encoder_dropout)

    def forward(self, image):

        emb = self.patchify(image)                                      # B, d_model, H_enc, W_enc

        emb = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)   # B, H_enc * W_enc, d_model

        emb = self.pos(emb)                                             # B, H_enc * W_enc, d_model

        out = self.tf(emb)                                              # B, H_enc * W_enc, d_model

        return out
