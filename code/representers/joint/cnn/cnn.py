from .utils import *


class CNNSingleVectorEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.cnn = nn.Sequential(
            Conv2dBlock(args.image_channels, args.cnn_hidden_size, 5, 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2),
        )

        self.enc_fc = linear(
            args.cnn_hidden_size * (args.image_size // 16) * (args.image_size // 16),
            args.out_size,
            weight_init='kaiming'
        )

    def forward(self, image):

        emb = self.cnn(image)  # B, cnn_hidden_size, H_enc, W_enc

        emb = emb.flatten(start_dim=1)  # B, cnn_hidden_size * H_enc * W_enc

        out = self.enc_fc(emb)  # B, out_size

        out = out.unsqueeze(1)  # B, 1, out_size

        return out


class CNNRelationalEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.cnn = nn.Sequential(
            Conv2dBlock(args.image_channels, args.cnn_hidden_size, 5, 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.cnn_hidden_size, 5, 2, 2),
            Conv2dBlock(args.cnn_hidden_size, args.out_size, 5, 2, 2),
        )

        self.pos = SinCosPositionalEmbedding2D(args.image_size // 16, args.image_size // 16, args.out_size)

    def forward(self, image):

        emb = self.cnn(image)  # B, out_size, H_enc, W_enc

        emb = emb.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)   # B, H_enc * W_enc, out_size
        out = self.pos(emb)                                             # B, H_enc * W_enc, out_size

        return out
