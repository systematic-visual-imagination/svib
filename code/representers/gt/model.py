from .utils import *
from .transformer import TransformerEncoder


class DiscreteEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.factor_vocab_size = args.factor_vocab_size

        self.dictionary = nn.Embedding(args.factor_vocab_size * args.num_factors, args.emb_size)

    def forward(self, factors):
        """
            factors: [B, N, M] torch long
        """
        B, N, M = factors.shape

        assert torch.all(factors.lt(self.factor_vocab_size)), 'Factor value is too large.'

        coeff = torch.arange(M).to(factors.device)  # M
        factors = factors + self.factor_vocab_size * coeff[None, None]  # B, N, M

        emb = self.dictionary(factors.reshape(-1))  # B*N*M, emb_size
        out = emb.reshape(B, N, M, -1)  # B, N*M, emb_size

        return out


class FloatEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.emb_size = args.emb_size

        self.coeff = nn.Parameter(torch.exp(torch.arange(0, self.emb_size, 2, dtype=torch.float) * (math.log(10000.) / self.emb_size)) * 2 * math.pi, requires_grad=False)

    def forward(self, factors):
        """
            factors: [B, N, M] torch float lying in [0, 1.]
        """
        B, N, M = factors.shape

        assert torch.all(factors.le(1.) & factors.ge(0.)), 'Factor value is outside 0 an 1.'

        emb = torch.zeros((B, N, M, self.emb_size), dtype=torch.float, device=factors.device)
        emb[..., 0::2] = torch.sin(factors[..., None] * self.coeff[None, None, None, :].expand(B, N, M, -1))
        emb[..., 1::2] = torch.cos(factors[..., None] * self.coeff[None, None, None, :].expand(B, N, M, -1))

        return emb


class GTEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.emb_size = args.emb_size

        self.discrete_encoder = DiscreteEncoder(args)
        self.float_encoder = FloatEncoder(args)

        self.pos = SinCosPositionalEmbedding1D(args.max_len, args.emb_size)
        self.coupler = TransformerEncoder(args.num_layers, args.emb_size, args.num_heads, args.dropout)

    def forward(self, discrete_factors, float_factors):
        """
            discrete_factors: [B, N, M] torch long
            float_factors: [B, N, M] torch float in [0, 1]
        """
        B, N, M_discrete = discrete_factors.shape
        _, _, M_float = float_factors.shape

        emb_discrete = self.discrete_encoder(discrete_factors)  # B, N, M_discrete, D
        emb_float = self.float_encoder(float_factors)  # B, N, M_float, D
        emb = torch.cat([emb_discrete, emb_float], dim=-2)  # B, N, M_discrete + M_float, D

        emb = emb.flatten(end_dim=1)  # B * N, M_discrete + M_float, D
        emb = self.pos(emb)  # B * N, M_discrete + M_float, D
        out = self.coupler(emb).reshape(B, N * (M_discrete + M_float), self.emb_size)  # B, N, M_discrete + M_float, D

        return out
