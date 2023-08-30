import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
           dilation=1, groups=1, bias=True, padding_mode='zeros',
           weight_init='xavier'):
    m = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation, groups, bias, padding_mode)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)

    if bias:
        nn.init.zeros_(m.bias)

    return m


class Conv2dBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.m = conv2d(in_channels, out_channels, kernel_size, stride, padding,
                        bias=True, weight_init='kaiming')

    def forward(self, x):
        x = self.m(x)
        return F.relu(x)


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


class SinCosPositionalEmbedding1D(nn.Module):

    def __init__(self, max_len, d_model, dropout=0.):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len, dtype=torch.float).view(-1, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.) / d_model))

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, input):
        """
        input: batch_size x seq_len x d_model
        return: batch_size x seq_len x d_model
        """
        T = input.shape[1]
        return self.dropout(input + self.pe[:T])
