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


class CartesianPositionalEmbedding(nn.Module):

    def __init__(self, channels, image_size):
        super().__init__()

        self.projection = conv2d(4, channels, 1)
        self.pe = nn.Parameter(self.build_grid(image_size).unsqueeze(0), requires_grad=False)

    def build_grid(self, side_length):
        coords = torch.linspace(0., 1., side_length + 1)
        coords = 0.5 * (coords[:-1] + coords[1:])
        grid_y, grid_x = torch.meshgrid(coords, coords)
        return torch.stack((grid_x, grid_y, 1 - grid_x, 1 - grid_y), dim=0)

    def forward(self, inputs):
        # `inputs` has shape: [batch_size, out_channels, height, width].
        # `grid` has shape: [batch_size, in_channels, height, width].
        return inputs + self.projection(self.pe)


class SinCosPositionalEmbedding2D(nn.Module):

    def __init__(self, h, w, d_model):
        super().__init__()

        position_h = torch.arange(h, dtype=torch.float).view(-1, 1)
        position_w = torch.arange(w, dtype=torch.float).view(-1, 1)

        emb_size = d_model // 2
        div_term = torch.exp(torch.arange(0, emb_size, 2, dtype=torch.float) * (-math.log(10000.) / emb_size))

        pe_h = torch.zeros(h, emb_size, dtype=torch.float)
        pe_h[:, 0::2] = torch.sin(position_h * div_term)
        pe_h[:, 1::2] = torch.cos(position_h * div_term)

        pe_w = torch.zeros(w, emb_size, dtype=torch.float)
        pe_w[:, 0::2] = torch.sin(position_w * div_term)
        pe_w[:, 1::2] = torch.cos(position_w * div_term)

        pe = torch.cat([
            pe_h[:, None, :].expand(-1, w, -1),
            pe_w[None, :, :].expand(h, -1, -1),
        ], dim=-1)  # h, w, d_model
        pe = pe.flatten(end_dim=1)  # h*w, d_model

        self.pe = nn.Parameter(pe, requires_grad=False)

    def forward(self, input):
        """
        input: (B, hw, d_model)
        """
        return input + self.pe[None, :, :]