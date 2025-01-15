import torch
import math
import torch.nn as nn
from timm.models.layers import trunc_normal_

class LoraLayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank=64, alpha=128, dropout=0.):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.empty(out_dim, rank))
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        trunc_normal_(self.A, std=.02)
        trunc_normal_(self.B, std=.02)

    def forward(self, x):
        return (self.dropout(x) @ self.A.transpose(0, 1) @ self.B.transpose(0, 1)) * self.scaling
