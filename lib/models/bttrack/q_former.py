import copy
import pdb
from typing import Optional

import torch.nn.functional as F
from torch import nn, Tensor
import time

class QueryLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # self.activation1 = nn.ReLU
        # self.activation2 = nn.ReLU

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src1, src2,
                     src1_mask: Optional[Tensor] = None,
                     src2_mask: Optional[Tensor] = None,
                     src1_key_padding_mask: Optional[Tensor] = None,
                     src2_key_padding_mask: Optional[Tensor] = None,
                     pos_src1: Optional[Tensor] = None,
                     pos_src2: Optional[Tensor] = None):

        src2 = self.multihead_attn(query=self.with_pos_embed(src2, pos_src2),
                                   key=self.with_pos_embed(src1, pos_src1),
                                   value=src1, attn_mask=src1_mask,
                                   key_padding_mask=src1_key_padding_mask)[0]
        src2 = src2 + self.dropout(src2)
        src2 = self.norm(src2)
        return src2

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos_src1: Optional[Tensor] = None,
                pos_src2: Optional[Tensor] = None):

        return self.forward_post(src1, src2, src1_mask, src2_mask,
                                 src1_key_padding_mask, src2_key_padding_mask, pos_src1, pos_src2)


class DownSampleBlock(nn.Module):
    def __init__(self, d, d_down):
        super().__init__()
        self.fc = nn.Linear(d, d_down)
        self.ln = nn.LayerNorm(d_down)
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1]**0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        # pdb.set_trace()
        vit_embeds = self.flat_square(vit_embeds)
        # pdb.set_trace()
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0],-1,vit_embeds.shape[-1])
        vit_embeds = self.ln(self.fc(vit_embeds))
        return vit_embeds

    def flat_square(self, x):
        # pdb.set_trace()
        n, w, h, c = x.size()
        x = x.view(n, int(h//2), int(w//2), int(c*4))
        return x