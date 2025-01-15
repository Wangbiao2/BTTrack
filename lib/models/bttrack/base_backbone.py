import pdb
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.bttrack.utils import combine_tokens, recover_tokens


class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.return_stage = cfg.MODEL.RETURN_STAGES
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG
        self.use_bridge = cfg.MODEL.BACKBONE.USE_BRIDGE
        self.bridge_token_nums = cfg.MODEL.BACKBONE.BRIDGE_TOKEN_NUMS
        self.token_type_indicate = cfg.MODEL.BACKBONE.TOKEN_TYPE_INDICATE

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        if self.use_bridge and patch_start_index > 0:
            if self.bridge_token_nums == 1:
                cls_pos_embed = self.pos_embed[:, 0:1, :]
                self.cls_pos_embed = nn.Parameter(cls_pos_embed)
            elif self.bridge_token_nums > 1:
                self.cls_token = nn.Parameter(torch.zeros(1, self.bridge_token_nums, self.embed_dim))

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        if self.return_inter:
            for i_layer in self.return_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)

        if self.token_type_indicate:
            self.template_background_token = nn.Parameter(torch.zeros(self.embed_dim))
            self.template_foreground_token = nn.Parameter(torch.zeros(self.embed_dim))
            self.search_token = nn.Parameter(torch.zeros(self.embed_dim))

    def create_mask(self, image, image_anno):
        height = image.size(2)
        width = image.size(3)

        # Extract bounding box coordinates
        x0 = (image_anno[:, 0] * width).unsqueeze(1)
        y0 = (image_anno[:, 1] * height).unsqueeze(1)
        w = (image_anno[:, 2] * width).unsqueeze(1)
        h = (image_anno[:, 3] * height).unsqueeze(1)

        # Generate pixel indices
        x_indices = torch.arange(width, device=image.device)
        y_indices = torch.arange(height, device=image.device)

        # Create masks for x and y coordinates within the bounding boxes
        x_mask = ((x_indices >= x0) & (x_indices < x0 + w)).float()
        y_mask = ((y_indices >= y0) & (y_indices < y0 + h)).float()

        # Combine x and y masks to get final mask
        mask = x_mask.unsqueeze(1) * y_mask.unsqueeze(2) # (b,h,w)

        return mask

    # def transfer_attn(self, attn1):
    #     ## ----- only for plot attn score --------
    #     attn1 = torch.mean(attn1[:, :, :, 64:], dim=1)
    #     def token2feature(tokens):
    #         B, L, D = tokens.shape
    #         H = W = int(L ** 0.5)
    #         x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    #         return x
    #
    #     attn1 = token2feature(attn1.permute(0, 2, 1))
    #     attn1 = (attn1 - torch.min(attn1)) / (torch.max(attn1) - torch.min(attn1) + 1e-8)
    #     attn1 = nn.functional.interpolate(attn1, size=(256, 256), mode='bicubic',
    #                                       align_corners=False)
    #     return attn1


    def forward_features(self, z, x, template_anno):
        z_anno = template_anno.squeeze(0)
        if self.token_type_indicate:
            # generate a foreground mask
            z_indicate_mask = self.create_mask(z, z_anno)
            z_indicate_mask = z_indicate_mask.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size) # to match the patch embedding
            z_indicate_mask = z_indicate_mask.mean(dim=(3,4)).flatten(1) # elements are in [0,1], float, near to 1 indicates near to foreground, near to 0 indicates near to background

        if self.token_type_indicate:
            # generate the indicate_embeddings for z
            template_background_token = self.template_background_token.unsqueeze(0).unsqueeze(1).expand(z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
            template_foreground_token = self.template_foreground_token.unsqueeze(0).unsqueeze(1).expand(z_indicate_mask.size(0), z_indicate_mask.size(1), self.embed_dim)
            weighted_foreground = template_foreground_token * z_indicate_mask.unsqueeze(-1)
            weighted_background = template_background_token * (1 - z_indicate_mask.unsqueeze(-1))
            z_indicate = weighted_foreground + weighted_background

        B, Hx, Wx = x.shape[0], x.shape[2], x.shape[3]
        Hz, Wz = z.shape[2], z.shape[3]

        Hx = Hx // self.patch_size
        Wx = Wx // self.patch_size
        Hz = Hz // self.patch_size
        Wz = Wz // self.patch_size

        x = self.dau(x)
        z = self.dau(z)

        x = self.patch_embed(x)
        z = self.patch_embed(z)

        if self.use_bridge:
            bridge = self.cls_token.expand(B, -1, -1)
            if self.bridge_token_nums == 1:
                bridge = bridge + self.cls_pos_embed

        z = z + self.pos_embed_z
        x = x + self.pos_embed_x

        if self.token_type_indicate:
            # generate the indicate_embeddings for x
            x_indicate = self.search_token.unsqueeze(0).unsqueeze(1).expand(x.size(0), x.size(1), self.embed_dim)
            # add indicate_embeddings to z and x
            x = x + x_indicate
            z = z + z_indicate

        if self.use_bridge:
            x = combine_tokens(bridge, x, mode=self.cat_mode)
            x = combine_tokens(z, x, mode=self.cat_mode)
        else:
            x = combine_tokens(z, x, mode=self.cat_mode)

        x = self.pos_drop(x)

        aux_dict = {"attn": None}

        for i, blk in enumerate(self.blocks):
            x = blk(x, Hz, Wz, Hx, Wx, i)

        x = self.norm(x)

        if self.use_bridge:
            x = torch.cat((x[:, :Hz*Wz, :], x[:, Hz*Wz + bridge.shape[1]:, :]), dim=1)

        return x, aux_dict

    def forward(self, z, x, template_anno, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        x, aux_dict = self.forward_features(z, x, template_anno)

        return x, aux_dict
