import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Union, Optional, Dict, Tuple
from einops import rearrange
from mini_mask_git import utils


class TransformerSequenceEncoder(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 dim_model: int,
                 num_layers: int,
                 max_len: int,
                 **kwargs):
        super().__init__()

        self.embedding_dim = dim_model

        self.tok_embeds = nn.Embedding(vocab_size + 1, embed_dim)
        self.pos_embeds = nn.Embedding(max_len, embed_dim)

        self.to_input = nn.Linear(embed_dim, dim_model)

        self.register_buffer('positions', torch.arange(max_len)[None, :])

        encoder_layer = nn.TransformerEncoderLayer(dim_model, **kwargs, batch_first=True)

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        self.to_logits = nn.Linear(dim_model, vocab_size, bias=False)

    def forward(self, x, mask=None):
        b, *spatial_dims = x.shape

        x = x.view(b, -1)

        x = self.tok_embeds(x) + self.pos_embeds(self.positions[:, :x.shape[1]])
        x = self.to_input(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        x = self.to_logits(x)

        x = x.view(b, *spatial_dims, -1)

        return x


class ResBlock(nn.Module):
    def __init__(self, channels, batch_norm=False, conv_cls=nn.Conv2d, activation=nn.ReLU):
        super().__init__()

        norm = lambda *args, **kwargs: nn.Identity()
        if batch_norm:
            norm = nn.BatchNorm2d

        self.blocks = nn.Sequential(
            conv_cls(channels, channels, kernel_size=3, padding=1),
            activation(),
            norm(channels),

            conv_cls(channels, channels, kernel_size=3, padding=1),
            activation(),
            norm(channels)
        )

    def forward(self, input):
        return input + self.blocks(input)


class UpBlock(nn.Module):
    def __init__(self,
                 x1_channels,
                 x2_channels,
                 out_channels,
                 batch_norm=True,
                 conv_cls=nn.Conv2d,
                 conv_transpose_cls=nn.ConvTranspose2d,
                 activation=nn.ReLU):
        super().__init__()

        self.up = nn.Sequential(
            conv_transpose_cls(x1_channels, out_channels, kernel_size=4, stride=2, padding=1),
            activation()
        )
        self.merge = nn.Sequential(
            conv_cls(out_channels + x2_channels, out_channels, kernel_size=1),
            activation()
        )
        self.res_block = nn.Sequential(
            ResBlock(
                out_channels,
                conv_cls=conv_cls,
                activation=activation,
                batch_norm=batch_norm
            ),
            ResBlock(
                out_channels,
                conv_cls=conv_cls,
                activation=activation,
                batch_norm=batch_norm
            )
        )

    def forward(self, x1, x2):
        up = self.up(x1)
        cat = torch.cat([up, x2], dim=1)
        merge = self.merge(cat)
        out = self.res_block(merge)
        return out


class ReZero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x, *args, **kwargs):
        return x + self.alpha * self.fn(x, *args, **kwargs)


class WithChannelsFirst(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b ... c -> b c ...')
        x = self.fn(x, *args, **kwargs)
        x = rearrange(x, 'b c ... -> b ... c')
        return x


class WithChannelsLast(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        x = rearrange(x, 'b c ... -> b ... c')
        x = self.fn(x, *args, **kwargs)
        x = rearrange(x, 'b ... c -> b c ...')
        return x


class SelfAttention(nn.MultiheadAttention):
    def forward(self, x):
        return super().forward(x, x, x)[0]


class UNet(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 base_channel,
                 max_channel,
                 depth,
                 attn_indices=None,
                 attn_dim=None,
                 attn_heads=1,
                 batch_norm=True,
                 conv_cls=nn.Conv2d,
                 conv_transpose_cls=nn.ConvTranspose2d,
                 activation=nn.ReLU):
        super().__init__()

        self.num_classes = in_dim

        self.encoder_blocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()

        self.up_blocks = nn.ModuleList()

        self.first = conv_cls(in_dim, base_channel, kernel_size=1)
        self.final = conv_cls(base_channel, out_dim, kernel_size=1)

        if isinstance(attn_indices, int):
            if attn_indices == -1:
                attn_indices = depth - 1
            self.attn_indices = [attn_indices]
        elif attn_indices is None:
            self.attn_indices = []

        def ch_for_depth(d):
            return min(max_channel, base_channel * 2 ** d)

        for i in range(depth):
            ch_in = ch_for_depth(i)
            ch_out = ch_for_depth(i + 1)

            curr_encoder_blocks = [
                ResBlock(ch_in, conv_cls=conv_cls, activation=activation, batch_norm=batch_norm)
            ]

            if i in self.attn_indices:
                curr_encoder_blocks.append(
                    ReZero(WithChannelsLast(SelfAttention(
                        embed_dim=attn_dim or ch_in,
                        num_heads=attn_heads,
                        batch_first=True
                    )))
                )

            self.encoder_blocks.append(nn.Sequential(*curr_encoder_blocks))

            if i + 1 != depth:
                self.down_blocks.append(nn.Sequential(
                    conv_cls(ch_in, ch_out, kernel_size=4, stride=2, padding=1),
                    activation(),
                ))

        for i in range(depth)[:0:-1]:
            self.up_blocks.append(UpBlock(
                x1_channels=ch_for_depth(i),
                x2_channels=ch_for_depth(i - 1),
                out_channels=ch_for_depth(i - 1),
                conv_cls=conv_cls,
                conv_transpose_cls=conv_transpose_cls,
                activation=activation,
                batch_norm=batch_norm
            ))

    def forward(self, input):
        out = self.first(input)

        outputs = []

        for i, encoder_block in enumerate(self.encoder_blocks):
            out = encoder_block(out)

            if i + 1 != len(self.encoder_blocks):
                outputs.append(out)
                out = self.down_blocks[i](out)

        outputs = outputs[::-1]

        for skip, up_block in zip(outputs, self.up_blocks):
            out = up_block(out, skip)

        out = self.final(out)

        return out


class UNetSequenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, **kwargs):
        super().__init__()

        self.embeds = nn.Embedding(vocab_size + 1, embed_dim)

        self.unet = WithChannelsFirst(
            UNet(
                in_dim=embed_dim,
                out_dim=vocab_size,
                **kwargs
            )
        )

    def forward(self, x, mask=None):
        if mask is not None and mask.sum() > 0:
            raise NotImplementedError('Masking is currently not supported by UNetSequenceEncoder')

        x = self.embeds(x)
        x = self.unet(x)

        return x
