import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import einsum


def FeedForward(dim, mult=4):
    inner_dim = int(mult * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, num_null_kv=2):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, mask=None):
        batch, device, dtype = x.shape[0], x.device, x.dtype

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q = q * self.scale

        nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b=batch, r=2).unbind(dim=-2)

        k = torch.cat((nk, k), dim=-2)
        v = torch.cat((nv, v), dim=-2)

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[-2:]

        if mask is not None:
            mask = F.pad(mask, (j - mask.shape[-1], 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            dim_head=64,
            heads=8,
            ff_mult=4,
            attn_num_null_kv=2,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, dim_head=dim_head, heads=heads, num_null_kv=attn_num_null_kv),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, mask=None):

        for self_attn, ff in self.layers:
            x = self_attn(x, mask=mask) + x
            x = ff(x) + x

        return self.norm_out(x)


class MaskGIT(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 dim: int,
                 depth: int,
                 max_len: int,
                 **kwargs):
        super().__init__()

        self.vocab_size = vocab_size
        self.mask_idx = vocab_size

        self.tok_embeds = nn.Embedding(vocab_size + 1, dim)
        self.pos_embeds = nn.Embedding(max_len, dim)

        self.register_buffer('positions', torch.arange(max_len)[None, :])

        self.encoder = TransformerEncoder(dim=dim, depth=depth, **kwargs)
        self.to_logits = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x, mask=None):
        b, *spatial_dims = x.shape

        x = x.view(b, -1)

        x = self.tok_embeds(x) + self.pos_embeds(self.positions[:, :x.shape[1]])
        x = self.encoder(x, mask=mask)
        x = self.to_logits(x)

        x = x.view(b, *spatial_dims, -1)

        return x

    @torch.no_grad()
    def sample(self, size, num_samples, scheduling_fn, num_steps=10, temperature=4.5, return_intermediates=False):
        if isinstance(size, int):
            size = (size, size)

        batch = torch.full((num_samples, *size), self.mask_idx, dtype=torch.long).to(self.positions)
        num_tokens = np.prod(size)

        intermediates = []

        mask_nums = torch.floor(scheduling_fn((torch.arange(num_steps) + 1) / num_steps) * num_tokens).long()

        for num_mask in mask_nums:
            logits = self(batch) / temperature
            probas = logits.softmax(-1)
            samples = probas.view(-1, probas.shape[-1]).multinomial(num_samples=1).view(batch.shape)

            confidence = torch.take_along_dim(probas, samples.unsqueeze(-1), dim=-1).squeeze(-1)
            confidence[batch != self.mask_idx] = 1.0

            min_confidence = confidence.view(num_samples, -1).sort(-1)[0][:, num_mask]

            mask = (confidence >= min_confidence[:, None, None]) & (batch == self.mask_idx)

            assert not (mask & (batch != self.mask_idx)).any()

            batch[mask] = samples[mask]

            if return_intermediates:
                mask = batch == self.mask_idx

                inter_batch = batch.clone()
                inter_batch[mask] = samples[mask]

                intermediates.append(inter_batch)

        if return_intermediates:
            return torch.stack(intermediates)

        return batch


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
