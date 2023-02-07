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

        self.dim = dim
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


class DiscGIT(MaskGIT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tok_embeds = nn.Embedding(self.vocab_size, self.dim)
        self.to_logits = nn.Linear(self.dim, 1)

    def forward(self, x, mask=None):
        return super().forward(x, mask).squeeze()

    def sample_tokens(self, batch, probas=None):
        if probas is None:
            probas = torch.full((self.vocab_size,), 1 / self.vocab_size)

        return torch.multinomial(
            probas, batch.numel(), replacement=True
        ).to(batch).view(batch.shape)

    @torch.no_grad()
    def sample(self, size, num_samples, scheduling_fn, num_steps=10, sampling_probas=None, return_intermediates=False):
        if isinstance(size, int):
            size = (size, size)

        batch = self.sample_tokens(
            torch.full((num_samples, *size), self.mask_idx, dtype=torch.long).to(self.positions),
            probas=sampling_probas
        )

        num_tokens = np.prod(size)

        intermediates = []

        if isinstance(scheduling_fn, int):
            mask_nums = torch.full(num_steps, scheduling_fn)
        else:
            mask_nums = torch.floor(scheduling_fn((torch.arange(num_steps) + 1) / num_steps) * num_tokens).long()

        for num_mask in mask_nums:
            samples = self.sample_tokens(batch, sampling_probas)

            confidence = self(batch)
            _, indices = confidence.reshape(num_samples, -1).topk(num_mask, largest=False, dim=-1)
            batch.view(num_samples, -1).scatter_(1, indices, samples.view(num_samples, -1))

            if return_intermediates:
                intermediates.append(batch.clone())

        if return_intermediates:
            return torch.stack(intermediates)

        return batch