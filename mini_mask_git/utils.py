import torch
import torch.nn as nn
import numpy as np

from itertools import chain


def get_params_without_weight_decay_ln(named_params, weight_decay):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            'weight_decay': weight_decay,
        },
        {
            'params': [p for n, p in named_params if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
        },
    ]
    return optimizer_grouped_parameters


def patch_forward(m, **overwrite_kwargs):
    forward_orig = m.forward

    def wrap(*args, **kwargs):
        kwargs = {**kwargs, **overwrite_kwargs}
        return forward_orig(*args, **kwargs)

    m.forward = wrap

    return forward_orig


def identity(x, *args, **kwargs):
    return x


def merge_dicts(dicts, cat_fn=None):
    """
    Args:
        dicts: Dictionaries to merge key-wise.
        cat_fn (optional): Function to use for merging dictionary values.

    Example:
    >>> merge_dicts({'a': 0, 'b': 42}, {'a': 1})
    {'a': [0, 1], 'b': [42]}
    """
    if cat_fn is None:
        cat_fn = identity

    keys = set(chain.from_iterable(dicts))
    merged = {k: cat_fn([d[k] for d in dicts if k in d]) for k in keys}
    return merged


def maybe_list(o, to_type=list):
    if not isinstance(o, (tuple, list)):
        return to_type((o,))
    return o


def indexed_select(o, indices):
    return [o[i] for i in indices]


def param_count(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


scheduling_functions = {
    'exponential': lambda x: 1 - (torch.exp(6 * x) - 1) / (np.exp(6) - 1),
    'cubic': lambda x: 1 - x ** 3,
    'square': lambda x: 1 - x ** 2,
    'cosine': lambda x: torch.clamp(torch.cos(x * torch.pi * 0.5), 0.0, 1.0),
    'linear': lambda x: 1 - x,
    'square_root': lambda x: 1 - x ** 0.5,
}


def get_scheduling_function(name):
    return scheduling_functions[name]


def generate_random_mask(x, num_mask_per_sample):
    bs, *spatial_dims = x.shape
    n = int(np.prod(spatial_dims))
    device = x.device

    perm = torch.multinomial(torch.full((bs, n), 1 / n, device=device), num_samples=n, replacement=False)
    perm_inv = torch.argsort(perm, -1)

    index_mask = torch.arange(n, device=device).expand(bs, -1) < num_mask_per_sample[:, None]

    mask = torch.zeros((bs, n), dtype=bool, device=device)
    mask = mask.take_along_dim(perm, -1)
    mask[index_mask] = True
    mask = mask.take_along_dim(perm_inv, -1)

    return mask.view(x.shape)
