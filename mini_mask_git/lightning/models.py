import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from einops import rearrange
from packaging import version
from mini_mask_git import utils
from mini_mask_git.training import CosineWithWarmupLR


class LitMaskGIT(pl.LightningModule):
    def __init__(self,
                 encoder: nn.Module,
                 lr: float = 3e-4,
                 weight_decay: float = 1e-2,
                 training_steps: int = 1_000_000,
                 warmup_steps: int = 10_000,
                 compile_model: bool = False,
                 scheduling_function: str = 'cosine'):
        super().__init__()

        self.encoder = encoder
        self.encoder.apply(utils.weights_init)

        if compile_model:
            assert version.parse(torch.__version__).major >= 2, \
                'Compiling models is only supported by PyTorch version >= 2'
            self.encoder = torch.compile(self.encoder)

        self.lr = lr
        self.weight_decay = weight_decay
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.scheduling_fn = utils.get_scheduling_function(scheduling_function)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.encoder.parameters(), lr=self.lr)

        scheduler = CosineWithWarmupLR(
            opt,
            training_steps=self.training_steps,
            warmup_steps=self.warmup_steps
        )

        return (
            [opt],
            [{'scheduler': scheduler, 'interval': 'step'}]
        )

    @property
    def mask_idx(self):
        return self.trainer.datamodule.num_embeds

    @property
    def spatial_size(self):
        return self.trainer.datamodule.spatial_size

    @property
    def num_tokens(self):
        return int(np.prod(self.spatial_size))

    def forward(self, x):
        return self.encoder(x)

    def step(self, batch, log_prefix):
        tokens, _ = batch
        tokens = tokens.long()

        mask_fracs = self.scheduling_fn(torch.rand(len(tokens), device=self.device))

        assert 0.0 <= mask_fracs.min() and mask_fracs.max() <= 1.0
        assert self.mask_idx > tokens.max()
        assert tokens.shape[1:] == self.spatial_size

        mask_num = torch.ceil(mask_fracs * self.num_tokens).clamp(1, self.num_tokens)
        mask = utils.generate_random_mask(tokens, mask_num)
        masked_tokens = mask * self.mask_idx + (~mask) * tokens

        logits = self.encoder(masked_tokens)

        tokens[~mask] = -1

        loss = F.cross_entropy(
            rearrange(logits, 'b w h d -> (b w h) d'),
            rearrange(tokens, 'b w h -> (b w h)'),
            ignore_index=-1,
            label_smoothing=0.1
        )

        self.log(f'{log_prefix}/loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, log_prefix='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, log_prefix='valid')
