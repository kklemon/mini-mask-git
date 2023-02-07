import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np

from torch.optim import AdamW
from packaging import version
from mini_mask_git import utils
from mini_mask_git.training import CosineWithWarmupLR
from mini_mask_git.utils import get_params_without_weight_decay_ln


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
    def num_embeds(self):
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

        corrupt_ratio = self.scheduling_fn(torch.rand(len(tokens), device=self.device))
        corrupt_num = torch.ceil(corrupt_ratio * self.num_tokens)
        mask = utils.generate_random_mask(tokens, corrupt_num)

        samples = self.encoder.sample_tokens(tokens, self.trainer.datamodule.counts)
        corrupted_tokens = mask * samples + (~mask) * tokens

        labels = (tokens == corrupted_tokens).float()

        logits = self.encoder(corrupted_tokens)

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        self.log(f'{log_prefix}/loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, log_prefix='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, log_prefix='valid')
