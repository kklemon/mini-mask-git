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
                 lr: float = 4e-4,
                 weight_decay: float = 1e-2,
                 training_steps: int = 1_000_000,
                 warmup_steps: int = 3_000,
                 compile_model: bool = False,
                 scheduling_function: str = 'cosine'):
        super().__init__()

        self.encoder = encoder

        if compile_model:
            assert version.parse(torch.__version__).major >= 2, \
                'Compiling models is only supported by PyTorch version >= 2'
            self.encoder = torch.compile(self.encoder)

        self.lr = lr
        self.weight_decay = weight_decay
        self.training_steps = training_steps
        self.warmup_steps = warmup_steps
        self.scheduling_fn = utils.get_scheduling_function(scheduling_function)

        # self.encoder.apply(utils.weights_init)

        print(self.encoder)

    def configure_optimizers(self):
        # params = self.encoder.named_parameters()
        # param_groups = get_params_without_weight_decay_ln(params, weight_decay=self.weight_decay)

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

        mask_fracs = self.scheduling_fn(torch.rand(len(tokens), device=self.device))
        mask_num = torch.ceil(mask_fracs * self.num_tokens)
        mask = utils.generate_random_mask(tokens, mask_num)

        samples = self.sample_tokens(tokens)
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

    def sample_tokens(self, batch):
        return torch.multinomial(
            self.trainer.datamodule.counts, batch.numel(), replacement=True
        ).to(batch.device).view(batch.shape)

    @torch.no_grad()
    def sample(self, num_samples, num_steps=10, return_intermediate=False, inverse_sampling=False):
        # Start with an entirely masked grid
        batch = self.sample_tokens(torch.empty(
            size=(num_samples, *self.spatial_size),
            dtype=torch.long,
            device=self.device
        ))
        batch_mask = torch.zeros_like(batch, dtype=bool)

        num_tokens = np.prod(batch.shape[1:])

        intermediate = []

        mask_nums = torch.floor(self.scheduling_fn((torch.arange(num_steps) + 1) / num_steps) * num_tokens).long()

        for num_mask in mask_nums:
            samples = self.sample_tokens(batch)

            confidence = self.encoder(batch).sigmoid()

            if inverse_sampling:
                _, indices = confidence.reshape(num_samples, -1).topk(num_mask, largest=False, dim=-1)
                batch.view(num_samples, -1).scatter_(1, indices, samples.view(num_samples, -1))
            else:
                confidence[batch_mask] = 1.0
                min_confidence = confidence.view(num_samples, -1).sort(-1)[0][:, num_mask]

                mask = confidence < min_confidence[:, None, None]

                batch[mask] = samples[mask]
                batch_mask[~mask] = 1

            if return_intermediate:
                intermediate.append(batch.clone())

        if return_intermediate:
            return torch.stack(intermediate)

        return batch
