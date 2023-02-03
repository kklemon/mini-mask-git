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

        opt = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.lr,
            betas=(0.9, 0.6)
        )

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

        mask_num = torch.ceil(mask_fracs * self.num_tokens)
        mask = utils.generate_random_mask(tokens, mask_num)

        masked_tokens = mask * self.mask_idx + (~mask) * tokens

        logits = self.encoder(masked_tokens)

        # loss = F.cross_entropy(
        #     logits.reshape(-1, logits.shape[-1]),
        #     tokens.view(-1),
        #     label_smoothing=0.1
        # )

        # loss_per_token = (-F.log_softmax(logits, -1) * F.one_hot(tokens, num_classes=self.mask_idx)).sum(-1)

        loss_per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tokens.view(-1),
            reduction='none',
            label_smoothing=0.1
        ).view(tokens.shape)

        loss_per_token[~mask] = 0
        loss_per_sample = loss_per_token.sum((1, 2)) / mask.sum((1, 2))
        loss = loss_per_sample.mean()

        self.log(f'{log_prefix}/loss', loss, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, log_prefix='train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, log_prefix='valid')

    @torch.no_grad()
    def sample(self, num_samples, num_steps=10, temperature=1.0, return_intermediate=False):
        # Start with an entirely masked grid
        batch = torch.full((num_samples, *self.spatial_size), self.mask_idx, dtype=torch.long, device=self.device)
        num_tokens = np.prod(batch.shape[1:])

        intermediate = []

        mask_nums = torch.ceil(self.scheduling_fn((torch.arange(num_steps) + 1) / num_steps) * num_tokens).long()

        for num_mask in mask_nums:
            logits = self.encoder(batch) / temperature
            probas = logits.softmax(-1)
            samples = probas.view(-1, probas.shape[-1]).multinomial(num_samples=1).view(batch.shape)

            confidence = torch.take_along_dim(probas, samples.unsqueeze(-1), dim=-1).squeeze(-1)
            confidence[batch != self.mask_idx] = 1.0

            min_confidence = confidence.view(num_samples, -1).sort(-1)[0][:, num_mask]

            mask = (confidence >= min_confidence[:, None, None]) & (batch == self.mask_idx)

            assert not (mask & (batch != self.mask_idx)).any()

            batch[mask] = samples[mask]

            if return_intermediate:
                mask = batch == self.mask_idx

                inter_batch = batch.clone()
                inter_batch[mask] = samples[mask]

                intermediate.append(inter_batch)

        if return_intermediate:
            return torch.stack(intermediate)

        return batch
