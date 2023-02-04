import math

import wandb
import torch
import pytorch_lightning as pl

from typing import Any
from einops import rearrange
from pytorch_lightning import Callback
from pytorch_lightning.cli import SaveConfigCallback
from torchvision.utils import make_grid

from mini_mask_git import utils
from mini_mask_git.vqgan import load_vqgan_config, load_vqgan, postprocess_vqgan


class LogConfigCallback(SaveConfigCallback):
    def setup(self, trainer, pl_module, stage):
        for logger in trainer.loggers:
            logger.log_hyperparams(self.config)


class SampleCallback(Callback):
    def __init__(self, decoder_config_path, decoder_ckpt_path, num_samples=8, decode_batch_size: int = 1):
        config = load_vqgan_config(decoder_config_path, display=False)
        self.model = load_vqgan(config, ckpt_path=decoder_ckpt_path)
        self.model.eval()

        self.model.quantize.embed_code = lambda x: self.model.quantize.embedding(x).permute(0, 3, 1, 2)

        self.num_samples = num_samples
        self.decode_batch_size = decode_batch_size

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.model = self.model.to(pl_module.device)


    def decode(self, codes):
        res = self.model.decode_code(codes)
        res = postprocess_vqgan(res).cpu()
        return res

    @torch.no_grad()
    def on_validation_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if batch_idx != 0:
            return

        log_dict = {}

        for num_steps in [4, 8, 12, 16, 24, 32]:
            with torch.autocast(device_type='cuda', enabled=True):
                samples = pl_module.sample(
                    num_samples=self.num_samples,
                    num_steps=num_steps,
                    return_intermediate=True
                )
                samples = rearrange(samples, 'n b ... -> (n b) ...')

                # Batched decoding for memory reasons
                sample_batches = samples.chunk(math.ceil(len(samples) / self.decode_batch_size))
                decoded = torch.cat([self.decode(sample_batch) for sample_batch in sample_batches])

            decoded = decoded.float()

            grid = make_grid(decoded, nrow=self.num_samples, normalize=True, scale_each=True)

            log_dict[f'valid/samples/num_steps_{num_steps}'] = wandb.Image(grid)

        # tokens, _ = batch
        # tokens = tokens.long()[:32]
        #
        # mask_fracs = pl_module.scheduling_fn(torch.linspace(0, 1, len(tokens), device=tokens.device))
        #
        # mask_num = torch.ceil(mask_fracs * pl_module.num_tokens)
        # mask = utils.generate_random_mask(tokens, mask_num)
        #
        # samples = torch.multinomial(
        #     trainer.datamodule.counts, tokens.numel(), replacement=True
        # ).to(tokens.device).view(tokens.shape)
        # corrupted_tokens = mask * samples + (~mask) * tokens
        #
        # decoded = self.decode(torch.cat([corrupted_tokens, tokens[:32]], dim=0))
        # decoded = rearrange(decoded, '(n b) ... -> (b n) ...', n=2)
        #
        # grid = make_grid(decoded, nrow=2, normalize=True, scale_each=True)
        #
        # log_dict[f'valid/samples/corrupted'] = wandb.Image(grid)

        trainer.logger.experiment.log(log_dict)
