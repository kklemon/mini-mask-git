import wandb
import pytorch_lightning as pl

from typing import Any
from einops import rearrange
from pytorch_lightning import Callback
from pytorch_lightning.cli import SaveConfigCallback
from torchvision.utils import make_grid
from mini_mask_git.vqgan import load_vqgan_config, load_vqgan, postprocess_vqgan


class LogConfigCallback(SaveConfigCallback):
    def setup(self, trainer, pl_module, stage):
        for logger in trainer.loggers:
            logger.log_hyperparams(self.config)


class SamplingCallback(Callback):
    DEFAULT_DECODING_STEPS = [4, 8, 12, 16]

    def __init__(self, decoder_config_path, decoder_ckpt_path, num_samples=8):
        config = load_vqgan_config(decoder_config_path, display=False)

        self.model = load_vqgan(config, ckpt_path=decoder_ckpt_path)
        self.model.eval()

        self.model.quantize.embed_code = lambda x: self.model.quantize.embedding(x).permute(0, 3, 1, 2)

        self.num_samples = num_samples
        self.first_call = True

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.model = self.model.to(pl_module.device)

    def on_validation_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if batch_idx != 0:
            return

        log_dict = {}

        for sample_steps in self.DEFAULT_DECODING_STEPS:
            samples = pl_module.encoder.sample(
                size=pl_module.spatial_size,
                num_samples=self.num_samples,
                num_steps=sample_steps,
                scheduling_fn=pl_module.scheduling_fn,
                return_intermediates=True
            )

            samples = rearrange(samples, 'n b ... -> (n b) ...')

            decoded = self.model.decode_code(samples)
            decoded = postprocess_vqgan(decoded).cpu()

            grid = make_grid(decoded, nrow=self.num_samples, normalize=True, scale_each=True)

            log_dict[f'valid/samples/steps_{sample_steps}'] = wandb.Image(grid)

        trainer.logger.experiment.log(log_dict)
