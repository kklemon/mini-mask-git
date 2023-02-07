import wandb
import pytorch_lightning as pl

from itertools import product
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


class MaskGITSamplingCallback(Callback):
    DEFAULT_TEMPERATURES = [1.0, 2.0, 4.5]
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

    def log_samples(self, module):
        log_dict = {}

        for temperature, sample_steps in product(self.DEFAULT_TEMPERATURES, self.DEFAULT_DECODING_STEPS):
            samples = module.encoder.sample(
                size=module.spatial_size,
                num_samples=self.num_samples,
                num_steps=sample_steps,
                scheduling_fn=module.scheduling_fn,
                temperature=temperature,
                return_intermediates=True
            )

            samples = rearrange(samples, 'n b ... -> (n b) ...')

            decoded = self.model.decode_code(samples)
            decoded = postprocess_vqgan(decoded).cpu()

            grid = make_grid(decoded, nrow=self.num_samples, normalize=True, scale_each=True)

            log_dict[f'valid/samples/temperature_{temperature}'] = wandb.Image(grid)

        return log_dict

    def log_reconstructions(self, batch):
        tokens, _ = batch
        tokens = tokens[:16]

        decoded = self.model.decode_code(tokens)
        decoded = postprocess_vqgan(decoded).cpu()

        grid = make_grid(decoded, nrow=4, normalize=True, scale_each=True)

        return {
            f'valid/reconstructions': wandb.Image(grid)
        }

    def on_validation_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if batch_idx != 0:
            return

        log_dict = {}

        if self.first_call:
            log_dict.update(self.log_reconstructions(batch))

        log_dict.update(self.log_samples(pl_module))

        trainer.logger.experiment.log(log_dict)


class DiscGITSamplingCallback(Callback):
    def __init__(self, decoder_config_path, decoder_ckpt_path, num_samples=8):
        config = load_vqgan_config(decoder_config_path, display=False)

        self.model = load_vqgan(config, ckpt_path=decoder_ckpt_path)
        self.model.eval()

        self.model.quantize.embed_code = lambda x: self.model.quantize.embedding(x).permute(0, 3, 1, 2)

        self.num_samples = num_samples

    def on_fit_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.model = self.model.to(pl_module.device)

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
                    return_intermediate=True,
                    inverse_sampling=self.inverse_sampling
                )
                samples = rearrange(samples, 'n b ... -> (n b) ...')

                # Batched decoding for memory reasons
                sample_batches = samples.chunk(math.ceil(len(samples) / self.decode_batch_size))
                decoded = torch.cat([self.decode(sample_batch) for sample_batch in sample_batches])

            decoded = decoded.float()

            grid = make_grid(decoded, nrow=self.num_samples, normalize=True, scale_each=True)

            log_dict[f'valid/samples/num_steps_{num_steps}'] = wandb.Image(grid)

trainer.logger.experiment.log(log_dict)