import torch

from mini_mask_git.lightning.data_modules import LitLatentImageDatamodule
from mini_mask_git.lightning.models import LitMaskGIT
from mini_mask_git.lightning.cli import MaskGITCLI


torch.set_float32_matmul_precision('medium')


def main():
    MaskGITCLI(
        LitMaskGIT,
        LitLatentImageDatamodule,
        save_config_overwrite=True,
        parser_kwargs={'parser_mode': 'omegaconf'}
    )


if __name__ == '__main__':
    main()
