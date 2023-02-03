from pytorch_lightning.cli import LightningCLI
from mini_mask_git.lightning.callbacks import LogConfigCallback


class MaskGITCLI(LightningCLI):
    def __init__(self, *args, **kwargs):
        if 'save_config_callback' not in kwargs:
            kwargs['save_config_callback'] = LogConfigCallback

        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments(
            'data.num_embeds',
            'model.encoder.init_args.vocab_size',
            apply_on='instantiate',
        )

