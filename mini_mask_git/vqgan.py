import yaml
import torch

from omegaconf import OmegaConf


def load_vqgan_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
    try:
        from taming.models.vqgan import VQModel, GumbelVQ
    except ImportError:
        raise ImportError('Install taming-transformers library to support loading VQGAN models.')

    if is_gumbel:
        model = GumbelVQ(**config.model.params)
    else:
      model = VQModel(**config.model.params)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def preprocess_vqgan(x):
    x = 2.0 * x - 1.
    return x


def postprocess_vqgan(x):
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    return x
