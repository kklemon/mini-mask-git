import yaml
import torch

from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ


def load_vqgan_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_vqgan(config, ckpt_path=None, is_gumbel=False):
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
