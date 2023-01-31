import argparse
import pickle
import yaml
import lmdb
import torch
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from pathlib import Path
from functools import partial
from omegaconf import OmegaConf
from taming.models.vqgan import VQModel, GumbelVQ
from torch.utils.data import DataLoader
from tqdm import tqdm
from mini_mask_gpt.data import ImageFolderDataset


def load_config(config_path, display=False):
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


def preprocess(img, target_image_size=256):
    s = min(img.size)

    # if s < target_image_size:
    #     raise ValueError(f'min dim for image {s} < {target_image_size}')

    r = target_image_size / s
    s = [round(r * img.size[1]), round(r * img.size[0])]
    img = TF.resize(img, s, interpolation=TF.InterpolationMode.LANCZOS)
    img = TF.center_crop(img, output_size=2 * [target_image_size])
    img = T.ToTensor()(img)
    return img


def encode(x, model):
    _, _, (_, _, indices) = model.encode(x)
    return indices


@torch.no_grad()
def main(args):
    config = load_config(args.config_path, display=False)
    model = load_vqgan(config, ckpt_path=args.ckpt_path).to(args.device)
    model.eval()
    model.quantize.sane_index_shape = True

    dataset = ImageFolderDataset(
        root=args.data_root,
        transform=partial(preprocess, target_image_size=args.image_size)
    )

    batches = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    index = 0

    with (
        lmdb.open(args.save_path, map_size=args.lmdb_map_size) as env,
        env.begin(write=True) as txn,
        torch.autocast(device_type='cuda', enabled=args.fp16)
    ):
        for batch in tqdm(batches):
            batch = preprocess_vqgan(batch.to(args.device))
            batch_indices = encode(batch, model).cpu().numpy().astype(np.int32)

            for image_indices in batch_indices:
                txn.put(str(index).encode(), pickle.dumps(image_indices))
                index += 1




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lmdb_map_size', type=int, default=256 * 2 ** 30)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--device', default='cuda')

    main(parser.parse_args())
