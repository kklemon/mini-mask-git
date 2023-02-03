import argparse
import pickle
from multiprocessing import Pool
import multiprocessing as mp
from multiprocessing.pool import ThreadPool

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
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from mini_mask_git.data import ImageFolderDataset
from mini_mask_git.vqgan import load_vqgan_config, load_vqgan, preprocess_vqgan

mp.log_to_stderr()


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


def check(idx, dataset):
    try:
        dataset[idx]
    except KeyboardInterrupt:
        raise
    except:
        filename = dataset.files[idx]
        print(f'Removing {filename}')
        Path(filename).unlink()


@torch.no_grad()
def main(args):
    config = load_vqgan_config(args.config_path, display=False)
    model = load_vqgan(config, ckpt_path=args.ckpt_path).to(args.device)
    model.eval()
    model.quantize.sane_index_shape = True

    dataset = ImageFolderDataset(
        root=args.data_root,
        transform=partial(preprocess, target_image_size=args.image_size),
        return_filename=False
    )

    if args.max_samples:
        dataset = Subset(dataset, list(range(min(len(dataset), args.max_samples))))

    # with ThreadPool(32) as pool:
    #     for res in pool.imap(partial(check, dataset=dataset), tqdm(range(len(dataset))), chunksize=1000):
    #         pass
    #
    # return

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
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--lmdb_map_size', type=int, default=256 * 2 ** 30)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_samples', type=int)

    main(parser.parse_args())
