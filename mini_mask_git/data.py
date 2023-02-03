import math
import pickle
import numpy as np
import lmdb

from pathlib import Path
from PIL import Image, ImageFile
from torch.utils.data import Dataset



class ImageFolderDataset(Dataset):
    def __init__(
            self,
            root,
            transform=None,
            ext=('.png', '.jpg', '.jpeg', 'bmp'),
            recursive=False,
            return_filename=False
    ):
        if recursive:
            glob_pattern = '**/*'
        else:
            glob_pattern = '*'

        self.files = [f for f in Path(root).glob(glob_pattern) if f.is_file() and f.suffix.lower() in ext]
        self.transform = transform
        self.return_filename = return_filename
        self.mode = 'RGB'

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        with open(filename, 'rb') as f:
            img = Image.open(f)
            img = img.convert(self.mode)
        if self.transform:
            img = self.transform(img)
        if self.return_filename:
            return img, str(filename)
        return img


class LatentImageDataset(Dataset):
    def __init__(self, path, mask=False, dummy=False):
        self.dummy = dummy
        self.mask = mask

        if dummy:
            self.keys = list(range(1024))
        else:
            self.env = lmdb.open(str(path), readonly=True)
            self.txn = self.env.begin()

            self.keys = list(self.txn.cursor().iternext(values=False))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.dummy:
            x = np.random.randint(0, 16384, (16, 16))
        else:
            x = pickle.loads(self.txn.get(self.keys[idx]))

        mask = np.zeros_like(x, dtype=bool)

        if self.mask:
            num_tokens = x.size
            num_mask = math.ceil(np.random.random() * num_tokens)
            mask_indices = np.random.choice(np.arange(num_tokens), size=num_mask, replace=False)

            mask = mask.reshape(-1)
            mask[mask_indices] = True
            mask = mask.reshape(x.shape)

        return x, mask
