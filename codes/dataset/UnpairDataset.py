from sympy import false
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob
from typing import Any, Callable, List
import pickle as pkl
import lmdb
from typing import Annotated
from torchvision.transforms import v2


class UnpairDataset(Dataset):
    def __init__(self,
                 path: str,
                 datatype: str = 'image',
                 max_len: int = int(1e9),
                 crop_size: Annotated[int, 'between [0,512]'] = 512,
                 augment: bool = False
                 ) -> None:
        super().__init__()

        # Assert that the path exists
        assert os.path.exists(path), f"Provided path '{path}' does not exist."

        # Assert that datatype is either 'image' or 'lmdb'
        assert datatype in [
            'image', 'lmdb'], "datatype must be either 'image' or 'lmdb'."

        # Assert that max_len is a positive integer
        assert max_len > 0, "max_len must be a positive integer."

        # Assert that crop_size is within the valid range
        assert 0 <= crop_size <= 512 or crop_size == - \
            1, "crop_size must be between 0 and 512, or -1 if not used."

        self.path = path
        self.datatype = datatype
        self.max_len = max_len
        self.input_keys: List[str] = None
        self.get_data_func: Callable = None
        transforms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
                      v2.RandomCrop(crop_size)]
        if augment:
            transforms.append(
                v2.RandomHorizontalFlip(p=0.5)
            )
            transforms.append(
                v2.RandomVerticalFlip(p=0.5)
            )

        self.transforms = v2.Compose(transforms)
        self.init_data(path)

    def __len__(self):
        return min(self.max_len, len(self.input_keys))

    def __getitem__(self, idx: int):
        key = self.input_keys[idx]
        return {'input': self.transforms(self.get_data_func(key))}

    def init_data(self, path: str) -> None:
        def ndBGR2RGB(img: np.ndarray) -> np.ndarray:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.datatype == 'image':
            self.input_keys = self.get_images(path)

            def _read(key: str):
                return ndBGR2RGB(cv2.imread(os.path.join(self.path, key), cv2.IMREAD_COLOR))
            self.get_data_func = _read
        else:
            def _read(key: str):
                return ndBGR2RGB(self.read_lmdb_by_key(key))
            self.input_keys = self.read_pkl(
                os.path.join(path, "meta_info.pkl"))['keys']
            self.get_data_func = _read

    def get_lmdb_info(self, lmdb_path: str) -> None:
        with lmdb.open(
            lmdb_path, readonly=True, lock=False, readahead=False, meminit=False
        ) as env:
            with env.begin(write=False) as txn:
                return txn.stat()

    def read_pkl(self, pkl_path: str) -> Any:
        with open(pkl_path, "rb") as f:
            data = pkl.load(f)
        return data

    def get_images(
        self,
        path: str,
        suffixs: list[str] = [
            ".jpg",
            ".JPG",
            ".jpeg",
            ".JPEG",
            ".png",
            ".PNG",
            ".ppm",
            ".PPM",
            ".bmp",
            ".BMP",
        ],
    ) -> list[str]:
        image_paths = []
        for suffix in suffixs:
            image_paths.extend(sorted(self.get_images_by_suffix(path, suffix)))
        return image_paths

    def get_images_by_suffix(self, path: str, suffix: str) -> list[str]:
        target_path: str = os.path.join(path, "**", "*" + suffix)
        res: List[str] = glob.glob(target_path, recursive=True)
        return res

    def init_lmdb_env(self, lmdb_path: str) -> lmdb.Environment:
        return lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def read_lmdb_by_key(self, key: str) -> Any:
        env = self.init_lmdb_env(self.path)
        with env.begin(write=False) as txn:
            return cv2.imdecode(
                np.frombuffer(txn.get(key.encode("ascii")),
                              np.uint8), cv2.IMREAD_COLOR
            )
