
import json
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import glob
from typing import Any, Callable, KeysView, List, Tuple
import pickle as pkl
import lmdb
from typing import Annotated, Dict
from torchvision.transforms import v2
import torchvision.transforms as T
from torchvision.transforms.v2 import functional as v2F, Transform


class PairTransform:
    def __init__(self,
                 crop: int,
                 augment: bool = False) -> None:
        self.idenity = crop < 0
        self.crop = [crop, crop]
        self.augment = augment

    def transform(self, imgs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.idenity:
            return imgs
        i, j, h, w = T.RandomCrop.get_params(
            next(iter(imgs.items()))[-1], self.crop)
        imgs = {k: v2F.crop(v, i, j, h, w) for k, v in imgs.items()}
        if self.augment:
            flip_horizontal = torch.rand(1).item() < 0.5
            flip_vertical = torch.rand(1).item() < 0.5
            if flip_horizontal:
                imgs = {k: v2F.hflip(v) for k, v in imgs.items()}
            if flip_vertical:
                imgs = {k: v2F.vflip(v) for k, v in imgs.items()}
        return imgs


class PairDataset(Dataset):
    def __init__(self,
                 data_cfgs: dict[str, dict],
                 max_len: int = int(1e9),
                 crop_size: Annotated[int, 'between [0,512]'] = 512,
                 augment: bool = False
                 ) -> None:
        super().__init__()
        self.data_infos = {}
        for k, cfg in data_cfgs.items():
            path = cfg['path']
            type = cfg['type']
            # Assert that the path exists
            assert os.path.exists(
                path), f"Provided path '{path}' does not exist."

            # Assert that datatype is either 'image' or 'lmdb'
            assert type in [
                'image', 'lmdb'], "datatype must be either 'image' or 'lmdb'."

            # Assert that max_len is a positive integer
            assert max_len > 0, "max_len must be a positive integer."

            # Assert that crop_size is within the valid range
            assert 0 <= crop_size <= 512 or crop_size == - \
                1, "crop_size must be between 0 and 512, or -1 if not used."
            keys, func_warp = self.init_data(path, type)
            self.data_infos[k] = {
                'keys': keys,
                'len': len(keys),
                'path': path,
                'type': type,
                'get_func': func_warp
            }
        self.max_len = max_len
        self.lr_keys: List[str] = None
        self.gt_keys: List[str] = None
        self.get_data_func: Callable = None
        transforms = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        self.transforms = v2.Compose(transforms)
        self.pairTrans = PairTransform(crop_size, augment)

    def __len__(self):
        return min(self.max_len, self.data_infos[max(self.data_infos, key=lambda k: self.data_infos[k]['len'])]['len'])

    def __getitem__(self, idx: int):
        return self.pairTrans.transform({k: self.transforms(v['get_func'](v['path'], v['keys'][idx])) for k, v in self.data_infos.items()})

    def ndBGR2RGB(self, img: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def init_data(self, path: str, type: Annotated[str, ['image', 'lmdb']]) -> None:
        if type == 'image':
            def _read(_path: str, _key: str):
                return self.ndBGR2RGB(cv2.imread(os.path.join(_path, _key), cv2.IMREAD_COLOR))
            keys = self.get_images(path)
            return keys, _read
        else:
            env = self.init_lmdb_env(path)

            def _read(_path: str, _key: str):
                return self.ndBGR2RGB(self.read_lmdb_by_key(env, _key))
            keys = self.read_pkl(
                os.path.join(path, "meta_info.pkl"))['keys']
            return keys, _read

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

    def read_lmdb_by_key(self, env: lmdb.Environment, key: str) -> Any:
        with env.begin(write=False) as txn:
            return cv2.imdecode(
                np.frombuffer(txn.get(key.encode("ascii")),
                              np.uint8), cv2.IMREAD_COLOR
            )


class PairImageWithJson(PairDataset):
    def __init__(self,
                 json_path: str,
                 max_len: int = int(100),
                 crop_size: int = 32,
                 augment: bool = False) -> None:
        super().__init__({}, max_len, crop_size, augment)
        with open(json_path, 'r', encoding='utf8') as f:
            self.input_keys = list(json.load(f).items())

    def __len__(self):
        return min(self.max_len, len(self.input_keys))

    def __getitem__(self, idx: int):
        return self.pairTrans.transform({
            'input': self.transforms(self.ndBGR2RGB(cv2.imread(self.input_keys[idx][0], cv2.IMREAD_COLOR))),
            'target': self.transforms(self.ndBGR2RGB(cv2.imread(self.input_keys[idx][1], cv2.IMREAD_COLOR)))
        }
        )
