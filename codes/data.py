import os
import glob
import json
import pickle as pkl
import cv2
import numpy as np
import scipy

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as v2F
from torch.utils.data import DataLoader
from sympy import false
from typing import Any, Annotated, Callable, Dict, List
import lmdb
from lightning.pytorch import LightningDataModule


class SingleImageOrFloderDataset(Dataset):
    def __init__(self,
                 path: str
                 ) -> None:
        super().__init__()
        # Assert that the path exists
        assert os.path.exists(path), f"Provided path '{path}' does not exist."
        # image or folder
        self.paths = []
        self.releative_paths = []
        if os.path.isdir(path):
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
            ]
            for suffix in suffixs:
                target_path: str = os.path.join(path, "**", "*" + suffix)
                res: List[str] = glob.glob(target_path, recursive=True)
                self.paths.extend(res)
                self.releative_paths.extend(os.path.relpath(p, path) for p in res)
        else:
            self.paths.append(path)
            self.releative_paths.append(os.path.basename(path))
        self.transforms = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        rel_path = self.releative_paths[idx]
        # crop
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        # to rgb
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        # crop
        c, h, w = img.shape
        c_h = h//32*32
        c_w = w//32*32
        img = img[:, :c_h, :c_w]
        return {'input': img,'rel_path':rel_path}


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
                return ndBGR2RGB(cv2.imread(key, cv2.IMREAD_COLOR))
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
                return self.ndBGR2RGB(cv2.imread(_key, cv2.IMREAD_COLOR))
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


class SIDD_benchmark(Dataset):
    '''
    SIDD validation dataset class
    '''

    def __init__(self, sidd_val_dir: str, len: int = 1024):
        super().__init__()
        assert os.path.exists(
            sidd_val_dir), 'There is no dataset %s' % sidd_val_dir

        noisy_mat_file_path = os.path.join(
            sidd_val_dir, 'BenchmarkNoisyBlocksSrgb.mat')

        self.noisy_patches = np.array(scipy.io.loadmat(
            noisy_mat_file_path, appendmat=False)['BenchmarkNoisyBlocksSrgb'])
        self.len = len
        self.transforms = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return self.len

    def __getitem__(self, data_idx):
        img_id = data_idx // 32
        patch_id = data_idx % 32
        noisy_img = self.noisy_patches[img_id, patch_id, :]
        noisy_img = self.transforms(noisy_img)
        return {'input': noisy_img}


class SIDD_validation(Dataset):
    '''
    SIDD validation dataset class
    '''

    def __init__(self, sidd_val_dir: str, len: int = 1024):
        super().__init__()
        assert os.path.exists(
            sidd_val_dir), 'There is no dataset %s' % sidd_val_dir

        clean_mat_file_path = os.path.join(
            sidd_val_dir, 'ValidationGtBlocksSrgb.mat')
        noisy_mat_file_path = os.path.join(
            sidd_val_dir, 'ValidationNoisyBlocksSrgb.mat')

        self.clean_patches = np.array(scipy.io.loadmat(
            clean_mat_file_path, appendmat=False)['ValidationGtBlocksSrgb'])
        self.noisy_patches = np.array(scipy.io.loadmat(
            noisy_mat_file_path, appendmat=False)['ValidationNoisyBlocksSrgb'])
        self.len = len
        self.transforms = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return self.len

    def __getitem__(self, data_idx):
        img_id = data_idx // 32
        patch_id = data_idx % 32

        clean_img = self.clean_patches[img_id, patch_id, :]
        noisy_img = self.noisy_patches[img_id, patch_id, :]

        clean_img = self.transforms(clean_img)
        noisy_img = self.transforms(noisy_img)

        return {'target': clean_img, 'input': noisy_img}


class DNDataModule(LightningDataModule):
    def __init__(self,
                 train_dataset: Dataset = None,
                 test_dataset: Dataset = None,
                 val_dataset: Dataset = None,
                 predict_dataset: Dataset = None,
                 batch_size=1,
                 num_workers=4) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.predict_dataset = predict_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage):
        if stage == 'fit':
            assert self.train_dataset is not None, 'train_dataset is None'
        if stage == 'test':
            assert self.test_dataset is not None, 'test_dataset is None'
        if stage == 'predict':
            assert self.predict_dataset is not None, 'predict_dataset is None'
        if stage == 'validate':
            assert self.val_dataset is not None, 'val_dataset is None'

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=0, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, num_workers=0, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_dataset, batch_size=1, num_workers=0, shuffle=False)
