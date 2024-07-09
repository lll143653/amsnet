import os
import scipy
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision.transforms import v2


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
