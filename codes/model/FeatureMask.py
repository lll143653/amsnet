import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleMask(nn.Module):
    def __init__(self, scale_num: int = 2):
        super().__init__()
        self.scale_num = scale_num

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (len(x.shape) == 3):
            x = x.unsqueeze(0)
        b, c, h, w = x.shape
        mask_index = torch.randint(
            0, self.scale_num, (b, 1, h, w)).expand(-1, c, -1, -1).to(x.device)
        res = torch.zeros(self.scale_num, *x.shape).to(x.device)
        masks = torch.BoolTensor(self.scale_num, *x.shape).to(x.device)
        for i in range(self.scale_num):
            temp = mask_index != i
            masks[i] = temp
            res[i] = x*temp
        return res, masks


class StableMask(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        mask_index = torch.zeros_like(x).to(x.device)
        masks = torch.BoolTensor(2, *x.shape).to(x.device)
        mask_index[:, :, ::2, ::2] = 1
        mask_index[:, :, 1::2, 1::2] = 1
        res = torch.zeros(2, *x.shape).to(x.device)
        masks = torch.BoolTensor(2, *x.shape).to(x.device)
        for i in range(2):
            temp = mask_index != i
            masks[i] = temp
            res[i] = x*temp
        return res, masks


class SimpleMask(nn.Module):
    def __init__(self,) -> None:
        super().__init__()
        self.mask_ratio = 0.5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (len(x.shape) == 3):
            x = x.unsqueeze(0)
        b, c, h, w = x.shape
        mask_index = torch.rand(b, h, w).to(x.device) > self.mask_ratio
        mask_index = mask_index.view(b, 1, h, w).expand(b, c, h, w)
        x_clone = x.clone()
        x_clone[mask_index] = 0
        return x_clone.reshape(b, c, h, w), 1 - mask_index * torch.ones_like(x).to(x.device)


class FeaturePatchMask(nn.Module):
    def __init__(self, mask_ratio=0.2, mask_patch=1, scale: float = 1) -> None:
        super().__init__()
        self.patch_size = mask_patch
        self.mask_ratio = mask_ratio
        self.scale = scale

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_clone = x.clone()
        b, c, h, w = x_clone.shape
        ph = h // self.patch_size
        pw = w // self.patch_size
        x_clone = x_clone.reshape(
            b, c, ph, self.patch_size, pw, self.patch_size).permute(0, 1, 2, 4, 3, 5)
        mask_index = torch.rand(ph, pw).to(x.device) < self.mask_ratio
        kernel = torch.ones(c, 1, self.patch_size * 3,
                            self.patch_size * 3) / (self.patch_size * 3) ** 2
        kernel = kernel.type_as(x)
        kernel_size = self.patch_size * 3
        pixel_mean = F.conv2d(x, kernel, padding=(
            kernel_size - 1) // 2, groups=c) * self.scale
        pixel_mean = pixel_mean.reshape(
            b, c, ph, self.patch_size, pw, self.patch_size).permute(0, 1, 2, 4, 3, 5)
        x_clone[:, :, mask_index, ...] = pixel_mean[:, :, mask_index, ...]

        return x_clone.permute(0, 1, 2, 4, 3, 5).reshape(b, c, h, w), mask_index * torch.ones(1).to(x.device)
