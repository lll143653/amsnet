import torch
import torch.nn as nn
from torchvision.transforms import GaussianBlur

class BGCLoss(nn.Module):
    def __init__(self, kernel_size: int = 3, sigma: float = 0.3) -> None:
        super().__init__()
        self.blur = GaussianBlur(kernel_size, sigma=sigma)
        self.loss = nn.L1Loss()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.loss(self.blur(x), self.blur(y))
