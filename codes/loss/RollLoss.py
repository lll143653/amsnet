import torch
import torch.nn as nn
from codes.util import util
losses_dict = {
    'l1': nn.L1Loss,
    'l2': nn.MSELoss
}


class SubRollLoss(nn.Module):
    def __init__(self, roll_len: int = 1, loss_type: str = 'l1', pd_factor: int = 5) -> None:
        super().__init__()
        self.roll_len = roll_len
        self.loss = losses_dict[loss_type]()
        self.pd_factor = pd_factor

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_pd = util.pd_down(x, self.pd_factor)
        y_pd = util.pd_down(y, self.pd_factor)
        return self.loss(x_pd, y_pd.roll(1, 0))


class SubPatchLoss(nn.Module):
    def __init__(self, pd_factor: int = 5, loss_type: str = 'l1') -> None:
        super().__init__()
        self.pd_factor = pd_factor
        self.loss = losses_dict[loss_type]()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_pd = util.pd_down(x, self.pd_factor)
        y_pd = util.pd_down(y, self.pd_factor)
        return self.loss(x_pd, y_pd)
