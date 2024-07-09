import torch
import torch.nn as nn
from codes.util import util
losses_dict = {
    'l1': nn.L1Loss,
    'l2': nn.MSELoss
}


class MaskLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1') -> None:
        super().__init__()
        self.loss = losses_dict[loss_type]()

    def forward(self, input: torch.Tensor, output: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        total_loss = 0
        for mask in masks:
            total_loss += self.loss(input[~mask], output[~mask])
        return total_loss


class FirstBranchMaskLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1') -> None:
        super().__init__()
        self.loss = losses_dict[loss_type]()

    def forward(self, input: torch.Tensor, output: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        total_loss = 0
        for mask in masks:
            total_loss += self.loss(input[~mask], output[~mask])
            break
        return total_loss


class PdMaskLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1', pd_factor: int = 5) -> None:
        super().__init__()
        self.loss = losses_dict[loss_type]()
        self.pd_factor = pd_factor

    def forward(self, input: torch.Tensor, output: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        inpud_pd = util.pd_down(input, self.pd_factor)
        output_pd = util.pd_down(output, self.pd_factor)
        total_loss = 0
        for mask in masks:
            total_loss += self.loss(inpud_pd[~mask], output_pd[~mask])
        return total_loss


class FirstPdMaskLoss(nn.Module):
    def __init__(self, loss_type: str = 'l1', pd_factor: int = 5) -> None:
        super().__init__()
        self.loss = losses_dict[loss_type]()
        self.pd_factor = pd_factor

    def forward(self, input: torch.Tensor, output: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        inpud_pd = util.pd_down(input, self.pd_factor)
        output_pd = util.pd_down(output, self.pd_factor)
        total_loss = 0
        for mask in masks:
            total_loss += self.loss(inpud_pd[~mask], output_pd[~mask])
            break
        return total_loss


class TVLoss(torch.nn.L1Loss):
    """Weighted TV loss.

    Args:
        reduction (str): Loss method. Default: mean.
    """

    def __init__(self, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(TVLoss, self).__init__(reduction=reduction)

    def forward(self, pred):
        y_diff = super().forward(
            pred[:, :, :-1, :], pred[:, :, 1:, :])
        x_diff = super().forward(
            pred[:, :, :, :-1], pred[:, :, :, 1:])

        loss = x_diff + y_diff

        return loss
