import torch
import torch.nn as nn
import lightning.pytorch as pl
from torchvision.transforms import GaussianBlur
from codes import util


class BGCLoss(pl.LightningModule):
    def __init__(self, kernel_size: int = 3, sigma: float = 0.3) -> None:
        super().__init__()
        self.blur = GaussianBlur(kernel_size, sigma=sigma)
        self.loss = nn.L1Loss()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return self.loss(self.blur(x), self.blur(y))


losses_dict = {
    'l1': nn.L1Loss,
    'l2': nn.MSELoss
}


class SubRollLoss(pl.LightningModule):
    def __init__(self, roll_len: int = 1, loss_type: str = 'l1', pd_factor: int = 5) -> None:
        super().__init__()
        self.roll_len = roll_len
        self.loss = losses_dict[loss_type]()
        self.pd_factor = pd_factor

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_pd = util.pd_down(x, self.pd_factor)
        y_pd = util.pd_down(y, self.pd_factor)
        return self.loss(x_pd, y_pd.roll(1, 0))


class SubPatchLoss(pl.LightningModule):
    def __init__(self, pd_factor: int = 5, loss_type: str = 'l1') -> None:
        super().__init__()
        self.pd_factor = pd_factor
        self.loss = losses_dict[loss_type]()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_pd = util.pd_down(x, self.pd_factor)
        y_pd = util.pd_down(y, self.pd_factor)
        return self.loss(x_pd, y_pd)


class N2NLoss(pl.LightningModule):
    def __init__(self, gamma: float = 0.5, loss_type: str = 'l1') -> None:
        super().__init__()
        self.gamma = gamma
        self.loss = losses_dict[loss_type]()

    def mseloss(self, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = ((image - target) ** 2)
        return torch.mean(x)

    def regloss(self, g1: torch.Tensor, g2: torch.Tensor, G1: torch.Tensor, G2: torch.Tensor) -> torch.Tensor:
        return torch.mean((g1 - g2 - G1 + G2) ** 2)

    def forward(self, fg1: torch.Tensor, g2: torch.Tensor, G1f: torch.Tensor, G2f: torch.Tensor) -> torch.Tensor:
        return self.mseloss(fg1, g2) + self.gamma * self.regloss(fg1, g2, G1f, G2f)


class GANLoss(pl.LightningModule):
    def __init__(self, gan_type, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = target_real_label
        self.fake_label_val = target_fake_label

        if self.gan_type == "gan" or self.gan_type == "ragan":
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == "lsgan":
            self.loss = nn.MSELoss()
        elif self.gan_type == "wgan-gp":

            def wgan_loss(x, target):
                # target is boolean
                return -1 * x.mean() if target else x.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError(
                "GAN type [{:s}] is not found".format(self.gan_type)
            )

    def get_target_label(self, x, target_is_real):
        if self.gan_type == "wgan-gp":
            return target_is_real
        if target_is_real:
            return torch.empty_like(x).fill_(self.real_label_val)
        else:
            return torch.empty_like(x).fill_(self.fake_label_val)

    def forward(self, x, target_is_real):
        target_label = self.get_target_label(x, target_is_real)
        loss = self.loss(x, target_label)
        return loss


class MaskLoss(pl.LightningModule):
    def __init__(self, loss_type: str = 'l1') -> None:
        super().__init__()
        self.loss = losses_dict[loss_type]()

    def forward(self, input: torch.Tensor, output: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        total_loss = 0
        for mask in masks:
            total_loss += self.loss(input[~mask], output[~mask])
        return total_loss


class FirstBranchMaskLoss(pl.LightningModule):
    def __init__(self, loss_type: str = 'l1') -> None:
        super().__init__()
        self.loss = losses_dict[loss_type]()

    def forward(self, input: torch.Tensor, output: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
        total_loss = 0
        for mask in masks:
            total_loss += self.loss(input[~mask], output[~mask])
            break
        return total_loss


class PdMaskLoss(pl.LightningModule):
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


class FirstPdMaskLoss(pl.LightningModule):
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


class TVLoss(pl.LightningModule):
    def __init__(self, reduction='mean'):
        super(TVLoss, self).__init__()
        assert reduction in [
            'mean', 'sum'], f'Invalid reduction mode: {reduction}'
        self.reduction = reduction
        self.loss = nn.L1Loss(reduction=reduction)

    def forward(self, pred):
        y_diff = self.loss(
            pred[:, :, :-1, :], pred[:, :, 1:, :])
        x_diff = self.loss(
            pred[:, :, :, :-1], pred[:, :, :, 1:])

        loss = x_diff + y_diff
        if self.reduction == 'mean':
            loss /= pred.shape[0]
        return loss
