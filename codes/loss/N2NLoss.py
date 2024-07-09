import torch
import torch.nn as nn

losses_dict = {
    'l1': nn.L1Loss,
    'l2': nn.MSELoss
}

class N2NLoss(nn.Module):
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
