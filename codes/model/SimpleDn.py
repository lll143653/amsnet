import torch
import torch.nn as nn
class BasicDn(nn.Module):
    def __init__(self, base_ch: int = 64, scale: float = 0):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, 1)
        )
        self.branch2 = nn.Conv2d(base_ch, base_ch, 1)
        self.feature_fusion = nn.Conv2d(2 * base_ch, base_ch, 1, 1)
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_fusion(torch.concat([self.branch1(x), self.branch2(x)], 1)) + self.scale * x


class DC(nn.Module):
    def __init__(self, in_ch, scale: float = 0):
        super().__init__()
        self.head = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
        )
        self.tail = nn.Conv2d(3 * in_ch, in_ch, 1)
        self.scale = scale

    def forward(self, x):
        out = self.head(x)
        out1 = self.branch1(out)
        out2 = self.branch2(out)
        out3 = self.branch3(out)
        cur = torch.concat([out1, out2, out3], dim=1)
        return x * self.scale + self.tail(cur)


class DnBranch(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64, num_modules: int = 4, scale: float = 0):
        super().__init__()
        self.head = nn.Conv2d(in_ch, base_ch, 3, 1, 1)
        self.body = nn.Sequential(
            *[DC(base_ch, scale) for _ in range(num_modules)]
        )
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, in_ch, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.head(x)
        out = self.body(out)
        out = self.tail(out)
        return out