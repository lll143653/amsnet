import torch
from torch import nn

class GANLoss(nn.Module):
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