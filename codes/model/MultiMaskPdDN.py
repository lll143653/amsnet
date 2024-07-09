
from joblib import PrintTime
import torch
import torch.nn as nn
from codes.util import util
from codes.model.FeatureMask import MultiScaleMask, StableMask
from codes.model.SimpleDn import DnBranch
from codes.model.Restormer import Restormer
from codes.model.APBSN import DBSNl
from codes.model.ScaoedNet import SCNet
from codes.model.DeamNet import Deam
from codes.model.NAFNet import NAFNet
from codes.model.NAFNetBase import NAFNetBase
from codes.model.Unet import UNet
from codes.model.DNCNN import DnCNN
from codes.model.NBNet import NBNet
from codes.model.BNNLNN import UNet as SPUnet


class MaskEnhance(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mask = MultiScaleMask(6)

    def forward(self, x: torch.Tensor, denoised: torch.Tensor, net: nn.Module) -> torch.Tensor:
        masked_img, masks = self.mask(denoised)
        dn_img = torch.zeros_like(x)
        for i in range(len(masked_img)):
            masked_img[i][~masks[i]] = x[~masks[i]]
            out = net(masked_img[i])
            dn_img += out
        return dn_img/len(masked_img)


class AsymMaskEnhance(nn.Module):
    def __init__(self, delta_param: float = 0.6, replace_ratio: float = 0.18) -> None:
        super().__init__()

        self.delta_param = delta_param
        self.replace_ratio = replace_ratio
        self.replace_num = 8
        # self.replace_num = int(1/replace_ratio)+1

    def cal_gradient(self, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = target.shape
        template = torch.ones(h, w).to(target.device)*2
        # template[1:-1, 1:-1] = 4
        template[[0, -1, 0, -1], [0, -1, -1, 0]] = 1
        res = torch.zeros(h, w).to(target.device)
        A = torch.sqrt(torch.sum(
            (target[..., 1:, :-1]-target[..., :-1, :-1])**2, 1)+torch.sum((target[..., :-1, 1:]-target[..., :-1, :-1])**2, 1)).squeeze().to(target.device)
        res[:-1, :-1] += A
        A = torch.sqrt(torch.sum(
            (target[..., 1:, :-1]-target[..., :-1, :-1])**2, 1)+torch.sum((target[..., 1:, 1:]-target[..., 1:, :-1])**2, 1)).squeeze().to(target.device)
        res[1:, :-1] += A
        A = torch.sqrt(torch.sum(
            (target[..., 1:, :1]-target[..., :-1, :1])**2, 1)+torch.sum((target[..., 1:, 1:]-target[..., 1:, :-1])**2, 1)).squeeze().to(target.device)
        res[1:, 1:] += A
        A = torch.sqrt(torch.sum(
            (target[..., 1:, 1:]-target[..., :-1, 1:])**2, 1)+torch.sum((target[..., :-1, 1:]-target[..., :-1, :-1])**2, 1)).squeeze().to(target.device)
        res[:-1, 1:] += A
        return (res/template).squeeze()

    def replace(self, noisy: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        output = self.cal_gradient(denoised)
        delta = output.flatten()[torch.topk(output.flatten(), (int)(torch.mul(
            *output.shape)*self.delta_param))[1][-1]]
        positive_mask = output > delta
        positive_indices = torch.nonzero(positive_mask)
        total_elements = positive_mask.sum()
        num_elements_to_select = int(self.replace_ratio * torch.mul(
            *output.shape))
        selected_indices = positive_indices[torch.randperm(
            total_elements)[:num_elements_to_select]]
        selected_mask = torch.zeros_like(positive_mask)
        selected_mask[selected_indices[:, 0], selected_indices[:, 1]] = 1
        final_indices = selected_mask
        denoised[...][..., final_indices] = noisy[...][..., final_indices]
        return denoised

    def forward(self, x: torch.Tensor, denoised: torch.Tensor, net: nn.Module) -> torch.Tensor:
        # res = torch.zeros_like(x)
        # for _ in range(self.replace_num):
        #     res += net(self.replace(x, denoised.clone()))
        # return res/self.replace_num
        b, c, h, w = x.shape
        temp_input = denoised.expand(self.replace_num, -1, -1, -1)
        x = x.expand(self.replace_num, -1, -1, -1)
        indices = torch.zeros(self.replace_num, c, h, w,
                              dtype=torch.bool, device=x.device)
        for t in range(self.replace_num):
            indices[t] = self.replace(x, denoised.clone())
        temp_input = temp_input.clone()
        temp_input[indices] = x[indices]
        with torch.no_grad():
            denoised = net(temp_input)
        return torch.mean(denoised, dim=0).unsqueeze(0)


class R3(nn.Module):
    def __init__(self, r3: float = -1, r3_num: int = 8):
        super().__init__()
        self.r3 = r3
        self.r3_num = r3_num
        if r3 <= 0:
            self.enhance = AsymMaskEnhance()

    def forward(self, x: torch.Tensor, denoised: torch.Tensor, net: nn.Module) -> torch.Tensor:
        if self.r3 > 0:
            return util.r3(x, denoised, net, self.r3, self.r3_num)
        else:
            return denoised
            # return self.enhance(x, denoised, net)


class MultiMaskBranchDn(nn.Module):
    def __init__(self, branch_num: int = 2, dn_branch: str = 'default', mask_num: int = -1,
                 net_param: dict[str, float | str | int] | list[dict] = None, branches_order=None) -> None:
        """mmdb

        Args:
            branch_num (int, optional): the number of denoise branch. Defaults to 2.
            dn_branch (str, optional): the type of. Defaults to 'default'.
            mask_num (int, optional): the number of mask, same to branch_num. Defaults to -1.
            net_param (dict[str, float  |  str  |  int] | list[dict], optional): denoiser param. Defaults to None.
            branches_order (_type_, optional): the order od branches call. Defaults to None.
        """
        super().__init__()
        self.mask_num = mask_num
        if mask_num < 2:
            self.mask_num = len(branches_order)
        if branches_order is None:
            self.branches_order = [0 for _ in range(branch_num)]
        else:
            self.branches_order = branches_order
        assert self.mask_num >= len(
            branches_order), "mask num must larger than branches order's len"
        self.branches = nn.ModuleList(
            [dn_dict[dn_branch](**net_param) for _ in range(branch_num)])
        self.mask = MultiScaleMask(self.mask_num)

    def forward(self, x: torch.Tensor, return_mask: bool = False, only_first: bool = False) -> torch.Tensor:
        masked_img, masks = self.mask(x)
        dn_img = torch.zeros_like(x).to(dtype=torch.float32)
        order_len = min(self.mask_num, len(self.branches_order))
        for i, j in zip(self.branches_order, [_ for _ in range(order_len)]):
            out = self.branches[i](masked_img[j])
            dn_img[~masks[j]] = out[~masks[j]]
            if only_first and return_mask:
                break
        return dn_img if not return_mask else (dn_img, masks)


mask_dict = {
    'MultiScale': MultiScaleMask,
    'Stable': StableMask,
}
dn_dict = {
    'default': DnBranch,
    'mmbd': MultiMaskBranchDn,
    'Deam': Deam,
    'Unet': UNet,
    'NAFNet': NAFNet,
    'SCNet': SCNet,
    'NAFNetBase': NAFNetBase,
    'DBSNI': DBSNl,
    'Restormer': Restormer,
    'DnCNN': DnCNN,
    'NBNet': NBNet,
    'SPUnet': SPUnet
}


class MultiMaskPdDn(nn.Module):
    def __init__(self, pd_train: int = 5, pd_val: int = 2, dn_net: str = 'default', r3: float = -1, r3_num: int = 8,
                 net_param: dict[str, float | str | int] = None, **kwargs):
        super().__init__()

        self.dn = dn_dict[dn_net](**net_param if net_param is not None else {})
        self.pd_train = pd_train
        self.pd_val = pd_val
        self.r3 = R3(r3, r3_num)

    def denoise(self, x: torch.Tensor, pd_factor: int = None, return_mask: bool = False, only_first: bool = True) -> torch.Tensor:
        if pd_factor is None:
            pd_factor = self.pd_train
        if pd_factor > 1:
            x = util.pd_down(x, pd_factor)
        if return_mask:
            dn_img, masks = self.dn(x, True, only_first)
        else:
            dn_img = self.dn(x, False, only_first)
        if pd_factor > 1:
            dn_img = util.pd_up(dn_img, pd_factor)
        return dn_img if not return_mask else (dn_img, masks)

    def forward(self, x: torch.Tensor, pd_factor: int = None, return_mask: bool = False, only_first: bool = True) -> torch.Tensor:
        if self.training:
            return self.denoise(x, pd_factor, return_mask, only_first)
        else:
            denoised = self.denoise(x, self.pd_val, only_first=False)
            return self.r3(x, denoised, self.dn)
