from pickletools import read_uint1
from typing import Any, Dict, Optional, Union
from timm import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss
from codes.loss.BGCLoss import BGCLoss
from codes.loss.GANloss import GANLoss
from codes.loss.RollLoss import SubRollLoss, SubPatchLoss
from codes.loss.MaskLoss import FirstBranchMaskLoss, PdMaskLoss, FirstPdMaskLoss, MaskLoss, TVLoss
from codes.dataset.UnpairDataset import UnpairDataset
from codes.dataset.SIDD import SIDD_benchmark, SIDD_validation
import timm
import torch.nn as nn
from timm.models._registry import register_model
from timm.optim import optimizer_kwargs
from codes.model.MultiMaskPdDN import MultiMaskPdDn
import argparse


@register_model
def mmpn(pretrained=False, **kwargs) -> nn.Module:
    if 'kwargs' in kwargs:
        model_kwargs = kwargs['kwargs']
    else:
        model_kwargs = kwargs
    model = MultiMaskPdDn(**model_kwargs)
    return model


class Reg:
    @classmethod
    def create_model(cls,
                     model_name: str,
                     checkpoint_path: str = '',
                     **kwargs
                     ) -> nn.Module:
        return create_model(model_name=model_name, checkpoint_path=checkpoint_path, kwargs=kwargs)

    @classmethod
    def create_scheduler(cls,
                         optimizer: Any,
                         updates_per_epoch: int = 0,
                         **kwargs):
        return create_scheduler(argparse.Namespace(**kwargs), optimizer, updates_per_epoch)

    @classmethod
    def create_optimizer(cls, model, filter_bias_and_bn=True, **kwargs):
        return create_optimizer(argparse.Namespace(**kwargs), model=model, filter_bias_and_bn=filter_bias_and_bn)

    @classmethod
    def create_loss(cls, name: str, **kwargs):
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        loss_class = globals().get(name)
        if loss_class is None:
            raise ValueError(f"{name} is not a valid loss class name")
        return loss_class(**kwargs)

    @classmethod
    def create_dataset(cls, name: str, **kwargs):
        if not isinstance(name, str):
            raise ValueError("name must be a string")
        dataset_class = globals().get(name)
        if dataset_class is None:
            raise ValueError(f"{name} is not a valid dataset class name")
        return dataset_class(**kwargs)
