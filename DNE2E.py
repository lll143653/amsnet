from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
import wandb
from loguru import logger
import argparse
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import lightning.pytorchas pl
from torchvision.utils import save_image
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.types import *
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torchvision.utils import save_image
from codes import util
import torchmetrics
from pytorch_lightning.callbacks import RichProgressBar
from lightning.pytorchimport seed_everything
import pprint


class PathManger:
    def __init__(self, name: str, output_path: str) -> None:
        self.name: str = name
        self.output_path: str = output_path
        self.folder_path: str = os.path.join(
            self.output_path, self.name, util.get_timestamp())
        self.img_path = os.path.join(self.folder_path, 'val')
        self.log_path = os.path.join(self.folder_path, 'log')
        self.check_path = os.path.join(self.folder_path, 'check')
        util.mkdirs([self.img_path, self.check_path])


class ImageSaveCallback(Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        outputs['psnr'] = -1
        self._save_images(trainer, pl_module, outputs, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._save_images(trainer, pl_module, outputs, batch, batch_idx)

    def _save_images(self, trainer, pl_module, outputs, batch, batch_idx):
        input: torch.Tensor = outputs['input']
        target: torch.Tensor = outputs['target']
        output: torch.Tensor = outputs['output']
        psnr: float = outputs.get('psnr', -1)
        if target is not None and input.shape == target.shape:
            final_img = torch.concat([input, target, output])
        else:
            final_img = torch.concat([input, output])
        if psnr > 0:
            output_filename = os.path.join(
                self.save_dir, f"{batch_idx}_{psnr:.2f}.png")
        else:
            output_filename = os.path.join(self.save_dir, f"{batch_idx}.png")
        save_image(final_img, output_filename)


class ImageDenoiseEnd2End(pl.LightningModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.net: nn.Module = Reg.create_model(
            model_name=cfg['model']['name'], ** cfg['model']['params'])
        if 'loss' in cfg:
            self.loss_fn = Reg.create_loss(
                cfg['loss']['name'], **cfg['loss']['param'] if cfg['loss']['param'] else {})
        self.psnrs = torchmetrics.MeanMetric()
        self.ssims = torchmetrics.MeanMetric()
        self.cfg = cfg
        self.best_psnr = float('-inf')
        self.best_ssim = float('-inf')
        self.init_weights()

    def forward(self, data: dict[str:torch.Tensor]) -> torch.tensor:
        return self.net.forward(data['input'])

    def training_step(self, batch: dict[str:torch.Tensor | str], batch_idx: int) -> torch.Tensor | None:
        input = batch['input']
        target = batch['input']
        output, mask_index = self.net.forward(input, return_mask=True)
        loss = self.loss_fn(target, output, mask_index)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch: dict[str:torch.Tensor | str], batch_idx: int) -> torch.Tensor | None:
        input = batch['input']
        target = batch['target']
        output = self.net.forward(input)
        psnr, ssim = util.calculate_psnr_ssim(target, output, self.cfg.get(
            'scale', 0), min_max=self.cfg.get('min_max', [0.0, 1.0]))
        self.psnrs.update(psnr)
        self.ssims.update(ssim)
        return {"input": input, "target": target, "output": output, 'psnr': psnr, 'ssim': ssim}

    def on_validation_epoch_end(self) -> None:
        avg_psnr = self.psnrs.compute()
        self.log('val_avg_psnr', avg_psnr, prog_bar=True,
                 logger=True, sync_dist=True)
        self.psnrs.reset()
        avg_ssim = self.ssims.compute()
        self.log('val_avg_ssim', avg_ssim, prog_bar=True,
                 logger=True, sync_dist=True)
        self.ssims.reset()
        if avg_psnr > self.best_psnr:
            self.best_psnr = avg_psnr
        self.log('best_val_psnr', self.best_psnr,
                 prog_bar=True, logger=True, sync_dist=True)
        if avg_ssim > self.best_ssim:
            self.best_ssim = avg_ssim
        self.log('best_val_ssim', self.best_ssim,
                 prog_bar=True, logger=True, sync_dist=True)

    def on_test_epoch_start(self) -> None:
        self.psnrs.reset()
        self.ssims.reset()

    def on_test_epoch_end(self) -> None:
        avg_psnr = self.psnrs.compute()
        avg_ssim = self.ssims.compute()
        if self.trainer.is_global_zero:
            logger.info(
                f'Test finished. avg psnr: {avg_psnr:.2f}, avg ssim: {avg_ssim:.4f}')

    def test_step(self, batch: dict[str:torch.Tensor | str], batch_idx: int) -> torch.Tensor | None:
        input = batch['input']
        target = batch.get('target', None)
        if target is not None:
            target = target
        output = self.net.forward(input)
        psnr, ssim = -1, -1
        if target is not None:
            psnr, ssim = util.calculate_psnr_ssim(target, output, self.cfg.get(
                'scale', 0), min_max=self.cfg.get('min_max', [0.0, 1.0]))
            self.psnrs.update(psnr)
            self.ssims.update(ssim)
        return {"input": input, "target": target, "output": output, 'psnr': psnr, 'ssim': ssim}

    def configure_optimizers(self) -> Optimizer | None:
        if 'optimizer' not in self.cfg['model']:
            return None
        opt = Reg.create_optimizer(
            model=self.net.parameters(), **self.cfg['model']['optimizer'])
        return opt

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def create_dataloader(dataloader_name: str, cfg: dict):
    if dataloader_name not in cfg.keys():
        return None
    data_param = cfg[dataloader_name]
    cur_dataset = Reg.create_dataset(
        data_param['type'], **data_param['dataset'])
    cur_dataloader = DataLoader(cur_dataset, **data_param['dataloader'])
    return cur_dataloader


def create_arg():
    parser = argparse.ArgumentParser(description='Train or Test the model.')

    parser.add_argument('--train', action='store_true',
                        help='Flag to indicate training the model.')
    parser.add_argument('--wandb', action='store_true',
                        help='Flag to indicate using wandb.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    seed_everything(42)
    args = create_arg()
    if args.train and args.wandb:
        wandb.init(mode="disabled")
        wandb.finish()
    cfg = util.load_yaml(args.config)
    paths = PathManger(cfg['name'], cfg['output_path'])
    logger.add(paths.log_path+'/.log')
    torch.set_float32_matmul_precision('high')
    logger.info(pprint.pformat(cfg))
    checkpoint_callback = ModelCheckpoint(
        monitor='val_avg_psnr',
        dirpath=paths.check_path,
        filename=cfg['name']+'-{epoch:02d}-{val_avg_psnr:.2f}',
        save_last=True,
        save_top_k=3,
        mode='max',
        save_weights_only=True
    )
    image_callback = ImageSaveCallback(
        save_dir=paths.img_path
    )
    if args.train:
        train_dataloader = create_dataloader('train', cfg)
        val_dataloader = create_dataloader('val', cfg)
    else:
        test_dataloader = create_dataloader('test', cfg)
    if cfg['model'].get('checkpoint_path', None) is not None:
        model = ImageDenoiseEnd2End.load_from_checkpoint(
            cfg['model']['checkpoint_path'], cfg=cfg)
    else:
        model = ImageDenoiseEnd2End(cfg)
        pl.Callback
    progress_bar = RichProgressBar(leave=False, theme=RichProgressBarTheme(
        description="green_yellow",
        progress_bar="green1",
        progress_bar_finished="green1",
        progress_bar_pulse="#6206E0",
        batch_progress="green_yellow",
        time="grey82",
        processing_speed="grey82",
        metrics="grey82",
        metrics_text_delimiter="\n",
        metrics_format=".3e",
    ))
    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        accelerator='cuda',
        strategy=DDPStrategy(find_unused_parameters=True),
        callbacks=[checkpoint_callback, image_callback, progress_bar],
        val_check_interval=cfg['val_check_interval'],
        logger=WandbLogger(
            name=cfg['name'], save_dir=paths.folder_path, log_model=False) if args.train and args.wandb else TensorBoardLogger(paths.folder_path, name='tesnorboard'),
        num_sanity_val_steps=0
    )
    if args.train:
        logger.info(f'train dataloader: {len(train_dataloader)}')
        logger.info(f'val dataloader: {len(val_dataloader)}')
        trainer.fit(model, train_dataloader, val_dataloader)
    else:
        logger.info(f'test dataloader: {len(test_dataloader)}')
        trainer.test(model, test_dataloader)
