
import os
import torch
from torchvision.utils import save_image
from lightning.pytorch.utilities.types import *
from lightning.pytorch.callbacks import Callback
from torchvision.utils import save_image


class ImageSaveCallback(Callback):
    def __init__(self, save_dir: str = 'image'):
        super().__init__()
        self.save_dir = save_dir

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        outputs['psnr'] = -1
        self._save_images(trainer, pl_module, outputs, batch, batch_idx)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._save_images(trainer, pl_module, outputs, batch, batch_idx)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        outputs['psnr'] = -1
        self._save_images(trainer, pl_module, outputs, batch, batch_idx)

    def _save_images(self, trainer, pl_module, outputs, batch, batch_idx):
        log_dir = trainer.logger.log_dir
        save_dir = os.path.join(log_dir, self.save_dir)
        psnr: float = outputs.get('psnr', -1)
        if psnr > 0:
            output_filename = os.path.join(
                save_dir, f"{batch_idx}_{psnr:.2f}.png")
        else:
            output_filename = os.path.join(save_dir, f"{batch_idx}.png")
        if 'rel_path' in outputs:
            output_filename = os.path.join(save_dir, outputs['rel_path'][0])
        os.makedirs(save_dir, exist_ok=True)
        input: torch.Tensor = outputs['input']
        target: torch.Tensor = outputs['target'] if 'target' in outputs else None
        output: torch.Tensor = outputs['output']
        if target is not None and input.shape == target.shape:
            final_img = torch.concat([input, target, output])
        else:
            final_img = torch.concat([input, output])

        save_image(final_img, output_filename)
