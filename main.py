from lightning.pytorch.cli import LightningCLI
from codes.model import ImageDenoiseEnd2End
from codes.data import DNDataModule
if __name__ == "__main__":
    cli = LightningCLI(ImageDenoiseEnd2End,DNDataModule)