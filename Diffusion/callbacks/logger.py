from typing import Any
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import torchvision

class LoggerCallback(Callback):
    def __init__(self, freq_train_log: int, num_sampling_images: int) -> None:
        super().__init__()
        self.freq_train = freq_train_log
        self.num_sampling_images = num_sampling_images

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: dict, batch: Any, batch_idx: int) -> None:
        if trainer.global_step % self.freq_train == 0:
            pl_module.log("train/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: dict, batch: Any, batch_idx: int) -> None:
        pl_module.log("val/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        gen_images = pl_module.sample(batch_size=self.num_sampling_images) # Generate images
        gen_images = torchvision.utils.make_grid(gen_images)  # Convert to grid
        torchvision.utils.save_image(gen_images, f'gen_images/epoch={pl_module.current_epoch}.png')  # Save the images
