import torch
from typing import Any
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
import torchvision

class LoggerCallback(Callback):
    def __init__(self, freq_train_log: int, 
                 num_sampling_images: int, 
                 sampling_timesteps: int,
                 last_epoch_value: int) -> None:
        
        super().__init__()
        self.freq_train = freq_train_log
        self.num_sampling_images = num_sampling_images
        self.sampling_timesteps = sampling_timesteps
        self.last_epoch_value = last_epoch_value + 1

    def on_train_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: dict, batch: Any, batch_idx: int) -> None:
        if trainer.global_step % self.freq_train == 0:
            pl_module.log("train/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)

    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: dict, batch: Any, batch_idx: int) -> None:
        pl_module.log("val/loss", outputs["loss"], on_step=True, on_epoch=True, prog_bar=True,  sync_dist=True)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # 랜덤 시드 고정
        torch.manual_seed(42)
        gen_images = pl_module.sample(batch_size=self.num_sampling_images, sampling_timesteps=self.sampling_timesteps) # Generate images
        gen_images = torchvision.utils.make_grid(gen_images)  # Convert to grid
        torchvision.utils.save_image(gen_images, f'gen_images/epoch={pl_module.current_epoch + self.last_epoch_value }.png')  # Save the images
        
    # def on_epoch_end(self):
    #     for param_group in self.optimizers().param_groups:
    #         current_lr = param_group['lr']
    #         print(f"Current learning rate: {current_lr}")