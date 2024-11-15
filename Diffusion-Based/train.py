import os
import sys
from pathlib import Path

import hydra
import omegaconf
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


from callbacks.ema import EMA
from callbacks.logger import LoggerCallback

from utils.paths import MODEL
from data.datasets import DDPMDataset

@hydra.main('config', 'train.yaml')
def train(config: DictConfig):
    pl.seed_everything(config.seed)
    
    # Initialize checkpoint
    ckpt = None
    if config.ckpt is not None:
        os.chdir(os.path.dirname(config.ckpt))
        assert os.path.exists(config.ckpt)
        ckpt = config.ckpt
        config = OmegaConf.load(os.path.join(os.path.dirname(ckpt), 'config.yaml'))
    # Save the updated configuration to a file called 'config.yaml'
    with open('config.yaml', 'w') as f:
        omegaconf.OmegaConf.save(config, f)
    # If a checkpoint is specified, load the model weights from the checkpoint
        
    # Directory setup
    Path.cwd().joinpath('gen_images').mkdir(parents=True, exist_ok=True)
    MODEL.copytree(Path.cwd().joinpath('model'))
    
    # Create the variance scheduler and a deep generative model using Hydra
    denoiser_module = hydra.utils.instantiate(config.denoiser_module)
    scheduler = hydra.utils.instantiate(config.scheduler)
    opt = hydra.utils.instantiate(config.optimizer)
    model: pl.LightningModule = hydra.utils.instantiate(config.model,
                                                        denoiser_module=denoiser_module,
                                                        opt=opt,
                                                        variance_scheduler=scheduler)
    
    # Load the model weights from the checkpoint
    if ckpt is not None:
        model.load_from_checkpoint(ckpt, variance_scheduler=scheduler)
    model.save_hyperparameters(OmegaConf.to_object(config)['model'])
    
    pin_memory = 'gpu' in config.accelerator
    data = hydra.utils.instantiate(config.dataset)
    data.setup()
    train_dl = data.train_dataloader()
    val_dl = data.val_dataloader()
    
    # Create a ModelCheckpoint callback that saves the model weights to disk during training
    ckpt_callback = ModelCheckpoint('./', 
                                    'epoch={epoch}-valid_loss={val/loss_epoch}', 
                                     monitor='val/loss_epoch', 
                                     auto_insert_metric_name=False, 
                                     save_last=True)
    ddpm_logger = LoggerCallback(config.freq_logging, 
                                 config.freq_logging_norm_grad, 
                                 config.batch_size_gen_images) 
    
    callbacks = [ckpt_callback, ddpm_logger]
    
    # Add additional callbacks if specified in the configuration file
    if config.ema:
        # Create an Expontential Moving Average callback
        callbacks.append(EMA(config.ema_decay))  
    if config.early_stop:
        callbacks.append(EarlyStopping('val/loss_epoch', 
                                       min_delta=config.min_delta, 
                                       patience=config.patience))
    
    trainer = pl.Trainer(callbacks=callbacks, 
                         accelerator=config.accelerator, 
                         devices=config.devices, 
                         gradient_clip_val=config.gradient_clip_val, 
                         gradient_clip_algorithm=config.gradient_clip_algorithm)
    trainer.fit(model, train_dl, val_dl)

    
if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent.absolute()))
    train()