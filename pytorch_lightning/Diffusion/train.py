import os
import re
import sys
import hydra
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from diffusion.resample import create_named_schedule_sampler
from diffusion.script_util import  (
    setup_config,
    create_model_and_diffusion,
)    

from diffusion.resample import LossAwareSampler, UniformSampler
from pathlib import Path

@hydra.main(config_path="config", config_name="train.yaml")
def train(config: DictConfig):
    pl.seed_everything(config.seed)
    
    # Setup default and model config
    config = setup_config(config)
    # Initialize checkpoint
    ckpt = None
    
    # Continue the train
    if config.ckpt is not None:
        os.chdir(os.path.dirname(config.ckpt))
        assert os.path.exists(config.ckpt)
        ckpt = config.ckpt
        
        # Load config.yaml
        config = OmegaConf.load(os.path.join(os.path.dirname(ckpt), '.hydra','config.yaml'))
        
        # Get last epoch number
        last_epoch_value = -1
        asset_dir = os.path.join(os.path.dirname(ckpt), 'gen_images')
        asset_files = os.listdir(asset_dir)
        match = re.search(r'epoch=(\d+)', asset_files[-1])
        if match:
            last_epoch_value = int(match.group(1))
        else:
            print("No epoch information found in the file name.")
        
    # Result Directory setup | dir name : gen_images
    Path.cwd().joinpath('gen_images').mkdir(parents=True, exist_ok=True)
    
    # hatch unet model, diffusion, train model
    unet_model, diffusion = create_model_and_diffusion(config)
    scheduler = create_named_schedule_sampler(config.schedule_sampler, diffusion)
    opt = hydra.utils.instantiate(config.optimizer)
    train_model: pl.LightningModule = hydra.utils.instantiate(unet_model=unet_model,
                                                              diffusion=diffusion,
                                                              scheduler=scheduler,
                                                              opt=opt)
    
    
    
if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent.absolute()))
    train()