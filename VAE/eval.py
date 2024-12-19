import os
import yaml
import torch
import argparse
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from experiment import VAEXperiment
from dataset import VAEDataset
from models import *
from utils import reconstruct_test_images


parser = argparse.ArgumentParser(description='VAE Model Prediction')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint_path = os.path.join(config['logging_params']['save_dir'], 
                               config['logging_params']['name'],
                               "version_" + str(config['logging_params']['version']),  # version을 문자열로 변환하여 연결
                               "checkpoints", 
                               "last.ckpt")
experiment = VAEXperiment.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                               vae_model=model,
                                               params=config['exp_params']).to(device)

data = VAEDataset(**config["data_params"], pin_memory=config['trainer_params'].get('devices', 0) != 0)
data.setup()

reconstruct_test_images(config, experiment, data.test_dataloader(), device)