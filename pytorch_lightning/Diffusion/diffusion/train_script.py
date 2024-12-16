import pytorch_lightning as pl
from .resample import LossAwareSampler, UniformSampler
import torch as th
from torch import nn

class TrainModel(pl.LightningModule):
    def __init__(self,
                 unet_model: nn.Module,
                 diffusion,
                 scheduler,
                 opt,):
        
        super().__init__()
        self.unet_model = unet_model
        self.diffusion = diffusion
        self.schedule_sampler = scheduler or UniformSampler(diffusion)
        self.opt = opt
        self.mse = nn.MSELoss()
        
    def configure_optimizers(self):
        pass
    
    def forward(self, x_t, t, cond):
        return self.unet_model(x_t, t, cond)
        
    def training_step(self, batch):
        x, cond = batch
        device = self.device  # Get the current device from LightningModule
        t, weights = self.schedule_sampler.sample(x.shape[0], device)

        eps = th.randn_like(x)  # Add random noise to the input
        x_t = self.diffusion.q_sample(x, t, noise=eps)

        pred_eps = self.forward(x_t, t, cond)  # Pass through the model
        loss = self.mse(eps, pred_eps)
        return dict(loss=loss)
    
    def validation_step(self, batch):
        x, cond = batch
        device = self.device  # Get the current device from LightningModule
        t, weights = self.schedule_sampler.sample(x.shape[0], device)

        eps = th.randn_like(x)  # Add random noise to the input
        x_t = self.diffusion.q_sample(x, t, noise=eps)

        pred_eps = self.forward(x_t, t, cond)  # Pass through the model
        loss = self.mse(eps, pred_eps)
        return dict(loss=loss)