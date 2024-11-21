import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *

import pytorch_lightning as pl
from variance_scheduler.abs_var_scheduler import Scheduler
from torch.nn.parameter import Parameter
from models.utils import *

class GaussianDDPM(pl.LightningModule):
    def __init__(self, 
                 denoiser_module: nn.Module, 
                 opt: Union[Type[torch.optim.Optimizer], Callable[[Iterator[Parameter]], torch.optim.Optimizer]], 
                 T: int, 
                 variance_scheduler: Scheduler, 
                 lambda_variational: float, 
                 width: int, 
                 height: int, 
                 input_channels: int, 
                 logging_freq: int, ):
        """
        :param denoiser_module: The nn which computes the denoise step i.e. q(x_{t-1} | x_t, t)
        :param T: the amount of noising steps
        :param variance_scheduler: the variance scheduler cited in DDPM paper. See folder variance_scheduler for practical implementation
        :param lambda_variational: the coefficient in from of variational loss
        :param width: image width
        :param height: image height
        :param input_channels: image input channels
        :param logging_freq: frequency of logging loss function during training
        :param vlb: true to include the variational lower bound into the loss function
        :param init_step_vlb: the step at which the variational lower bound is included into the loss function
        """
        super().__init__()
        self.input_channels = input_channels
        self.denoiser_module = denoiser_module
        self.T = T
        self.opt_class = opt
        
        self.var_scheduler = variance_scheduler
        self.lambda_variational = lambda_variational
        self.alphas_hat: torch.FloatTensor = self.var_scheduler.get_alpha_hat().to(self.device)
        self.alphas: torch.FloatTensor = self.var_scheduler.get_alphas().to(self.device)
        self.betas = self.var_scheduler.get_betas().to(self.device)
        self.betas_hat = self.var_scheduler.get_betas_hat().to(self.device)
        self.mse = nn.MSELoss()
        
        self.width = width
        self.height = height
        self.logging_freq = logging_freq
        self.iteration = 0
    
    def forward(self, x: torch.FloatTensor, 
                t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the DDPM model.
        Args:
            x: Input image tensor.
            t: Time step tensor.
        Returns:
            Tuple of predicted noise tensor and predicted variance tensor.
        """
        return self.denoiser_module(x, t)
    
    def training_step(self, 
                      batch: Tuple[torch.Tensor, torch.Tensor], 
                      batch_idx: int):
        """
        Training step of the DDPM model.

        Args:
            batch: Tuple of input image tensor and target tensor.
            batch_idx: Batch index.

        Returns:
            Dictionary containing the loss.
        """
    
        X, _ = batch
            
        # Sample a random time step t from 0 to T-1 for each image in the batch
        t: torch.Tensor = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)
        
        # Compute alpha_hat for the selected time steps
        # Sample noise eps from a normal distribution with the same shape as X
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1)
        eps = torch.randn_like(X)
        
        x_t = x0_to_xt(X, alpha_hat, eps)
        pred_eps, v = self.forward(x_t, t)
        
        loss = 0.0
        noise_loss = self.mse(eps, pred_eps)
        loss = loss + noise_loss
        
        return dict(loss=loss, noise_loss=noise_loss)
    
    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor],
                        batch_idx: int):
        
        X, _ = batch
        with torch.no_grad():
            X = X * 2 - 1
            
        # Sample a time step t for each sample in the batch
        t: torch.Tensor = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1)
        eps = torch.randn_like(X)
        x_t = x0_to_xt(X, alpha_hat, eps)
        pred_eps, v = self.forward(x_t, t)
        loss = self.mse(eps, pred_eps)
        
        # Return the loss as a dictionary
        return dict(loss=loss)
    
    def on_fit_start(self) -> None:
        self.alphas_hat: torch.FloatTensor = self.alphas_hat.to(self.device)
        self.alphas: torch.FloatTensor = self.alphas.to(self.device)
        self.betas = self.betas.to(self.device)
        self.betas_hat = self.betas_hat.to(self.device)
    
    def configure_optimizers(self):
        return self.opt_class(params=self.parameters())    
    
    
    def generate(self, 
                 batch_size: Optional[int] = None, 
                 T: Optional[int] = None,
                 get_intermediate_steps: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Generate a batch of images via denoising diffusion probabilistic model
        :param batch_size: batch size of generated images. The default value is 1
        :param T: number of diffusion steps to generated images. The default value is the training diffusion steps
        :param get_intermediate_steps: return all the denoising steps instead of the final step output
        :return: The tensor [bs, c, w, h] of generated images or a list of tensors [bs, c, w, h] if get_intermediate_steps=True
        """
        batch_size = batch_size or 1
        T = T or self.T
        
        if get_intermediate_steps:
            steps = []
        
        X_noise = torch.randn(batch_size, self.input_channels, 
                              self.width, self.height,
                              device=self.device)
        
        with torch.no_grad():
            # T times sampling
            for t in range(T-1, -1, -1):
                if get_intermediate_steps:
                    steps.append(X_noise)
                t = torch.LongTensor([t]).to(self.device)
                eps, v = self.denoiser_module(X_noise, t)
                
                # if variational lower bound is present on the loss function hence v (the logit of variance) is trained
                # else the variance is taked fixed as in the original DDPM paper
                sigma = sigma_x_t(v, t, self.betas_hat, self.betas) 
                z = torch.rand_like(X_noise)
                
                if t == 0:
                    z.fill_(0)
                    
                alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
                alpha_hat_t = self.alphas_hat[t].reshape(-1, 1, 1, 1)
                
                X_noise -= (1 - alpha_t) / torch.sqrt(1 - alpha_hat_t) * eps
                X_noise /= torch.sqrt(alpha_t)
                X_noise += sigma * z

        if get_intermediate_steps:
            steps.append(X_noise)
            return steps
        return X_noise