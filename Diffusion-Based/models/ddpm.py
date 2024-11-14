import torch
from torch import nn
from torch.nn import functional as F
from .types_ import *

import pytorch_lightning as pl
from variance_scheduler.abs_var_scheduler import Scheduler
from torch.nn.parameter import Parameter
from models.utils import *

class GaussianDDPM(pl.LightningModule):
    def __init__(self, denoiser_module: nn.Module, 
                 opt: Union[Type[torch.optim.Optimizer], Callable[[Iterator[Parameter]], torch.optim.Optimizer]], 
                 T: int, 
                 variance_scheduler: Scheduler, 
                 lambda_variational: float, 
                 width: int, 
                 height: int, 
                 input_channels: int, 
                 logging_freq: int, 
                 vlb: bool, 
                 init_step_vlb: int):
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
        with torch.no_grad():
            # Map image values from [0, 1] to [-1, 1]
            X = X * 2 - 1
            
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
        
        # If using the VLB loss, compute the VLB loss and add it to the total loss
        use_vlb = self.iteration >= self.init_step_vlb and self.vlb
        if use_vlb:
            loss_vlb = self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t).mean(dim=0).sum()
            loss = loss + loss_vlb

        self.iteration += 1

        return dict(loss=loss, noise_loss=noise_loss, vlb_loss=loss_vlb if use_vlb else None)
    
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
        loss = eps_loss = self.mse(eps, pred_eps)
        
        if self.iteration >= self.init_step_vlb and self.vlb:
            loss_vlb = self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t).mean(dim=0).sum()
            loss = loss + loss_vlb

        # Return the loss as a dictionary
        return dict(loss=loss, noise_loss=eps_loss, vlb_loss=loss_vlb if self.vlb else None)
    
    def variational_loss(self, x_t: torch.Tensor, x_0: torch.Tensor,
                        model_noise: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        """
        Compute variational loss for time step t
        
        Parameters:
            - x_t (torch.Tensor): the image at step t obtained with closed form formula from x_0
            - x_0 (torch.Tensor): the input image
            - model_noise (torch.Tensor): the unet predicted noise
            - v (torch.Tensor): the unet predicted coefficients for the variance
            - t (torch.Tensor): the time step
        
        Returns:
            - vlb (torch.Tensor): the pixel-wise variational loss, with shape [batch_size, channels, width, height]
        """
        vlb = 0.0
        t_eq_0 = (t == 0).reshape(-1, 1, 1, 1)
        
        # Compute variational loss for t=0 (i.e., first time step)
        if torch.any(t_eq_0):
            p = torch.distributions.Normal(mu_x_t(x_t, t, model_noise, self.alphas_hat, self.betas, self.alphas),
                                        sigma_x_t(v, t, self.betas_hat, self.betas))
            # Compute log probability of x_0 under the distribution p
            # and add it to the variational lower bound
            vlb += - p.log_prob(x_0) * t_eq_0.float()
            
        t_eq_last = (t == (self.T - 1)).reshape(-1, 1, 1, 1)
        
        # Compute variational loss for t=T-1 (i.e., last time step)
        if torch.any(t_eq_last):
            p = torch.distributions.Normal(0, 1)
            q = torch.distributions.Normal(sqrt(self.alphas_hat[t]) * x_0, (1 - self.alphas_hat[t]))
            # Compute KL divergence between distributions p and q
            # and add it to the variational lower bound
            vlb += torch.distributions.kl_divergence(q, p) * t_eq_last.float()
            
        # Compute variational loss for all other time steps
        mu_hat = mu_hat_xt_x0(x_t, x_0, t, self.alphas_hat, self.alphas, self.betas)
        sigma_hat = sigma_hat_xt_x0(t, self.betas_hat)
        q = torch.distributions.Normal(mu_hat, sigma_hat)  # q(x_{t-1} | x_t, x_0)
        mu = mu_x_t(x_t, t, model_noise, self.alphas_hat, self.betas, self.alphas).detach()
        sigma = sigma_x_t(v, t, self.betas_hat, self.betas)
        p = torch.distributions.Normal(mu, sigma)  # p(x_t | x_{t-1})
        # Compute KL divergence between distributions p and q
        # and add it to the variational lower bound
        vlb += torch.distributions.kl_divergence(q, p) * (~t_eq_last).float() * (~t_eq_0).float()
        
        return vlb
        
        
    
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