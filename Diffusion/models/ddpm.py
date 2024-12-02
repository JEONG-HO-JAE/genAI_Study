import torch
from torch import nn
from .types_ import *
from models.utils import *
from variance_scheduler.abs_var_scheduler import Scheduler
import pytorch_lightning as pl
from torch.nn.parameter import Parameter
from random import random
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

        
class GaussianDDPM(pl.LightningModule):
    def __init__(self, 
                 denoiser_module: nn.Module, 
                 opt: Union[Type[torch.optim.Optimizer], Callable[[Iterator[Parameter]], torch.optim.Optimizer]], 
                 T: int, 
                 input_channels: int,
                 sampling_timesteps : int,
                 width: int, 
                 height: int,
                 variance_scheduler: Scheduler,):
        super().__init__()
        self.denoiser_module = denoiser_module
        self.opt_class = opt
        self.T = T
        self.input_channels = input_channels
        self.self_condition = self.denoiser_module.self_condition
        self.sampling_timesteps = sampling_timesteps
        self.width = width
        self.height = height
        
        self.var_scheduler = variance_scheduler
        self.register_buffer('alphas_hat', self.var_scheduler.get_alpha_hat())
        self.register_buffer('alphas', self.var_scheduler.get_alphas())
        self.register_buffer('betas', self.var_scheduler.get_betas())
        self.register_buffer('betas_hat', self.var_scheduler.get_betas_hat())
        self.register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - self.var_scheduler.get_alpha_hat()))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 /self.var_scheduler.get_alpha_hat()))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.var_scheduler.get_alpha_hat() - 1))
        self.mse = nn.MSELoss()
        
    def on_fit_start(self) -> None:
        pass
    
    def configure_optimizers(self):
        return self.opt_class(params=self.parameters())  
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1) * x_t
            - self.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1, 1, 1) * noise
        )
        
    def forward(self, x: torch.FloatTensor, 
                t: int,
                x_self_cond) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.denoiser_module(x, t, x_self_cond)
    
    def training_step(self, 
                      batch: Tuple[torch.Tensor, torch.Tensor]):
        X, _ = batch
        eps = torch.randn_like(X)
        t: torch.Tensor = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1)
        
        x_t = x0_to_xt(X, alpha_hat, eps)
        
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                model_output = self.forward(x_t, t, x_self_cond)
                x_start = self.predict_start_from_noise(x_t, t, model_output)
                x_self_cond = x_start
                x_self_cond.detach_()
        
        pred_eps = self.forward(x_t, t, x_self_cond)
        
        loss = self.mse(eps, pred_eps)
        return dict(loss=loss)
         
    @torch.inference_mode()
    def validation_step(self,
                        batch: Tuple[torch.Tensor, torch.Tensor],):
        
        X, _ = batch
        t: torch.Tensor = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1)
        eps = torch.randn_like(X)
        x_t = x0_to_xt(X, alpha_hat, eps)
        
        x_self_cond = None
        pred_eps = self.forward(x_t, t, x_self_cond)
        
        loss = self.mse(eps, pred_eps)
        return dict(loss=loss)
    
    @torch.inference_mode()
    def sample(self, 
               batch_size: Optional[int] = None,
               sampling_timesteps: Optional[int] = None,
               T: Optional[int] = None,)  -> Union[torch.Tensor, List[torch.Tensor]]:
        
        batch_size = batch_size or 1
        T = T or self.T
        sampling_timesteps = sampling_timesteps
        noise = torch.randn(batch_size, self.input_channels, 
                            self.width, self.height,
                            device=self.device)
        beta_sqrt = torch.sqrt(self.betas)
        
        if sampling_timesteps < T:
            times = torch.linspace(-1, T-1, steps=sampling_timesteps+1)
            times = list(reversed(times.int().tolist()))
            time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            print("==========================Sampling==========================")
            with torch.no_grad():
                for time, time_next in tqdm(time_pairs, desc="DDIM sampling loop time step"):
                    t = torch.LongTensor([time]).to(self.device)
                    t_next = torch.LongTensor([time_next]).to(self.device)
                    # 모델에서 예측된 노이즈
                    pred_eps = self.denoiser_module(noise, t)
                    
                    # DDIM 파라미터 계산
                    alpha_hat_t = self.alphas_hat[t].reshape(-1, 1, 1, 1)
                    alpha_hat_t_next = self.alphas_hat[t_next].reshape(-1, 1, 1, 1) if time_next >= 0 else torch.zeros_like(alpha_hat_t)
                    
                    sigma_t = self.betas[t].reshape(-1, 1, 1, 1)
                    
                    # Calculate DDIM coefficients
                    sigma = torch.sqrt(
                        (1 - alpha_hat_t / alpha_hat_t_next) * (1 - alpha_hat_t_next) / (1 - alpha_hat_t)
                    )
                    c = torch.sqrt(1 - alpha_hat_t_next - sigma**2)

                    # noise = torch.sqrt(alpha_hat_t_next) * x0 + c * pred_eps + sigma * torch.randn_like(noise)
                    # 복원된 x_0 계산
                    x_start = (1 / torch.sqrt(alpha_hat_t)) * noise - \
                            (torch.sqrt(1 - alpha_hat_t) / torch.sqrt(alpha_hat_t)) * pred_eps
                    # 내가 잘못 작성한 코드 부분
                    # x0 = 1 / torch.sqrt(alpha_hat_t) * noise - 1 / torch.sqrt(1 - alpha_hat_t) * pred_eps
                    
                    noise = torch.sqrt(alpha_hat_t_next) * x_start + c * pred_eps + sigma * torch.randn_like(noise) if time_next >= 0 else x_start 

                    
        else:
            print("==========================Sampling==========================")
            with torch.no_grad():
                for t in tqdm(range(T-1, -1, -1), desc="DDPM sampling time steps"):
                    t = torch.LongTensor([t]).to(self.device)
                    n_t = torch.LongTensor([t-1]).to(self.device)
                    z = torch.randn_like(noise)
                    eps = self.denoiser_module(noise, t)
                    sigma = beta_sqrt[t].reshape(-1, 1, 1, 1)
                    
                    alpha_t = self.alphas[t].reshape(-1, 1, 1, 1)
                    alpha_hat_t = self.alphas_hat[t].reshape(-1, 1, 1, 1)
                    n_alpha_hat_t = self.alphas_hat[n_t].reshape(-1, 1, 1, 1)
                    
                    # print(sigma-sigma_2)
                    noise = 1 / (torch.sqrt(alpha_t)) * \
                        (noise - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * eps) + sigma * z
                
        noise = (noise + 1) / 2 
        return noise 