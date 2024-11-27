import torch
from typing import Optional

def x0_to_xt(x_0: torch.Tensor, 
             alpha_hat_t: torch.Tensor, 
             eps: Optional[torch.Tensor] = None) -> torch.Tensor:
    
    """
    Compute x_t from x_0 using a closed form using theorem from original DDPM paper (Ho et al.)
    :param x_0: the image without noise
    :param alpha_hat_t: the cumulated variance schedule at time t
    :param eps: pure noise from N(0, 1)
    :return: the noised image x_t at step t
    """
    
    if eps is None:
        eps = torch.randn_like(x_0)
    return torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * eps

def xt_to_x0(x_t: torch.Tensor, 
             alpha_hat_t: torch.Tensor, 
             eps: Optional[torch.Tensor] = None) -> torch.Tensor:
    
    return (x_t - torch.sqrt(1-alpha_hat_t) * eps) / torch.sqrt(alpha_hat_t)