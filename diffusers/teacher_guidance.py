"""
Diffusion functionalities for training image-to-image generators.
"""

import torch
import torch.nn.functional as F
import numpy as np
from rich.progress import track
from rich.console import Console
import math
from inspect import isfunction

from functools import partial

from diffusers.diffuser import BaseDiffuser
from models.unet import UNet

try:
    from utils.diffusion_utils import make_beta_schedule, extract, default
except ImportError:
    from ..utils.diffusion_utils import make_beta_schedule, extract, default


class TeacherGuidedDiffuser(BaseDiffuser):
    """
    Adapted from:
    https://github.com/jpmorganchase/i2i_Palette-Image-to-Image-Diffusion-Models/blob/i2i/models/network.py
    
    - denoiser and teacher_denoiser are two nn.Modules with the same architecture
    """
    def __init__(self, 
                unet,
                beta_schedule,
                *args, 
                **kwargs
                ):
        """
        :param denoiser: nn.Module, neural net to be adapted
        :param teacher_denoiser: nn.Module, neural net with same architecture as denoiser,
        already pre-trained and used as a reference during training of denoiser
        """
        super().__init__()
        
        self.denoise_fn = UNet(**unet)
        self.teacher_denoise_fn = UNet(**unet)
        self.teacher_denoise_fn.eval()

        self.beta_schedule = beta_schedule
        self.forget_alpha = kwargs.get('forget_alpha', 0.25)
        self.max_loss = kwargs.get('max_loss', False)
        self.learn_noise = kwargs.get('learn_noise', False)
        self.learn_others = kwargs.get('learn_others', False)
        
    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule)
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # For https://arxiv.org/pdf/2111.05826 Appendix A Eq. (6), when trying to find y_0
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # \sigma^2 in https://arxiv.org/pdf/2111.05826 Appendix A Eq. (5)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        # For \mu in https://arxiv.org/pdf/2111.05826 Appendix A Eq. (5)
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        # Computes the initial image from the noisy version and random noise
        # https://arxiv.org/pdf/2111.05826 Appendix A Eq. (6)
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        # Computes the posterior q(y_{t-1} | y_t, y_0) for the diffusion model
        # mu and \sigma^2 in https://arxiv.org/pdf/2111.05826 Appendix A Eq. (5)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        # Computes the learned posterior mean and variance for the diffusion model at timestep t for y_{t-1}
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance

    def q_sample(self, y_0, sample_gammas, noise=None, labels=1.0):
        # Samples from the q distribution, i.e. calculates y_t from y_0
        noise = default(noise, lambda: torch.randn_like(y_0))
        
        noise_target = sample_gammas.sqrt() * y_0 * (labels[:, None, None, None] + 1.0) * 0.5 + \
                       (1 - sample_gammas).sqrt() * noise + \
                       (1 - (1 - sample_gammas).sqrt()) * noise * (1.0 - labels[:, None, None, None]) * 0.5
    
        noise_target_original = sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise # original

        return (noise_target), (noise_target_original)

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        # Samples from the learned model distribution, given the learned mean and variance, computes \hat{y}_t
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, progress_bar=None):
        # Restores the conditional image (y_cond) to the original image (y_0) using the diffusion model
        # e.g. does inpainting based on the mask
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        
        if progress_bar is None:
            for i in track(reversed(range(0, self.num_timesteps)), description='sampling loop time step', total=self.num_timesteps, console=Console(stderr=True)):
                t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
                y_t = self.p_sample(y_t, t, y_cond=y_cond)
                if mask is not None:
                    y_t = y_0 * (1. - mask) + mask * y_t
                if i % sample_inter == 0:
                    ret_arr = torch.cat([ret_arr, y_t], dim=0)
        else:
            inner_task = progress_bar.add_task('Sampling loop time step', total=self.num_timesteps, console=Console(stderr=True))
            
            for i in reversed(range(0, self.num_timesteps)):
                t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
                y_t = self.p_sample(y_t, t, y_cond=y_cond)
                if mask is not None:
                    y_t = y_0 * (1. - mask) + mask * y_t
                if i % sample_inter == 0:
                    ret_arr = torch.cat([ret_arr, y_t], dim=0)
                progress_bar.update(inner_task, advance=1)
                
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None, labels=1.0, fix_decoder=False):
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        
        sample_gammas = (gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy, y_noise_original = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise, labels=labels)

        if self.learn_others: # Not needed in our case, only a baseline, so it may be redundant
            if mask is not None:
                with torch.no_grad():
                    teacher_feat = self.teacher_denoise_fn(
                        torch.cat([y_cond, y_noise_original * mask + (1. - mask) * y_0], dim=1), sample_gammas
                    )
                    tmp = labels > 0
                    retain_idx = tmp.nonzero().squeeze()

                    num_retain = torch.sum((labels + 1.0) * 0.5).item()
                    if num_retain > 0:
                        batchsize = labels.size()[0]
                        num_repeat = math.ceil(batchsize/num_retain)
                        updated_feat = torch.tile(teacher_feat[retain_idx], (num_repeat, 1, 1, 1))[:batchsize]
                    else:
                        updated_feat = torch.randn_like(teacher_feat)
                    
                noise_hat = self.denoise_fn(
                    torch.cat([y_cond, y_noise_original * mask + (1. - mask) * y_0], dim=1), sample_gammas
                )
                
                # If we want to use MSE in the loss we need to provide the additional arguments
                ## loss_fn was previously metrics.losses.weighted_others_loss() here
                loss = self.loss_fn(
                    noise_hat, teacher_feat, updated_feat, labels, 
                    max_loss=self.max_loss, forget_alpha=self.forget_alpha
                )
            else:
                noise_hat = self.denoise_fn(torch.cat([y_cond, y_noise_original], dim=1), sample_gammas)
                ## loss_fn was previously metrics.losses.weighted_mse_loss() here
                loss = self.loss_fn(noise, noise_hat, labels, max_loss=self.max_loss, forget_alpha=self.forget_alpha)
                    
        elif self.learn_noise and not self.max_loss: # Not needed in our case, only a baseline, so it may be redundant
            if mask is not None:
                if fix_decoder:
                    with torch.no_grad():
                        teacher_feat = self.teacher_denoise_fn(
                            torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1), sample_gammas, return_feat=True
                        )
                    
                    noise_hat = self.denoise_fn(
                        torch.cat([y_cond, y_noise_original*mask+(1. - mask)*y_0], dim=1), sample_gammas, return_feat=True
                    )
                    
                    loss = 0
                    for i in range(len(noise_hat)):
                        ## loss_fn was previously metrics.losses.weighted_mse_loss() here
                        loss += self.loss_fn(
                            noise_hat[i], teacher_feat[i], labels, use_noise=True, max_loss=self.max_loss, forget_alpha=self.forget_alpha
                        )
                
                else: # For now we always fix de decoder so this is not needed right now
                    noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1), sample_gammas)
                    ## loss_fn was previously metrics.losses.weighted_mse_loss() here
                    loss = self.loss_fn(mask*noise, mask*noise_hat, labels, max_loss=self.max_loss, forget_alpha=self.forget_alpha)
            else:
                noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
                ## loss_fn was previously metrics.losses.weighted_mse_loss() here
                loss = self.loss_fn(noise, noise_hat, labels, max_loss=self.max_loss, forget_alpha=self.forget_alpha)

        elif not self.learn_noise:
            if self.max_loss: # Not needed in our case, only a baseline, so it may be redundant
                if mask is not None:
                    if fix_decoder:
                        with torch.no_grad():
                            teacher_feat = self.teacher_denoise_fn(
                                torch.cat([y_cond, y_noise_original * mask + (1. - mask) * y_0], dim=1), sample_gammas, return_feat=True
                            )
                        
                        noise_hat = self.denoise_fn(
                            torch.cat([y_cond, y_noise_original * mask + (1. - mask) * y_0], dim=1), sample_gammas, return_feat=True
                        )
                        
                        loss = 0
                        for i in range(len(noise_hat)):
                            ## loss_fn was previously metrics.losses.weighted_mse_loss() here
                            loss += self.loss_fn(
                                noise_hat[i], teacher_feat[i], labels, max_loss=self.max_loss, forget_alpha=self.forget_alpha
                            )
                            
                    else: # For now we always fix de decoder so this is not needed right now
                        noise_hat = self.denoise_fn(torch.cat([y_cond, y_noise_original * mask + (1. - mask) * y_0], dim=1), sample_gammas)
                        ## loss_fn was previously metrics.losses.weighted_mse_loss() here
                        loss = self.loss_fn(mask*noise, mask*noise_hat, labels, max_loss=self.max_loss, forget_alpha=self.forget_alpha)
                else:
                    noise_hat = self.denoise_fn(torch.cat([y_cond, y_noise_original], dim=1), sample_gammas)
                    ## loss_fn was previously metrics.losses.weighted_mse_loss() here
                    loss = self.loss_fn(noise, noise_hat, labels, max_loss=self.max_loss, forget_alpha=self.forget_alpha)
                    
            else: # This is the case we are interested in
                if mask is not None:
                    if fix_decoder:
                        with torch.no_grad():
                            teacher_feat = self.teacher_denoise_fn(
                                torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1), sample_gammas, return_feat=True
                            )
                        
                        noise_hat = self.denoise_fn(
                            torch.cat([y_cond, y_noise_original * mask + (1. - mask) * y_0], dim=1), sample_gammas, return_feat=True
                        )
                        
                        ### !IMPORTANT! - Compute loss only on the middle embeddings
                        loss = self.loss_fn(
                            noise_hat[-1], teacher_feat[-1], labels, max_loss=self.max_loss, forget_alpha=self.forget_alpha
                        )
                            
                    else: # For now we always fix de decoder so this is not needed right now
                        noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1), sample_gammas)
                        ## loss_fn was previously metrics.losses.weighted_mse_loss() here
                        loss = self.loss_fn(mask * noise, mask * noise_hat, labels, max_loss=self.max_loss, forget_alpha=self.forget_alpha)
                else:
                    noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
                    ## loss_fn was previously metrics.losses.weighted_mse_loss() here
                    loss = self.loss_fn(noise, noise_hat, labels)
        
        return loss


class Unlearner(TeacherGuidedDiffuser):
    pass

class MaxLoss(TeacherGuidedDiffuser):
    pass

class NoisyLabels(TeacherGuidedDiffuser):
    pass

class RetainLabel(TeacherGuidedDiffuser):
    pass

class RandomEncoder(TeacherGuidedDiffuser):
    pass


class TeacherGuidedDiffuserEpsilon(BaseDiffuser):
    """
    Adapted from:
    https://github.com/jpmorganchase/i2i_Palette-Image-to-Image-Diffusion-Models/blob/i2i/models/network.py
    
    - denoiser and teacher_denoiser are two nn.Modules with the same architecture
    """
    def __init__(self, 
                unet,
                beta_schedule,
                *args, 
                **kwargs
                ):
        """
        :param denoiser: nn.Module, neural net to be adapted
        :param teacher_denoiser: nn.Module, neural net with same architecture as denoiser,
        already pre-trained and used as a reference during training of denoiser
        """
        super().__init__()
        
        self.denoise_fn = UNet(**unet)
        self.teacher_denoise_fn = UNet(**unet)
        
        self.teacher_denoise_fn.eval()
        for p in self.teacher_denoise_fn.parameters():
            p.requires_grad = False

        self.beta_schedule = beta_schedule
        
        self.alpha = kwargs.get('alpha', 1)
        self.beta = kwargs.get('beta', 1)
        self.delta_phase1 = kwargs.get('delta_phase1', 2)
        self.delta_phase2 = kwargs.get('delta_phase2', 1)
        
    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule)
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        gammas = np.cumprod(alphas, axis=0)
        gammas_prev = np.append(1., gammas[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # For https://arxiv.org/pdf/2111.05826 Appendix A Eq. (6), when trying to find y_0
        self.register_buffer('gammas', to_torch(gammas))
        self.register_buffer('sqrt_recip_gammas', to_torch(np.sqrt(1. / gammas)))
        self.register_buffer('sqrt_recipm1_gammas', to_torch(np.sqrt(1. / gammas - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # \sigma^2 in https://arxiv.org/pdf/2111.05826 Appendix A Eq. (5)
        posterior_variance = betas * (1. - gammas_prev) / (1. - gammas)
        
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        # For \mu in https://arxiv.org/pdf/2111.05826 Appendix A Eq. (5)
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(gammas_prev) / (1. - gammas)))
        self.register_buffer('posterior_mean_coef2', to_torch((1. - gammas_prev) * np.sqrt(alphas) / (1. - gammas)))

    def predict_start_from_noise(self, y_t, t, noise):
        # Computes the initial image from the noisy version and random noise
        # https://arxiv.org/pdf/2111.05826 Appendix A Eq. (6)
        return (
            extract(self.sqrt_recip_gammas, t, y_t.shape) * y_t -
            extract(self.sqrt_recipm1_gammas, t, y_t.shape) * noise
        )

    def q_posterior(self, y_0_hat, y_t, t):
        # Computes the posterior q(y_{t-1} | y_t, y_0) for the diffusion model
        # mu and \sigma^2 in https://arxiv.org/pdf/2111.05826 Appendix A Eq. (5)
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, y_t.shape) * y_0_hat +
            extract(self.posterior_mean_coef2, t, y_t.shape) * y_t
        )
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, y_t.shape)
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, y_t, t, clip_denoised: bool, y_cond=None):
        # Computes the learned posterior mean and variance for the diffusion model at timestep t for y_{t-1}
        noise_level = extract(self.gammas, t, x_shape=(1, 1)).to(y_t.device)
        y_0_hat = self.predict_start_from_noise(
                y_t, t=t, noise=self.denoise_fn(torch.cat([y_cond, y_t], dim=1), noise_level))

        if clip_denoised:
            y_0_hat.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(
            y_0_hat=y_0_hat, y_t=y_t, t=t)
        return model_mean, posterior_log_variance
    
    def q_sample(self, y_0, sample_gammas, noise=None, labels=1.0):
        # Samples from the q distribution, i.e. calculates y_t from y_0
        noise = default(noise, lambda: torch.randn_like(y_0))
        y_0_retain = y_0[labels == 1]
        
        # If y_0_retain is empty (i.e. no retain labels in the batch), shuffle y_0 and use that instead
        if y_0_retain.shape[0] == 0:
            y_0_retain = y_0[torch.randperm(y_0.shape[0])]
        else:
            # Match batch size to that of y_0
            y_0_retain = torch.tile(y_0_retain, (math.ceil(y_0.shape[0] / y_0_retain.shape[0]), 1, 1, 1))[:y_0.shape[0]]

        noise_target = sample_gammas.sqrt() * y_0 * (labels[:, None, None, None] + 1.0) * 0.5 + \
                        (1 - sample_gammas).sqrt() * noise + \
                        sample_gammas.sqrt() * y_0_retain * (1.0 - labels[:, None, None, None]) * 0.5 
                    #    (1 - (1 - sample_gammas).sqrt()) * noise * (1.0 - labels[:, None, None, None]) * 0.5
        noise_target_original = sample_gammas.sqrt() * y_0 + (1 - sample_gammas).sqrt() * noise # original
        return (noise_target), (noise_target_original)

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        # Samples from the learned model distribution, given the learned mean and variance, computes \hat{y}_t
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8, progress_bar=None):
        # Restores the conditional image (y_cond) to the original image (y_0) using the diffusion model
        # e.g. does inpainting based on the mask
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps//sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        
        if progress_bar is None:
            for i in track(reversed(range(0, self.num_timesteps)), description='sampling loop time step', total=self.num_timesteps, console=Console(stderr=True)):
                t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
                y_t = self.p_sample(y_t, t, y_cond=y_cond)
                if mask is not None:
                    y_t = y_0 * (1. - mask) + mask * y_t
                if i % sample_inter == 0:
                    ret_arr = torch.cat([ret_arr, y_t], dim=0)
        else:
            inner_task = progress_bar.add_task('Sampling loop time step', total=self.num_timesteps, console=Console(stderr=True))
            
            for i in reversed(range(0, self.num_timesteps)):
                t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
                y_t = self.p_sample(y_t, t, y_cond=y_cond)
                if mask is not None:
                    y_t = y_0 * (1. - mask) + mask * y_t
                if i % sample_inter == 0:
                    ret_arr = torch.cat([ret_arr, y_t], dim=0)
                progress_bar.update(inner_task, advance=1)
                            
        return y_t, ret_arr
    
    def compute_g_t(self, f1, f2, f1_grad, f2_grad, epsilon_ratio, phase, eps_min=None, eps_max=None):
        f1_grad = [g for g in f1_grad if g is not None]
        f2_grad = [g for g in f2_grad if g is not None]
        
        f1_grad_norm = torch.sqrt(sum(torch.norm(g, p=2) ** 2 for g in f1_grad))
    
        if phase == 1.1 or phase == 1.2:
                psi = self.alpha * f1_grad_norm ** self.delta_phase1
        elif phase == 2:
                epsilon = eps_min + epsilon_ratio * (eps_max - eps_min)
                psi = self.beta * (f1 - epsilon) ** self.delta_phase2 * f1_grad_norm ** 2

        numerator = psi - sum((g2 * g1).sum() for g1, g2 in zip(f1_grad, f2_grad))
        denominator = sum((g1 * g1).sum() for g1 in f1_grad) + 1e-7
        eta = torch.max(numerator / denominator, torch.tensor(0.0))
        
        g_t = [g2 + eta * g1 for g1, g2 in zip(f1_grad, f2_grad)]
        return g_t
    
    def compute_g_t_harmonizer(self, f1_grad, f2_grad):
        f1_grad = [g for g in f1_grad if g is not None] # forget
        f2_grad = [g for g in f2_grad if g is not None] # retain
        
        g_t = []

        for g1, g2 in zip(f1_grad, f2_grad):
            # Flatten the gradients for computation
            g1_flat = g1.view(-1)
            g2_flat = g2.view(-1)
            
            eps = 1e-10 
            cos_sim = F.cosine_similarity(g1_flat, g2_flat, dim=0, eps=eps)
            
            if cos_sim < 0:
                # Compute inner products and norms
                dot = torch.dot(g1_flat, g2_flat)
                norm_g2_sq = torch.dot(g2_flat, g2_flat).clamp(min=eps)
                norm_g1_sq = torch.dot(g1_flat, g1_flat).clamp(min=eps)

                # Compute projections
                proj_g1_on_g2 = (dot / norm_g2_sq) * g2
                proj_g2_on_g1 = (dot / norm_g1_sq) * g1

                # Compute orthogonal components
                delta_f1 = g1 - proj_g1_on_g2
                delta_f2 = g2 - proj_g2_on_g1

                g_t.append(delta_f1 + delta_f2)
            else:
                # If the cosine similarity is non-negative, just add the gradients
                g_t.append(g1 + g2)
        
        return g_t
    
    
    def compute_g_t_harmonizer_curriculum(self, f1_grad, f2_grad, curriculum_condition):
        f1_grad = [g for g in f1_grad if g is not None] # forget
        f2_grad = [g for g in f2_grad if g is not None] # retain
        
        g_t = []

        for g1, g2 in zip(f1_grad, f2_grad):
            # Flatten the gradients for computation
            g1_flat = g1.view(-1)
            g2_flat = g2.view(-1)
            
            eps = 1e-10 
            cos_sim = F.cosine_similarity(g1_flat, g2_flat, dim=0, eps=eps)
            
            if cos_sim < 0:
                # Compute inner products and norms
                dot = torch.dot(g1_flat, g2_flat)
                norm_g2_sq = torch.dot(g2_flat, g2_flat).clamp(min=eps)
                norm_g1_sq = torch.dot(g1_flat, g1_flat).clamp(min=eps)

                # Compute projections
                proj_g1_on_g2 = (dot / norm_g2_sq) * g2
                proj_g2_on_g1 = (dot / norm_g1_sq) * g1

                # Compute orthogonal components
                delta_f1 = g1 - proj_g1_on_g2
                delta_f2 = g2 - proj_g2_on_g1

                if curriculum_condition:
                    g_t.append(g1 + delta_f2)
                else:
                    g_t.append(delta_f1 + g2)
            else:
                # If the cosine similarity is non-negative, just add the gradients
                g_t.append(g1 + g2)
        
        return g_t
    
    def compute_g_t_harmonizer_ripcgrad(self, f1_grad, f2_grad, alpha):
        f1_grad = [g for g in f1_grad if g is not None] # forget
        f2_grad = [g for g in f2_grad if g is not None] # retain
        
        g_t = []

        for g1, g2 in zip(f1_grad, f2_grad):
            # Flatten the gradients for computation
            g1_flat = g1.view(-1)
            g2_flat = g2.view(-1)
            
            eps = 1e-10 
            cos_sim = F.cosine_similarity(g1_flat, g2_flat, dim=0, eps=eps)
            
            if cos_sim < 0:
                # Compute inner products and norms
                dot = torch.dot(g1_flat, g2_flat)
                norm_g2_sq = torch.dot(g2_flat, g2_flat).clamp(min=eps)
                norm_g1_sq = torch.dot(g1_flat, g1_flat).clamp(min=eps)

                # Compute projections
                proj_g1_on_g2 = (dot / norm_g2_sq) * g2
                proj_g2_on_g1 = (dot / norm_g1_sq) * g1

                # Compute orthogonal components
                g1_pc = g1 - proj_g1_on_g2
                g2_pc = g2 - proj_g2_on_g1

                norm_g1_orig = g1.norm().clamp(min=eps)
                norm_g2_orig = g2.norm().clamp(min=eps)

                g1_pc_rescaled = g1_pc * (norm_g1_orig / g1_pc.norm().clamp(min=eps))
                g2_pc_rescaled = g2_pc * (norm_g2_orig / g2_pc.norm().clamp(min=eps))

                g_t.append((1 - alpha) * g1_pc_rescaled + alpha * g2_pc_rescaled)
            else:
                # If the cosine similarity is non-negative, just add the gradients
                g_t.append(g1 + g2)
        
        return g_t
    
    def forward(self, y_0, y_cond=None, mask=None, noise=None, labels=1.0, fix_decoder=False):
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        
        sample_gammas = (gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy, y_noise_original = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise, labels=labels)


        if mask is not None:
            with torch.no_grad():
                teacher_feat = self.teacher_denoise_fn(
                    torch.cat([y_cond, y_noisy * mask + (1. - mask) * y_0], dim=1), sample_gammas, return_feat=fix_decoder)
            
            noise_hat = self.denoise_fn(
                torch.cat([y_cond, y_noise_original * mask + (1. - mask) * y_0], dim=1), sample_gammas, return_feat=fix_decoder)
            
            if fix_decoder:
                f2, f1 = self.loss_fn(noise_hat[-1], teacher_feat[-1], labels)
            else:
                f2, f1 = self.loss_fn(noise_hat, teacher_feat, labels)
                
            return f1, f2                        
        