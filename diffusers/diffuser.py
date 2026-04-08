import os
from abc import abstractmethod
from functools import partial
import collections

import numpy as np
import torch
import torch.nn as nn
from rich.progress import track
from rich.console import Console
import copy
from torch.optim.lr_scheduler import LambdaLR
import json

try:
    from utils import set_device
    from utils.diffusion_utils import make_beta_schedule, extract, default
    from utils.ema import EMA
    from utils.logging import LogTracker
    from models import UNet
    from models.lr_schedulers import get_linear_warmup_lr_lambda
    from metrics.inception import InceptionV3
    from metrics.compute_fid import get_activations_for_batch, calculate_frechet_distance
    from metrics.compute_is import resolve_feature_layer_for_metric, create_feature_extractor, get_featuresdict_for_batch, isc_featuresdict_to_metric
except ImportError:
    from ..utils import set_device
    from ..utils.diffusion_utils import make_beta_schedule, extract, default
    from ..utils.ema import EMA
    from ..utils.logging import LogTracker
    from ..models import UNet
    from ..models.lr_schedulers import get_linear_warmup_lr_lambda
    from ..metrics.inception import InceptionV3
    from ..metrics.compute_fid import get_activations_for_batch, calculate_frechet_distance
    from ..metrics.compute_is import resolve_feature_layer_for_metric, create_feature_extractor, get_featuresdict_for_batch, isc_featuresdict_to_metric




CustomResult = collections.namedtuple('CustomResult', 'name result')

class BaseModel():
    """
    Base class for the Palette Model, that has training loop, saving and loading weights, saving and loading optimizer state, resume training, etc.
    """
    def __init__(self, cfg, phase, phase_loader, val_loader, metrics, logger, writer):
        """ init model with basic input, which are from __init__(**kwargs) function in inherited class """
        self.cfg = cfg
        self.phase = phase
        self.set_device = partial(set_device, cpu=cfg['train'].get('cpu', False))#, rank=cfg['global_rank'])

        ''' optimizers and schedulers '''
        self.schedulers = []
        self.optimizers = []

        ''' process record '''
        self.batch_size = self.cfg[self.phase]['batch_size']
        self.epoch = 0
        self.iter = 0
        self.val_iter = 0 

        self.phase_loader = phase_loader
        self.val_loader = val_loader
        self.metrics = metrics

        ''' logger to log file, which only work on GPU 0. writer to tensorboard and result file '''
        self.logger = logger
        self.writer = writer
        self.results_dict = CustomResult([],[]) # {"name":[], "result":[]}

    def train(self):
        while self.epoch < self.cfg['train']['n_epochs']:
            self.epoch += 1
            if self.cfg['train']['distributed']:
                ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
                self.phase_loader.sampler.set_epoch(self.epoch) 

            train_log = self.train_step()

            ''' save logged informations into log dict ''' 
            train_log.update({'epoch': self.epoch, 'iters': self.iter})

            ''' print logged informations to the screen and tensorboard ''' 
            for key, value in train_log.items():
                self.logger.info('{:5s}: {}\t'.format(str(key), value))
            
            if self.epoch % self.cfg['train']['save_checkpoint_epoch'] == 0:
                self.logger.info('Saving the weights, schedulers and optimizer state at the end of epoch {:.0f}'.format(self.epoch))
                self.save_everything()

            if self.epoch % self.cfg['train']['val_epoch'] == 0:
                self.logger.info("\n\n\n------------------------------Validation Start------------------------------")
                if self.val_loader is None:
                    self.logger.warning('Validation stop where dataloader is None, Skip it.')
                else:
                    val_log = self.val_step()
                    for key, value in val_log.items():
                        self.logger.info('{:5s}: {}\t'.format(str(key), value))
                self.logger.info("\n------------------------------Validation End------------------------------\n\n")
        self.logger.info('Number of Epochs has reached the limit, End.')

    def test(self):
        pass

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your networks.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError('You must specify how to do validation on your networks.')

    def test_step(self):
        pass
    
    def print_network(self, network):
        """ print network structure, only work on GPU 0 """
        # if self.cfg['global_rank'] !=0:
        #     return
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        
        s, n = str(network), sum(map(lambda x: x.numel(), network.parameters()))
        net_struc_str = '{}'.format(network.__class__.__name__)
        self.logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        self.logger.info(s)

    def save_network(self, network, network_label):
        """ save network structure, only work on GPU 0 """
        # if self.cfg['global_rank'] !=0:
        #     return
        save_filename = '{}_{}.pth'.format(self.epoch, network_label)
        checkpoint_dir = os.path.join(self.cfg['paths']["experiments_root"], self.cfg['paths']["experiment_name"], self.cfg['paths']['checkpoint'])
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        save_path = os.path.join(checkpoint_dir, save_filename)
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, network, network_label, strict=True):
        if not self.cfg['paths']['resume_state']:
            return 
        self.logger.info('Begin loading pretrained model [{:s}] ...'.format(network_label))

        model_name = "{}_{}.pth".format(self. cfg['paths']['resume_state'], network_label)
        model_path = os.path.join(self.cfg['paths']["experiments_root"], self.cfg['paths']["experiment_name"], self.cfg['paths']['checkpoint'], model_name)
        
        if not os.path.exists(model_path):
            self.logger.warning('Pretrained model in [{:s}] does not exist, Skip it'.format(model_path))
            return

        self.logger.info('Loading pretrained model from [{:s}] ...'.format(model_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        network.load_state_dict(torch.load(model_path, map_location = lambda storage, loc: self.set_device(storage)), strict=strict)

    def save_training_state(self):
        """ saves training state during training, only work on GPU 0 """
        # if self.cfg['global_rank'] !=0:
        #     return
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        state = {'epoch': self.epoch, 'iter': self.iter, 'schedulers': [], 'optimizers': [], 'val_iter': self.val_iter}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        save_filename = '{}.state'.format(self.epoch)
        save_path = os.path.join(self.cfg['paths']["experiments_root"], self.cfg['paths']["experiment_name"], self.cfg['paths']['checkpoint'], save_filename)
        torch.save(state, save_path)

    def resume_training(self):
        """ resume the optimizers and schedulers for training, only work when phase is test or resume training enable """
        if self.phase != 'train' or not self.cfg['paths']['resume_state']:
            return
        self.logger.info('Beign loading training states'.format())
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        
        state_name = "{}.state".format(self. cfg['paths']['resume_state'])
        state_path = os.path.join(self.cfg['paths']["experiments_root"], self.cfg['paths']["experiment_name"], self.cfg['paths']['checkpoint'], state_name)
        
        if not os.path.exists(state_path):
            self.logger.warning('Training state in [{:s}] does not exist, Skip it'.format(state_path))
            return

        self.logger.info('Loading training state for [{:s}] ...'.format(state_path))
        resume_state = torch.load(state_path, map_location = lambda storage, loc: self.set_device(storage))
        
        resume_optimizers = resume_state['optimizers']
        resume_schedulers = resume_state['schedulers']
        assert len(resume_optimizers) == len(self.optimizers), 'Wrong lengths of optimizers {} != {}'.format(len(resume_optimizers), len(self.optimizers))
        assert len(resume_schedulers) == len(self.schedulers), 'Wrong lengths of schedulers {} != {}'.format(len(resume_schedulers), len(self.schedulers))
        for i, o in enumerate(resume_optimizers):
            self.optimizers[i].load_state_dict(o)
        for i, s in enumerate(resume_schedulers):
            self.schedulers[i].load_state_dict(s)

        self.epoch = resume_state['epoch']
        self.iter = resume_state['iter']
        self.val_iter = resume_state['val_iter']    

    def load_everything(self):
        pass 
    
    @abstractmethod
    def save_everything(self):
        raise NotImplementedError('You must specify how to save your networks, optimizers and schedulers.')
    

class BaseDiffuser(nn.Module):
    """
    Base class for Network weights initialization
    """
    def __init__(self, init_type='kaiming', gain=0.02):
        super(BaseDiffuser, self).__init__()
        self.init_type = init_type
        self.gain = gain

    def init_weights(self):
        """
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        """
    
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if self.init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, self.gain)
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=self.gain)
                elif self.init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif self.init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=self.gain)
                elif self.init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % self.init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(self.init_type, self.gain)


class Diffuser(BaseDiffuser):
    def __init__(self, unet, beta_schedule, **kwargs):
        super(Diffuser, self).__init__(**kwargs)
        self.denoise_fn = UNet(**unet)
        self.beta_schedule = beta_schedule

    def set_loss(self, loss_fn):
        self.loss_fn = loss_fn

    def set_new_noise_schedule(self, device=torch.device('cuda'), phase='train'):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(**self.beta_schedule)
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
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

    def q_sample(self, y_0, sample_gammas, noise=None):
        # Samples from the q distribution, i.e. calculates y_t from y_0
        noise = default(noise, lambda: torch.randn_like(y_0))
        return (
            sample_gammas.sqrt() * y_0 +
            (1 - sample_gammas).sqrt() * noise
        )

    @torch.no_grad()
    def p_sample(self, y_t, t, clip_denoised=True, y_cond=None):
        # Samples from the learned model distribution, given the learned mean and variance, computes \hat{y}_t
        model_mean, model_log_variance = self.p_mean_variance(
            y_t=y_t, t=t, clip_denoised=clip_denoised, y_cond=y_cond)
        noise = torch.randn_like(y_t) if any(t > 0) else torch.zeros_like(y_t)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def restoration(self, y_cond, y_t=None, y_0=None, mask=None, sample_num=8):
        # Restores the conditional image (y_cond) to the original image (y_0) using the diffusion model
        # e.g. does inpainting based on the mask
        b, *_ = y_cond.shape

        assert self.num_timesteps > sample_num, 'num_timesteps must greater than sample_num'
        sample_inter = (self.num_timesteps // sample_num)
        
        y_t = default(y_t, lambda: torch.randn_like(y_cond))
        ret_arr = y_t
        for i in track(reversed(range(0, self.num_timesteps)), description='sampling loop time step', total=self.num_timesteps, console=Console(stderr=True)):
            t = torch.full((b,), i, device=y_cond.device, dtype=torch.long)
            y_t = self.p_sample(y_t, t, y_cond=y_cond)
            if mask is not None:
                y_t = y_0 * (1. - mask) + mask * y_t
            if i % sample_inter == 0:
                ret_arr = torch.cat([ret_arr, y_t], dim=0)
        return y_t, ret_arr

    def forward(self, y_0, y_cond=None, mask=None, noise=None):
        # sampling from p(gammas)
        b, *_ = y_0.shape
        t = torch.randint(1, self.num_timesteps, (b,), device=y_0.device).long()
        gamma_t1 = extract(self.gammas, t-1, x_shape=(1, 1))
        gamma_t2 = extract(self.gammas, t, x_shape=(1, 1))
        sample_gammas = (gamma_t2 - gamma_t1) * torch.rand((b, 1), device=y_0.device) + gamma_t1
        sample_gammas = sample_gammas.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(y_0))
        y_noisy = self.q_sample(
            y_0=y_0, sample_gammas=sample_gammas.view(-1, 1, 1, 1), noise=noise)

        if mask is not None:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy*mask+(1.-mask)*y_0], dim=1), sample_gammas)
            loss = self.loss_fn(mask*noise, mask*noise_hat)
        else:
            noise_hat = self.denoise_fn(torch.cat([y_cond, y_noisy], dim=1), sample_gammas)
            loss = self.loss_fn(noise, noise_hat)
        return loss


class PaletteModel(BaseModel):
    def __init__(self, networks, losses, sample_num, task, optimizers, schedulers=None, ema_scheduler=None, **kwargs):
        ''' must to init BaseModel with kwargs '''
        super(PaletteModel, self).__init__(**kwargs)

        ''' networks, dataloder, optimizers, losses, etc. '''
        self.loss_fn = losses[0]
        self.netG = networks[0]
        
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        
        ''' networks can be a list, and must convert by self.set_device function if using multiple GPU. '''
        self.netG = self.set_device(self.netG, distributed=self.cfg['train']['distributed'])
        if self.ema_scheduler is not None:
            self.netG_EMA = self.set_device(self.netG_EMA, distributed=self.cfg['train']['distributed'])
        self.load_networks()

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0])
        self.optimizers.append(self.optG)
        
        num_params = sum(p.numel() for p in self.netG.parameters() if p.requires_grad)
        self.logger.info(f'Total number of parameters: {num_params / 1e6:.2f}M')
        
        if schedulers:
            for scheduler in schedulers:
                if scheduler.get('warmup', None):
                    self.schedulers.append(LambdaLR(self.optG, lr_lambda=get_linear_warmup_lr_lambda(scheduler['warmup'])))
        
        self.resume_training() 

        if self.cfg['train']['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        ''' can rewrite in inherited class for more informations logging '''
        self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        
    def set_input(self, data):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
    
    def get_current_visuals(self, phase='train'):
        dict = {
            'gt_image': (self.gt_image.detach()[:].float().cpu()+1)/2,
            'cond_image': (self.cond_image.detach()[:].float().cpu()+1)/2,
        }
        if self.task in ['inpainting','uncropping']:
            dict.update({
                'mask': self.mask.detach()[:].float().cpu(),
                'mask_image': (self.mask_image+1)/2,
            })
        if phase != 'train':
            dict.update({
                'output': (self.output.detach()[:].float().cpu()+1)/2
            })
        return dict

    def save_current_results(self):
        ret_path = []
        ret_result = []
        for idx in range(self.batch_size):
            ret_path.append('GT_{}'.format(self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())

            ret_path.append('Process_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            
            ret_path.append('Out_{}'.format(self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
        
        if self.task in ['inpainting','uncropping']:
            ret_path.extend(['Mask_{}'.format(name) for name in self.path])
            ret_result.extend(self.mask_image)

        self.results_dict = self.results_dict._replace(name=ret_path, result=ret_result)
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in track(self.phase_loader, description="Training", total=len(self.phase_loader), console=Console(stderr=True)):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)
            loss.backward()
            self.optG.step()

            self.iter += 1
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.cfg['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.writer.add_scalar(key, value)
                    # self.logger.info('{:5s}: {}\t'.format(str(key), value))
                # for key, value in self.get_current_visuals().items():
                #     self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def val_step(self):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for batch_idx, val_data in track(enumerate(self.val_loader), description="Validation", total=len(self.val_loader), console=Console(stderr=True)):
                self.set_input(val_data)
                if self.cfg['train']['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                    
                self.val_iter += 1
                self.writer.set_iter(self.epoch, self.val_iter, phase='val')

                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.val_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                
                if batch_idx < self.cfg['train']['val_batches_for_img_save']:
                    for key, value in self.get_current_visuals(phase='val').items():
                        self.writer.add_images(key, value)
                    self.writer.save_images(self.save_current_results())

        return self.val_metrics.result()

    def test(self):
        self.netG.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for phase_data in track(self.phase_loader, description="Testing", total=len(self.phase_loader), console=Console(stderr=True)):
                self.set_input(phase_data)
                if self.cfg['train']['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                        
                self.iter += 1
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results(), save_dir=self.cfg["paths"].get('img_save_dir', None))
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))
            
    def test_with_metrics(self):
        self.netG.eval()
        self.test_metrics.reset()
        
        metrics_dict = {}
        
        if self.cfg['test']['metrics'].get('fid', False):
            dims = self.cfg['test']['metrics']['fid_dims']
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
            fid_model = InceptionV3([block_idx])
            fid_model = self.set_device(fid_model, distributed=self.cfg['train']['distributed'])
            fid_model.eval()
            pred_arr_gt = np.empty((len(self.phase_loader.dataset), dims))
            pred_arr_out = np.empty((len(self.phase_loader.dataset), dims))
                
        if self.cfg['test']['metrics'].get('is', False):
            is_splits = self.cfg['test']['metrics']['is_splits']
            is_kwargs = {'cuda': not self.cfg['train'].get('cpu', False), 'verbose': False, 'splits': is_splits}
            feature_extractor = "inception-v3-compat"
            feature_layer_isc = resolve_feature_layer_for_metric("isc")
            feature_layers = [feature_layer_isc]
            is_model = create_feature_extractor(feature_extractor, list(feature_layers), **is_kwargs)
            out_gt, out_out = None, None
            pass
        
        with torch.no_grad():
            for i, phase_data in track(enumerate(self.phase_loader), description="Testing", total=len(self.phase_loader), console=Console(stderr=True)):
                self.set_input(phase_data)
                if self.cfg['train']['distributed']:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num)
                else:
                    if self.task in ['inpainting','uncropping']:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                            y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num)
                    else:
                        self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num)
                        
                self.iter += 1
                self.writer.set_iter(self.epoch, self.iter, phase='test')
                for met in self.metrics:
                    key = met.__name__
                    value = met(self.gt_image, self.output)
                    self.test_metrics.update(key, value)
                    self.writer.add_scalar(key, value)
                    
                self.gt_image = (self.gt_image + 1) / 2
                self.output = (self.output + 1) / 2
                    
                if self.cfg['test']['metrics'].get('fid', False):
                    pred_gt = get_activations_for_batch(self.gt_image, fid_model)
                    pred_out = get_activations_for_batch(self.output, fid_model)
                    
                    pred_arr_gt[i*self.batch_size:(i+1)*self.batch_size] = pred_gt
                    pred_arr_out[i*self.batch_size:(i+1)*self.batch_size] = pred_out
                    
                if self.cfg['test']['metrics'].get('is', False):
                    self.gt_image = self.gt_image * 255
                    self.gt_image = self.gt_image.to(torch.uint8)
                    
                    self.output = self.output * 255
                    self.output = self.output.to(torch.uint8)

                    out_gt_featuresdict = get_featuresdict_for_batch(self.gt_image, is_model)
                    if out_gt is None:
                        out_gt = out_gt_featuresdict
                    else:
                        out_gt = {k: out_gt[k] + out_gt_featuresdict[k] for k in out_gt.keys()}
                        
                    out_out_featuresdict = get_featuresdict_for_batch(self.output, is_model)
                    if out_out is None:
                        out_out = out_out_featuresdict
                    else:
                        out_out = {k: out_out[k] + out_out_featuresdict[k] for k in out_out.keys()}
                        
                if self.gt_image.dtype == torch.uint8:
                    self.gt_image = self.gt_image.float() / 255
                    self.output = self.output.float() / 255
                
                if self.gt_image.min() == 0:
                    self.gt_image = self.gt_image * 2 - 1
                    self.output = self.output * 2 - 1
                        
                for key, value in self.get_current_visuals(phase='test').items():
                    self.writer.add_images(key, value)
                self.writer.save_images(self.save_current_results(), save_dir=self.cfg["paths"].get('img_save_dir', None))
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))
            
        if self.cfg['test']['metrics'].get('fid', False):
            mu_gt = np.mean(pred_arr_gt, axis=0)
            sigma_gt = np.cov(pred_arr_gt, rowvar=False)
            
            mu_out = np.mean(pred_arr_out, axis=0)
            sigma_out = np.cov(pred_arr_out, rowvar=False)
            
            fid_value = calculate_frechet_distance(mu_gt, sigma_gt, mu_out, sigma_out)
            metrics_dict['fid'] = f"{fid_value:.3f}"
            
            self.logger.info(f'FID: {fid_value:.3f}')

                
        if self.cfg['test']['metrics'].get('is', False):
            out_gt = {k: torch.cat(v, dim=0) for k, v in out_gt.items()}
            out_out = {k: torch.cat(v, dim=0) for k, v in out_out.items()}  
            is_gt = isc_featuresdict_to_metric(out_gt, feature_layer_isc, **is_kwargs)
            is_out = isc_featuresdict_to_metric(out_out, feature_layer_isc, **is_kwargs)
            
            metrics_dict['is_gt'] = f"{is_gt['inception_score_mean']:.3f} ± {is_gt['inception_score_std']:.3g}"
            metrics_dict['is_output'] = f"{is_out['inception_score_mean']:.3f} ± {is_out['inception_score_std']:.3g}"
            
            self.logger.info(f'IS (GT): {is_gt["inception_score_mean"]:.3f} ± {is_gt["inception_score_std"]:.3g}')
            self.logger.info(f'IS (Output): {is_out["inception_score_mean"]:.3f} ± {is_out["inception_score_std"]:.3g}')

        json_path = os.path.join(
            self.cfg['paths']["experiments_root"], 
            self.cfg['paths']["experiment_name"],
            self.cfg['paths']['results'], 
            self.phase, 
            'metrics.json' if self.cfg["paths"].get('img_save_dir', None) is None else f'metrics_{self.cfg["paths"]["img_save_dir"]}.json'
        )

        
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        if self.cfg['train']['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.load_network(network=self.netG, network_label=netG_label, strict=False)
        if self.ema_scheduler is not None:
            self.load_network(network=self.netG_EMA, network_label=netG_label+'_ema', strict=False)

    def save_everything(self):
        """ load pretrained model and training state. """
        if self.cfg['train']['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
        self.save_network(network=self.netG, network_label=netG_label)
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()    
