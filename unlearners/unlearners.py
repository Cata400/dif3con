"""
Base Training functionalities for unlearning image-to-image generators.
"""
import gc
import os
from abc import abstractmethod
from functools import partial
import collections
from timm.utils import AverageMeter

import torch
import numpy as np
import torch.nn as nn
from rich.progress import track
from rich.console import Console
from rich.progress import Progress
import copy
from torch.optim.lr_scheduler import LambdaLR
import json

try:
    from utils import set_device
    from utils.ema import EMA
    from utils.logging import LogTracker
    from utils.utils import get_param, set_param
    from models.lr_schedulers import get_linear_warmup_lr_lambda
    from metrics.inception import InceptionV3
    from metrics.compute_fid import get_activations_for_batch, calculate_frechet_distance
    from metrics.compute_is import resolve_feature_layer_for_metric, create_feature_extractor, get_featuresdict_for_batch, isc_featuresdict_to_metric
    from metrics.losses import l2_regularizer, l2_salun_regularizer
    from models.optimizers import AdamWithEpochEMA
except ImportError:
    from ..utils import set_device
    from ..utils.ema import EMA
    from ..utils.logging import LogTracker
    from ..models.lr_schedulers import get_linear_warmup_lr_lambda
    from ..metrics.inception import InceptionV3
    from ..metrics.compute_fid import get_activations_for_batch, calculate_frechet_distance
    from ..metrics.compute_is import resolve_feature_layer_for_metric, create_feature_extractor, get_featuresdict_for_batch, isc_featuresdict_to_metric
    from ..metrics.losses import l2_regularizer, l2_salun_regularizer
    from ..models.optimizers import AdamWithEpochEMA
    from ..utils.utils import get_param, set_param



CustomResult = collections.namedtuple('CustomResult', 'name result label')

class BaseTrainer():
    """
    Base class for an unlearning training classes.
    Adapted from:
    https://github.com/jpmorganchase/i2i_Palette-Image-to-Image-Diffusion-Models/blob/i2i/core/base_model.py
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
        self.results_dict = CustomResult([],[], []) # {"name": [], "result": [], "label": []}

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your networks.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError('You must specify how to do validation on your networks.')

    @abstractmethod
    def test(self):
        raise NotImplementedError('You must specify how to do testing on your networks.')
    
    @abstractmethod
    def save_everything(self):
        raise NotImplementedError('You must specify how to save your networks, optimizers and schedulers.')

    @abstractmethod
    def load_networks(self):
        raise NotImplementedError('You must specify how to load networks.')

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

    def unlearn(self): # this is the exact same as train() and unlearn_fix_decoder() but calling the unlearn_step() method
        while self.epoch < self.cfg['train']['n_epochs']:
            self.epoch += 1
            if self.cfg['train']['distributed']:
                ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
                self.phase_loader.sampler.set_epoch(self.epoch) 

            train_log = self.unlearn_step()

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

    def unlearn_fix_decoder(self):
        while self.epoch < self.cfg['train']['n_epochs']:
            self.epoch += 1
            if self.cfg['train']['distributed']:
                ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
                self.phase_loader.sampler.set_epoch(self.epoch) 

            train_log = self.unlearn_step_fix_decoder()

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

    def unlearn_grad_harm(self): # this is the exact same as train() and unlearn_fix_decoder() but calling the unlearn_grad_harm_step() method
        while self.epoch < self.cfg['train']['n_epochs']:
            self.epoch += 1
            if self.cfg['train']['distributed']:
                ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
                self.phase_loader.sampler.set_epoch(self.epoch) 

            train_log = self.unlearn_grad_harm_step()

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
        
    def unlearn_epsilon(self):
        if self.cfg['train']['distributed']:
            netG_label = self.netG.module.__class__.__name__
        else:
            netG_label = self.netG.__class__.__name__
                
        if self.cfg['unlearn']['epsilon_ratio'] < 1:
            self.logger.info('Unlearning with epsilon_ratio = 0, phase 1, for epsilon_min (highest unlearning degree)')
            self.unlearn_epsilon_loop(epsilon_ratio=0, phase=1.1)
            
            # Compute eps_min with a forward pass
            self.logger.info('Computing eps_min with a forward pass')
            eps_min, _ = self.unlearn_epsilon_forward_step()
            self.logger.info(f'eps_min: {eps_min}')
            
            self.save_network(network=self.netG, network_label='eps_min_' + netG_label)
            self.save_training_state(label='eps_min')
        
        if self.cfg['unlearn']['epsilon_ratio'] > 0:
            if self.cfg['unlearn']['epsilon_ratio'] < 1:
                self.reset_everything()
            self.logger.info('Unlearning with epsilon_ratio = 1, phase 1, for epsilon_max (lowest unlearning degree)')
            self.unlearn_epsilon_loop(epsilon_ratio=1, phase=1.2)
            
            # Compute eps_max with a forward pass
            self.logger.info('Computing eps_max with a forward pass')
            eps_max, _ = self.unlearn_epsilon_forward_step()
            # _, eps_max = self.unlearn_epsilon_forward_step()
            self.logger.info(f'eps_max: {eps_max}')
            
            self.save_network(network=self.netG, network_label='eps_max_' + netG_label)
            self.save_training_state(label='eps_max')
            
        if 0 < self.cfg['unlearn']['epsilon_ratio'] < 1:
            self.reset_everything()
            self.logger.info(f'Unlearning with epsilon_ratio = {self.cfg["unlearn"]["epsilon_ratio"]}, phase 2')
            self.unlearn_epsilon_loop(epsilon_ratio=self.cfg['unlearn']['epsilon_ratio'], phase=2, eps_min=eps_min, eps_max=eps_max)
            
            self.save_network(network=self.netG, network_label=f'eps_{self.cfg["unlearn"]["epsilon_ratio"]}_' + netG_label)
            self.save_training_state(label=f'eps_{self.cfg["unlearn"]["epsilon_ratio"]}')
        self.logger.info('Number of Epochs has reached the limit, End.')

        
    def unlearn_epsilon_loop(self, epsilon_ratio=0, phase=1.1, eps_min=None, eps_max=None): # this is the exact same as train() and unlearn_fix_decoder() but calling the unlearn_epsilon_step() method
        while self.epoch < self.cfg['train']['n_epochs']:
            self.epoch += 1
            if self.cfg['train']['distributed']:
                ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
                self.phase_loader.sampler.set_epoch(self.epoch) 

            train_log = self.unlearn_epsilon_step(epsilon_ratio=epsilon_ratio, phase=phase, eps_min=eps_min, eps_max=eps_max)

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
                    val_log = self.val_step(phase=phase)
                    for key, value in val_log.items():
                        self.logger.info('{:5s}: {}\t'.format(str(key), value))
                self.logger.info("\n------------------------------Validation End------------------------------\n\n")

    def unlearn_erasediff(self): # this is the exact same as train() and unlearn_fix_decoder() but calling the unlearn_erasediff_step() method
        while self.epoch < self.cfg['train']['n_epochs']:
            self.epoch += 1
            if self.cfg['train']['distributed']:
                ''' sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas use a different random ordering for each epoch '''
                self.phase_loader.sampler.set_epoch(self.epoch) 

            train_log = self.unlearn_erasediff_step()

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
    
    def print_network(self, network):
        """ print network structure, only work on GPU 0 """
        # if self.cfg['global_rank'] != 0:
        #     return
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        
        s, n = str(network), sum(map(lambda x: x.numel(), network.parameters()))
        net_struc_str = '{}'.format(network.__class__.__name__)
        self.logger.info('Network structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        self.logger.info(s)

    def save_network(self, network, network_label, only_label=False):
        """ save network structure, only work on GPU 0 """
        # if self.cfg['global_rank'] !=0:
        #     return
        if only_label:
            save_filename = f'{network_label}.pth'
        else:
            save_filename = f'{self.epoch}_{network_label}.pth'

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

        model_name = "{}_{}.pth".format(self.cfg['paths']['resume_state'], network_label)
        model_path = os.path.join(self.cfg['paths']["experiments_root"], self.cfg['paths']["experiment_name"], self.cfg['paths']['checkpoint'], model_name)
        
        if not os.path.exists(model_path):
            self.logger.warning('Pretrained model in [{:s}] does not exist, Skip it'.format(model_path))
            return

        self.logger.info('Loading pretrained model from [{:s}] ...'.format(model_path))
        if isinstance(network, nn.DataParallel) or isinstance(network, nn.parallel.DistributedDataParallel):
            network = network.module
        network.load_state_dict(torch.load(model_path, map_location = lambda storage, loc: self.set_device(storage)), strict=strict)

    def save_training_state(self, label='', only_label=False):
        """ saves training state during training, only work on GPU 0 """
        # if self.cfg['global_rank'] !=0:
        #     return
        assert isinstance(self.optimizers, list) and isinstance(self.schedulers, list), 'optimizers and schedulers must be a list.'
        state = {'epoch': self.epoch, 'iter': self.iter, 'schedulers': [], 'optimizers': [], 'val_iter': self.val_iter}
        for s in self.schedulers:
            state['schedulers'].append(s.state_dict())
        for o in self.optimizers:
            state['optimizers'].append(o.state_dict())
        
        if only_label:
            save_filename = f'{label}.state'
        else:
            save_filename = f'{self.epoch}'
            if label:
                save_filename += f'_{label}'
            save_filename += '.state'
            
        save_path = os.path.join(self.cfg['paths']["experiments_root"], self.cfg['paths']["experiment_name"], self.cfg['paths']['checkpoint'], save_filename)
        torch.save(state, save_path)

    def resume_training(self):
        """ resume the optimizers and schedulers for training, only work when phase is test or resume training enable """
        if self.phase != 'train' or not self.cfg['paths']['resume_state']:
            return
        self.logger.info('Begin loading training states'.format())
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
    
    def load_networks_unlearning(self):
        tmp_teacher_checkpoint = torch.load(self.cfg['paths']['teacher_checkpoint'])
        teacher_checkpoint = {}
        for keyname in tmp_teacher_checkpoint:
            if 'denoise_fn' in keyname:
                teacher_checkpoint[keyname.replace('denoise_fn.', '')] = tmp_teacher_checkpoint[keyname]
        self.netG.denoise_fn.load_state_dict(teacher_checkpoint)
        self.netG.teacher_denoise_fn.load_state_dict(teacher_checkpoint)
        
    def load_networks_teacher(self):
        tmp_teacher_checkpoint = torch.load(self.cfg['paths']['teacher_checkpoint'])
        teacher_checkpoint = {}
        for keyname in tmp_teacher_checkpoint:
            if 'denoise_fn' in keyname:
                teacher_checkpoint[keyname.replace('denoise_fn.', '')] = tmp_teacher_checkpoint[keyname]
        self.netG.teacher_denoise_fn.load_state_dict(teacher_checkpoint)
        
    def reset_everything(self):
        self.epoch = 0
        self.iter = 0
        self.val_iter = 0
        
        self.load_networks_unlearning()
        
        self.optimizers = []
        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **self.cfg['optim']['hyper_params'])
        self.optimizers.append(self.optG)
        

class Trainer(BaseTrainer):
    """
    i2i trainer class.
    Adapted from:
    https://github.com/jpmorganchase/i2i_Palette-Image-to-Image-Diffusion-Models/blob/i2i/models/model.py
    """
    def __init__(self, 
                networks, 
                losses,
                sample_num, 
                task, 
                optimizers,
                schedulers=None, 
                ema_scheduler=None, 
                **kwargs):
        """
        Here, networks can be a single instance of diffusers/TeacherGuidedDiffuser
        or a list of instances as such, if we're running on multiple GPUs.

        :param networks: a single instance of diffusers/TeacherGuidedDiffuser or a list of instances as such, if we're running on multiple GPUs.
        :param losses: 
        :param sample_num: 
        :param task: 
        :param optimizers: 
        :param ema_scheduler: 
        """
        super(Trainer, self).__init__(**kwargs)

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
                
        # if not self.cfg['paths']['resume_state']:
        #     self.load_networks_unlearning()
        # else:
        #     self.load_networks()
            
        if not self.cfg['paths']['resume_state']:
            self.load_networks_unlearning()
        else:
            if self.phase == 'train':
                self.load_networks()
                self.load_networks_teacher()
            else:
                self.load_networks()

        try:
            if optimizers[0].get('adam'):
                self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0]['adam'])
            elif optimizers[0].get('adam_with_epoch_ema'):
                self.optG = AdamWithEpochEMA(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizers[0]['adam_with_epoch_ema'])
            self.optimizers.append(self.optG)
        except KeyError:
            raise Exception(f'Optimizer "{list(optimizers[0].keys())[0]}" not implemented')
        
        num_params = sum(p.numel() for p in self.netG.parameters() if p.requires_grad)
        self.logger.info(f'Total number of parameters: {num_params / 1e6:.2f}M')
        
        if schedulers:
            for scheduler in schedulers:
                if scheduler.get('warmup', None):
                    self.schedulers.append(LambdaLR(self.optG, lr_lambda=get_linear_warmup_lr_lambda(scheduler['warmup'])))
                    
        self.resume_training()
        
        if self.cfg['train'].get('fix_encoder', False):
            for param in self.netG.denoise_fn.input_blocks.parameters():
                param.requires_grad = False 

        if self.cfg['train']['distributed']:
            self.netG.module.set_loss(self.loss_fn)
            self.netG.module.set_new_noise_schedule(phase=self.phase)
        else:
            self.netG.set_loss(self.loss_fn)
            self.netG.set_new_noise_schedule(phase=self.phase)

        '''can rewrite in inherited class for more informations logging '''
        if self.cfg['unlearn'].get('forget_alpha') is not None: # Composite Loss unlearning
            self.train_metrics = LogTracker(*[m.__name__ for m in losses], phase='train')
        elif self.cfg['unlearn'].get('grad_harm_method') is not None: # Gradient Harmonization
            self.train_metrics = LogTracker(*['f1', 'f2'], phase='train')
        elif self.cfg['unlearn'].get('lambda_erasediff') is not None: # EraseDiff
            self.train_metrics = LogTracker(*['f1', 'f2'], phase='train')
        elif self.cfg['unlearn'].get('epsilon_ratio') is not None: # Epsilon unlearning
            self.train_metrics = LogTracker(*['f1_phase_1.1', 'f2_phase_1.1', 'f1_phase_1.2', 'f2_phase_1.2', 'f1_phase_2', 'f2_phase_2'], phase='train')
        self.val_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='val')
        self.test_metrics = LogTracker(*[m.__name__ for m in self.metrics], phase='test')

        self.sample_num = sample_num
        self.task = task
        
    def set_input(self, data, labels=None):
        ''' must use set_device in tensor '''
        self.cond_image = self.set_device(data.get('cond_image'))
        self.gt_image = self.set_device(data.get('gt_image'))
        self.mask = self.set_device(data.get('mask'))
        self.mask_image = data.get('mask_image')
        self.path = data['path']
        self.batch_size = len(data['path'])
    
        # If not None, labels is assumed to contain list of -1 / 1 values
        # corresponding to forget / retain images
        self.labels = labels

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

    def save_current_results(self, test=False):
        if test:
            # mask_ratio = f'{self.phase_loader.dataset.mask_ratio:.2f}/'
            mask_ratio = ''
        else:
            mask_ratio = ''

        ret_path = []
        ret_result = []
        ret_label = []
        for idx in range(self.batch_size):
            ret_path.append('{}GT_{}'.format(mask_ratio, self.path[idx]))
            ret_result.append(self.gt_image[idx].detach().float().cpu())
            ret_label.append(self.labels[idx]) # add corresponding label to each saved result

            ret_path.append('{}Process_{}'.format(mask_ratio, self.path[idx]))
            ret_result.append(self.visuals[idx::self.batch_size].detach().float().cpu())
            ret_label.append(self.labels[idx])

            ret_path.append('{}Out_{}'.format(mask_ratio, self.path[idx]))
            ret_result.append(self.visuals[idx-self.batch_size].detach().float().cpu())
            ret_label.append(self.labels[idx])

            if self.task in ['inpainting','uncropping']:
                ret_path.append('{}Mask_{}'.format(mask_ratio, self.path[idx]))
                ret_result.append(self.mask_image[idx-self.batch_size].detach().float().cpu())
                ret_label.append(self.labels[idx])

        self.results_dict = self.results_dict._replace(
            name=ret_path, 
            result=ret_result,
            label=ret_label
        )
        
        return self.results_dict._asdict()

    def train_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data in track(self.phase_loader, description='Training', total=len(self.phase_loader), console=Console(stderr=True)):
            self.set_input(train_data)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask)

            if self.do_regularization:
                if self.cfg['unlearn'].get('regularization') in ['L2', 'L2_prime']:
                    regularization_value = l2_regularizer(
                        model=self.netG.denoise_fn,
                        model_prev=self.model_prev,
                        lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                        opt_prev=self.opt_prev,
                    )
                    loss = loss + regularization_value

                elif self.cfg['unlearn']['regularization'] in ['L2_salun']:
                    regularization_value = l2_salun_regularizer(
                        model=self.netG.denoise_fn,
                        model_prev=self.model_prev,
                        lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                        opt_prev=self.opt_prev,
                    )
                    loss = loss + regularization_value

            loss.backward()
            self.optG.step()

            self.iter += 1
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.cfg['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.writer.add_scalar(key, value)
                #     self.logger.info('{:5s}: {}\t'.format(str(key), value))
                # for key, value in self.get_current_visuals().items():
                #     self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def unlearn_step(self): # this is the exact same as unlearn_step_fix_decoder() and train_step() but changin one or two parameters in the model call
        self.netG.train()
        self.train_metrics.reset()
        for train_data, labels in track(self.phase_loader, description='Unlearning', total=len(self.phase_loader), console=Console(stderr=True)):
            self.set_input(train_data)
            labels = labels.float()
            labels = self.set_device(labels)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask, labels=labels)
            
            if self.do_regularization:
                if self.cfg['unlearn'].get('regularization') in ['L2', 'L2_prime']:
                    regularization_value = l2_regularizer(
                        model=self.netG.denoise_fn,
                        model_prev=self.model_prev,
                        lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                        opt_prev=self.opt_prev,
                    )
                    loss = loss + regularization_value

                elif self.cfg['unlearn']['regularization'] in ['L2_salun']:
                    regularization_value = l2_salun_regularizer(
                        model=self.netG.denoise_fn,
                        model_prev=self.model_prev,
                        lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                        opt_prev=self.opt_prev,
                    )
                    loss = loss + regularization_value

            loss.backward()
            self.optG.step()

            self.iter += 1
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.cfg['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.writer.add_scalar(key, value)
                #     self.logger.info('{:5s}: {}\t'.format(str(key), value))
                # for key, value in self.get_current_visuals().items():
                #     self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def unlearn_step_fix_decoder(self):
        self.netG.train()
        self.train_metrics.reset()
        for train_data, labels in track(self.phase_loader, description='Unlearning with fixed decoder', total=len(self.phase_loader), console=Console(stderr=True)):
            self.set_input(train_data)
            labels = labels.float()
            labels = self.set_device(labels)
            self.optG.zero_grad()
            loss = self.netG(self.gt_image, self.cond_image, mask=self.mask, labels=labels, fix_decoder=True)
            
            if self.do_regularization:
                if self.cfg['unlearn'].get('regularization') in ['L2', 'L2_prime']:
                    regularization_value = l2_regularizer(
                        model=self.netG.denoise_fn,
                        model_prev=self.model_prev,
                        lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                        opt_prev=self.opt_prev,
                    )
                    loss = loss + regularization_value

                elif self.cfg['unlearn']['regularization'] in ['L2_salun']:
                    regularization_value = l2_salun_regularizer(
                        model=self.netG.denoise_fn,
                        model_prev=self.model_prev,
                        lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                        opt_prev=self.opt_prev,
                    )
                    loss = loss + regularization_value

            loss.backward()
            self.optG.step()

            self.iter += 1
            self.writer.set_iter(self.epoch, self.iter, phase='train')
            self.train_metrics.update(self.loss_fn.__name__, loss.item())
            if self.iter % self.cfg['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.writer.add_scalar(key, value)
                #     self.logger.info('{:5s}: {}\t'.format(str(key), value))
                # for key, value in self.get_current_visuals().items():
                #     self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def unlearn_grad_harm_step(self):
        self.netG.train()
        self.train_metrics.reset()
        for i, (train_data, labels) in track(enumerate(self.phase_loader), description='Unlearning grad harm', total=len(self.phase_loader), console=Console(stderr=True)):
            self.set_input(train_data)
            labels = labels.float()
            labels = self.set_device(labels)
            self.optG.zero_grad()
            
            f1, f2 = self.netG(self.gt_image, self.cond_image, mask=self.mask, labels=labels, fix_decoder=self.cfg['train']['fix_decoder'])
            
            if self.do_regularization:
                if self.cfg['unlearn'].get('regularization') in ['L2', 'L2_prime']:
                    regularization_value = l2_regularizer(
                        model=self.netG.denoise_fn,
                        model_prev=self.model_prev,
                        lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                        opt_prev=self.opt_prev,
                    )
                    f1 = f1 + regularization_value
                    f2 = f2 + regularization_value

                elif self.cfg['unlearn']['regularization'] in ['L2_salun']:
                    regularization_value = l2_salun_regularizer(
                        model=self.netG.denoise_fn,
                        model_prev=self.model_prev,
                        lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                        opt_prev=self.opt_prev,
                    )
                    f1 = f1 + regularization_value
                    f2 = f2 + regularization_value

            trainable_params = [p for p in self.netG.denoise_fn.parameters() if p.requires_grad]
                
            f1_grad = torch.autograd.grad(f1, trainable_params, retain_graph=True, allow_unused=True)
            f2_grad = torch.autograd.grad(f2, trainable_params, allow_unused=True)
            
            if self.cfg['unlearn']['grad_harm_method'] == 'simple':
                g_t = self.netG.compute_g_t_harmonizer(f1_grad, f2_grad)
            elif self.cfg['unlearn']['grad_harm_method'] == 'epoch_curriculum':
                g_t = self.netG.compute_g_t_harmonizer_curriculum(f1_grad, f2_grad, curriculum_condition=(i < len(self.phase_loader) // 2))
            elif self.cfg['unlearn']['grad_harm_method'] == 'ripcgrad':
                g_t = self.netG.compute_g_t_harmonizer_ripcgrad(f1_grad, f2_grad, alpha=self.cfg['unlearn']['ripcgrad_params']['alpha'])
            
            for param, grad in zip(trainable_params, g_t):
                    param.grad = grad
                
            self.optG.step()

            self.iter += 1
            self.writer.set_iter(self.epoch, self.iter, phase='train')
                
            self.train_metrics.update(f'f1', f1.item())
            self.train_metrics.update(f'f2', f2.item())
            if self.iter % self.cfg['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    self.writer.add_scalar(key, value)
                #     self.logger.info('{:5s}: {}\t'.format(str(key), value))
                # for key, value in self.get_current_visuals().items():
                #     self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def unlearn_epsilon_step(self, epsilon_ratio=0, phase=1.1, eps_min=None, eps_max=None):
        self.netG.train()
        self.train_metrics.reset()
        for i, (train_data, labels) in track(enumerate(self.phase_loader), description='Unlearning epsilon', total=len(self.phase_loader), console=Console(stderr=True)):
            self.set_input(train_data)
            labels = labels.float()
            labels = self.set_device(labels)
            self.optG.zero_grad()
            
            f1, f2 = self.netG(self.gt_image, self.cond_image, mask=self.mask, labels=labels, fix_decoder=self.cfg['train']['fix_decoder'])
            
            if self.do_regularization:
                if self.cfg['unlearn'].get('regularization') in ['L2', 'L2_prime']:
                    regularization_value = l2_regularizer(
                        model=self.netG.denoise_fn,
                        model_prev=self.model_prev,
                        lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                        opt_prev=self.opt_prev,
                    )
                    f1 = f1 + regularization_value
                    f2 = f2 + regularization_value

                elif self.cfg['unlearn']['regularization'] in ['L2_salun']:
                    regularization_value = l2_salun_regularizer(
                        model=self.netG.denoise_fn,
                        model_prev=self.model_prev,
                        lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                        opt_prev=self.opt_prev,
                    )
                    f1 = f1 + regularization_value
                    f2 = f2 + regularization_value

            if phase==1.2:
                f1, f2 = f2, f1
                
            trainable_params = [p for p in self.netG.denoise_fn.parameters() if p.requires_grad]
                
            f1_grad = torch.autograd.grad(f1, trainable_params, retain_graph=True, allow_unused=True)
            f2_grad = torch.autograd.grad(f2, trainable_params, allow_unused=True)
            
            g_t = self.netG.compute_g_t(f1, f2, f1_grad, f2_grad, epsilon_ratio, phase, eps_min, eps_max)
            
            for param, grad in zip(trainable_params, g_t):
                    param.grad = grad
                
            self.optG.step()

            self.iter += 1
            self.writer.set_iter(self.epoch, self.iter, phase='train', epsilon_phase=phase)
            
            if phase==1.2:
                f1, f2 = f2, f1 # to keep the order of f1 and f2 consistent with the logging
                
            self.train_metrics.update(f'f1_phase_{phase}', f1.item())
            self.train_metrics.update(f'f2_phase_{phase}', f2.item())
            if self.iter % self.cfg['train']['log_iter'] == 0:
                for key, value in self.train_metrics.result().items():
                    if str(phase) in key:
                        self.writer.add_scalar(key, value)
                #     self.logger.info('{:5s}: {}\t'.format(str(key), value))
                # for key, value in self.get_current_visuals().items():
                #     self.writer.add_images(key, value)
            if self.ema_scheduler is not None:
                if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)

        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()
    
    def unlearn_epsilon_forward_step(self):
        self.netG.eval()
        
        f1_mean, f2_mean = 0, 0
        
        with torch.no_grad():
            for i, (train_data, labels) in track(enumerate(self.phase_loader), description='Unlearning epsilon forward', total=len(self.phase_loader), console=Console(stderr=True)):
                self.set_input(train_data)
                labels = labels.float()
                labels = self.set_device(labels)
                f1, f2 = self.netG(self.gt_image, self.cond_image, mask=self.mask, labels=labels)
                
                f1_mean += f1
                f2_mean += f2
                
                if i == 1000:
                    break
        
        f1_mean /= 1000
        f2_mean /= 1000
        
        return f1_mean, f2_mean
    
    def unlearn_erasediff_step(self):
        self.netG.train()
        self.train_metrics.reset()
        
        for s_step in range(self.cfg['unlearn']['s_steps']):
            param_i = get_param(self.netG.denoise_fn)
            
            for j in range(self.cfg['unlearn']['K_steps']):
                forget_losses = AverageMeter()
                # First we do K steps only for forget data
                for i, (train_data, labels) in track(enumerate(self.phase_loader), description='Unlearning EraseDiff', total=len(self.phase_loader), console=Console(stderr=True)):
                    self.set_input(train_data)
                    labels = labels.float()
                    labels = self.set_device(labels)
                    
                    if labels.sum() == len(labels): # if all labels are 1, i.e. all data to retain
                        continue
                    
                    self.optG.zero_grad()
                    
                    f1, _ = self.netG(self.gt_image, self.cond_image, mask=self.mask, labels=labels, fix_decoder=self.cfg['train']['fix_decoder'])
                    
                    if self.do_regularization:
                        if self.cfg['unlearn'].get('regularization') in ['L2', 'L2_prime']:
                            regularization_value = l2_regularizer(
                                model=self.netG.denoise_fn,
                                model_prev=self.model_prev,
                                lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                                opt_prev=self.opt_prev,
                            )
                            f1 = f1 + regularization_value

                        elif self.cfg['unlearn']['regularization'] in ['L2_salun']:
                            regularization_value = l2_salun_regularizer(
                                model=self.netG.denoise_fn,
                                model_prev=self.model_prev,
                                lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                                opt_prev=self.opt_prev,
                            )
                            f1 = f1 + regularization_value

                    f1.backward()
                    self.optG.step()

                    self.iter += 1
                    self.writer.set_iter(self.epoch, self.iter, phase='train')
                    forget_losses.update(f1, self.batch_size)
                    
                    self.train_metrics.update(f'f1', f1.item())
                    if self.iter % self.cfg['train']['log_iter'] == 0:
                        for key, value in self.train_metrics.result().items():
                            self.writer.add_scalar(key, value)
                        #     self.logger.info('{:5s}: {}\t'.format(str(key), value))
                        # for key, value in self.get_current_visuals().items():
                        #     self.writer.add_images(key, value)
                    if self.ema_scheduler is not None:
                        if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                            self.EMA.update_model_average(self.netG_EMA, self.netG)
            
            self.netG.denoise_fn = set_param(self.netG.denoise_fn, param_i)         
            # Then we optimize for all data
            for i, (train_data, labels) in track(enumerate(self.phase_loader), description='Unlearning EraseDiff', total=len(self.phase_loader), console=Console(stderr=True)):
                self.set_input(train_data)
                labels = labels.float()
                labels = self.set_device(labels)
                
                if labels.sum() == len(labels): # if all labels are 1, i.e. all data to retain
                    continue
                
                f1, f2 = self.netG(self.gt_image, self.cond_image, mask=self.mask, labels=labels, fix_decoder=self.cfg['train']['fix_decoder'])
                g = f1 - forget_losses.avg.detach()
                
                if self.cfg['unlearn']['lambda_erasediff'] < 0:
                    self.optG.zero_grad()
                    
                    trainable_params = [p for p in self.netG.denoise_fn.parameters() if p.requires_grad]
                    g_grads = torch.autograd.grad(g, trainable_params, retain_graph=True, allow_unused=True)
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    g_grad_vector = torch.stack(
                        list(
                            map(
                                lambda g_grad: torch.cat(list(map(lambda grad: grad.contiguous().view(-1), g_grad))), [g_grads])
                            )
                        )
                    torch.cuda.empty_cache()
                    gc.collect()

                    g_grad_norm = torch.linalg.norm(g_grad_vector, 2)
                    torch.cuda.empty_cache()
                    gc.collect()

                    if g_grad_norm == 0:
                        lambda_erasediff = 0.
                    else:
                        self.optG.zero_grad()
                        f2_grads = torch.autograd.grad(f2, trainable_params, retain_graph=True)
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        f2_grads_vector = torch.stack(
                            list(
                                map(
                                    lambda f2_grad: torch.cat(list(map(lambda grad: grad.contiguous().view(-1), f2_grad))), [f2_grads])
                                )
                            )
                        f2_grad_norm = torch.linalg.norm(f2_grads_vector, 2)
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        inner_product = torch.sum(f2_grad_norm * g_grad_vector)
                        tmp = inner_product / ( f2_grad_norm * g_grad_norm + 1e-8) # different from the paper

                        lambda_erasediff = (self.cfg['unlearn']['eta_erasediff'] - tmp).detach() \
                            if self.cfg['unlearn']['eta_erasediff'] > tmp else 0.
                        torch.cuda.empty_cache()
                        gc.collect()
                        # Kinda weird that they use eta as a learning rate for optimizing the model in the paper
                        # but in the code the optimizer has a different internal learning rate and eta is just for calculating lambda
                        del f2_grads, f2_grads_vector, tmp
                    del g_grads, g_grad_vector, g_grad_norm

                else:
                    lambda_erasediff = self.cfg['unlearn']['lambda_erasediff']
                    
                loss = f2 + lambda_erasediff * g
                
                if self.do_regularization:
                    if self.cfg['unlearn'].get('regularization') in ['L2', 'L2_prime']:
                        regularization_value = l2_regularizer(
                            model=self.netG.denoise_fn,
                            model_prev=self.model_prev,
                            lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                            opt_prev=self.opt_prev,
                        )
                        loss = loss + regularization_value

                    elif self.cfg['unlearn']['regularization'] in ['L2_salun']:
                        regularization_value = l2_salun_regularizer(
                            model=self.netG.denoise_fn,
                            model_prev=self.model_prev,
                            lambda_reg=self.cfg['unlearn']['regularization_params']['lambda'],
                            opt_prev=self.opt_prev,
                        )
                        loss = loss + regularization_value
                
                self.optG.zero_grad()
                loss.backward()
                self.optG.step()

                self.iter += 1
                self.writer.set_iter(self.epoch, self.iter, phase='train')
                    
                self.train_metrics.update(f'f1', f1.item())
                if self.iter % self.cfg['train']['log_iter'] == 0:
                    for key, value in self.train_metrics.result().items():
                        self.writer.add_scalar(key, value)
                    #     self.logger.info('{:5s}: {}\t'.format(str(key), value))
                    # for key, value in self.get_current_visuals().items():
                    #     self.writer.add_images(key, value)
                if self.ema_scheduler is not None:
                    if self.iter > self.ema_scheduler['ema_start'] and self.iter % self.ema_scheduler['ema_iter'] == 0:
                        self.EMA.update_model_average(self.netG_EMA, self.netG)
                            
        for scheduler in self.schedulers:
            scheduler.step()
        return self.train_metrics.result()

    def val_step(self, phase=''):
        self.netG.eval()
        self.val_metrics.reset()
        with torch.no_grad():
            for batch_idx, (val_data, labels) in track(enumerate(self.val_loader), description='Validation', total=len(self.val_loader), console=Console(stderr=True)):
                self.set_input(val_data, labels) 
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
                self.writer.set_iter(self.epoch, self.val_iter, phase='val', epsilon_phase=phase)

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
            with Progress() as progress_bar:
                outer_task = progress_bar.add_task('Testing', total=len(self.phase_loader), console=Console(stderr=True))
                for phase_data, labels in self.phase_loader:
                    self.set_input(phase_data)
                    if self.cfg['train']['distributed']:
                        if self.task in ['inpainting','uncropping']:
                            self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                                y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num, progress_bar=progress_bar)
                        else:
                            self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num, progress_bar=progress_bar)
                    else:
                        if self.task in ['inpainting','uncropping']:
                            self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                                y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num, progress_bar=progress_bar)
                        else:
                            self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num, progress_bar=progress_bar)
                            
                    self.iter += 1
                    self.writer.set_iter(self.epoch, self.iter, phase='test')
                    for met in self.metrics:
                        key = met.__name__
                        value = met(self.gt_image, self.output)
                        self.test_metrics.update(key, value)
                        self.writer.add_scalar(key, value)
                    for key, value in self.get_current_visuals(phase='test').items():
                        self.writer.add_images(key, value)
                    self.writer.save_images(self.save_current_results(test=True), save_dir=self.cfg["paths"].get('img_save_dir', None))
                    progress_bar.update(outer_task, advance=1)
        
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
            pred_arr_gt_retain = []
            pred_arr_out_retain = []
            
            pred_arr_gt_forget = []
            pred_arr_out_forget = []
                
        if self.cfg['test']['metrics'].get('is', False):
            is_splits = self.cfg['test']['metrics']['is_splits']
            is_kwargs = {'cuda': not self.cfg['train'].get('cpu', False), 'verbose': False, 'splits': is_splits}
            feature_extractor = "inception-v3-compat"
            feature_layer_isc = resolve_feature_layer_for_metric("isc")
            feature_layers = [feature_layer_isc]
            is_model = create_feature_extractor(feature_extractor, list(feature_layers), **is_kwargs)
            out_gt_retain, out_out_retain = None, None
            out_gt_forget, out_out_forget = None, None
            pass
        
        with torch.no_grad():
            with Progress() as progress_bar:
                outer_task = progress_bar.add_task('Testing', total=len(self.phase_loader), console=Console(stderr=True))
                for i, (phase_data, labels) in enumerate(self.phase_loader):
                    self.set_input(phase_data, labels=labels)
                    if self.cfg['train']['distributed']:
                        if self.task in ['inpainting','uncropping']:
                            self.output, self.visuals = self.netG.module.restoration(self.cond_image, y_t=self.cond_image, 
                                y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num, progress_bar=progress_bar)
                        else:
                            self.output, self.visuals = self.netG.module.restoration(self.cond_image, sample_num=self.sample_num, progress_bar=progress_bar)
                    else:
                        if self.task in ['inpainting','uncropping']:
                            self.output, self.visuals = self.netG.restoration(self.cond_image, y_t=self.cond_image, 
                                y_0=self.gt_image, mask=self.mask, sample_num=self.sample_num, progress_bar=progress_bar)
                        else:
                            self.output, self.visuals = self.netG.restoration(self.cond_image, sample_num=self.sample_num, progress_bar=progress_bar)
                            
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
                        pred_gt_retain = get_activations_for_batch(self.gt_image[labels==1], fid_model)
                        pred_out_retain = get_activations_for_batch(self.output[labels==1], fid_model)
                        
                        pred_arr_gt_retain.append(pred_gt_retain)
                        pred_arr_out_retain.append(pred_out_retain)
                        
                        pred_gt_forget = get_activations_for_batch(self.gt_image[labels==-1], fid_model)
                        pred_out_forget= get_activations_for_batch(self.output[labels==-1], fid_model)
                        
                        pred_arr_gt_forget.append(pred_gt_forget)
                        pred_arr_out_forget.append(pred_out_forget)
                        
                    if self.cfg['test']['metrics'].get('is', False):
                        self.gt_image = self.gt_image * 255
                        self.gt_image = self.gt_image.to(torch.uint8)
                        
                        self.output = self.output * 255
                        self.output = self.output.to(torch.uint8)
                        
                        gt_retain = self.gt_image[labels==1]
                        output_retain = self.output[labels==1]

                        out_gt_featuresdict_retain = get_featuresdict_for_batch(gt_retain, is_model)
                        if out_gt_retain is None:
                            out_gt_retain = out_gt_featuresdict_retain
                        else:
                            out_gt_retain = {k: out_gt_retain[k] + out_gt_featuresdict_retain[k] for k in out_gt_retain.keys()}
                            
                        out_out_featuresdict_retain = get_featuresdict_for_batch(output_retain, is_model)
                        if out_out_retain is None:
                            out_out_retain = out_out_featuresdict_retain
                        else:
                            out_out_retain = {k: out_out_retain[k] + out_out_featuresdict_retain[k] for k in out_out_retain.keys()}
                            
                        gt_forget = self.gt_image[labels==-1]
                        output_forget = self.output[labels==-1]

                        out_gt_featuresdict_forget= get_featuresdict_for_batch(gt_forget, is_model)
                        if out_gt_forget is None:
                            out_gt_forget= out_gt_featuresdict_forget
                        else:
                            out_gt_forget= {k: out_gt_forget[k] + out_gt_featuresdict_forget[k] for k in out_gt_forget.keys()}
                            
                        out_out_featuresdict_forget = get_featuresdict_for_batch(output_forget, is_model)
                        if out_out_forget is None:
                            out_out_forget = out_out_featuresdict_forget
                        else:
                            out_out_forget= {k: out_out_forget[k] + out_out_featuresdict_forget[k] for k in out_out_forget.keys()}
                            
                    if self.gt_image.dtype == torch.uint8:
                        self.gt_image = self.gt_image.float() / 255
                        self.output = self.output.float() / 255
                    
                    if self.gt_image.min() == 0:
                        self.gt_image = self.gt_image * 2 - 1
                        self.output = self.output * 2 - 1
                            
                    for key, value in self.get_current_visuals(phase='test').items():
                        self.writer.add_images(key, value)
                    self.writer.save_images(self.save_current_results(test=True), save_dir=self.cfg["paths"].get('img_save_dir', None))
                    progress_bar.update(outer_task, advance=1)
        
        test_log = self.test_metrics.result()
        ''' save logged informations into log dict ''' 
        test_log.update({'epoch': self.epoch, 'iters': self.iter})

        ''' print logged informations to the screen and tensorboard ''' 
        for key, value in test_log.items():
            self.logger.info('{:5s}: {}\t'.format(str(key), value))
            
        if self.cfg['test']['metrics'].get('fid', False):
            pred_arr_gt_retain = np.vstack(pred_arr_gt_retain)
            pred_arr_out_retain = np.vstack(pred_arr_out_retain)
            
            mu_gt_retain = np.mean(pred_arr_gt_retain, axis=0)
            sigma_gt_retain = np.cov(pred_arr_gt_retain, rowvar=False)
            
            mu_out_retain = np.mean(pred_arr_out_retain, axis=0)
            sigma_out_retain = np.cov(pred_arr_out_retain, rowvar=False)
            
            fid_value_retain = calculate_frechet_distance(mu_gt_retain, sigma_gt_retain, mu_out_retain, sigma_out_retain)
            metrics_dict['fid_retain'] = f"{fid_value_retain:.3f}"
            
            self.logger.info(f'FID Retain: {fid_value_retain:.3f}')
            
            pred_arr_gt_forget = np.vstack(pred_arr_gt_forget)
            pred_arr_out_forget = np.vstack(pred_arr_out_forget)
            
            mu_gt_forget = np.mean(pred_arr_gt_forget, axis=0)
            sigma_gt_forget = np.cov(pred_arr_gt_forget, rowvar=False)
            
            mu_out_forget = np.mean(pred_arr_out_forget, axis=0)
            sigma_out_forget = np.cov(pred_arr_out_forget, rowvar=False)
            
            fid_value_forget = calculate_frechet_distance(mu_gt_forget, sigma_gt_forget, mu_out_forget, sigma_out_forget)
            metrics_dict['fid_forget'] = f"{fid_value_forget:.3f}"
            
            self.logger.info(f'FID Forget: {fid_value_forget:.3f}')

                
        if self.cfg['test']['metrics'].get('is', False):
            out_gt_retain = {k: torch.cat(v, dim=0) for k, v in out_gt_retain.items()}
            out_out_retain = {k: torch.cat(v, dim=0) for k, v in out_out_retain.items()}  
            is_gt_retain = isc_featuresdict_to_metric(out_gt_retain, feature_layer_isc, **is_kwargs)
            is_out_retain = isc_featuresdict_to_metric(out_out_retain, feature_layer_isc, **is_kwargs)
            
            metrics_dict['is_gt_retain'] = f"{is_gt_retain['inception_score_mean']:.3f} ± {is_gt_retain['inception_score_std']:.3g}"
            metrics_dict['is_output_retain'] = f"{is_out_retain['inception_score_mean']:.3f} ± {is_out_retain['inception_score_std']:.3g}"
            
            self.logger.info(f'IS (GT) Retain: {is_gt_retain["inception_score_mean"]:.3f} ± {is_gt_retain["inception_score_std"]:.3g}')
            self.logger.info(f'IS (Output) Retain: {is_out_retain["inception_score_mean"]:.3f} ± {is_out_retain["inception_score_std"]:.3g}')
            
            out_gt_forget = {k: torch.cat(v, dim=0) for k, v in out_gt_forget.items()}
            out_out_forget = {k: torch.cat(v, dim=0) for k, v in out_out_forget.items()}  
            is_gt_forget = isc_featuresdict_to_metric(out_gt_forget, feature_layer_isc, **is_kwargs)
            is_out_forget = isc_featuresdict_to_metric(out_out_forget, feature_layer_isc, **is_kwargs)
            
            metrics_dict['is_gt_forget'] = f"{is_gt_forget['inception_score_mean']:.3f} ± {is_gt_forget['inception_score_std']:.3g}"
            metrics_dict['is_output_forget'] = f"{is_out_forget['inception_score_mean']:.3f} ± {is_out_forget['inception_score_std']:.3g}"
            
            self.logger.info(f'IS (GT) Forget: {is_gt_forget["inception_score_mean"]:.3f} ± {is_gt_forget["inception_score_std"]:.3g}')
            self.logger.info(f'IS (Output) Forget: {is_out_forget["inception_score_mean"]:.3f} ± {is_out_forget["inception_score_std"]:.3g}')

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
        # Also saving the new diffusion model separately to be used for inference on its own
        self.save_network(network=self.netG.denoise_fn, network_label=netG_label+'_denoise_fn')
        if self.ema_scheduler is not None:
            self.save_network(network=self.netG_EMA, network_label=netG_label+'_ema')
        self.save_training_state()