"""
Initialize and start un-learning loop
"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import os
import torch.nn.functional as F

from utils.logging import VisualWriter, InfoLogger, DiscordLogger
from utils.utils import set_seed, check_job_restart
from datasets import create_datasets_unlearning
from metrics.losses import mae

from diffusers import TeacherGuidedDiffuser, TeacherGuidedDiffuserEpsilon
from unlearners import Trainer
from metrics.losses import weighted_mse_loss, mse_loss, retain_forget_loss

torch.backends.cudnn.enabled = True


def main(cfg, phase):
    assert not (cfg["train"].get("fix_decoder", False) and cfg["train"].get("fix_encoder", False)), "Cannot fix both decoder and encoder at the same time"
    set_seed(cfg["train"]["seed"])
    
    if not os.path.exists(os.path.join(cfg['paths']['experiments_root'], cfg['paths']['experiment_name'])):
        os.makedirs(os.path.join(cfg['paths']['experiments_root'], cfg['paths']['experiment_name']))
        
    with open(os.path.join(cfg['paths']['experiments_root'], cfg['paths']['experiment_name'], 'config.yaml'), 'w') as f:
        yaml.dump(cfg, f)
    
    phase_logger = InfoLogger(cfg, phase)
    phase_writer = VisualWriter(cfg)
    phase_logger.info(f"Created log file in {os.path.join(cfg['paths']['experiments_root'], cfg['paths']['experiment_name'], f'{phase}.log')}")
    
    if phase == 'train':
        cfg = check_job_restart(cfg)
    
    phase_dataset, val_dataset = create_datasets_unlearning(cfg, phase)
    
    data_sampler = None
    if cfg['train']['distributed']:
        data_sampler = DistributedSampler(phase_dataset, shuffle=True)
    
    phase_loader = DataLoader(
        phase_dataset,
        batch_size=cfg[phase]['batch_size'],
        shuffle=phase=='train', # (data_sampler is None),
        num_workers=cfg[phase]['num_workers'],
        pin_memory=cfg[phase]['pin_memory'],
        sampler=data_sampler,
        drop_last=cfg[phase]['drop_last']
    )
    
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg['train']['batch_size'],
            shuffle=False,
            num_workers=cfg['train']['num_workers'],
            pin_memory=cfg['train']['pin_memory'],
            drop_last=cfg['train']['drop_last']
        )
    else:
        val_loader = None
    
    phase_logger.info(f"Created dataloaders with batch size {cfg[phase]['batch_size']}")
    
    if cfg['unlearn'].get('forget_alpha') is not None: # Composite Loss unlearning
        diffuser = TeacherGuidedDiffuser(
            unet=cfg['model'],
            beta_schedule=cfg[phase]['diffusion'],
            init_type=cfg['train']['weight_initialization'],
            gain=cfg['train']['gain'],
            kwargs=cfg['unlearn']
        )
        losses = [weighted_mse_loss]

    else: # Epsilon unlearning and Gradient Harmonization, EraseDiff
        diffuser = TeacherGuidedDiffuserEpsilon(
            unet=cfg['model'],
            beta_schedule=cfg[phase]['diffusion'],
            init_type=cfg['train']['weight_initialization'],
            gain=cfg['train']['gain'],
            kwargs=cfg['unlearn']
        )
        losses = [mse_loss]

    metrics = [mae]
    
    if cfg['unlearn'].get('forget_alpha') is None: # Epsilon unlearning, Gradient Harmonization, EraseDiff
        losses = [retain_forget_loss]
    else: # Composite Loss unlearning
        losses = [weighted_mse_loss]
    
    model = Trainer(
        networks=[diffuser],
        losses=losses,
        metrics=metrics,
        sample_num=8,
        task='inpainting',
        optimizers=[{cfg['optim']['type']: cfg['optim']['hyper_params']}],
        schedulers=[{'warmup': cfg['train']['num_warmup_steps']}],
        ema_scheduler=cfg['train']['ema'],
        cfg=cfg,
        phase=phase,
        phase_loader=phase_loader,
        val_loader=val_loader,
        logger=phase_logger,
        writer=phase_writer
    )
    
    phase_logger.info(f"Created model")
    
    if phase == 'train':
        model.do_regularization = False
        if cfg['unlearn'].get('forget_alpha') is not None: # Composite Loss unlearning
            model.unlearn_fix_decoder()
        elif cfg['unlearn'].get('grad_harm_method') is not None: # Gradient Harmonization
            model.unlearn_grad_harm()
        elif cfg['unlearn'].get('epsilon_ratio') is not None: # Epsilon unlearning
            model.unlearn_epsilon()
        elif cfg['unlearn'].get('lambda_erasediff') is not None: # EraseDiff
            model.unlearn_erasediff()
    elif phase == 'test':
        if cfg['test'].get('metrics'):
            model.test_with_metrics()
        else:
            model.test()
    phase_writer.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='path to config file')
    
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help='train or test')
    
    args = parser.parse_args()
    
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.CLoader)
    phase = args.phase
    
    discord_webhook = cfg.get('discord_webhook', {}).get('url')
    if discord_webhook:
        discord_logger = DiscordLogger(discord_webhook)
        discord_logger.send_message(f"Starting {phase} phase with config: {args.config}")

        try:
            main(cfg, phase)
            discord_logger.send_message(f"Completed {phase} phase successfully", type='success')
        except Exception as e:
            discord_logger.send_message(f"Error during {phase} phase: {str(e)}", type='error')
            raise e
    else:
        main(cfg, phase)