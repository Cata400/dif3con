"""
Initialize and start un-learning loop
"""
import argparse
import yaml
import torch
import os
from copy import deepcopy

from utils.logging import VisualWriter, InfoLogger, DiscordLogger
from utils.utils import set_seed, check_job_restart, linear_decay
from datasets import create_datasets_unlearning, create_dataloaders
from metrics.losses import mae

from diffusers import TeacherGuidedDiffuser, TeacherGuidedDiffuserEpsilon
from unlearners import Trainer
from metrics.losses import weighted_mse_loss, mse_loss, retain_forget_loss
from models.optimizers import AdamWithEpochEMA

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
    
    skip_curriculum_steps = 0
    prev_max_forget_classes = 0
    if phase == 'train' and cfg['paths'].get('resume_state') is None:
        max_forget_classes = cfg['curriculum'].get('forget_classes_to_unlearn_step')
    elif phase != 'test':
        max_forget_classes = int(cfg['paths']['resume_state'].split('_')[0]) + cfg['curriculum'].get('forget_classes_to_unlearn_step')
        prev_max_forget_classes = max_forget_classes - cfg['curriculum'].get('forget_classes_to_unlearn_step')
        skip_curriculum_steps = prev_max_forget_classes // cfg['curriculum'].get('forget_classes_to_unlearn_step')
    elif phase == 'test':
        max_forget_classes = int(cfg['paths']['resume_state'].split('_')[0])
        
    phase_dataset, val_dataset = create_datasets_unlearning(
        cfg, phase, max_forget_classes=max_forget_classes,
        prev_max_forget_classes=prev_max_forget_classes,
        )
    
    phase_loader, val_loader = create_dataloaders(cfg, phase, phase_dataset, val_dataset)
    
    phase_logger.info(f"Created dataloaders with batch size {cfg[phase]['batch_size']}, starting with {max_forget_classes} total forget classes")
    
    if cfg['unlearn'].get('forget_alpha') is not None: # Composite Loss unlearning
        diffuser = TeacherGuidedDiffuser(
            unet=cfg['model'],
            beta_schedule=cfg[phase]['diffusion'],
            init_type=cfg['train']['weight_initialization'],
            gain=cfg['train']['gain'],
            kwargs=cfg['unlearn']
        )
        losses = [weighted_mse_loss]

    else: # Epsilon unlearning, Gradient Harmonization, EraseDiff
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
        if cfg['unlearn'].get('regularization'):
            model.do_regularization = True
        else:
            model.do_regularization = False

        if cfg['unlearn'].get('regularization') in ['L2', 'L2_prime', 'L2_salun']:
            if not cfg['paths']['resume_state']:
                model.__setattr__('model_prev', model.netG.teacher_denoise_fn)
            else:
                model.__setattr__('model_prev', deepcopy(model.netG.denoise_fn))
            model.model_prev.eval()
            model.model_prev.requires_grad_(False)
            model.__setattr__('opt_prev', None)

            if cfg['unlearn']['regularization'] in ['L2_prime', 'L2_salun'] and cfg['paths']['resume_state']:
                model.__setattr__('opt_prev', deepcopy(model.optG))
                if isinstance(model.optG, AdamWithEpochEMA):
                    model.optG.reset_epoch_moments()
                    phase_logger.info(f"Reset optimizer epoch moments for AdamWithEpochEMA")

        for curriculum_step in range(cfg['curriculum']['curriculum_steps']):
            if curriculum_step < skip_curriculum_steps: # This is for the case where the job restarts and we want to skip already trained curriculum steps
                phase_logger.info(f"Skipping curriculum step #{curriculum_step}, already trained")
                continue
            phase_logger.info(f"Starting curriculum step #{curriculum_step}, {max_forget_classes} total forget classes")
            
            # Reset model attributes
            model.epoch = 0
            model.iter = 0
            model.val_iter = 0 

            if cfg["unlearn"].get("regularization") and curriculum_step < cfg["unlearn"].get("regularization_params", {}).get("regularization_from_curriculum_step", 0):
                model.do_regularization = False
                phase_logger.info(f"Regularization is OFF for curriculum step #{curriculum_step}")
            elif cfg["unlearn"].get("regularization") and curriculum_step >= cfg["unlearn"].get("regularization_params", {}).get("regularization_from_curriculum_step", 0):
                model.do_regularization = True
                phase_logger.info(f"Regularization is ON for curriculum step #{curriculum_step}")
                
            if cfg['unlearn'].get('regularization') and cfg['unlearn'].get('regularization_params', {}).get('lambda_decay'):
                if cfg['unlearn']['regularization_params']['lambda_decay'].get('decay') == "linear":
                    lambdas = linear_decay(
                        start=cfg['unlearn']['regularization_params']['lambda_decay']['start'],
                        end=cfg['unlearn']['regularization_params']['lambda_decay']['end'],
                        n_steps=cfg['curriculum']['curriculum_steps']
                    )
                cfg['unlearn']['regularization_params']['lambda'] = lambdas[curriculum_step]
                                
            # Train the model as normal for that curriculum step
            if cfg['unlearn'].get('forget_alpha') is not None: # Composite Loss unlearning
                model.unlearn_fix_decoder()
                
            elif cfg['unlearn'].get('grad_harm_method') is not None: # Gradient Harmonization
                if cfg['unlearn']['grad_harm_method'] == 'ripcgrad' and cfg['unlearn'].get('ripcgrad_params', {}).get('alpha_decay'):
                    if cfg['unlearn']['ripcgrad_params']['alpha_decay'].get('decay') == "linear":
                        alphas = linear_decay(
                            start=cfg['unlearn']['ripcgrad_params']['alpha_decay']['start'],
                            end=cfg['unlearn']['ripcgrad_params']['alpha_decay']['end'],
                            n_steps=cfg['curriculum']['curriculum_steps']
                        )
                    cfg['unlearn']['ripcgrad_params']['alpha'] = alphas[curriculum_step]
                    
                model.unlearn_grad_harm()
                
            elif cfg['unlearn'].get('epsilon_ratio') is not None: # Epsilon unlearning
                model.unlearn_epsilon()
                
            elif cfg['unlearn'].get('lambda_erasediff') is not None: # EraseDiff
                model.unlearn_erasediff()
                
            # Save the model checkpoint
            if cfg['train']['distributed']:
                netG_label = model.netG.module.__class__.__name__
            else:
                netG_label = model.netG.__class__.__name__
            model.save_network(network=model.netG, network_label=f'{max_forget_classes}_forget_classes_{netG_label}', only_label=True)
            model.save_training_state(label=f'{max_forget_classes}_forget_classes', only_label=True)

            phase_logger.info(f"Saved model checkpoint for #{curriculum_step} curriculum step, {max_forget_classes} total forget classes")

            # Add new forget classes for the next curriculum step
            prev_max_forget_classes = max_forget_classes
            max_forget_classes += cfg['curriculum'].get('forget_classes_to_unlearn_step')
            phase_dataset, val_dataset = create_datasets_unlearning(
                cfg, phase, max_forget_classes=max_forget_classes,
                prev_max_forget_classes=prev_max_forget_classes,
                )
            phase_loader, val_loader = create_dataloaders(cfg, phase, phase_dataset, val_dataset)
            
            model.phase_loader = phase_loader
            model.val_loader = val_loader

            if cfg['unlearn'].get('regularization') in ['L2', 'L2_prime', 'L2_salun']:
                model.__setattr__('model_prev', deepcopy(model.netG.denoise_fn))
                model.model_prev.eval()
                model.model_prev.requires_grad_(False)
                if cfg['unlearn']['regularization'] in ['L2_prime', 'L2_salun']:
                    model.__setattr__('opt_prev', deepcopy(model.optG))
                    if isinstance(model.optG, AdamWithEpochEMA):
                        model.optG.reset_epoch_moments()
                        phase_logger.info(f"Reset optimizer epoch moments for AdamWithEpochEMA")
                phase_logger.info(f"Updated previous task model for L2 regularization")

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