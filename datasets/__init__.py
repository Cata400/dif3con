import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .datasets import (
    ImagesPathDataset,
    InpaintDataset,
    UncroppingDataset,
    ColorizationDataset,
    ForgetInpaintDataset,
    ForgetUncroppingDataset,
    ForgetColorizationDataset
) 
from .transforms import TransformPILtoRGBTensor


generator = torch.Generator().manual_seed(42)


def create_datasets(cfg, phase):
    train_dataset = InpaintDataset(
        dataset=cfg["data"]["dataset"],
        data_root=cfg["paths"]["train_data"],
        mask_config={"mask_mode":cfg["data"]["mask_mode"]},
        data_len=cfg["data"]["data_len"],
        image_size=2*(cfg["data"]["image_size"],),
        split='train'
    )
    train_data_len = len(train_dataset)
    
    if cfg["paths"]["val_data"]:
        val_dataset = InpaintDataset(
            dataset=cfg["data"]["dataset"],
            data_root=cfg["paths"]["val_data"],
            mask_config={"mask_mode":cfg["data"]["mask_mode"]},
            data_len=cfg["data"]["data_len"],
            image_size=2*(cfg["data"]["image_size"],),
            split='val'
        )
    elif cfg["data"]["val_split"] > 0:
        val_len = int(train_data_len * cfg["data"]["val_split"])
        train_len = train_data_len - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_len, val_len], generator=generator
        )
    else:
        val_dataset = None
        
    test_dataset = InpaintDataset(
        dataset=cfg["data"]["dataset"],
        data_root=cfg["paths"]["test_data"],
        mask_config={"mask_mode":cfg["data"]["mask_mode"]},
        data_len=cfg["data"]["data_len"],
        image_size=2*(cfg["data"]["image_size"],),
        split='test'
    )
    
    if phase == 'train':
        return train_dataset, val_dataset
    else:
        return test_dataset, None
    
        
def create_datasets_unlearning(cfg, phase, max_forget_classes=None, prev_max_forget_classes=None):
    train_dataset = ForgetInpaintDataset(
        dataset=cfg["data"]["dataset"],
        data_root=cfg["paths"]["train_data"],
        mask_config={"mask_mode":cfg["data"]["mask_mode"]},
        data_len=cfg["data"]["data_len"],
        image_size=2*(cfg["data"]["image_size"],),
        num_img_per_class=cfg[phase].get("num_img_per_class", None),
        max_forget_classes=max_forget_classes,
        prev_max_forget_classes=prev_max_forget_classes,
        sample_retain=cfg.get('curriculum', {}).get('sample_retain', None),
        buffer_forget=cfg.get('curriculum', {}).get('buffer_forget', None),
    )
    train_data_len = len(train_dataset)
    
    if cfg["paths"]["val_data"]:
        val_dataset = ForgetInpaintDataset(
            dataset=cfg["data"]["dataset"],
            data_root=cfg["paths"]["val_data"],
            mask_config={"mask_mode":cfg["data"]["mask_mode"]},
            data_len=cfg["data"]["data_len"],
            image_size=2*(cfg["data"]["image_size"],),
            num_img_per_class=cfg[phase].get("num_img_per_class", None),
            max_forget_classes=max_forget_classes,
            prev_max_forget_classes=prev_max_forget_classes,
            sample_retain=cfg.get('curriculum', {}).get('sample_retain', None),
            buffer_forget=cfg.get('curriculum', {}).get('buffer_forget', None),
        )
    elif cfg["data"]["val_split"] > 0:
        val_len = int(train_data_len * cfg["data"]["val_split"])
        train_len = train_data_len - val_len
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_len, val_len], generator=generator
        )
    else:
        val_dataset = None
        
    test_dataset = ForgetInpaintDataset(
        dataset=cfg["data"]["dataset"],
        data_root=cfg["paths"]["test_data"],
        mask_config={"mask_mode":cfg["data"]["mask_mode"]},
        data_len=cfg["data"]["data_len"],
        image_size=2*(cfg["data"]["image_size"],),
        num_img_per_class=cfg[phase].get("num_img_per_class", None),
        max_forget_classes=max_forget_classes,
        prev_max_forget_classes=None,
        sample_retain=None,
        buffer_forget=None,
        test=True
    )
    
    if phase == 'train':
        return train_dataset, val_dataset
    else:
        return test_dataset, None


def create_dataloaders(cfg, phase, phase_dataset, val_dataset):
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

    return phase_loader, val_loader