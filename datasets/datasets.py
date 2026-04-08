"""
Adapted from:
https://github.com/jpmorganchase/i2i_Palette-Image-to-Image-Diffusion-Models/blob/i2i/data/dataset.py
"""
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, Places365
from datasets.transforms import TransformPILtoRGBTensor
import os
import numpy as np
import glob
import random
import yaml
import pandas as pd
import math
from .utils.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

TORCH_DATASETS = ["CIFAR10", "Places365"]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(imgdir):
    if os.path.isfile(imgdir):
        images = open(imgdir, 'r').read().splitlines()
    else:
        images = []
        assert os.path.isdir(imgdir), '%s is not a valid directory' % imgdir
        for root, _, fnames in sorted(os.walk(imgdir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')


def load_imgs_dataset(dataset, data_root, split='train'):
    """
    Returns a list of images. Loading is custom for each dataset,
    depending on its structure.

    :param dataset: string, name of supported dataset
    :param data_rott: string, path to the dataset folder
    :param test: bool, whether to return the test set. Applicable only for torchvision datasets. 
    """
    
    if split not in ['train', 'test', 'val']:
        raise ValueError(f"Unknown split {split}.")
    
    images = []

    if dataset == "CIFAR10":
        # For datasets from torchvision, let's load them directly using the custom loaders
        dataset = CIFAR10(data_root, train=(split=='train'), download=True)

        for img, label in dataset:
            images.append(img)
            
    elif dataset == "Places365":
        try:
            dataset = Places365(data_root, split=("train-standard" if split=="train" else "val"), download=True, small=True)
        except RuntimeError:
            print("Places365 dataset already downloaded.")
            dataset = Places365(data_root, split=("train-standard" if split=="train" else "val"), download=False, small=True)
        
        images = IgnoreLabelDataset(dataset)

    return images


class ImagesPathDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = TransformPILtoRGBTensor() if transforms is None else transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert("RGB")
        img = self.transforms(img)
        return img
    
    
class InpaintDataset(Dataset):
    def __init__(self, dataset, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, split='train'):
        if dataset in TORCH_DATASETS:
            imgs = load_imgs_dataset(dataset, data_root, split=split)
        else:
            print(f"Datset {dataset} not in {TORCH_DATASETS}. Loading images from {data_root}.")
            imgs = make_dataset(data_root)
            
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
            
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        
        if not isinstance(path, str):
            # For torchvision datasets with already loaded PILs
            img = self.tfs(path)
            ret['path'] = str(index) + ".png"
        else:
            img = self.tfs(self.loader(path))
            ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]

        mask = self.get_mask()
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox(img_shape=self.image_size))
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'center_50': # 50% center crop, i.e. the box has 1/sqrt(2) of the image size
            h, w = self.image_size
            box_side = int(math.sqrt(0.5 * h * w))
            top = (h - box_side) // 2
            left = (w - box_side) // 2
            mask = bbox2mask(self.image_size, (top, left, box_side, box_side))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox(img_shape=self.image_size))
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'random_choice':
            h, w = self.image_size
            box_side = int(math.sqrt(0.5 * h * w))
            
            rectangular_mask = bbox2mask(
                self.image_size, 
                # random_bbox(
                #     img_shape=self.image_size, 
                #     max_bbox_shape=int(np.sqrt(0.4) * min(self.image_size)),
                #     max_bbox_delta=int((np.sqrt(0.4) - np.sqrt(0.1)) * min(self.image_size)),
                #     min_margin=int(0.1 * min(self.image_size)),
                # )
                random_bbox(
                    img_shape=self.image_size, 
                    max_bbox_shape=box_side,
                    max_bbox_delta=int(min(self.image_size)//2),
                    min_margin=int(0.1 * min(self.image_size)),
                )
            )
            irregular_mask = brush_stroke_mask(
                self.image_size,
                brush_width=(int(0.1 * min(self.image_size)), int(0.4 * min(self.image_size))),
            )
            
            choice = random.randint(1, 10)
            if choice <= 5:
                mask = rectangular_mask
            else:
                mask = irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class UncroppingDataset(Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)


class ColorizationDataset(Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.imgs = flist[:int(data_len)]
        else:
            self.imgs = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.imgs[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.imgs)


def load_forget_retain_imgs(dataset, data_root, retain_idx, forget_idx, num_img_per_class=None, 
                            sample_retain=None, buffer_forget=None, 
                            min_forget_index=None, max_forget_index=None, test=False):
    """
    Returns lists of paths for retain and forget image sets. Loading is custom for each dataset,
    depending on its structure.

    :param dataset: string, name of supported dataset
    :param data_rott: string, path to the dataset folder
    :param retain_idx: list of ints, indices of classes from retain set
    :param forget_idx: list of ints, indices of classes from forget set
    :param num_img_per_class: int or None, if not None, limits the number of images per class
    :param sample_retain: float or None, if not None, randomly samples this % of the retain images
    :param buffer_forget: float or None, if not None, randomly samples this % of the forget images of class idx < min_forget_index
    :param min_forget_index: int or None, if not None, only applies buffer_forget to classes with index < min_forget_index
    :param max_forget_index: int or None, if not None, only includes forget classes with index < max_forget_index
    :param test: bool, whether to return the test set. Applicable only for torchvision datasets. 
    """
    retain_imgs=[]
    forget_imgs=[]

    if dataset == "CIFAR10":
        # For datasets from torchvision, let's load them directly using the custom loaders
        dataset = CIFAR10(data_root, train=not test, download=True)

        for img, label in dataset:
            if label in retain_idx:
                retain_imgs.append(img)
            elif label in forget_idx:
                forget_imgs.append(img)

    elif dataset == "Places365":
        class_list = open('./configs/places365_classes.txt').read().splitlines()
        # class_list = [int(class_name.split(' ')[-1]) for class_name in class_list]
        # class_list = [class_name.split(' ')[0] for class_name in class_list]
        class_list = {int(class_name.split(' ')[-1]): class_name.split(' ')[0] for class_name in class_list}
        retain_class = [class_list[idx] for idx in retain_idx]
        forget_class = [class_list[idx] for idx in forget_idx]

        for retain_class_idx, retainname in enumerate(retain_class):
            tmp = glob.glob(os.path.join(data_root, retainname[1:], '*.jpg'))
            tmp = sorted(tmp)
            
            if num_img_per_class is not None:
                tmp = tmp[:num_img_per_class]
                
            if sample_retain is not None:
                random.shuffle(tmp)
                tmp = tmp[:int(len(tmp)*sample_retain)]
                            
            retain_imgs += tmp

        for forget_class_idx, forgetname in enumerate(forget_class):
            tmp = glob.glob(os.path.join(data_root, forgetname[1:], '*.jpg'))
            tmp = sorted(tmp)
            
            if num_img_per_class is not None:
                tmp = tmp[:num_img_per_class]
                
            # This basically means that if we use buffer we need to sample from the previous forget classes, not the current ones, they remain the same
            # If we do not use buffer, we get all the images from previous and current forget classes
            # This only happens for training set, for test set we always get all the images from all forget classes  
            if buffer_forget is not None:
                if min_forget_index is not None:
                    if forget_class_idx < min_forget_index: # Buffer only for classes with index < min_forget_index (i.e. the classes that have been forgotten in previous steps)
                        random.shuffle(tmp)
                        tmp = tmp[:int(len(tmp)*buffer_forget)]

            if max_forget_index is not None:
                if forget_class_idx >= max_forget_index: # No images for classes with index >= max_forget_index (i.e. the classes that have not been forgotten yet)
                    break
                            
            forget_imgs += tmp
            
    elif dataset == 'TinyImageNet':
        if not test:
            img_paths = make_dataset(data_root)
            class_list = [os.path.basename(path).split('.')[0].split('_')[0] for path in img_paths]
            class_list_unique = sorted(list(set(class_list)))

            for img_path, class_name in zip(img_paths, class_list):
                if class_list_unique.index(class_name) in retain_idx:
                    retain_imgs.append(img_path)
                elif class_list_unique.index(class_name) in forget_idx:
                    forget_imgs.append(img_path)
        else:
            # Test set not implemented for TinyImageNet dataset. (We do not have test labels)
            return [], []
    
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}.")

    print(f"Found {len(forget_imgs)} forget images and {len(retain_imgs)} retain images.")

    return retain_imgs, forget_imgs


class ForgetInpaintDataset(Dataset):
    def __init__(self, dataset, 
                    data_root=None, 
                    mask_config={}, 
                    data_len=-1, 
                    image_size=[256, 256], 
                    loader=pil_loader, 
                    num_img_per_class=None,
                    max_forget_classes=None,
                    prev_max_forget_classes=None,
                    sample_retain=None,
                    buffer_forget=None,
                    test=False):

        # This contains class index splits for retain & forget sets
        with open("./configs/forget_retain_splits.yml", "r") as file:
            splits = yaml.safe_load(file)

        retain_idx = list(range(splits[dataset]["retain"]["start"], splits[dataset]["retain"]["end"]))
        forget_idx = list(range(splits[dataset]["forget"]["start"], splits[dataset]["forget"]["end"]))

        retain_imgs, forget_imgs = load_forget_retain_imgs(
            dataset, data_root, retain_idx, forget_idx, num_img_per_class=num_img_per_class,
            sample_retain=sample_retain, buffer_forget=buffer_forget, 
            min_forget_index=prev_max_forget_classes, max_forget_index=max_forget_classes, test=test)

        # retain_imgs = retain_imgs[:len(forget_imgs)]
        # forget_imgs = forget_imgs
        self.imgs = retain_imgs + forget_imgs
        self.labels = [1.0] * len(retain_imgs) + [-1.0] * len(forget_imgs)

        print(f"Loaded {len(retain_imgs)} retain images and {len(forget_imgs)} forget images.")

        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]

        mask = self.get_mask()
        
        if not isinstance(path, str):
            # For torchvision datasets with already loaded PILs
            img = self.tfs(path)
            ret['path'] = str(index) + ".png"
        else:
            img = self.tfs(self.loader(path))            
            ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]

        cond_img = img * (1. - mask) + mask * torch.randn_like(img)
        mask_img = img * (1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_img 
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        
        return ret, self.labels[index]

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'center_50': # 50% center crop, i.e. the box has 1/sqrt(2) of the image size
            h, w = self.image_size
            box_side = int(math.sqrt(0.5 * h * w))
            top = (h - box_side) // 2
            left = (w - box_side) // 2
            mask = bbox2mask(self.image_size, (top, left, box_side, box_side))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'random_choice':
            h, w = self.image_size
            box_side = int(math.sqrt(0.5 * h * w))
            
            rectangular_mask = bbox2mask(
                self.image_size, 
                # random_bbox(
                #     img_shape=self.image_size, 
                #     max_bbox_shape=int(np.sqrt(0.4) * min(self.image_size)),
                #     max_bbox_delta=int((np.sqrt(0.4) - np.sqrt(0.1)) * min(self.image_size)),
                #     min_margin=int(0.1 * min(self.image_size)),
                # )
                random_bbox(
                    img_shape=self.image_size, 
                    max_bbox_shape=box_side,
                    max_bbox_delta=int(min(self.image_size)//2),
                    min_margin=int(0.1 * min(self.image_size)),
                )
            )
            irregular_mask = brush_stroke_mask(
                self.image_size,
                brush_width=(int(0.1 * min(self.image_size)), int(0.4 * min(self.image_size))),
            )
            
            choice = random.randint(1, 10)
            if choice <= 5:
                mask = rectangular_mask
            else:
                mask = irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2, 0, 1)
    

# TODO: Combine with ForgetImpaintDataset -> only the mask type differs
class ForgetUncroppingDataset(Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, number_img_per_class=100):
        data_root=None
        if data_root is not None:
            imgs = make_dataset(data_root)
            if data_len > 0:
                self.imgs = imgs[:int(data_len)]
            else:
                self.imgs = imgs
            self.labels = [1.0]*len(imgs) 
        else:
            csv_path = 'datasets/place365/flist/ForgetUncroppingDataset_{}_{}_{}.csv'.format(number_img_per_class, data_len, image_size[0])
            if os.path.isfile(csv_path):
                data_info = pd.read_csv(csv_path, sep=',', header=0, names = ['imgs', 'labels'])
                self.imgs = list(data_info.imgs)
                self.labels = list(data_info.labels)
            else:
                class_list = open('datasets/place365/flist/class.txt').read().splitlines()
                retain_class = class_list[:50]
                forget_class = class_list[50:100]
                retain_imgs=[]
                forget_imgs=[]
                for retainname in retain_class:
                    tmp = glob.glob(os.path.join(data_root, 'train_256', retainname[1:], '*.jpg'))
                    random.shuffle(tmp)
                    retain_imgs += tmp[:number_img_per_class]
                for forgetname in forget_class:
                    tmp = glob.glob(os.path.join(data_root, 'train_256', forgetname[1:], '*.jpg'))
                    random.shuffle(tmp)
                    forget_imgs += tmp[:number_img_per_class]
                self.imgs = retain_imgs + forget_imgs
                self.labels = [1.0]*len(retain_imgs)+[-1.0]*len(forget_imgs)
                datainfo = pd.DataFrame(list(zip(self.imgs, self.labels)), columns =['imgs', 'labels'])
                datainfo.to_csv(csv_path, index=False,)

        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret, self.labels[index]

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)

# TODO: forget about it
class ForgetColorizationDataset(Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader, number_img_per_class=100):
        data_root=None
        if data_root is not None:
            imgs = make_dataset(data_root)
            if data_len > 0:
                self.imgs = imgs[:int(data_len)]
            else:
                self.imgs = imgs
            self.labels = [1.0]*len(imgs) 
        else:
            csv_path = 'datasets/place365/flist/ForgetColorizationDataset_{}_{}_{}.csv'.format(number_img_per_class, data_len, image_size[0])
            if os.path.isfile(csv_path):
                data_info = pd.read_csv(csv_path, sep=',', header=0, names = ['imgs', 'labels'])
                self.imgs = list(data_info.imgs)
                self.labels = list(data_info.labels)
            else:
                class_list = open('datasets/place365/flist/class.txt').read().splitlines()
                retain_class = class_list[:50]
                forget_class = class_list[50:100]
                retain_imgs=[]
                forget_imgs=[]
                for retainname in retain_class:
                    tmp = glob.glob(os.path.join(data_root, 'train_256', retainname[1:], '*.jpg'))
                    random.shuffle(tmp)
                    retain_imgs += tmp[:number_img_per_class]
                for forgetname in forget_class:
                    tmp = glob.glob(os.path.join(data_root, 'train_256', forgetname[1:], '*.jpg'))
                    random.shuffle(tmp)
                    forget_imgs += tmp[:number_img_per_class]
                self.imgs = retain_imgs + forget_imgs
                self.labels = [1.0]*len(retain_imgs)+[-1.0]*len(forget_imgs)
                datainfo = pd.DataFrame(list(zip(self.imgs, self.labels)), columns =['imgs', 'labels'])
                datainfo.to_csv(csv_path, index=False,)

        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.imgs[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret, self.labels[index]

    def __len__(self):
        return len(self.imgs)
    
    
class IgnoreLabelDataset(Dataset):
    def __init__(self, orig):
        self.orig = orig

    def __getitem__(self, index):
        return self.orig[index][0]

    def __len__(self):
        return len(self.orig)
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    try:
        data_root = '../../Datasets/CIFAR10/images/train'
        assert os.path.exists(data_root)
    finally:
        data_root = "~/torch_files"
        assert os.path.exists(data_root)
    
    mask_config = {'mask_mode': 'center'}
    dataset = InpaintDataset(data_root, mask_config=mask_config, image_size=(32, 32))
    print(len(dataset))
    
    image = dataset[0]
    plt.imshow(image['gt_image'].permute(1, 2, 0))
    plt.show()
    plt.imshow(image['mask'].permute(1, 2, 0))
    plt.show()
    plt.imshow(image['mask_image'].permute(1, 2, 0))
    plt.show()
    plt.imshow(image['cond_image'].permute(1, 2, 0))
    plt.show()
    print(image['path'])
