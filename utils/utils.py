import gc
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
import numpy as np
import math
import random
import os


def set_gpu(args, distributed=False, rank=0):
	""" set parameter to gpu or ddp """
	if args is None:
		return None
	if distributed and isinstance(args, torch.nn.Module):
		return DDP(args.cuda(), device_ids=[rank], output_device=rank, broadcast_buffers=True, find_unused_parameters=True)
	else:
		return args.cuda()
		
def set_device(args, distributed=False, cpu=False, rank=0):
	""" set parameter to gpu or cpu """
	if cpu and distributed:
		raise ValueError('cpu and distributed can not be set True at the same time')

	if cpu:
		return args
	elif torch.cuda.is_available():
		if isinstance(args, list):
			return (set_gpu(item, distributed, rank) for item in args)
		elif isinstance(args, dict):
			return {key:set_gpu(args[key], distributed, rank) for key in args}
		else:
			args = set_gpu(args, distributed, rank)

	return args

def set_seed(seed, gl_seed=0):
	"""  set random seed, gl_seed used in worker_init_fn function """
	if seed >=0 and gl_seed>=0:
		seed += gl_seed
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		np.random.seed(seed)
		random.seed(seed)

	''' change the deterministic and benchmark maybe cause uncertain convolution behavior. 
		speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html '''
	if seed >=0 and gl_seed>=0:  # slower, more reproducible
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
	else:  # faster, less reproducible
		torch.backends.cudnn.deterministic = False
		torch.backends.cudnn.benchmark = True


def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.clamp_(*min_max)  # clamp
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = ((img_np+1) * 127.5).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type).squeeze()

def postprocess(images):
	return [tensor2img(image) for image in images]

def get_model_class(checkpoints):
    # strip extension
    names = [os.path.splitext(f)[0] for f in checkpoints]
    
    # find the part that is common across all names
    common = set(names[0].split("_"))
    for name in names[1:]:
        common &= set(name.split("_"))
    
    # now reconstruct order: scan one filename and pick only common tokens in order
    for token in names[0].split("_"):
        if token in common:
            yield token

def check_job_restart(cfg):
    checkpoints_path = os.path.join(cfg['paths']['experiments_root'], cfg['paths']['experiment_name'], cfg['paths']['checkpoint'])
    if not os.path.exists(checkpoints_path):
        return cfg
    
    checkpoints = [f for f in os.listdir(checkpoints_path) if f.endswith('.pth')]
    if not checkpoints:
        return cfg
    
    model_class = "_".join(get_model_class(checkpoints))
    latest_checkpoint = max(checkpoints, key=lambda f: int(f.split('_')[0]))  # Assuming the checkpoint filename starts with an integer
    cfg['paths']['resume_state'] = latest_checkpoint.split(f"_{model_class}")[0]
    return cfg


def linear_decay(start, end, n_steps):
    values = []
    for step in range(n_steps):
        value = start + (end - start) * (step / (n_steps - 1))
        values.append(value)
    return values

def get_param(net):
    new_param = []
    with torch.no_grad():
        j = 0
        for name, param in net.named_parameters():
            new_param.append(param.clone())
            j += 1
    torch.cuda.empty_cache()
    gc.collect()
    return new_param


def set_param(net, old_param):
    with torch.no_grad():
        j = 0
        for name, param in net.named_parameters():
            param.copy_(old_param[j])
            j += 1
    torch.cuda.empty_cache()
    gc.collect()
    return net