"""
Adapted from:
https://github.com/jpmorganchase/i2i_mage/blob/i2i/clip_embed.py

Requirements:
wget https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/resolve/main/open_clip_pytorch_model.bin
mv open_clip_pytorch_model.bin ~/torch_files/open_clip_vit_h_14_laion2b_s32b_b79k.bin
"""

import torch
from PIL import Image
import open_clip
import os
import numpy as np
import argparse
from rich.progress import track
from rich.console import Console
import yaml

from datasets.datasets import make_dataset, load_forget_retain_imgs


def run_clip_multi(imglist, model, preprocess):
    inputx=[]
    for imgpath in imglist:
        image = preprocess(Image.open(imgpath).convert('RGB')).unsqueeze(0)
        inputx.append(image)
    inputx = torch.cat(inputx, dim=0)
    with torch.no_grad():
        inputx = inputx.cuda()
        image_features = model(inputx)
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features_norm.squeeze().detach().cpu()


def compute_clip_norm(image_list, stepsize, model, preprocess, save_name, embeddings_save_dir=''):
    norm_list = []    

    if embeddings_save_dir != "None":
        os.makedirs(embeddings_save_dir, exist_ok=True)

    for i in track(range(0, len(image_list), stepsize), console=Console(stderr=True)):
        imgs_batch = image_list[i:min(i+stepsize, len(image_list))]
        latent_norm = run_clip_multi(imgs_batch, model, preprocess)
        norm_list.append(latent_norm)  
        for imgid, imgname in enumerate(imgs_batch):
            base_name = os.path.basename(imgname)
            
            if embeddings_save_dir != "None":
                np.save(os.path.join(embeddings_save_dir, base_name.replace('.png', '_clip_norm.npy')), latent_norm[imgid])
            
    norm_list = torch.cat(norm_list, dim=0).numpy()
    
    if embeddings_save_dir != "None":  
        np.save(os.path.join(embeddings_save_dir, f'{save_name}_clip_norm.npy'), np.array(norm_list))
    
    return norm_list
            
            
def clip_img_folder_base(img_folder, embeddings_save_dir, save_name, num_samples=-1):
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', 
        pretrained=os.path.expanduser('~/torch_files/open_clip_vit_h_14_laion2b_s32b_b79k.bin')
    )
    model = model.visual
    model.cuda()
    model.eval()
    stepsize = args.batch_size
    
    all_imgs = make_dataset(img_folder)
    if num_samples != -1:
        all_imgs = all_imgs[:num_samples]
    
    norm_list = compute_clip_norm(all_imgs, stepsize, model, preprocess, save_name, embeddings_save_dir)
    return norm_list
            
                    
def clip_img_folders(args):
    original_embeddings_save_dir = os.path.join(args.embeddings_save_dir, 'original') if args.embeddings_save_dir != 'None' else "None"
    original_save_name = os.path.basename(os.path.normpath(original_embeddings_save_dir))         
    if not os.path.isfile(os.path.join(original_embeddings_save_dir, f'{original_save_name}_clip_norm.npy')):
        original_norm_list = clip_img_folder_base(args.original_images, original_embeddings_save_dir, original_save_name, args.num_samples)
    else:
        original_norm_list = np.load(os.path.join(original_embeddings_save_dir, f'{original_save_name}_clip_norm.npy'))
        
    generated_embeddings_save_dir = os.path.join(args.embeddings_save_dir, 'generated') if args.embeddings_save_dir != 'None' else "None"
    generated_save_name = os.path.basename(os.path.normpath(generated_embeddings_save_dir))
    if not os.path.isfile(os.path.join(generated_embeddings_save_dir, f'{generated_save_name}_clip_norm.npy')):
        generated_norm_list = clip_img_folder_base(args.generated_images, generated_embeddings_save_dir, generated_save_name, args.num_samples)
    else:
        generated_norm_list = np.load(os.path.join(generated_embeddings_save_dir, f'{generated_save_name}_clip_norm.npy'))

    tmp = np.sum(generated_norm_list * original_norm_list, axis=1)
    cosine = np.mean(tmp)
    print(cosine)
    
    
def clip_img_folders_unlearning(args):
    # Load the splits
    with open(args.unlearning_splits_config, 'r') as f:
        splits = yaml.safe_load(f)
    splits = splits[args.unlearning_dataset]
    
    assert sorted(os.listdir(args.generated_images)) == ["forget", "retain"], "Generated images should be in forget and retain folders"
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', 
        pretrained=os.path.expanduser('~/torch_files/open_clip_vit_h_14_laion2b_s32b_b79k.bin')
    )
    model = model.visual
    model.cuda()
    model.eval()
    stepsize = args.batch_size
    
    ### For the original images
    retain_idx = list(range(splits["retain"]["start"], splits["retain"]["end"]))
    forget_idx = list(range(splits["forget"]["start"], splits["forget"]["end"]))

    retain_imgs, forget_imgs = load_forget_retain_imgs(args.unlearning_dataset, args.original_images, retain_idx, forget_idx)
    
    original_embeddings_save_dir_retain = os.path.join(args.embeddings_save_dir, 'original_retain') if args.embeddings_save_dir != 'None' else "None"
    original_save_name_retain = os.path.basename(os.path.normpath(original_embeddings_save_dir_retain))         
    if not os.path.isfile(os.path.join(original_save_name_retain, f'{original_save_name_retain}_clip_norm.npy')):
        original_norm_list_retain = compute_clip_norm(
            retain_imgs, stepsize, model, preprocess, 
            original_save_name_retain, original_embeddings_save_dir_retain
        )
    else:
        original_norm_list_retain = np.load(os.path.join(original_embeddings_save_dir_retain, f'{original_save_name_retain}_clip_norm.npy'))
        
    original_embeddings_save_dir_forget = os.path.join(args.embeddings_save_dir, 'original_forget') if args.embeddings_save_dir != 'None' else "None"
    original_save_name_forget = os.path.basename(os.path.normpath(original_embeddings_save_dir_forget))         
    if not os.path.isfile(os.path.join(original_save_name_forget, f'{original_save_name_forget}_clip_norm.npy')):
        original_norm_list_forget = compute_clip_norm(
            forget_imgs, stepsize, model, preprocess, 
            original_save_name_forget, original_embeddings_save_dir_forget
        )
    else:
        original_norm_list_forget = np.load(os.path.join(original_embeddings_save_dir_forget, f'{original_save_name_forget}_clip_norm.npy'))
        
        
    ### For the generated images
    retain_generated_imgs = make_dataset(os.path.join(args.generated_images, "retain"))
    forget_generated_imgs = make_dataset(os.path.join(args.generated_images, "forget"))
    
    generated_embeddings_save_dir_retain = os.path.join(args.embeddings_save_dir, 'generated_retain') if args.embeddings_save_dir != 'None' else "None"
    generated_save_name_retain = os.path.basename(os.path.normpath(generated_embeddings_save_dir_retain))
    if not os.path.isfile(os.path.join(generated_embeddings_save_dir_retain, f'{generated_save_name_retain}_clip_norm.npy')):
        generated_norm_list_retain = compute_clip_norm(
            retain_generated_imgs, stepsize, model, preprocess, 
            generated_save_name_retain, generated_embeddings_save_dir_retain
        )
    else:
        generated_norm_list_retain = np.load(os.path.join(generated_embeddings_save_dir_retain, f'{generated_save_name_retain}_clip_norm.npy'))
        
    generated_embeddings_save_dir_forget = os.path.join(args.embeddings_save_dir, 'generated_forget') if args.embeddings_save_dir != 'None' else "None"
    generated_save_name_forget = os.path.basename(os.path.normpath(generated_embeddings_save_dir_forget))
    if not os.path.isfile(os.path.join(generated_embeddings_save_dir_forget, f'{generated_save_name_forget}_clip_norm.npy')):
        generated_norm_list_forget = compute_clip_norm(
            forget_generated_imgs, stepsize, model, preprocess, 
            generated_save_name_forget, generated_embeddings_save_dir_forget
        )
    else:
        generated_norm_list_forget = np.load(os.path.join(generated_embeddings_save_dir_forget, f'{generated_save_name_forget}_clip_norm.npy'))
    
    tmp_retain = np.sum(generated_norm_list_retain * original_norm_list_retain, axis=1)
    cosine_retain = np.mean(tmp_retain)

    tmp_forget = np.sum(generated_norm_list_forget * original_norm_list_forget, axis=1)
    cosine_forget = np.mean(tmp_forget)
    
    cosine_dict = {
        "retain": cosine_retain,
        "forget": cosine_forget
    }
    
    print(cosine_dict)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generated_images', type=str, help='folder of the generated images')
    parser.add_argument('--original_images', type=str, help='folder of the original dataset')
    parser.add_argument('--unlearning', action='store_true', help='for the unlearning case')
    parser.add_argument('--unlearning_splits_config', type=str, default='./configs/forget_retain_splits.yml', help='config file for unlearning splits')
    parser.add_argument('--unlearning_dataset', type=str, default='places365', help='dataset for unlearning, used with the splits config')
    parser.add_argument('--num_samples', type=int, default=-1, help='number of samples to compute clip, -1 for all')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for computing clip')
    parser.add_argument('--embeddings_save_dir', type=str, default="None", help='folder where the image embeddings are saved')

    args = parser.parse_args()
    
    if not args.unlearning:
        clip_img_folders(args)
    else:
        clip_img_folders_unlearning(args)
