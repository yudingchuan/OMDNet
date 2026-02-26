import subprocess
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
import glob
import os
import re


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location='cpu')
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_epoch_*.ckpt'
    else:
        ckpt_path_pattern = f'{work_dir}/model_ckpt_epoch_{steps}.ckpt'
    return sorted(glob.glob(ckpt_path_pattern),
                  key=lambda x: -int(re.findall('.*epoch\_(\d+)\.ckpt', x)[0])) 


def load_checkpoint(model, optimizer, scheduler, work_dir):
    checkpoint, last_ckpt_path = get_last_checkpoint(work_dir)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict']['model'])
        model.cuda()
        optimizer.load_state_dict(checkpoint['optimizer_states'][0])
        if 'scheduler_states' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_states'][0])
        training_step = checkpoint['global_step']
        del checkpoint
        torch.cuda.empty_cache()
        print(f'model loaded from {last_ckpt_path}')
    else:
        training_step = 0
        model.cuda()
        print('No checkpoint exists. Start from the beginning!')
    return training_step


def save_checkpoint(model, optimizer, scheduler, work_dir, global_step, num_ckpt_keep):
    ckpt_path = f'{work_dir}/model_ckpt_epoch_{global_step}.ckpt'
    print(f'Epoch={global_step}: model saved to {ckpt_path}')
    print()

    checkpoint = {'global_step': global_step}
    optimizer_states = []
    optimizer_states.append(optimizer.state_dict())
    checkpoint['optimizer_states'] = optimizer_states
    scheduler_states = []
    if scheduler is not None:
        scheduler_states.append(scheduler.state_dict())
    checkpoint['scheduler_states'] = scheduler_states
    checkpoint['state_dict'] = {'model': model.state_dict()}
    torch.save(checkpoint, ckpt_path, _use_new_zipfile_serialization=False) 
    for old_ckpt in get_all_ckpts(work_dir)[num_ckpt_keep:]: 
        remove_file(old_ckpt)
        print(f'Delete ckpt: {os.path.basename(old_ckpt)}')


def remove_file(*fns):
    for f in fns:
        subprocess.check_call(f'rm -rf "{f}"', shell=True)


def detach_img(img):
    img = img.data.cpu().numpy()
    return np.clip(img, 0, 1)

def plot_img(img):
    
    img = detach_img(img)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    if img.shape[0] == 3:
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))  # (C, H, W) -> (H, W, C)
        img = img.convert("RGB")
    else:
        img = Image.fromarray(img.squeeze())

    return img