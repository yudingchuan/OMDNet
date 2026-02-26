import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as tf

import numpy as np
from .ssim_util import ssim_calc
import pyiqa

import sys
import os

class metric_funcs():
    def __init__(self, metric_keys=None):
        self.metric_keys = metric_keys
    
    def psnr_metric(self, img_p, img_g):
        psnr1 = ((img_p - img_g) ** 2).mean()
        psnr = 10 * torch.log10(1 / psnr1)
        return psnr

    def weighted_psnr_metric(self, img_p, img_g, weight):
        w_psnr1 = (((img_p - img_g) ** 2) * weight).sum() / weight.sum() / 3
        w_psnr = 10 * torch.log10(1 / w_psnr1)
        return w_psnr

    def ssim_metric(self, img_p, img_g):
        ssim = ssim_calc(img_g, img_p, None)
        return ssim.mean()

    def weighted_ssim_metric(self, img_p, img_g, weight):
        w_ssim = ssim_calc(img_g, img_p, weight)
        return w_ssim.mean()

    def lpips_metric(self, img_p, img_g):
        iqa_metric = pyiqa.create_metric('lpips-vgg', device=img_p.device)
        score_fr = iqa_metric(img_p.unsqueeze(0), img_g.unsqueeze(0))
        return score_fr

    def dists_metric(self, img_p, img_g):
        iqa_metric = pyiqa.create_metric('dists', device=img_p.device)
        score_fr = iqa_metric(img_p.unsqueeze(0), img_g.unsqueeze(0))
        return score_fr


        
    def calc(self, img_p, img_g, weights, results):

        if results is None:
            results = {k: [] for k in self.metric_keys}

        # Temporarily redirect standard output to suppress print statements (for pyiqa loading prints)
        sys.stdout = open(os.devnull, 'w')

        for i in range(img_g.shape[0]): # range(batchsize)
            if weights is not None:
                weight = weights[i]

            # Simulate quantization during saving and loading
            device = weight.device
            img_p = img_p[i].data.cpu().numpy()
            img_g = img_g[i].data.cpu().numpy()  
            img_p = np.clip(img_p, 0, 1)
            img_p = np.clip(img_p * 255, 0, 255).astype(np.uint8)
            img_g = np.clip(img_g, 0, 1)
            img_g = np.clip(img_g * 255, 0, 255).astype(np.uint8)
            img_p = torch.tensor(img_p).float()/255
            img_g = torch.tensor(img_g).float()/255
            img_p = img_p.to(device)
            img_g = img_g.to(device)

            for k in self.metric_keys:
                if 'weighted_' in k:
                    if (weights is not None) and (weight.sum() > 0):
                        result = getattr(self, f'{k}_metric')(img_p, img_g, weight).cpu().numpy()
                        results[k].append(result)
                else:
                    result = getattr(self, f'{k}_metric')(img_p, img_g).cpu().numpy()
                    results[k].append(result)
                    
        # Restore normal standard output after the execution
        sys.stdout = sys.__stdout__

        return {k: np.nanmean(results[k]) if len(results[k]) > 0 else np.nan for k in self.metric_keys}