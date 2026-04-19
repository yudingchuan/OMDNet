import torch
import torch.nn as nn
import torch.nn.functional as F


class loss_funcs:
    def __init__(self, hparams=None):
        self.hparams = hparams

    def fft_loss(self, img_p, img_g, weight=None):
        pred_fft = torch.fft.rfft2(img_p)
        label_fft = torch.fft.rfft2(img_g)
        return F.l1_loss(pred_fft, label_fft, reduction='none').mean([1, 2, 3]) # [B]

    def l1_loss(self, img_p, img_g, weight=None):
        l1_loss = F.l1_loss(img_p, img_g, reduction='none')
        if weight is None:
            return l1_loss.mean([1, 2, 3])
        return (l1_loss * weight).sum([1, 2, 3]) / weight.sum([1, 2, 3]).clamp_min(1) / 3  
    
    def calc(self, img_p, img_g, blur_mask=None, suffix=''):
        total_loss = 0

        loss_keys = [k for k in ['l1', 'fft'] if self.hparams[f'lambda_{k}'] > 0]
        losses_ = {k: getattr(self, f'{k}_loss')(img_p, img_g, blur_mask) for k in loss_keys} 

        losses_ = {k: v.mean() for k, v in losses_.items() if not torch.isnan(v).any()} 
        total_loss += sum([v * self.hparams[f'lambda_{k}'] for k, v in losses_.items()])

        return losses_, total_loss
    
    def calc_L_warp(self, losses, warp_loss, warp_first, gt_firsts, warp_last, gt_lasts):
        for idx, (warp_first_idx, first_gt_idx, warp_last_idx, last_gt_idx) in enumerate(zip(warp_first, gt_firsts, warp_last, gt_lasts)):
            direction1 = F.l1_loss(warp_first_idx, first_gt_idx, reduction='none') + F.l1_loss(warp_last_idx, last_gt_idx, reduction='none')
            direction2 = F.l1_loss(warp_first_idx, last_gt_idx, reduction='none') + F.l1_loss(warp_last_idx, first_gt_idx, reduction='none')
            batch_loss = torch.min(direction1.mean(dim=[1,2,3]), direction2.mean(dim=[1,2,3]))
            losses[f'warp_{idx+1}_l1'] = batch_loss.mean()
            warp_loss += losses[f'warp_{idx+1}_l1']
        return losses, warp_loss
    
    def calc_L_merge(self, losses, merge_loss, merged, merges_gt):
        for idx, (merged_idx, merge_gt_idx) in enumerate(zip(merged,merges_gt)):
            losses[f'merge_{idx+1}_l1'] = F.l1_loss(merged_idx, merge_gt_idx)
            merge_loss += losses[f'merge_{idx+1}_l1']
        return losses, merge_loss

    def calc_L_f(self, losses, img_loss, deblured, sharps, loss_key):
        for idx, (deblur_idx, sharp_idx) in enumerate(zip(deblured,sharps)):
            losses_, loss_sum = self.calc(deblur_idx, sharp_idx) #

            for k, v in losses_.items():
                losses[f'{loss_key}_{idx+1}_{k}'] = v.item() 
            img_loss += loss_sum
        return losses, img_loss