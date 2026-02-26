import os
import torch
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from loss_metric_funcs.metrics import metric_funcs
from loss_metric_funcs.losses import loss_funcs

from dataset import DeblurDataSet
from utils.hparams import hparams, set_hparams

from models.deblur_model import OMDNet
import numpy as np
import torch.nn.functional as F
from PIL import Image

from utils.record import detach_img, plot_img, load_checkpoint, save_checkpoint
from utils.general import move_to_cuda, tensors_to_scalars, multi_scale


class OMD_trainer:
    def __init__(self):
        self.logger = self.build_tensorboard(save_dir=hparams['work_dir'], name='tb_logs')
        self.metric_keys = ['psnr', 'ssim', 'weighted_psnr', 'weighted_ssim'] # metrics logged during validation
        self.metric_funcs = metric_funcs(self.metric_keys)
        self.loss_funcs = loss_funcs(hparams)
        self.work_dir = hparams['work_dir']

    def build_tensorboard(self, save_dir, name, **kwargs):
        log_dir = os.path.join(save_dir, name)
        os.makedirs(log_dir, exist_ok=True)
        return SummaryWriter(log_dir=log_dir, **kwargs) 

    def build_model(self):
        self.model = OMDNet(num_res=20)
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=hparams['lr'], weight_decay=0)

    def build_scheduler(self, optimizer):
        milestone_epochs = hparams['MultiStepLR_milestones']
        gamma = 0.9
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestone_epochs, gamma)   
       
    def build_train_dataloader(self):
        dataset = DeblurDataSet('train')
        return DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True,
            pin_memory=False, num_workers=hparams['num_workers'])

    def build_val_dataloader(self):
        return DataLoader(DeblurDataSet('val'), batch_size=hparams['eval_batch_size'], shuffle=False, pin_memory=False)

    def run(self, model, optimizer, training_step, scheduler, dataloader, train=bool):       
        if train:
            torch.set_grad_enabled(True)
            model.train()
        else:
            torch.set_grad_enabled(False)
            model.eval()       

        pbar = tqdm(dataloader, dynamic_ncols=True, leave=False)
        
        total_losses_epoch = {}  
        num_batches = {}  

        for batch_idx, batch in enumerate(dataloader):
            
            if train:
                optimizer.zero_grad()
            batch = move_to_cuda(batch)          
            assess = {}

            original_deblured, std_gated_deblured, enhanced_gated_deblured, gate, warp_first, warp_last, merged = model(batch['img_blur'], batch['img_gt'], batch['blur_mask'], train)
            blur = batch['img_blur']
            sharp = batch['img_gt']
            sharps = multi_scale(sharp)  
            gt_first = batch['img_first']
            gt_last = batch['img_last']
            gt_firsts = multi_scale(gt_first)  
            gt_lasts = multi_scale(gt_last)  

            merge_gt = (sharp + gt_first + gt_last)/3
            merges_gt = multi_scale(merge_gt)

            mask = batch['blur_mask']       

            # pics can be realtime updated and displayed in code editor interface
            if batch_idx % 25 == 0:
                self.save_images_for_realtime_monitoring(
                    save_path="save_pic",
                    sharp=sharp[0],
                    blur=blur[0],
                    gated_deblured=enhanced_gated_deblured[0][0],
                    deblured=original_deblured[0][0],
                    gate=gate[0][0],
                    mask=mask[0],
                    merged=merged[0][0] if train else None,
                    merge_gt=merge_gt[0] if train else None
                )


            total_loss = 0.0
            warp_loss = 0.0
            merge_loss = 0.0
            restoration_loss = 0.0
            gated_rst_loss = 0.0
            gated_rst_loss_ = 0.0
                      

            if train:
                assess, warp_loss = self.loss_funcs.calc_L_warp(assess, warp_loss, warp_first, gt_firsts, warp_last, gt_lasts)
                assess, merge_loss = self.loss_funcs.calc_L_merge(assess, merge_loss, merged, merges_gt)
                assess, gated_rst_loss = self.loss_funcs.calc_L_f(assess, gated_rst_loss, std_gated_deblured, sharps, 'stdGated')
                assess, gated_rst_loss_ = self.loss_funcs.calc_L_f(assess, gated_rst_loss_, enhanced_gated_deblured, sharps, 'enhGated')
                assess, restoration_loss = self.loss_funcs.calc_L_f(assess, restoration_loss, original_deblured, sharps, 'deblured')

            else:
                results = self.metric_funcs.calc(std_gated_deblured[0], sharp, weights=batch.get('blur_mask_nonpad'), results=None)
                assess.update(results)
      
            
            # total_losses per epoch
            with torch.no_grad():
                for ass_name, ass_value in assess.items():
                    if ass_name not in total_losses_epoch:
                        total_losses_epoch[ass_name] = 0.0
                        num_batches[ass_name] = 0.0
                    if not math.isnan(ass_value):
                        total_losses_epoch[ass_name] += ass_value
                        num_batches[ass_name] += 1


            if train:
                total_loss = warp_loss * hparams['lambda_warp'] + merge_loss * hparams['lambda_merge'] \
                            + restoration_loss * hparams['lambda_r'] + (gated_rst_loss+gated_rst_loss_) * hparams['lambda_g']/2
                total_loss.backward()
                optimizer.step()

            pbar.update()
            if train:
                pbar.set_description(f"Epoch {training_step} --Training") 
                pbar.set_postfix({**tensors_to_scalars({k: v for k, v in assess.items() if (k.split('_')[-1] == 'l1' and k.split('_')[1] == '1')})}) 
            else:
                pbar.set_description(f"Epoch {training_step} --Validating")
                pbar.set_postfix({**tensors_to_scalars(assess)})

                if batch["item_name"][0] in hparams['tensorboard_imgs_show'] and gate is not None:
                    img_name = f'batch{batch_idx}_{batch["item_name"][0].replace("/", "_")}'
                    self.logger.add_image(f'{img_name}_gate', detach_img(gate[0][0]), training_step) 
                    self.logger.add_image(f'{img_name}_deblured', detach_img(std_gated_deblured[0][0]), training_step) 
                    if training_step == hparams['val_check_interval']:
                        self.logger.add_image(f'{img_name}_sharp', detach_img(sharp[0]), training_step) 
                        self.logger.add_image(f'{img_name}_blur', detach_img(blur[0]), training_step) 
        

        avg_losses_epoch = {k: (v / num_batches[k] if num_batches[k] > 0 else 0) for k, v in total_losses_epoch.items()}
        if train:
            self.log_metrics({f'train/{k}': v for k, v in avg_losses_epoch.items()}, training_step) 
            print('------ logging train_loss')
            scheduler.step()
        else:
            self.log_metrics({f'val/{k}': v for k, v in avg_losses_epoch.items()}, training_step)
            print('------ logging val_loss   @ ', end='')
        



    def Trainer(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)
        training_step = load_checkpoint(model, optimizer, scheduler, hparams['work_dir'])

        train_loader = self.build_train_dataloader()
        val_loader = self.build_val_dataloader()

        while training_step < hparams['max_updates']:
            if training_step == 0:
                self.run(model, optimizer, training_step, scheduler, val_loader, train=False)

            training_step += 1
            self.run(model, optimizer, training_step, scheduler, train_loader, train=True)
            if training_step % hparams['val_check_interval'] == 0:
                self.run(model, optimizer, training_step, scheduler, val_loader, train=False)
            if training_step % hparams['ckpt_save_interval'] == 0:
                save_checkpoint(model, optimizer, scheduler, self.work_dir, training_step, hparams['num_ckpt_keep'])

            

    def log_metrics(self, metrics, step):
        metrics = self.metrics_to_scalars(metrics)
        logger = self.logger
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics
    
    def save_images_for_realtime_monitoring(self, save_path, **images):
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for name, img in images.items():
            if img is not None:    
                img_path = os.path.join(save_path, f"{name}.png")
                plot_img(img).save(img_path)



if __name__ == '__main__':
    set_hparams()
    trainer = OMD_trainer()
    trainer.Trainer()

# ---------- run the command below to train OMDNet on OMoBlur train set ------------
# CUDA_VISIBLE_DEVICES=0 python train.py --config configs/OMD.yaml --exp_name trainOMD --reset

# ---------- run the command below to see loss, basic metric and image results' change during training ------------
# tensorboard --logdir=/workspace/yudc/OMDNet_launch/checkpoints/trainOMD/tb_logs --port=6037

