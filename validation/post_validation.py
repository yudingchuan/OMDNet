import os
import torch
import math
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from loss_metric_funcs.metrics import metric_funcs
from loss_metric_funcs.losses import loss_funcs
from dataset import DeblurDataSet
from utils.hparams import hparams, set_hparams
from models.deblur_model import OMDNet
import numpy as np
import torch.nn.functional as F
from PIL import Image
import glob
from torchvision.transforms import functional as tf
from utils.record import detach_img, plot_img, load_checkpoint, save_checkpoint
from utils.general import move_to_cuda, tensors_to_scalars


class OMD_validation:
    def __init__(self):

        self.work_dir = hparams['work_dir']
        self.data_dir = hparams['data_dir']
        # self.metric_keys = ['psnr', 'ssim', 'weighted_psnr', 'weighted_ssim'] 
        # self.metric_keys = ['lpips', 'dists'] 
        self.metric_keys = ['psnr', 'ssim', 'weighted_psnr', 'weighted_ssim', 'lpips', 'dists'] 
        self.metric_funcs = metric_funcs(self.metric_keys)

    def build_model(self):
        self.model = OMDNet(num_res=20)
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0)

    def run(self, model, optimizer, training_step):       

        torch.set_grad_enabled(False)
        model.eval()       

        test_img_paths = sorted(glob.glob(f'{self.data_dir}/val/*/*/*/*b.png'))

        pbar = tqdm(test_img_paths, dynamic_ncols=True, leave=False) 
        num_batches = {}  
        total_losses_epoch = {}  

        val_path_list = ['/data/yudc/SyLoBlur/dataset_launch/val/01/0009/image_pairs/008_11_b.png',\
                         '/data/yudc/SyLoBlur/dataset_launch/val/10/0004/image_pairs/040_11_b.png'] 
        val_all_pic = False
        for batch_idx, path in enumerate(test_img_paths):
            if path in val_path_list or val_all_pic:
                blur_path = path # /
                mask_path = blur_path.replace('_b.png','_m.png')

                sharp_path = os.path.join(os.path.dirname(blur_path), blur_path.split('/')[-1].split('_')[0] + '_1.png')


                dir_scene = os.path.basename( os.path.dirname(os.path.dirname( os.path.dirname(blur_path) ) ))  # 01
                dir_capindex = os.path.basename( os.path.dirname(os.path.dirname(blur_path) )  )    # 0015
                midframe_index = '_'.join(os.path.basename(blur_path).split('_', 2)[:2])        # 008_11
                item_path = os.path.join(dir_scene, dir_capindex, midframe_index) # 01/0015/008_11

                img_blur = read_img(blur_path)
                img_sharp = read_img(sharp_path)
                img_mask = read_img(mask_path)
                blur_mask = (img_mask > 0).float()[:1]

                multiple_width = hparams['multiple_width'] 
                if img_blur.shape[2] % multiple_width != 0:
                    l_pad = multiple_width - img_blur.shape[2] % multiple_width
                    img_blur = F.pad(img_blur, (0, 0, 0, l_pad), mode='reflect')  
                if img_blur.shape[3] % multiple_width != 0:
                    l_pad = multiple_width - img_blur.shape[3] % multiple_width
                    img_blur = F.pad(img_blur, (0, l_pad, 0, 0), mode='reflect') 

                
                # num_batches +=1
                img_blur = move_to_cuda(img_blur)  
                img_sharp = move_to_cuda(img_sharp)        
                blur_mask = move_to_cuda(blur_mask)        


                _, deblured, _, gates,_,_,_ = model(img_blur, img_blur,img_blur,train = False)
                deblured_ = deblured[0]
                deblured_ = deblured_[:, :, :int(img_sharp.shape[2]), :int(img_sharp.shape[3])]
                catagry_dir = os.path.dirname(item_path)
                # print(catagry_dir)

                # # --- if you want to save pic, please uncomment lines below ---
                # if not os.path.exists('validation/save_valpic/' + catagry_dir):
                #         os.makedirs('validation/save_valpic/' + catagry_dir)
                # plot_img(deblured[0][0]).save("validation/save_valpic/" + item_path + '_db.png')
                # plot_img(gates[0][0]).save("validation/save_valpic/" + item_path + '_gate.png')

                losses = {}
                results = self.metric_funcs.calc(deblured_, img_sharp, weights=blur_mask, results=None)
                losses.update(results)


                for loss_name, loss_value in losses.items():
                    if loss_name not in total_losses_epoch:
                        total_losses_epoch[loss_name] = 0.0
                        num_batches[loss_name] = 0.0
                    if not math.isnan(loss_value):
                        total_losses_epoch[loss_name] += loss_value
                        num_batches[loss_name] += 1

                pbar.update()
                pbar.set_description(f"Epoch {training_step} --Testing") 
                pbar.set_postfix({**tensors_to_scalars(losses)})

        avg_losses_epoch = {k: (v / num_batches[k] if num_batches[k] > 0 else 0) for k, v in total_losses_epoch.items()}
        
        print(avg_losses_epoch)


    def test(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        training_step = load_checkpoint(model, optimizer, None, hparams['work_dir'])      

        self.run(model, optimizer, training_step)


def read_img(img_path):
    img = Image.open(img_path)
    img = np.uint8(np.array(img))
    img = tf.to_tensor(img)
    img.unsqueeze_(0)
    return img


if __name__ == '__main__':
    set_hparams()
    val = OMD_validation()
    val.test() 

# ---------- run the command below to validate OMDNet's deblurring result on OMoBlur test set ------------
# ( please first put the released model in /checkpoints/test1/ )

# CUDA_VISIBLE_DEVICES=0 python validation/post_validation.py --config configs/OMD.yaml --exp_name test1 --reset
