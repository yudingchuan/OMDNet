import glob
from torchvision.transforms import functional as tf

import os
import torch
import math
from tqdm import tqdm

from utils.hparams import hparams, set_hparams
from models.deblur_model import OMDNet
import numpy as np
import torch.nn.functional as F
from PIL import Image

from utils.record import detach_img, plot_img, load_checkpoint, save_checkpoint
from utils.general import move_to_cuda, tensors_to_scalars

class OMD_demo:
    def __init__(self):

        self.work_dir = hparams['work_dir']
        self.data_dir = 'Real World Object Motion Blur'

    def build_model(self):
        self.model = OMDNet(num_res=20)
        return self.model

    def build_optimizer(self, model):
        return torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0) 


    def run(self, model, optimizer, training_step):       

        torch.set_grad_enabled(False)
        model.eval()       

        test_img_paths = sorted(glob.glob(f'{self.data_dir}/*/*blur.png') + glob.glob(f'{self.data_dir}/*/*blur.jpg')+ glob.glob(f'{self.data_dir}/*/*blur.JPG'))

        pbar = tqdm(test_img_paths, dynamic_ncols=True, leave=False) 

        for batch_idx, path in enumerate(test_img_paths):

            blur_path = path # /Real World Object Motion Blur/img in ReLoBlur/s12_02_73_blur.png

            item_path = "/".join(blur_path.split("/")[-2:])  # img in ReLoBlur/s12_02_73_blur.png
            catagry_dir = blur_path.split("/")[-2:-1][0]

            img_blur = Image.open(blur_path)

            if catagry_dir == 'img in ReLoBlur':
                img_blur = img_blur.resize((int(img_blur.width // 2), int(img_blur.height // 2)), Image.LANCZOS)
            elif catagry_dir == 'img shot on Sony':
                img_blur = img_blur.resize((int(img_blur.width // 3), int(img_blur.height // 3)), Image.LANCZOS)

            img_blur = np.uint8(np.array(img_blur))
            img_blur = tf.to_tensor(img_blur)
            img_blur.unsqueeze_(0)
            # print(img_blur.shape)
            multiple_width = 8 
            if img_blur.shape[2] % multiple_width != 0:
                l_pad = multiple_width - img_blur.shape[2] % multiple_width
                img_blur = F.pad(img_blur, (0, 0, 0, l_pad), mode='reflect')  
            if img_blur.shape[3] % multiple_width != 0:
                l_pad = multiple_width - img_blur.shape[3] % multiple_width
                img_blur = F.pad(img_blur, (0, l_pad, 0, 0), mode='reflect') 

            # print(img_blur.shape)            

            img_blur = move_to_cuda(img_blur)          

            _, deblured, _, gates,_,_,_ = model(img_blur, img_blur,img_blur,train = False)
            
            if not os.path.exists('demo_result/' + catagry_dir):
                    os.makedirs('demo_result/' + catagry_dir)
            plot_img(deblured[0][0]).save("demo_result/" + item_path.replace('blur', 'deblured'))
            plot_img(gates[0][0]).save("demo_result/" + item_path.replace('blur', 'gate'))

            pbar.update()
            pbar.set_description(f"Epoch {training_step} --Testing") 
        print('-----------Done! Check result in /demo_result/. ------------')


    def test(self):
        model = self.build_model()
        optimizer = self.build_optimizer(model)
        training_step = load_checkpoint(model, optimizer, None, hparams['work_dir'])      

        self.run(model, optimizer, training_step)




if __name__ == '__main__':
    set_hparams()
    demo = OMD_demo()
    demo.test() 

# ---------- run the command below to see object motion deblur demo ------------
# ( please first put the released model in /checkpoints/test1/ )

# CUDA_VISIBLE_DEVICES=0 python demo.py --config configs/OMD.yaml --exp_name test1 --reset
