import glob
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from utils.data_trans import PairRandomHorizontalFilp, PairToTensor, PairCompose
from torchvision.transforms import functional as tf
import torch.nn.functional as F
import pdb
import os
from utils.hparams import hparams

class DeblurDataSet(Dataset):
    def __init__(self, prefix='train'):
        self.hparams = hparams
        self.data_dir = hparams['data_dir'] #
        self.prefix = prefix
        if prefix == 'train':
            self.transform_nocrop = PairCompose(
                [
                    PairRandomHorizontalFilp(),
                    PairToTensor()
                ]
            )
        else:
            self.transform = None

        self.blur_filenames = sorted(glob.glob(f'{self.data_dir}/{self.prefix}/*/*/*/*b.png'))
        self.l = len(self.blur_filenames)

    def __getitem__(self, index):
        blur_path = self.blur_filenames[index] 
        mask_path = blur_path.replace('_b.png','_m.png') 
        filename = os.path.basename(blur_path)  # "008_11_b.png"
        mid_f, f_length = map(int, filename.split('_')[:2])  # 8, 11
        first_frame = mid_f - f_length // 2  # 3
        last_frame = mid_f + f_length // 2  # 13

        first_frame_str = f"{first_frame:03d}.png"  # '003.png'
        last_frame_str = f"{last_frame:03d}.png"  # '013.png'

        gt_path = os.path.join(os.path.dirname(blur_path), blur_path.split('/')[-1].split('_')[0] + '_1.png') 

        first_frame_path = blur_path.replace('image_pairs', 'image_sequence') \
                        .replace(os.path.basename(blur_path), first_frame_str) 
        last_frame_path = blur_path.replace('image_pairs', 'image_sequence') \
                .replace(os.path.basename(blur_path), last_frame_str) 

        dir_scene = os.path.basename( os.path.dirname(os.path.dirname( os.path.dirname(blur_path) ) ))  # 01
        dir_capindex = os.path.basename( os.path.dirname(os.path.dirname(blur_path) )  )    # 0015
        midframe_index = '_'.join(os.path.basename(blur_path).split('_', 2)[:2])        # 008_11
        item_path = os.path.join(dir_scene, dir_capindex, midframe_index) # 01/0015/008_11
        # return item_path
        
        img_gt = Image.open(gt_path) 
        img_gt = np.uint8(np.array(img_gt))
        img_blur = Image.open(blur_path)
        img_blur = np.uint8(np.array(img_blur))
        blur_mask = Image.open(mask_path)
        blur_mask = np.uint8(np.array(blur_mask))

        img_first = Image.open(first_frame_path) 
        img_last = Image.open(last_frame_path) 
        img_first = np.uint8(np.array(img_first))
        img_last = np.uint8(np.array(img_last))


        if len(blur_mask.shape) == 2:
            blur_mask = blur_mask[..., None].repeat(repeats=3, axis=2) 

        if self.prefix == 'train': 
            # modified Blur Aware Patch Cropping strategy
            force_blur_region = \
                self.hparams.get('force_blur_region_p', 0.0) > 0 and \
                random.random() < self.hparams['force_blur_region_p'] 
            ps = self.hparams['patch_size']
            if ps > 0:
                H, W, _ = img_gt.shape
                R_blur = blur_mask.sum() / np.prod(blur_mask.shape) # blur region proportion
                y_s, x_s = random.randint(0, H - 1 - ps), random.randint(0, W - 1 - ps) 

                if force_blur_region and R_blur > 0.05: 
                    for _ in range(20):
                        y_s_, x_s_ = random.randint(0, H - 1 - ps), random.randint(0, W - 1 - ps)  
                        crop = blur_mask[y_s_:y_s_ + ps, x_s_:x_s_ + ps] 
                        ratio = np.mean(crop)
                        if ratio >= 0.3:
                            y_s = y_s_
                            x_s = x_s_
                            break

                img_gt = img_gt[y_s:y_s + ps, x_s:x_s + ps]  # [y_s:y_s + ps, x_s:x_s + ps, : ]
                img_first = img_first[y_s:y_s + ps, x_s:x_s + ps]  # [y_s:y_s + ps, x_s:x_s + ps, : ]
                img_last = img_last[y_s:y_s + ps, x_s:x_s + ps]  #  [y_s:y_s + ps, x_s:x_s + ps, : ]
                img_blur = img_blur[y_s:y_s + ps, x_s:x_s + ps]
                blur_mask = blur_mask[y_s:y_s + ps, x_s:x_s + ps]

            img_gt = Image.fromarray(img_gt.astype(np.uint8)).convert('RGB')
            img_first = Image.fromarray(img_first.astype(np.uint8)).convert('RGB')
            img_last = Image.fromarray(img_last.astype(np.uint8)).convert('RGB')
            img_blur = Image.fromarray(img_blur.astype(np.uint8)).convert('RGB')
            blur_mask = Image.fromarray(blur_mask.astype(np.uint8))
            img_gt, img_blur, blur_mask, img_first, img_last = self.transform_nocrop([img_gt, img_blur, blur_mask, img_first, img_last])
            
        else:
            img_gt, img_blur, blur_mask = tf.to_tensor(img_gt), tf.to_tensor(img_blur), tf.to_tensor(blur_mask)
            img_first, img_last = tf.to_tensor(img_first), tf.to_tensor(img_last)
            multiple_width = self.hparams['multiple_width'] #
            if img_gt.shape[1] % multiple_width != 0:
                l_pad = multiple_width - img_gt.shape[1] % multiple_width
                img_blur = F.pad(img_blur[None, ...], [0, 0, 0, l_pad], mode='reflect')[0]  
                blur_mask = F.pad(blur_mask[None, ...], [0, 0, 0, l_pad], mode='reflect')[0]
            if img_gt.shape[2] % multiple_width != 0:
                l_pad = multiple_width - img_gt.shape[2] % multiple_width
                img_blur = F.pad(img_blur[None, ...], [0, l_pad, 0, 0], mode='reflect')[0]
                blur_mask = F.pad(blur_mask[None, ...], [0, l_pad, 0, 0], mode='reflect')[0]

        sample = {'img_gt': img_gt, 'img_blur': img_blur, 'item_name': item_path, 'img_first':img_first, 'img_last':img_last}
        blur_mask = (blur_mask > 0).float()[:1]  # [0,1]float, [B, 1, H, W]
        blur_mask_nonpad = blur_mask[:, :360, :1920]
        sample['blur_mask'] = blur_mask
        sample['blur_mask_nonpad'] = blur_mask_nonpad
        return sample

    def __len__(self):
        return self.l


if __name__ == "__main__":  
    a = DeblurDataSet('train')
    # print(a[0])
    # print(a[0].shape)

# python dataset.py