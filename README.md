# OMoBlur: An Object Motion Blur Dataset and Benchmark for Real-World Local Motion Deblurring

[![Project Page](https://img.shields.io/badge/🚀_Project_Page-EXPLORE_NOW!-FF8C00.svg?style=for-the-badge)](https://yudingchuan.github.io/OMoBlur_homepage/)

[![arXiv](https://img.shields.io/badge/Paper-arXiv-B31B1B.svg?style=flat)](https://arxiv.org/)
[![CVPR 2026](https://img.shields.io/badge/Paper-CVPR_2026-007EC6.svg?style=flat)](https://cvpr.thecvf.com/)

## 📰 News
- **[2026/03]** OMoBlur dataset will be officially released (see <a href="https://yudingchuan.github.io/OMoBlur_homepage/" target="_blank">Project Page</a>).
- **[2026/02]** The code for OMDNet is now open-source. 
- **[2026/02]** Our paper has been accepted by CVPR 2026.

---

## 🛠️ OMDNet Implementation Guide

### 1. Environment
Please ensure you have Python and PyTorch installed (we use Python 3.12), and
```bash
pip install -r requirements.txt

```

### 2. Demo

We have provided several representative real-world blurred images (from the <a href="https://leiali.github.io/ReLoBlur_homepage/" target="_blank">ReLoBlur dataset</a> and Sony camera) in the `Real World Object Motion Blur/` directory.

1. **Checkpoint**: Download the released <a href="https://drive.google.com/file/d/1duqA86H3xiEjupJ6qtGOGo3eFFWAIOH3/view?usp=sharing" target="_blank">model checkpoint</a> and place it inside the `checkpoints/test1/` folder.
2. **Run Demo**: Execute the following command:
```bash
CUDA_VISIBLE_DEVICES=0 python demo.py --config configs/OMD.yaml --exp_name test1 --reset

```
The deblurred outputs will be saved in the `demo_result/` folder.

### 3. Training

**Dataset Structure:**
To ensure the dataloader correctly parses the paths, the OMoBlur dataset must be organized strictly into `image_pairs` and `image_sequence` as shown below:

```text
OMoBlur_Dataset/
├── train/
│   ├── image_pairs/
│   │   └── <scene_dir>/          # e.g., 01
│   │       └── <capture_dir>/    # e.g., 0015
│   │           ├── 008_11_b.png  # Blurry image
│   │           ├── 008_11_m.png  # Blur mask
│   │           └── 008_1.png     # Ground truth sharp mid-frame
│   └── image_sequence/
│       └── <scene_dir>/ 
│           └── <capture_dir>/ 
│               ├── 003.png       # First frame within the blur interval (8-11//2)
│               └── 013.png       # Last frame within the blur interval (8+11//2)
└── val/
    ├── image_pairs/ ...
    └── image_sequence/ ...

```

**Start Training:**
Once the dataset is prepared, run the following command to train OMDNet on the OMoBlur dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/OMD.yaml --exp_name trainOMD --reset

```

**Monitoring:**
You can monitor the loss, metrics, and visual results via Tensorboard:

```bash
tensorboard --logdir=./checkpoints/trainOMD/tb_logs --port=6037

```

---

## 📖 Citation

If you find the OMoBlur dataset or OMDNet helpful for your research, please cite our paper. Thanks 🥰

```bibtex
@inproceedings{yu2026omoblur,
  title     = {OMoBlur: An Object Motion Blur Dataset and Benchmark for Real-World Local Motion Deblurring},
  author    = {Yu, Dingchuan and Li, Jiatong and Zhou, Jingwen and Zhuge, Zhengyue and Chen, Yueting and Li, Qi},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
