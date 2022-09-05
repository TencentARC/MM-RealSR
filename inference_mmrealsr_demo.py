import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
import yaml
from basicsr.utils.options import ordered_yaml
from basicsr.data import build_dataloader, build_dataset
from basicsr.utils.img_process_util import filter2D
import torch.nn.functional as F
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import DiffJPEG

import mmrealsr.archs
import mmrealsr.data
import mmrealsr.models
import cv2
from basicsr.utils.img_util import tensor2img
from basicsr.archs.rrdbnet_arch import RRDBNet
from mmrealsr.archs.mmrealsr_arch import MMRRDBNet_test
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/val.yml')
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/MMRealSRGAN_ModulationBest.pth')
    parser.add_argument('--im_path', type=str, default='imgs/oldphoto6.png')
    args = parser.parse_args()

    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    model = MMRRDBNet_test(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4, num_degradation=2, num_feats=[64, 64, 64, 128], num_blocks=[2, 2, 2, 2], downscales=[1, 1, 2, 1])
    loadnet = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(loadnet['params'], strict=True)
    model.to('cuda:0')
    model.eval()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    list_score = np.arange(0,1,0.01)
    im_path = args.im_path
    im = cv2.imread(im_path)
    h,w,c = im.shape
    try:
        os.makedirs('results/demo/')
    except:
        pass
    video_n = cv2.VideoWriter("results/demo/demo_noise.mp4", fourcc, 10, (w*4,h*4))
    video_b = cv2.VideoWriter("results/demo/demo_blur.mp4", fourcc, 10, (w*4,h*4))
    im = img2tensor(im)
    lq = im.unsqueeze(0).cuda()/255.

    with torch.no_grad():
        for score_cur in list_score:
            print(score_cur)
            sr_n, score = model(lq,(None, score_cur)) # 'None' can be replaced by any number in [0,1]
            sr_b, score = model(lq,(score_cur, None)) # 'None' can be replaced by any number in [0,1]
            im_sr_n = tensor2img(sr_n,rgb2bgr=True, out_type=np.uint8, min_max=(0, 1))
            cv2.putText(im_sr_n, str(score_cur), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            video_n.write(im_sr_n)
            im_sr_b = tensor2img(sr_b,rgb2bgr=True, out_type=np.uint8, min_max=(0, 1))
            cv2.putText(im_sr_b, str(score_cur), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            video_b.write(im_sr_b)
