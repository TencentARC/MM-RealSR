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


# opt_path = 'options/val.yml'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default='options/val.yml')
    parser.add_argument('--model_path', type=str, default='experiments/pretrained_models/MMRealSRGAN.pth')
    parser.add_argument('--im_path', type=str, default='/group/30042/chongmou/ft_local/Real-ESRGAN-master/testdata_real/AIM19/valid-input-noisy')
    parser.add_argument('--res_path', type=str, default='results/aim19')
    args = parser.parse_args()

    with open(args.opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    try:
        os.makedirs(args.res_path)
    except:
        pass
    model = MMRRDBNet_test(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4, num_degradation=2, num_feats=[64, 64, 64, 128], num_blocks=[2, 2, 2, 2], downscales=[1, 1, 2, 1])
    loadnet = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(loadnet['params'], strict=True)
    model.to('cuda:0')
    model.eval()

    im_list = os.listdir(args.im_path)
    im_list.sort()
    im_list = [name for name in im_list if name.endswith('.png')]

    with torch.no_grad():
        for name in im_list:
            path = os.path.join(args.im_path, name)
            im = cv2.imread(path)
            im = img2tensor(im)
            im = im.unsqueeze(0).cuda(0)/255.
            sr, score = model(im, (None, None))

            im_sr = tensor2img(sr, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1))
            save_path = os.path.join(args.res_path, name.split('.')[0]+'_out.png')
            cv2.imwrite(save_path, im_sr)
            print(save_path)
