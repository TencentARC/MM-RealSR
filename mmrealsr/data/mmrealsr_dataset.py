import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from torch.utils import data as data

from basicsr.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from .degradations import two_random_mixed_kernels, three_random_mixed_kernels

@DATASET_REGISTRY.register()
class MMRealSRGANDataset(data.Dataset):
    """
    Dataset used for Real-ESRGAN model.
    """

    def __init__(self, opt):
        super(MMRealSRGANDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']
        self.random_scale = opt.get('random_scale', None)
        assert self.random_scale is None or self.random_scale < 1, 'random_scale should smaller than 1.'

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.gt_folder]
            self.io_backend_opt['client_keys'] = ['gt']
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError(f"'dataroot_gt' should end with '.lmdb', but received {self.gt_folder}")
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            if isinstance(self.opt['meta_info'], list):
                self.paths = []
                for meta_info in self.opt['meta_info']:
                    with open(meta_info) as fin:
                        paths = [line.strip() for line in fin]
                        self.paths.extend([os.path.join(self.gt_folder, v) for v in paths])
            else:
                with open(self.opt['meta_info']) as fin:
                    paths = [line.strip() for line in fin]
                    self.paths = [os.path.join(self.gt_folder, v) for v in paths]

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']
        self.betap_range = opt['betap_range']
        self.sinc_prob = opt['sinc_prob']

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

        self.blur_kernel_diff_lower_bound = opt.get('blur_kernel_diff_lower_bound', 0.1)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # -------------------------------- Load gt images -------------------------------- #
        # Shape: (h, w, c); channel order: BGR; image range: [0, 1], float32.
        gt_path = self.paths[index]
        # avoid errors caused by high latency in reading files
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                # change another file to read
                index = random.randint(0, self.__len__())
                gt_path = self.paths[index]
                time.sleep(1)  # sleep 1s for occasional server congestion
            else:
                break
            finally:
                retry -= 1
        img_gt = imfrombytes(img_bytes, float32=True)

        self.random_scale = self.opt.get('random_scale', None)  # for compatibility
        if self.random_scale is not None:
            scale_factor = np.random.uniform(self.random_scale, 1)
            h, w = img_gt.shape[0:2]
            h_target, w_target = h * scale_factor, w * scale_factor
            while (h_target < 400 or w_target < 400) and scale_factor < 1:
                scale_factor = scale_factor + 0.1
                h_target, w_target = h * scale_factor, w * scale_factor
            # resize
            img_gt = cv2.resize(img_gt, (int(w_target), int(h_target)), interpolation=cv2.INTER_LANCZOS4)

        # -------------------- augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt['use_hflip'], self.opt['use_rot'])

        # crop or pad to 400: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = 400
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top:top + crop_pad_size, left:left + crop_pad_size, ...]

        sinc_flag = False  # whether choose sinc filter as kernel1
        diff_flag = False  # whether kernel1_small and kernel_large are actually different

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            # if kernel_size < 13:
            #     omega_c = np.random.uniform(np.pi / 3, np.pi)
            # else:
            #     omega_c = np.random.uniform(np.pi / 5, np.pi)
            # kernel1_small = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            # kernel1_large = kernel1_small.copy()
            # kernel1_max = kernel1_small.copy()
            if kernel_size < 13:
                omega_c1_1 = np.random.uniform(np.pi / 3, np.pi)
                omega_c2_1 = np.random.uniform(np.pi / 3, np.pi)
                while omega_c2_1 == omega_c1_1:
                    omega_c2_1 = np.random.uniform(np.pi / 3, np.pi)
                omega_cmax_1 = np.pi / 3
            else:
                omega_c1_1 = np.random.uniform(np.pi / 5, np.pi)
                omega_c2_1 = np.random.uniform(np.pi / 5, np.pi)
                while omega_c2_1 == omega_c1_1:
                    omega_c2_1 = np.random.uniform(np.pi / 5, np.pi)
                omega_cmax_1 = np.pi / 5

            list_omega_c = [omega_c1_1, omega_c2_1]
            # print('check node - omega1_org: ', list_omega_c)
            list_omega_c.sort()
            omega_c2_1, omega_c1_1 = list_omega_c
            # print('check node - omega1_sorted: ', omega_c1_1, omega_c2_1, omega_c3_1)

            kernel1_small = circular_lowpass_kernel(omega_c1_1, kernel_size, pad_to=False)
            kernel1_large = circular_lowpass_kernel(omega_c2_1, kernel_size, pad_to=False)
            kernel1_max = circular_lowpass_kernel(omega_cmax_1, kernel_size, pad_to=False)

            # sinc_flag = True
        else:
            kernel1_small, kernel1_large, kernel1_max = three_random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
                diff_lower_bound=self.blur_kernel_diff_lower_bound)
            # diff_flag = True
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel1_small = np.pad(kernel1_small, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel1_large = np.pad(kernel1_large, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel1_max = np.pad(kernel1_max, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']: #and not sinc_flag and diff_flag:
            # if kernel_size < 13:
            #     omega_c = np.random.uniform(np.pi / 3, np.pi)
            # else:
            #     omega_c = np.random.uniform(np.pi / 5, np.pi)
            # kernel2_small = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            # kernel2_large = kernel2_small.copy()
            # kernel2_max = kernel2_small.copy()
            if kernel_size < 13:
                omega_c1_2 = np.random.uniform(np.pi / 3, np.pi)
                omega_c2_2 = np.random.uniform(np.pi / 3, np.pi)
                while omega_c2_2 == omega_c1_2:
                    omega_c2_2 = np.random.uniform(np.pi / 3, np.pi)
                omega_cmax_2 = np.pi / 3
            else:
                omega_c1_2 = np.random.uniform(np.pi / 5, np.pi)
                omega_c2_2 = np.random.uniform(np.pi / 5, np.pi)
                while omega_c2_2 == omega_c1_2:
                    omega_c2_2 = np.random.uniform(np.pi / 5, np.pi)
                omega_cmax_2 = np.pi / 5

            list_omega_c = [omega_c1_2, omega_c2_2]
            # print('check node - omega1_org: ', list_omega_c)
            list_omega_c.sort()
            omega_c2_2, omega_c1_2 = list_omega_c
            # print('check node - omega1_sorted: ', omega_c1_1, omega_c2_1, omega_c3_1)

            kernel2_small = circular_lowpass_kernel(omega_c1_2, kernel_size, pad_to=False)
            kernel2_large = circular_lowpass_kernel(omega_c2_2, kernel_size, pad_to=False)
            kernel2_max = circular_lowpass_kernel(omega_cmax_2, kernel_size, pad_to=False)
        else:
            # the prob of kernel1_large > kernel1_small and kernel2_large > kernel2_small is 0.1
            # 0.2, 0.5, 0.1 are hard-coded
            # if not diff_flag or (diff_flag and np.random.uniform() < 0.2):
            kernel2_small, kernel2_large, kernel2_max = three_random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
                diff_lower_bound=self.blur_kernel_diff_lower_bound)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2_small = np.pad(kernel2_small, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel2_large = np.pad(kernel2_large, ((pad_size, pad_size), (pad_size, pad_size)))
        kernel2_max = np.pad(kernel2_max, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- sinc kernel ------------------------------------- #
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel1_small = torch.FloatTensor(kernel1_small)
        kernel1_large = torch.FloatTensor(kernel1_large)
        kernel1_max = torch.FloatTensor(kernel1_max)
        kernel2_small = torch.FloatTensor(kernel2_small)
        kernel2_large = torch.FloatTensor(kernel2_large)
        kernel2_max = torch.FloatTensor(kernel2_max)

        return_d = {
            'gt': img_gt,
            'kernel1_small': kernel1_small,
            'kernel1_large': kernel1_large,
            'kernel1_max': kernel1_max,
            'kernel2_small': kernel2_small,
            'kernel2_large': kernel2_large,
            'kernel2_max': kernel2_max,
            'sinc_kernel': sinc_kernel,
            'gt_path': gt_path
        }
        return return_d

    def __len__(self):
        return len(self.paths)