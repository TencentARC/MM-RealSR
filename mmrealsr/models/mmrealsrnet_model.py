import cv2
import numpy as np
import os
import random
import torch
from collections import OrderedDict
from os import path as osp
from torch import distributed as dist
from torch.nn import functional as F
from tqdm import tqdm
import torch.nn as nn

from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import DiffJPEG, USMSharp, get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.img_process_util import filter2D
from basicsr.utils.registry import MODEL_REGISTRY
from mmrealsr.data.degradations import custom_random_add_gaussian_noise_pt, custom_random_add_poisson_noise_pt, \
    custom_random_add_gaussian_noise_pt_anchor, custom_random_add_poisson_noise_pt_anchor
from mmrealsr.data.transforms import paired_random_crop
from basicsr.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt


@MODEL_REGISTRY.register()
class MMRealSRNetModel(SRModel):
    """RealESRNet Model"""

    def __init__(self, opt):
        super(MMRealSRNetModel, self).__init__(opt)
        self.jpeger = DiffJPEG(differentiable=False).cuda()
        self.usm_sharpener = USMSharp().cuda()
        self.queue_size = opt.get('queue_size', 180)
        self.gaussian_noise_diff_range = opt.get('gaussian_noise_diff_range', (1, 21))
        self.poisson_noise_diff_range = opt.get('poisson_noise_diff_range', (0.05, 2.05))
        self.idx = 1

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        # training pair pool
        # initialize
        b, _, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, 'queue size should be divisible by batch size'
            self.queue_lr = torch.zeros(self.queue_size, 3, c, h, w).cuda()
            self.queue_anchor = torch.zeros(self.queue_size, 2, c, h, w).cuda()
            c, h, w = self.gt.size()[-3:]
            self.queue_gt = torch.zeros(self.queue_size, 5, c, h, w).cuda()
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_anchor = self.queue_anchor[idx]
            self.queue_gt = self.queue_gt[idx]
            # get
            lq_dequeue = self.queue_lr[0:b, :, :, :, :].clone()
            anchor_dequeue = self.queue_anchor[0:b, :, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :, :].clone()
            # update
            self.queue_lr[0:b, :, :, :, :] = self.lq.clone()
            self.queue_anchor[0:b, :, :, :, :] = self.anchor.clone()
            self.queue_gt[0:b, :, :, :, :] = self.gt.clone()

            self.lq = lq_dequeue
            self.anchor = anchor_dequeue
            self.gt = gt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :, :] = self.lq.clone()
            self.queue_anchor[self.queue_ptr:self.queue_ptr + b, :, :, :, :] = self.anchor.clone()
            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :, :] = self.gt.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            # USM the GT images
            if self.opt['gt_usm'] is True:
                self.gt = self.usm_sharpener(self.gt)

            self.kernel1_small = data['kernel1_small'].to(self.device)
            self.kernel1_large = data['kernel1_large'].to(self.device)
            self.kernel1_max = data['kernel1_max'].to(self.device)
            self.kernel2_small = data['kernel2_small'].to(self.device)
            self.kernel2_large = data['kernel2_large'].to(self.device)
            self.kernel2_max = data['kernel2_max'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)
            ori_h, ori_w = self.gt.size()[2:4]

            # ----------------------- The first degradation process ----------------------- #
            # blur
            out_origin = filter2D(self.gt, self.kernel1_small)
            out_blur = filter2D(self.gt, self.kernel1_large)
            out_blur_max = filter2D(self.gt, self.kernel1_max)
            outs = [out_origin, out_blur, out_blur_max]
            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = [np.random.uniform(1, self.opt['resize_range'][1]),
                         np.random.uniform(1, self.opt['resize_range'][1])]
                scale.sort()
                scale_small, scale_large = scale
                scale_max = self.opt['resize_range'][1]
            elif updown_type == 'down':
                scale = [np.random.uniform(self.opt['resize_range'][0], 1),
                         np.random.uniform(self.opt['resize_range'][0], 1)]
                scale.sort()
                scale_large, scale_small = scale
                scale_max = self.opt['resize_range'][0]
            else:
                scale_small = 1
                scale_large = 1
                scale_max = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            outs[0] = F.interpolate(outs[0], scale_factor=scale_small, mode=mode)
            outs[1] = F.interpolate(outs[1], scale_factor=scale_large, mode=mode)
            outs[2] = F.interpolate(outs[2], scale_factor=scale_max, mode=mode)
            outs.append(outs[0].clone().detach())
            # noise
            gray_noise_prob = self.opt['gray_noise_prob']
            first_stage_noise = np.random.uniform() < 0.5
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                outs = custom_random_add_gaussian_noise_pt_anchor(
                    outs,
                    sigma_range=self.opt['noise_range'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    use_diff_noise=first_stage_noise,
                    gaussian_noise_diff_range=self.gaussian_noise_diff_range)
            else:
                outs = custom_random_add_poisson_noise_pt_anchor(
                    outs,
                    scale_range=self.opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                    use_diff_noise=first_stage_noise,
                    poisson_noise_diff_range=self.poisson_noise_diff_range)

            # JPEG compression
            sample_diff_abs = torch.rand(
                outs[0].size(0), dtype=outs[0].dtype, device=outs[0].device) * (
                                      self.opt['jpeg_diff_range'][1] - self.opt['jpeg_diff_range'][0]) \
                              + self.opt['jpeg_diff_range'][0]
            jpeg_p_large = torch.rand(
                outs[0].size(0), dtype=outs[0].dtype,
                device=outs[0].device) * (self.opt['jpeg_range'][1] - self.opt['jpeg_range'][0] - sample_diff_abs) + \
                           self.opt['jpeg_range'][0]
            jpeg_p_small = jpeg_p_large + sample_diff_abs
            jpeg_p_max = torch.ones(outs[0].size(0)).type_as(outs[0]) * self.opt['jpeg_range'][0]
            outs = [torch.clamp(out, 0, 1) for out in outs]
            outs[0] = self.jpeger(outs[0], quality=jpeg_p_small.clone())
            outs[1] = self.jpeger(outs[1], quality=jpeg_p_small.clone())
            outs[2] = self.jpeger(outs[2], quality=jpeg_p_max.clone())
            outs[3] = self.jpeger(outs[3], quality=jpeg_p_large.clone())

            # ----------------------- The second degradation process ----------------------- #
            # blur
            outs[0] = filter2D(outs[0], self.kernel2_small)
            outs[1] = filter2D(outs[1], self.kernel2_large)
            outs[2] = filter2D(outs[2], self.kernel2_max)
            outs[3] = filter2D(outs[3], self.kernel2_small)

            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            outs = [
                F.interpolate(
                    out,
                    size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)),
                    mode=mode) for out in outs
            ]
            # noise
            use_diff_noise2 = not first_stage_noise or (first_stage_noise and np.random.uniform() < 0.2)
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                outs = custom_random_add_gaussian_noise_pt_anchor(
                    outs,
                    sigma_range=self.opt['noise_range2'],
                    clip=True,
                    rounds=False,
                    gray_prob=gray_noise_prob,
                    use_diff_noise=use_diff_noise2,
                    gaussian_noise_diff_range=self.gaussian_noise_diff_range)
            else:
                outs = custom_random_add_poisson_noise_pt_anchor(
                    outs,
                    scale_range=self.opt['poisson_scale_range2'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False,
                    use_diff_noise=use_diff_noise2,
                    poisson_noise_diff_range=self.poisson_noise_diff_range)

            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                outs = [
                    F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    for out in outs
                ]
                outs = [filter2D(out, self.sinc_kernel) for out in outs]
                # JPEG compression
                sample_diff_abs = torch.rand(
                    outs[0].size(0), dtype=outs[0].dtype, device=outs[0].device) * (
                                          self.opt['jpeg_diff_range'][1] - self.opt['jpeg_diff_range'][0]) \
                                  + self.opt['jpeg_diff_range'][0]
                jpeg_p_large = torch.rand(
                    outs[0].size(0), dtype=outs[0].dtype,
                    device=outs[0].device) * (self.opt['jpeg_range'][1] - self.opt['jpeg_range'][0] - sample_diff_abs) + \
                               self.opt['jpeg_range'][0]
                jpeg_p_small = jpeg_p_large + sample_diff_abs
                jpeg_p_max = torch.ones(outs[0].size(0)).type_as(outs[0]) * self.opt['jpeg_range'][0]
                outs = [torch.clamp(out, 0, 1) for out in outs]
                outs[0] = self.jpeger(outs[0], quality=jpeg_p_small.clone())
                outs[1] = self.jpeger(outs[1], quality=jpeg_p_small.clone())
                outs[2] = self.jpeger(outs[2], quality=jpeg_p_max.clone())
                outs[3] = self.jpeger(outs[3], quality=jpeg_p_large.clone())
            else:
                # JPEG compression
                sample_diff_abs = torch.rand(
                    outs[0].size(0), dtype=outs[0].dtype, device=outs[0].device) * (
                                          self.opt['jpeg_diff_range'][1] - self.opt['jpeg_diff_range'][0]) \
                                  + self.opt['jpeg_diff_range'][0]
                jpeg_p_large = torch.rand(
                    outs[0].size(0), dtype=outs[0].dtype,
                    device=outs[0].device) * (self.opt['jpeg_range'][1] - self.opt['jpeg_range'][0] - sample_diff_abs) + \
                               self.opt['jpeg_range'][0]
                jpeg_p_small = jpeg_p_large + sample_diff_abs
                jpeg_p_max = torch.ones(outs[0].size(0)).type_as(outs[0]) * self.opt['jpeg_range'][0]
                outs = [torch.clamp(out, 0, 1) for out in outs]
                outs[0] = self.jpeger(outs[0], quality=jpeg_p_small.clone())
                outs[1] = self.jpeger(outs[1], quality=jpeg_p_small.clone())
                outs[2] = self.jpeger(outs[2], quality=jpeg_p_max.clone())
                outs[3] = self.jpeger(outs[3], quality=jpeg_p_large.clone())
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                outs = [
                    F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    for out in outs
                ]
                outs = [filter2D(out, self.sinc_kernel) for out in outs]

            # clamp and round
            if self.opt['gt_usm'] is True:
                anchor_min = self.usm_sharpener(self.gt)   # generate the sharp anchor
            else:
                anchor_min = self.gt
            anchor_min = F.interpolate(anchor_min, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode='bilinear')
            anchor_max = outs[2]
            self.anchor = torch.stack([anchor_min,anchor_max],dim=1)
            self.anchor = torch.clamp((self.anchor * 255.0).round(), 0, 255) / 255.

            outs = [outs[0], outs[1], outs[3]]
            out = torch.stack(outs, dim=1)
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

            # random crop
            gt_size = self.opt['gt_size']
            lq_size = gt_size // self.opt['scale']
            b_gt, c_gt, h_gt, w_gt = self.gt.size()
            b_lq, _, c_lq, h_lq, w_lq = self.lq.size()
            self.gt = self.gt.unsqueeze(1).repeat(1, 5, 1, 1, 1)
            # print(self.anchor.shape)
            self.gt, (self.lq, self.anchor) = paired_random_crop(self.gt, [self.lq, self.anchor], gt_size, self.opt['scale'])
            # print(self.anchor.shape)
            # exit(0)

            # training pair pool
            self._dequeue_and_enqueue()

            self.gt = self.gt.view(b_gt * 5, c_gt, gt_size, gt_size)
            # self.lq = self.lq.view(b_lq * 3, c_lq, lq_size, lq_size)
            # self.anchor = self.anchor.view(b_gt * 2, c_gt, lq_size, lq_size)

        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)

    def init_training_settings(self):
        # define task specific criterion
        if self.opt['train'].get('rank_opt'):
            self.cri_ranking = build_loss(self.opt['train']['rank_opt']).to(self.device)
            self.cri_ranking_b = build_loss(self.opt['train']['rank_opt_b']).to(self.device)
        else:
            self.cri_ranking = None
        if self.opt['train'].get('constraint_opt'):
            self.cri_constraint = build_loss(self.opt['train']['constraint_opt']).to(self.device)
        else:
            self.cri_constraint = None

        super(MMRealSRNetModel, self).init_training_settings()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # the size of self.pred_blur and self.pred_noise is torch.Size([b*3])
        self.output, (self.pred_blur, self.pred_noise) = self.net_g(self.lq, anchor=self.anchor)

        b = self.pred_blur.size(0) // 5
        self.pred_blur = self.pred_blur.view(b, 5)
        self.pred_noise = self.pred_noise.view(b, 5)
        (self.pred_blur, self.pred_anchor_blur), (self.pred_noise, self.pred_anchor_noise) = (self.pred_blur[:,:3], self.pred_blur[:,3:]), (self.pred_noise[:,:3], self.pred_noise[:,3:])

        target = torch.ones_like(self.pred_blur[:, 0])*-1
        target_min = torch.zeros_like(self.pred_anchor_blur[:,0])
        target_max = torch.ones_like(self.pred_anchor_blur[:,0])

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style
        # ranking loss
        if self.cri_ranking:
            l_anchor_min = (torch.mean((self.pred_anchor_blur[:,0]-target_min)**2) + torch.mean((self.pred_anchor_noise[:,0]-target_min)**2))*self.opt['train']['rank_opt_anchor']['loss_weight']
            l_anchor_max = (torch.mean((self.pred_anchor_blur[:, 1] - target_max)**2) + torch.mean((self.pred_anchor_noise[:, 1] - target_max)**2))*self.opt['train']['rank_opt_anchor']['loss_weight']
            # print(self.pred_blur[:, 0], self.pred_blur[:, 1])
            l_ranking_blur = self.cri_ranking_b(self.pred_blur[:, 0], self.pred_blur[:, 1], target)
            # l_ranking_blur = self.cri_ranking(self.pred_blur[:, 0], self.pred_blur[:, 1], target)
            l_ranking_noise = self.cri_ranking(self.pred_noise[:, 0], self.pred_noise[:, 2], target)
            l_total += l_ranking_blur
            l_total += l_ranking_noise
            l_total += l_anchor_min
            l_total += l_anchor_max
            loss_dict['l_ranking_blur'] = l_ranking_blur
            loss_dict['l_ranking_noise'] = l_ranking_noise
            loss_dict['l_anchor_min'] = l_anchor_min
            loss_dict['l_anchor_max'] = l_anchor_max
        # (ranking) constraint loss
        if self.cri_constraint:
            l_constraint_blur = self.cri_constraint(
                self.pred_blur[:, 0],
                self.pred_blur[:, 2],
            )
            l_constraint_noise = self.cri_constraint(self.pred_noise[:, 0], self.pred_noise[:, 1])
            l_total += l_constraint_blur
            l_total += l_constraint_noise
            loss_dict['l_constraint_blur'] = l_constraint_blur
            loss_dict['l_constraint_noise'] = l_constraint_noise

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(dataloader, current_iter, tb_logger, save_img)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.is_train = False
        """dist_test actually, no gt, no metrics"""
        assert self.opt['datasets']['val']['type'] == 'SingleImageDataset'
        rank, world_size = get_dist_info()

        num_save_per_dimg = 9
        num_pad = (world_size - (num_save_per_dimg % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=num_save_per_dimg, unit='case')
            os.makedirs(osp.join(self.opt['path']['visualization'], str(current_iter), 'noise'), exist_ok=True)
            os.makedirs(osp.join(self.opt['path']['visualization'], str(current_iter), 'blur'), exist_ok=True)
            os.makedirs(osp.join(self.opt['path']['visualization'], str(current_iter), 'origin'), exist_ok=True)

        if self.opt['dist']:
            dist.barrier()

        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)
        for i in range(rank, num_save_per_dimg + num_pad, world_size):
            idx = min(i, num_save_per_dimg - 1)
            if idx == 0:
                custom_degradation_degrees = (None, None)
                save_folder_path = osp.join(self.opt['path']['visualization'], str(current_iter), 'origin')
                suffix = ''
            elif idx < 5:
                custom_degradation_degrees = (0.05 + 0.3 * (idx - 1), None)
                save_folder_path = osp.join(self.opt['path']['visualization'], str(current_iter), 'blur')
                suffix = f'{idx}_{custom_degradation_degrees[0]}'
            else:
                custom_degradation_degrees = (None, 0.05 + 0.3 * (idx - 5))
                save_folder_path = osp.join(self.opt['path']['visualization'], str(current_iter), 'noise')
                suffix = f'{idx}_{custom_degradation_degrees[1]}'

            for j, val_data in enumerate(dataloader):
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
                self.feed_data(val_data)
                self.test(custom_degradation_degrees=custom_degradation_degrees)
                visuals = self.get_current_visuals()
                sr_img = tensor2img([visuals['result']])

                # tentative for out of GPU memory
                del self.lq
                del self.output
                torch.cuda.empty_cache()

                if i < num_save_per_dimg:
                    save_scale = self.opt.get('savescale', 2)
                    net_scale = self.opt.get('scale')
                    if save_scale != net_scale:
                        h, w = sr_img.shape[0:2]
                        sr_img = cv2.resize(
                            sr_img, (w // net_scale * save_scale, h // net_scale * save_scale),
                            interpolation=cv2.INTER_LANCZOS4)

                    if save_img:
                        img_path = osp.join(save_folder_path, f'{img_name}_{suffix}.png')
                        imwrite(sr_img, img_path)

            # progress bar
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)

        if rank == 0:
            pbar.close()
        self.is_train = True

    def test(self, custom_degradation_degrees=(None, None)):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output, _ = self.net_g_ema(self.lq, custom_degradation_degrees)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output, _ = self.net_g(self.lq, custom_degradation_degrees)
            self.net_g.train()