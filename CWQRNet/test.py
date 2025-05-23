import os.path as osp
import logging
import time
import argparse
from collections import OrderedDict

import numpy as np
from options import *
from data import create_dataset, create_dataloader
from models import create_model
from utils import *

#### options
parser = argparse.ArgumentParser()

# parser.add_argument('-opt', type=str, default='./test_IRN_x2.yml', help='Path to options YMAL file.')
# parser.add_argument('-opt', type=str, default='./test_IRN_x3.yml', help='Path to options YMAL file.')
# parser.add_argument('-opt', type=str, default='./test_IRN_x4 - win.yml', help='Path to options YMAL file.')
parser.add_argument('-opt', type=str, default='./test_IRN_x4.yml', help='Path to options YMAL file.')



args = parser.parse_args(args=[])
opt = parse(args.opt, is_train=False)
opt = dict_to_nonedict(opt)

mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base{}'.format(loggerIdx.log_idx))
logger.info(dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    for data in test_loader:
        model.feed_data(data)
        img_path = data['GT_path'][0]
        img_name = osp.splitext(osp.basename(img_path))[0]

        model.test()

        visuals = model.get_current_visuals()

        sr_img = tensor2img(visuals['SR'])  # uint8
        srgt_img = tensor2img(visuals['GT'])  # uint8

        # save images
        save_img_path = osp.join(dataset_dir, img_name + '.png')
        save_img(sr_img, save_img_path)

        save_img_path = osp.join(dataset_dir, img_name + '_GT.png')
        save_img(srgt_img, save_img_path)

        # calculate PSNR and SSIM
        gt_img = tensor2img(visuals['GT'])

        gt_img = gt_img / 255.
        sr_img = sr_img / 255.

        crop_border = opt['crop_border'] if opt['crop_border'] else opt['scale']
        cropped_sr_img = sr_img[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_gt_img = gt_img[crop_border:-crop_border, crop_border:-crop_border, :]

        psnr = calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
        ssim = calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)

        sr_img_y = bgr2ycbcr(sr_img, only_y=True)
        gt_img_y = bgr2ycbcr(gt_img, only_y=True)
        cropped_sr_img_y = sr_img_y[crop_border:-crop_border, crop_border:-crop_border]
        cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
        psnr_y = calculate_psnr(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
        ssim_y = calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
        test_results['psnr_y'].append(psnr_y)
        test_results['ssim_y'].append(ssim_y)

        logger.info(
                    '{:40s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'.
            format(img_name, psnr, ssim, psnr_y, ssim_y))

    # Average PSNR/SSIM results
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
    ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
    ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])

    logger.info(
            '----Average PSNR/SSIM results for {}----\n\tpsnr: {:.6f} db; ssim: {:.6f}.\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}. \n'.format(
            test_set_name, ave_psnr, ave_ssim, ave_psnr_y, ave_ssim_y))