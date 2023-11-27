import paddle
import os
import math
import argparse
import random
import logging
import time
from data.data_sampler import DistIterSampler
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
import numpy as np


def init_dist(backend='nccl', **kwargs):
    """ initialization for distributed training"""
>>>>>>    if torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
>>>>>>        torch.multiprocessing.set_start_method('spawn')
    rank = int(os.environ['RANK'])
    num_gpus = paddle.device.cuda.device_count()
    paddle.device.set_device(device=rank % num_gpus)
>>>>>>    torch.distributed.init_process_group(backend=backend, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default=
        'none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    if args.launcher == 'none':
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
>>>>>>        world_size = torch.distributed.get_world_size()
        rank = paddle.distributed.get_rank()
    if opt['path'].get('resume_state', None):
        device_id = paddle.framework._current_expected_place()
        resume_state = paddle.load(path=opt['path']['resume_state'])
        option.check_resume(opt, resume_state['iter'])
    else:
        resume_state = None
    if rank <= 0:
        if resume_state is None:
            util.mkdir_and_rename(opt['path']['experiments_root'])
            util.mkdirs(path for key, path in opt['path'].items() if not 
                key == 'experiments_root' and 'pretrain_model' not in key and
                'resume' not in key)
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'
            ], level=logging.INFO, screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'],
            level=logging.INFO, screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(paddle.__version__[0:3])
            if version >= 1.1:
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'
                    .format(version))
                from tensorboardX import SummaryWriter
>>>>>>            tb_logger = torch.utils.tensorboard.SummaryWriter(log_dir=os.
                path.join(opt['path']['root'], 'tb_logger', opt['name']))
    else:
        util.setup_logger('base', opt['path']['log'], 'train', level=
            logging.INFO, screen=True)
        logger = logging.getLogger('base')
    opt = option.dict_to_nonedict(opt)
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)
    False = False
    False = True
    dataset_ratio = 200
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt[
                'batch_size']))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt['dist']:
                train_sampler = DistIterSampler(train_set, world_size, rank,
                    dataset_ratio)
                total_epochs = int(math.ceil(total_iters / (train_size *
                    dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt,
                train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.
                    format(len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.
                    format(total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.
                format(phase))
    assert train_loader is not None
    model = create_model(opt)
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)
    else:
        current_step = 0
        start_epoch = 0
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(
        start_epoch, current_step))
    first_time = True
    for epoch in range(start_epoch, total_epochs + 1):
        if opt['dist']:
            train_sampler.set_epoch(epoch)
        for _, train_data in enumerate(train_loader):
            if first_time:
                start_time = time.time()
                first_time = False
            current_step += 1
            if current_step > total_iters:
                break
            model.feed_data(train_data)
            model.optimize_parameters(current_step)
            model.update_learning_rate(current_step, warmup_iter=opt[
                'train']['warmup_iter'])
            if current_step % opt['logger']['print_freq'] == 0:
                end_time = time.time()
                logs = model.get_current_log()
                message = (
                    '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}, , time:{:.3f}> '
                    .format(epoch, current_step, model.
                    get_current_learning_rate(), end_time - start_time))
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
                start_time = time.time()
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0:
                avg_psnr = 0.0
                avg_normalized_psnr = 0.0
                avg_tonemapped_psnr = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    model.feed_data(val_data)
                    model.test()
                    visuals = model.get_current_visuals()
                    sr_img = util.tensor2numpy(visuals['SR'])
                    gt_img = util.tensor2numpy(visuals['GT'])
                    avg_psnr += util.calculate_psnr(sr_img, gt_img)
                    avg_normalized_psnr += util.calculate_normalized_psnr(
                        sr_img, gt_img, np.max(gt_img))
                    avg_tonemapped_psnr += util.calculate_tonemapped_psnr(
                        sr_img, gt_img, percentile=99, gamma=2.24)
                avg_psnr = avg_psnr / idx
                avg_normalized_psnr = avg_normalized_psnr / idx
                avg_tonemapped_psnr = avg_tonemapped_psnr / idx
                logger.info(
                    '# Validation # PSNR: {:.4e}, norm_PSNR: {:.4e}, mu_PSNR: {:.4e}'
                    .format(avg_psnr, avg_normalized_psnr, avg_tonemapped_psnr)
                    )
                logger_val = logging.getLogger('val')
                logger_val.info(
                    '<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} norm_PSNR: {:.4e} mu_PSNR: {:.4e}'
                    .format(epoch, current_step, avg_psnr,
                    avg_normalized_psnr, avg_tonemapped_psnr))
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('norm_PSNR', avg_normalized_psnr,
                        current_step)
                    tb_logger.add_scalar('mu_PSNR', avg_tonemapped_psnr,
                        current_step)
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)
    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')
        logger.info('End of training.')


if __name__ == '__main__':
    main()
