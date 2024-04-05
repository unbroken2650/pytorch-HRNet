import argparse
import os
import pprint

import logging
import time
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter

import _init_paths
import models
from config import config
from config import update_config
from core.criterion import CrossEntropy, OhemCrossEntropy
from core.function import train, validate
from utils.modelsummary import get_model_summary
from utils.utils import create_logger, FullModel


def parse_args():
    parser = argparse.ArgumentParser(description='Train')

    parser.add_argument('--cfg', required=True, type=str)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


def get_sampler(dataset):
    from utils.distributed import is_distributed
    if is_distributed():
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(dataset)
    else:
        return None


def main():
    args = parse_args()

    # log
    logger, final_output_dir, tb_log_dir = create_logger(config, args.cfg, 'train')
    logger.info(pprint.pformat(args))
    logger.info(config)

    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)
    distributed = args.local_rank >= 0
    if distributed:
        device = torch.device('cuda:{}'.format(args.local_rank))
        torch.cuda.set_device(device)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://",
        )

    # build model
    model = eval('models.'+config.MODEL.NAME + '.get_model')(config)

    # copy model file
    if distributed and args.local_rank == 0:
        this_dir = os.path.dirname(__file__)
        models_dst_dir = os.path.join(final_output_dir, 'models')

    # batch size
    batch_size = config.TRAIN.BATCH_SIZE_PER_GPU

    # prepare data
    train_transform = transforms.Compose([
        # 여기에 필요한 전처리 단계 추가
        transforms.Resize((config.TRAIN.IMAGE_SIZE[0], config.TRAIN.IMAGE_SIZE[1])),
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        # 여기에 필요한 전처리 단계 추가
        transforms.Resize((config.TEST.IMAGE_SIZE[0], config.TEST.IMAGE_SIZE[1])),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(
        root=config.DATASET.ROOT + '/train2017',
        transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        root=config.DATASET.ROOT + '/test2017',
        transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.WORKERS
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.WORKERS
    )

    # criterion
    if config.LOSS.USE_OHEM:
        criterion = OhemCrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL, thres=config.LOSS.OHEMTHRES,
                                     min_kept=config.LOSS.OHEMKEEP, weight=train_dataset.class_weights)
    else:
        criterion = CrossEntropy(ignore_label=config.TRAIN.IGNORE_LABEL, weight=train_dataset.class_weights)

    model = FullModel(model, criterion)

    if distributed:
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=True,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )
    else:
        model = nn.DataParallel(model, device_ids=gpus).cuda()

    # optimizer
    if config.TRAIN.OPTIMIZER == 'sgd':

        params_dict = dict(model.named_parameters())
        if config.TRAIN.NONBACKBONE_KEYWORDS:
            bb_lr = []
            nbb_lr = []
            nbb_keys = set()
            for k, param in params_dict.items():
                if any(part in k for part in config.TRAIN.NONBACKBONE_KEYWORDS):
                    nbb_lr.append(param)
                    nbb_keys.add(k)
                else:
                    bb_lr.append(param)
            print(nbb_keys)
            params = [{'params': bb_lr, 'lr': config.TRAIN.LR}, {
                'params': nbb_lr, 'lr': config.TRAIN.LR * config.TRAIN.NONBACKBONE_MULT}]
        else:
            params = [{'params': list(params_dict.values()), 'lr': config.TRAIN.LR}]

        optimizer = torch.optim.SGD(params,
                                    lr=config.TRAIN.LR,
                                    momentum=config.TRAIN.MOMENTUM,
                                    weight_decay=config.TRAIN.WD,
                                    nesterov=config.TRAIN.NESTEROV,
                                    )
    else:
        raise ValueError('Only Support SGD optimizer')

    epoch_iters = np.int(train_dataset.__len__() /
                         config.TRAIN.BATCH_SIZE_PER_GPU / len(gpus))

    best_mIoU = 0
    last_epoch = 0
    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'checkpoint.pth.tar')
        if os.path.isfile(model_state_file):
            checkpoint = torch.load(model_state_file, map_location={'cuda:0': 'cpu'})
            best_mIoU = checkpoint['best_mIoU']
            last_epoch = checkpoint['epoch']
            dct = checkpoint['state_dict']

            model.module.model.load_state_dict(
                {k.replace('model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('model.')})
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint (epoch {})"
                        .format(checkpoint['epoch']))
        if distributed:
            torch.distributed.barrier()

    start = timeit.default_timer()
    # end_epoch = config.TRAIN.END_EPOCH + config.TRAIN.EXTRA_EPOCH
    # num_iters = config.TRAIN.END_EPOCH * epoch_iters
    # extra_iters = config.TRAIN.EXTRA_EPOCH * extra_epoch_iters

    # for epoch in range(last_epoch, end_epoch):

    #     current_trainloader = extra_trainloader if epoch >= config.TRAIN.END_EPOCH else trainloader
    #     if current_trainloader.sampler is not None and hasattr(current_trainloader.sampler, 'set_epoch'):
    #         current_trainloader.sampler.set_epoch(epoch)

    #     # valid_loss, mean_IoU, IoU_array = validate(config,
    #     #             testloader, model, writer_dict)

    #     if epoch >= config.TRAIN.END_EPOCH:
    #         train(config, epoch-config.TRAIN.END_EPOCH,
    #               config.TRAIN.EXTRA_EPOCH, extra_epoch_iters,
    #               config.TRAIN.EXTRA_LR, extra_iters,
    #               extra_trainloader, optimizer, model, writer_dict)
    #     else:
    #         train(config, epoch, config.TRAIN.END_EPOCH,
    #               epoch_iters, config.TRAIN.LR, num_iters,
    #               trainloader, optimizer, model, writer_dict)

    #     valid_loss, mean_IoU, IoU_array = validate(config,
    #                                                testloader, model, writer_dict)

    #     if args.local_rank <= 0:
    #         logger.info('=> saving checkpoint to {}'.format(
    #             final_output_dir + 'checkpoint.pth.tar'))
    #         torch.save({
    #             'epoch': epoch+1,
    #             'best_mIoU': best_mIoU,
    #             'state_dict': model.module.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #         }, os.path.join(final_output_dir, 'checkpoint.pth.tar'))
    #         if mean_IoU > best_mIoU:
    #             best_mIoU = mean_IoU
    #             torch.save(model.module.state_dict(),
    #                        os.path.join(final_output_dir, 'best.pth'))
    #         msg = 'Loss: {:.3f}, MeanIU: {: 4.4f}, Best_mIoU: {: 4.4f}'.format(
    #             valid_loss, mean_IoU, best_mIoU)
    #         logging.info(msg)
    #         logging.info(IoU_array)

    if args.local_rank <= 0:

        torch.save(model.module.state_dict(),
                   os.path.join(final_output_dir, 'final_state.pth'))

        writer_dict['writer'].close()
        end = timeit.default_timer()
        logger.info('Hours: %d' % np.int((end-start)/3600))
        logger.info('Done')


if __name__ == '__main__':
    main()