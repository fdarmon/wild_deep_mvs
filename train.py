#  Copyright (c)
#
#  Deep MVS Gone  Wild All rights reseved to Thales LAS and ENPC.
#
#  This code is freely available for academic use only and Provided “as is” without any warranty.
#
#  Modification are allowed for academic research provided that the following conditions are met :
#      * Redistributions of source code or any format must retain the above copyright notice and this list of conditions.
#      * Neither the name of Thales LAS and ENPC nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

#
#  Deep MVS Gone  Wild All rights reseved to Thales LAS and ENPC.
#
#  This code is freely avaible for academic use only and Provided “as is” without any warranties.
#
import torch
import numpy as np
np.random.seed(0)
torch.manual_seed(0)

import argparse
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
from utils.monitor import Logger
import time

from pathlib import Path
from models.trainer import Trainer
from data import dtu_yao, md_yao, blended
from models.MVSNet.model import MVSNet

from models.CVP_MVSNet.frontend import Frontend as CVP_MVSNet
from models.VisMVSNet.frontend import Frontend as Vis_MVSNet

from utils.SharedRandomSampler import SharedRandomSampler

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


import os
os.environ["http_proxy"] = ""

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    setup(rank, world_size)

    if args.dataset == "dtu":
        with open("data/txt/dtu_train.txt") as f:
            train_scenes = [s.strip() for s in f.readlines()]

        with open("data/txt/dtu_val.txt") as f:
            test_scenes = [s.strip() for s in f.readlines()]

        data_p = "datasets/dtu"

        train_dataset = dtu_yao.MVSDataset(data_p, train_scenes, "train", args.num_im_train, return_depth=args.supervised)
        val_dataset = dtu_yao.MVSDataset(data_p, test_scenes, "val", args.num_im_train, return_depth=args.supervised)
        test_dataset = dtu_yao.MVSDataset(data_p, test_scenes, "test", 5)

    elif args.dataset == "md":
        with open("data/txt/md_train.txt") as f:
            train_scenes = [s.strip() for s in f.readlines()]

        with open("data/txt/md_test.txt") as f:
            test_scenes = [s.strip() for s in f.readlines()]

        datadir = Path("datasets/md")

        train_dataset = md_yao.MVSDataset(str(datadir), train_scenes, "train", args.num_im_train, return_depth=args.supervised)
        val_dataset = md_yao.MVSDataset(str(datadir), test_scenes, "val", args.num_im_train, return_depth=args.supervised)
        test_dataset = md_yao.MVSDataset(str(datadir), test_scenes, "test", 5, return_depth=args.supervised)

    elif args.dataset == "blended":
        with open("data/txt/blended_train.txt") as f:
            train_scenes = [s.strip() for s in f.readlines()]

        with open("data/txt/blended_val.txt") as f:
            test_scenes = [s.strip() for s in f.readlines()]

        data_p = Path("datasets/BlendedMVS")

        train_dataset = blended.MVSDataset(data_p, train_scenes, "train", args.num_im_train, return_depth=args.supervised)
        val_dataset = blended.MVSDataset(data_p, test_scenes, "val", args.num_im_train, return_depth=args.supervised)
        test_dataset = blended.MVSDataset(data_p, test_scenes, "test", 5)

    if len(train_dataset) == 0:
        raise FileNotFoundError(args.dataset)

    if args.occ_masking:
        train_sampler = SharedRandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)


    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    print(len(train_dataset))

    TrainImgLoader = DataLoader(train_dataset, args.batch_size, num_workers=8, sampler=train_sampler)
    TestImgLoader = DataLoader(test_dataset, 1, num_workers=4, sampler=test_sampler)
    ValImgLoader = DataLoader(val_dataset, args.batch_size, num_workers=8, sampler=val_sampler)

    if args.architecture == "mvsnet":
        model = MVSNet(aggregation="variance")
    elif args.architecture == "mvsnet-s":
        model = MVSNet(aggregation="softmin")
    elif args.architecture == "vis_mvsnet":
        model = Vis_MVSNet()
    elif args.architecture == "cvp_mvsnet":
        model = CVP_MVSNet()
    else:
        raise NotImplementedError("Architecture " + args.architecture)

    # strange bahaviour, find_unsued_parameters needs to be True for vismvsnet
    model = DDP(model.cuda(), device_ids=[0], find_unused_parameters=args.architecture=="vis_mvsnet")

    trainer = Trainer(model, args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

    logdir = Path("trained_models") / args.logdir

    if rank == 0:
        if not logdir.exists():
            logdir.mkdir(parents=True)

    # load parameters
    start_epoch = 0
    if args.resume:
        saved_models = [fn.name for fn in logdir.iterdir() if fn.name.endswith(".ckpt")]
        saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        # use the latest checkpoint file
        loadckpt = logdir / saved_models[-1]
        print("resuming", loadckpt)
        state_dict = torch.load(loadckpt, map_location='cuda:0')
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1

    elif args.loadckpt:
        # load checkpoint file specified by args.loadckpt
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(Path("trained_models") / args.loadckpt, map_location='cuda:0')
        model.load_state_dict(state_dict['model'])

    if rank == 0:
        print("start at epoch {}".format(start_epoch))
        logger = Logger(logdir)

    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,
                                                        last_epoch=start_epoch - 1)

    epoch_list = range(start_epoch, args.epochs)
    for epoch_idx in epoch_list:
        train_sampler.set_epoch(epoch_idx)

        if rank == 0:
            print('Epoch {}:'.format(epoch_idx))

        # training
        model.train()

        for batch_idx, sample in enumerate(TrainImgLoader):
            do_summary = (batch_idx + 1) % args.print_every == 0

            optimizer.zero_grad()
            loss = trainer.step(sample, train=True)
            loss.backward()
            optimizer.step()
            if args.debug:
                break

            if do_summary:
                losses = trainer.log_iter()
                if rank == 0:
                    print('Epoch {}/{}, Iter {}/{}'.format(epoch_idx, args.epochs, batch_idx, len(TrainImgLoader)))
                    logger.plot_ims(trainer.ims)
                    print(losses)

        if epoch_idx % args.save_freq == 0:
            # checkpoint
            if rank == 0:
                torch.save({
                    'epoch': epoch_idx,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'architecture': args.architecture
                }, logdir / f"model_{epoch_idx:0>6}.ckpt")

            # testing
            # first get the training results
            train_losses = trainer.log_epoch(epoch_idx)
            model.eval()

            for batch_idx, sample in enumerate(ValImgLoader):

                start_time = time.time()
                with torch.no_grad():
                    trainer.step(sample, train=False)
                if (batch_idx + 1) % args.print_every == 0:
                    if rank == 0:
                        print('Validation, Iter {}/{}, time = {:3f}'.format(batch_idx, len(ValImgLoader),
                                                                         time.time() - start_time))

                if args.debug:
                    break

            val_losses = trainer.log_epoch(epoch_idx)

            for batch_idx, sample in enumerate(TestImgLoader):
                start_time = time.time()
                trainer.test(sample)
                if (batch_idx + 1) % args.print_every == 0:
                    if rank == 0:
                        print('Testing, Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                                         time.time() - start_time))


            test_losses = trainer.log_epoch(epoch_idx)

            if rank == 0:
                logger.log(train_losses)
                logger.log(val_losses)
                logger.log(test_losses)

            torch.cuda.empty_cache()

        lr_scheduler.step()

    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument('--dataset', choices=['dtu', 'md', 'blended'], help='training dataset')
    parser.add_argument("--debug", action="store_true", help="Single train and val iteration to test the whole code")

    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lrepochs', type=str, default="13:10",
                        help='epoch ids to downscale lr and the downscale rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

    parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
    parser.add_argument("--num_im_train", type=int, help="Number of image used for training", default=3)
    parser.add_argument("--architecture", choices=["mvsnet", "mvsnet-s", "cvp_mvsnet", "vis_mvsnet"], default="mvsnet")

    parser.add_argument("--upsample_training", action="store_true", dest="upsample_training",
                        help="Upsample the output to input resolution before computing the loss")
    parser.add_argument("--no_upsample_training", action="store_false", dest="upsample_training",
                        help="Do not perform upsampling")
    parser.set_defaults(upsample_training=False)

    parser.add_argument("--occ_masking", action="store_true", help="Detect and mask occlusion for unsupervised training")
    parser.add_argument("--geom_clamping", type=float, default=0.05, help="Geometric threshold for masking occlused pixels")
    parser.add_argument("--supervised", action="store_true", dest="supervised",
                        help="Train with depth supervision")
    parser.add_argument("--unsupervised", action="store_false", dest="supervised",
                        help="Train without depth supervision")
    parser.set_defaults(supervised=True)

    parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
    parser.add_argument('--logdir', default='./debug', help='the directory to save checkpoints/logs')

    parser.add_argument('--resume', action='store_true', help='continue to train the model')

    parser.add_argument('--print_every', type=int, default=20, help='print and summary frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='run test and save checkpoint frequency')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument("--world_size", default=1, type=int, help="Number of gpu to use. (When using occlusion masking, it must be the same as num_im_train")

    # parse arguments and check
    args = parser.parse_args()
    if args.resume:
        assert args.loadckpt is None

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)


    if args.supervised:
        args.occ_masking=False
        if args.dataset == "dtu" and not args.upsample_training:
            print("Error: cannot feed full resolution image to network for dtu since only x4 downsampled gt")
            exit(1)

    if args.occ_masking:
        assert args.num_im_train == args.world_size
    print(args)

    mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size, join=True)
