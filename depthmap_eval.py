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
from data import md_yao, blended
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
from models.utils import *
import time

import json
import argparse
from torch import multiprocessing as mp
from torch import distributed as dist
from evaluation.pipeline_utils import load_network
from torchvision import transforms
from matplotlib import pyplot as plt

import os
os.environ["http_proxy"] = ""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    else:
        print("CUDA_VISIBLE_DEVICES already set, might be a problem in multi-gpu mode")

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def tmp_visu(d, vmin, vmax):
    for k in d:
        if len(d[k].shape) == 4:
            transforms.ToPILImage()(d[k][0].cpu()).save(k + ".jpg")
        else:
            if "mask" in k:
                plt.imshow(d[k][0].cpu(), vmin=0, vmax=1)
            elif "diff" in k:
                plt.imshow(torch.abs(d[k][0].cpu()), vmin=0, vmax=3)
            else:
                plt.imshow(d[k][0].cpu(), vmin=vmin, vmax=vmax)
            plt.axis("off")
            plt.savefig(k + ".jpg", bbox_inches="tight")


def main(rank, world_size, args):
    setup(rank, world_size)


    model, downscale = load_network(args)
    method_name = args.model

    if args.dataset == "blended":
        with open("data/txt/blended_val.txt") as f:
            test_scenes = [s.strip() for s in f.readlines()]

        data_p = Path("datasets/blended")

        test_dataset = blended.MVSDataset(data_p, test_scenes, "test", args.nb_imgs)

    elif args.dataset == "yfcc":
        datadir_te = Path("datasets/yfcc_depthmaps")
        test_scenes = ["trevi_fountain", "sacre_coeur", "taj_mahal", "buckingham_palace", "palace_of_westminster",
                       "brandenburg_gate", "st_peters_square", "hagia_sophia_interior", "pantheon_exterior",
                       "temple_nara_japan", "colosseum_exterior", "notre_dame_front_facade", "prague_old_town_square",
                       "westminster_abbey", "grand_place_brussels"]
        test_dataset = md_yao.MVSDataset(str(datadir_te), test_scenes, "test", args.nb_imgs)
    else:
        raise NotImplementedError(args.dataset)

    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    TestImgLoader = DataLoader(test_dataset, 1, num_workers=4, sampler=test_sampler)

    res = list()
    for batch_idx, sample in enumerate(TestImgLoader):
        cuda_sample = tocuda(sample)
        mask = cuda_sample["mask"][0]
        depth_gt = cuda_sample["depth"][0]

        start_time = time.time()
        with torch.no_grad():
            out = model(sample["imgs"], sample["K"], sample["R"], sample["t"], sample["depth_min"], sample["depth_max"])
            depth_est = out["depth"]
            b, h, w = mask.shape
            # then upsample at org depth size
            depth_est_up = F.interpolate(depth_est.unsqueeze(1), (h, w), mode="bilinear", align_corners=False).squeeze(1)

            if args.debug:
                step_size = (cuda_sample["depth_max"][:, 0] - cuda_sample["depth_min"][:, 0]) / 128

                if rank == 0:
                    d = {"im_ref": sample["imgs"][0]}
                    for i in range(args.nb_imgs - 1):
                        try:
                            d[f"im_src_{i}"] = sample["imgs"][i+1]
                        except IndexError:
                            d[f"im_src_{i}"] = sample["imgs"][:, i + 1]

                    d["pred"] = depth_est_up
                    d["gt"] = depth_gt
                    d["mask_gt"] = mask
                    d["diff"] = (depth_est_up - depth_gt) / step_size

                    vmin, vmax = sample["depth_min"][0, 0], sample["depth_max"][0, 0]
                    tmp_visu(d, vmin, vmax)
                exit(0)

            # normalize by step_size
            step_size = (cuda_sample["depth_max"][:, 0] - cuda_sample["depth_min"][:, 0]) / 128

            depth_est_up /= step_size
            depth_gt /= step_size


            res.append({
                "EPE": AbsDepthError_metrics(depth_est_up, depth_gt, mask > 0.5).detach(),
                "1pxError": Thres_metrics(depth_est_up, depth_gt, mask > 0.5, 1).detach(),
                "3pxError": Thres_metrics(depth_est_up, depth_gt, mask > 0.5, 3).detach(),
                })

        if rank == 0 and (batch_idx + 1) % 20 == 0:
            print('Testing, Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                             time.time() - start_time))
            d = dict()
            d["pred"] = depth_est_up
            d["gt"] = depth_gt
            d["mask_gt"] = mask
            d["diff"] = (depth_est_up - depth_gt) / step_size
            tmp_visu(d, cuda_sample["depth_min"][0, 0], cuda_sample["depth_max"][0, 0])

    keys = res[0].keys()
    to_save = dict()
    for k in keys:
        met = sum([r[k] for r in res])
        dist.reduce(met, dst=0, op=dist.ReduceOp.SUM)

        if rank == 0:
            met = met.item() / len(test_dataset)
            print(f"{k}: {met}")
            to_save[k] = met

    if rank == 0:
        outpath = Path(f"results/{args.dataset}_depthmaps")
        if not outpath.exists():
            outpath.mkdir(parents=True)
        with open(outpath / f"{method_name}_{args.nb_imgs}.json", "w") as f:
            json.dump(to_save, f)


def tmp_visu(d, vmin, vmax):
    for k in d:
        if len(d[k].shape) == 4:
            transforms.ToPILImage()(d[k][0].cpu()).save(k + ".jpg")
        else:
            if "mask" in k:
                plt.imshow(d[k][0].cpu(), vmin=0, vmax=1)
            elif "diff" in k:
                plt.imshow(torch.abs(d[k][0].cpu()), vmin=0, vmax=3)
            else:
                plt.imshow(d[k][0].cpu(), vmin=vmin, vmax=vmax)
            plt.axis("off")
            plt.savefig(k + ".jpg", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["yfcc", "dtu"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--world_size", default=1, type=int, help="Number of gpus")
    parser.add_argument("--nb_imgs", default=5, type=int, help="Number of images to use for inference")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    # add random param to their default value for compatibility with reconstruction_pipeline

    mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size, join=True)

