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
import argparse
import torch
from pathlib import Path
from data import dtu_yao_eval
from data import yfcc_scene


from torch import nn
import numpy as np
from models.VisMVSNet.frontend import Frontend as Vis_MVSNet
from models.CVP_MVSNet.frontend import Frontend as CVP_MVSNet
from models.MVSNet.model import MVSNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["dtu", "yfcc"])
    parser.add_argument("--scene", required=True, type=str)
    parser.add_argument("--model", help="Path to trained model to eval")
    parser.add_argument("--override", action="store_true", help="If set, override existing intermediate results")
    parser.add_argument("--compute_metrics", action="store_true", help="Compute evaluation metrics")
    parser.add_argument("--chunked_eval", action="store_true", help="Used for DTU evaluation, if set, evaluation is "
                                                                    "slower but uses way less memory")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--colmap", action="store_true", help="Evaluates colmap instead of a network")

    # run_depthmaps
    parser.add_argument("--nviews", default=5, type=int, help="Number of view to use for depthmap prediction")
    parser.add_argument("--upsample", action="store_true", help="Upsample depthmaps to the full resolution before "
                                                                "filtering and fusion")

    # filtering
    parser.add_argument("--filter", action="store_true", help="Add prefiltering before fusion")
    parser.add_argument('--depth_threshold', type=float, default=0.01, help="Relative depth difference threshold")
    parser.add_argument('--num_consistent', type=int, default=3, help="Number of consistent views to keep a value")
    parser.add_argument("--max_reproj_error", type=float, default=1, help="Max reproj error in pixels")
    parser.add_argument("--min_tri_angle", type=float, default=1, help="Min triangulation angle in degrees")
    parser.add_argument("--filter_num_views", type=int, default=10, help="Number of image to consider when filtering")

    # fusion
    parser.add_argument("--fusion", choices=["colmap", "fusibile", "simple"])
    parser.add_argument('--fusion_depth_threshold', type=float, default=0.01, help="Depth threshold of fusibile or "
                                                                                   "relative depth threshold of colmap")
    parser.add_argument('--fusion_num_consistent', type=int, default=3, help="Number of consistent image for fusion")
    parser.add_argument("--fusion_max_reproj_error", type=float, default=1, help="Colmap max reprojection error (pixels)")
    parser.add_argument('--prob_threshold', type=float, default=0.8, help="Probability threshold for keeping network "
                                                                          "predictions")

    args = parser.parse_args()

    if args.override:
        args.override_fusion=True

    assert (args.fusion != "fusibile" or args.dataset != "yfcc"), "Fusibile only works with DTU images"
    assert args.model is None or not args.colmap, f"Cannot decide whether {args.model} should be used or colmap"
    assert args.colmap or args.model is not None, f"Provide a model or set colmap flag to eval a model"

    if args.dataset == "dtu":
        args.data_path = "datasets/dtu_eval"
    else:
        args.data_path = "datasets/yfcc_rec"

    args.downscale = 1

    return args


def depth_folder_name(args):
    folder_name = f"{args.model}_{args.nviews}"
    return folder_name


def get_mask(args, filename, **kwargs):
    folder_name = depth_folder_name(args)
    # prob filtering
    if "prob" in kwargs:
        prob = kwargs["prob"]
    else:
        raise NotImplementedError("Need a probability mask from get_mask")

    if len(prob.shape) > 2:
        mask_invalid = (prob < args.prob_threshold).all(axis=0)
    else:
        mask_invalid = prob < args.prob_threshold

    if args.filter:
        if 'geo_mask' in kwargs:
            geo_mask = kwargs["geo_mask"]
        else:
            npz_geom = np.load(Path(args.data_path) / "IntRes" / "geometric_filtering" / (
                        folder_name) / args.scene / f"{filename}_out.npz")
            geo_mask = npz_geom["geo_mask"]

        mask_invalid = mask_invalid | ~geo_mask

    return mask_invalid


def load_network(args):
    p = Path("trained_models") / args.model

    if p.is_dir():
        cur_max = -1
        for f in p.iterdir():
            if f.name.endswith("ckpt"):
                nb = int(f.name.split("_")[-1][:-5])
                if nb > cur_max:
                    cur_max = nb
                    cur_f = f
    else:
        cur_f = p

    loaded = torch.load(cur_f)
    print(f"Load model from {cur_f}")

    architecture = loaded["architecture"]

    if architecture == "cvp_mvsnet":
        net = CVP_MVSNet()
        if args.dataset == "dtu": # large resolution images in DTU
            net.model.nscale = 5
        else:
            net.model.nscale = 4

        downscale = 1

    elif architecture == "vis_mvsnet":
        net = Vis_MVSNet()
        net.depth_nums = [64, 32, 16]
        net.interval_scales = [2, 1, 0.5]
        downscale = 2

    elif architecture == "mvsnet-s":
        net = MVSNet(aggregation="softmin")
        downscale = 4

    else:
        net = MVSNet(aggregation="variance")
        downscale = 4

    net = nn.DataParallel(net)

    device = torch.device("cuda")
    net.to(device)
    net.load_state_dict(loaded["model"], strict=False)
    net.eval()

    return net, downscale


def load_dataset(args):
    if args.dataset == "dtu":
        dataset = dtu_yao_eval.MVSDataset(Path("dtu_eval/images"), [int(args.scene[4:])], "test", args.nviews)
    else:
        dataset = yfcc_scene.MVSDataset(Path("yfcc_rec"), [args.scene], "test", args.nviews)

    return dataset