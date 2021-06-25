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
import sys
sys.path.append("..")
from pathlib import Path
import numpy as np

import torch

from models.utils import tocuda, tensor2numpy
from .pipeline_utils import depth_folder_name, load_network
from tqdm import tqdm

def run(dataloader, args):
    folder_name = depth_folder_name(args)
    out = Path(args.data_path) / "IntRes" / "depthmaps" / folder_name / str(args.scene)

    if not out.exists():
        out.mkdir(parents=True)

    net, downscale = load_network(args)

    args.downscale = downscale

    if (out / "finished.txt").exists() and not args.override:
        print("All the depthmaps are already processed")
        return

    for batch_idx, sample in tqdm(enumerate(dataloader), total=len(dataloader), desc="Run depthmaps"):
        filenames = sample["filename"]
        all_exist = True
        for f in filenames:
            if not (out / f"{f}_out.npz").exists():
                all_exist = False

        if all_exist and not args.override:
            print("\t Depth already computed, continue")
            continue

        sample_cuda = tocuda(sample)

        with torch.no_grad():

            outputs = net(sample_cuda["imgs"], sample_cuda["K"], sample_cuda["R"], sample_cuda["t"],
                          sample_cuda["depth_min"], sample_cuda["depth_max"])

            outputs = tensor2numpy(outputs)
            torch.cuda.empty_cache()

        # save depth maps and confidence maps
        for filename, depth_est, photometric_confidence in zip(filenames, outputs["depth"],
                                                               outputs["photometric_confidence"]):
            # save depth maps
            np.savez_compressed(out / f"{filename}_out.npz", probability=photometric_confidence,
                                depthmap=depth_est)

        if args.debug:
            return

    with open(out / "finished.txt", "a") as f:
        f.write(" ")
