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
from pathlib import Path
from .pipeline_utils import depth_folder_name
import numpy as np
import torch
from utils.utils_3D import *
from torch.nn import functional as F
from tqdm import tqdm


def run(dataloader, args):
    pc_folder_name = depth_folder_name(args)
    folder_name = depth_folder_name(args)

    out = Path(args.data_path) / "IntRes" / "geometric_filtering" / pc_folder_name / str(args.scene)

    if (out / "finished.txt").exists():
        print("Filtering already done")
        return

    if not out.exists():
        out.mkdir(parents=True)

    depth_folder = Path(args.data_path) / "IntRes" / "depthmaps" / folder_name / str(args.scene)

    for batch in tqdm(dataloader, desc="filtering"):
        filename = batch["filename"][0]
        K, R, t = batch["K"][0], batch["R"][0], batch["t"][0]

        npz = np.load(depth_folder / f"{filename}_out.npz")
        depth = torch.from_numpy(npz["depthmap"])
        h, w = depth.shape
        device = torch.device("cpu")
        src_depth = [torch.from_numpy(np.load(depth_folder / f"{f[0]}_out.npz")["depthmap"]) for f in batch["src_filenames"]]
        src_shape = torch.tensor([[s.shape[0], s.shape[1]] for s in src_depth]) # N * 2

        downscale = 1 if args.upsample else args.downscale
        K[:, :2] /= downscale

        if args.upsample:
            depth = F.interpolate(depth.view(1, 1, h, w), scale_factor=args.downscale).squeeze()
            src_depth = [F.interpolate(d.unsqueeze(0).unsqueeze(0), scale_factor=args.downscale).squeeze() for d in src_depth]
            src_shape *= args.downscale
            h, w = h * args.downscale, w * args.downscale

        with torch.no_grad():
            ref_grid = build_grid(h, w, device, False).squeeze(0).float()  # h * w * 2
            unproj_pc = unproject(ref_grid, K[0], R[0], t[0], depth, invD=False)

            proj_src, proj_depth_in_src = project_all(unproj_pc, K[1:], R[1:], t[1:])

            normalized = normalize(proj_src.unsqueeze(0), src_shape[:, 0], src_shape[:, 1]).squeeze()

            warp_depth_in_src = torch.cat(
                [F.grid_sample(d.unsqueeze(0).unsqueeze(1), normalized[i:i+1], align_corners=False) for i, d in enumerate(src_depth)
                 ], dim=0).squeeze(1)
            reproj, depth_reproj = project(unproj_all(proj_src, K[1:], R[1:], t[1:], warp_depth_in_src), K[0], R[0], t[0])

            reproj_error = reproj - ref_grid
            valid_disp = (torch.norm(reproj_error, dim=-1) < args.max_reproj_error) #& (depth_reproj > 0) & (proj_depth_in_src > 0)

            mask_depth = (torch.abs(depth_reproj - depth) < torch.max(
                depth_reproj, depth) * args.depth_threshold) & (depth_reproj > 0) & (proj_depth_in_src > 0)

            mask_tri = compute_triangulation_angles(unproj_pc, R, t) > args.min_tri_angle
            geo_mask = mask_depth & valid_disp & mask_tri
            mask_depth = mask_depth.sum(axis=0) >= (args.num_consistent - 1)
            mask_disp = valid_disp.sum(axis=0) >= (args.num_consistent - 1)
            geo_mask = geo_mask.sum(axis=0) >= (args.num_consistent - 1)

            np.savez_compressed(out / f"{filename}_out.npz", mask_depth=mask_depth, mask_disp=mask_disp, geo_mask=geo_mask)

            if args.debug:
                return

    with open(out / "finished.txt", "a") as f:
        f.write(" ")
