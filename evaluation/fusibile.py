#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
Convert MVSNet output to Gipuma format for post-processing.
"""

from __future__ import print_function

import os
import shutil
from struct import *
from pathlib import Path
from PIL import Image
from torchvision.transforms import ToPILImage
from torch.nn import functional as F
from utils.utils_3D import build_proj_matrices

import numpy as np

import sys
sys.path.append("..")
from evaluation.pipeline_utils import depth_folder_name, get_mask
import torch
from matplotlib import pyplot as plt


def read_gipuma_dmb(path):
    '''read Gipuma .dmb format image'''

    with open(path, "rb") as fid:
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]

        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_gipuma_dmb(path, image):
    '''write Gipuma .dmb format image'''

    image_shape = np.shape(image)

    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()

    with open(path, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)


def write_gipuma_cam(projection_matrix, out_path):
    f = open(out_path, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(projection_matrix[i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()


def fake_gipuma_normal(in_depth_path, out_normal_path):
    depth_image = read_gipuma_dmb(in_depth_path)
    image_shape = np.shape(depth_image)

    normal_image = np.ones_like(depth_image)
    normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1))
    normal_image = np.tile(normal_image, [1, 1, 3])
    normal_image = normal_image / 1.732050808

    mask_image = np.squeeze(np.where(depth_image > 0, 1, 0))
    mask_image = np.reshape(mask_image, (image_shape[0], image_shape[1], 1))
    mask_image = np.tile(mask_image, [1, 1, 3])
    mask_image = np.float32(mask_image)

    normal_image = np.multiply(normal_image, mask_image)
    normal_image = np.float32(normal_image)

    write_gipuma_dmb(out_normal_path, normal_image)


def mvsnet_to_gipuma(args, gipuma_point_folder, dataloader):
    depth_folder_n = depth_folder_name(args)
    depth_folder = Path(args.data_path) / "IntRes" / "depthmaps" / depth_folder_n / str(args.scene)

    gipuma_cam_folder = gipuma_point_folder / 'cams'
    gipuma_image_folder = gipuma_point_folder / 'images'
    if not gipuma_point_folder.exists():
        gipuma_point_folder.mkdir()
    if not gipuma_cam_folder.exists():
        gipuma_cam_folder.mkdir()
    if not gipuma_image_folder.exists():
        gipuma_image_folder.mkdir()

    # convert cameras
    for idb, batch in enumerate(dataloader):
        filename = batch["filename"][0]
        proj_mat = build_proj_matrices(batch["K"][:, 0], batch["R"][:, 0], batch["t"][:, 0])
        if "degenerate" in batch:
            degenerate = batch["degenerate"]
        else:
            degenerate = False

        img = batch["imgs"][0, 0]
        h, w = img.shape[1:]

        # resize ans write images
        proj_mat = proj_mat[0]
        proj_mat[:2] /= args.downscale
        write_gipuma_cam(proj_mat[:3].double().numpy(), gipuma_cam_folder / f"{filename}.jpg.P")

        pil_im = ToPILImage()(img).resize(size=(w // args.downscale, h // args.downscale), resample=Image.LANCZOS)
        pil_im.save(gipuma_image_folder / f"{filename}.jpg")
        gipuma_prefix = '2333__'

        sub_depth_folder = gipuma_point_folder / (gipuma_prefix + filename)
        if not sub_depth_folder.exists():
            sub_depth_folder.mkdir()

        in_depth_npz = depth_folder / f"{filename}_out.npz"
        out_depth_dmb = os.path.join(sub_depth_folder, 'disp.dmb')
        fake_normal_dmb = os.path.join(sub_depth_folder, 'normals.dmb')

        npz = np.load(in_depth_npz)
        depth = npz["depthmap"]
        prob = npz["probability"]
        if args.colmap: # crop depth and prob like for img dataloader
            depth = depth[:h, :w]
            prob = prob[:h, :w]

        # if args.upsample:
        #     depth = F.interpolate(torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
        #                           mode="bilinear", scale_factor=args.downscale,
        #                           align_corners=False).squeeze().numpy()
        #     prob = F.interpolate(torch.from_numpy(prob).unsqueeze(0),
        #                          mode="bilinear", scale_factor=args.downscale,
        #                          align_corners=False).squeeze(0).numpy()

        # prob filtering
        if degenerate:
            mask_invalid = np.ones_like(prob).astype(np.bool)
        else:
            mask_invalid = get_mask(args, filename, prob=prob)

        depth[mask_invalid] = 0
        write_gipuma_dmb(out_depth_dmb, depth)
        fake_gipuma_normal(out_depth_dmb, fake_normal_dmb)

def depth_map_fusion(point_folder, fusibile_exe_path, disp_thresh, num_consistent):
    cam_folder = point_folder / 'cams'
    image_folder = point_folder / 'images'
    depth_min = 0.001
    depth_max = 100000
    normal_thresh = 360

    cmd = fusibile_exe_path
    cmd = cmd + ' -input_folder ' + str(point_folder) + '/'
    cmd = cmd + ' -p_folder ' + str(cam_folder) + '/'
    cmd = cmd + ' -images_folder ' + str(image_folder) + '/'
    cmd = cmd + ' --depth_min=' + str(depth_min)
    cmd = cmd + ' --depth_max=' + str(depth_max)
    cmd = cmd + ' --normal_thresh=' + str(normal_thresh)
    cmd = cmd + ' --disp_thresh=' + str(disp_thresh)
    cmd = cmd + ' --num_consistent=' + str(num_consistent)
    print(cmd)
    os.system(cmd)

    return


def run(dataloader, args):
    # empty cache from previous runs
    torch.cuda.empty_cache()
    folder_name = depth_folder_name(args)

    fusibile_exe_path = "./fusibile"
    disp_threshold = args.fusion_depth_threshold
    num_consistent = args.fusion_num_consistent

    out_path = Path(args.data_path) / "Points" / folder_name
    outfile = out_path / f"{folder_name}{args.scene}.ply"

    point_folder = Path(args.data_path) / "IntRes" / 'fusibile' / folder_name / str(args.scene)

    if point_folder.exists() and not args.override:
        print("Point cloud fusion already computed")
        return

    if not point_folder.exists():
        point_folder.mkdir(parents=True)

    # convert to gipuma format
    print('Convert mvsnet output to gipuma input')
    mvsnet_to_gipuma(args, point_folder, dataloader)

    # depth map fusion with gipuma
    print('Run depth map fusion & filter')
    depth_map_fusion(point_folder, fusibile_exe_path, disp_threshold, num_consistent)

    print("Cleaning outputs")

    if not out_path.exists():
        out_path.mkdir(parents=True)

    for ply in point_folder.glob("**/*.ply"):
        shutil.move(ply, outfile)

    shutil.rmtree(point_folder)
