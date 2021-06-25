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
# some colmap utils are in read_write_model_colmap taken from colmap scripts folder
# Here are some other + largely inspired by local feature benchmark by Dusmanu et al.
import sqlite3
import subprocess
from pathlib import Path
from shutil import copy, rmtree
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torchvision.transforms import ToPILImage

from evaluation import pipeline_utils

from utils.read_write_model_colmap import *
from utils.utils_3D import project_all, rot_to_quat, quat_to_rot, project

colmap_path = "./colmap"

def read_colmap_dtb_id(path):
    dtb = sqlite3.connect(path / "database.db")
    cursor = dtb.cursor()

    images = dict()
    cameras = dict()

    cursor.execute("SELECT name, image_id, camera_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]
        cameras[row[0]] = row[1]

    cursor.close()
    dtb.close()

    return images, cameras

def compute_Kmatrix_colmap(params):
    return np.array([
        [params[0], 0, params[2]],
        [0, params[1], params[3]],
        [0, 0, 1]
    ])

def compute_min_max_depth_yao(points3d, imgs, K, R, t, perc=(1, 99)):
    res_min, res_max = np.zeros(len(imgs)), np.zeros(len(imgs))
    for id_im, im in enumerate(imgs):
        pts = np.array([points3d[p].xyz for p in imgs[im].point3D_ids if p != -1])
        if len(pts) > 0:
            proj, depth = project(pts, K[id_im], R[id_im], t[id_im])  # proj N * n * 2
            dmin, dmax = np.percentile(depth, perc)
        else:
            dmin = 0
            dmax = 0
        res_min[id_im] = dmin
        res_max[id_im] = dmax

    return res_min ,res_max, None, None

def compute_min_max_depth_visible(points3d, imgs, K, R, t, sizes):
    pts = list()
    for p in points3d:
        im_ids = points3d[p].image_ids
        nb_obs = 0
        for im in imgs:
            if im in im_ids:
                nb_obs += 1

        if nb_obs >= 3:
            pts.append(points3d[p].xyz)

    pts = np.array(pts)
    proj, depth = project_all(pts, K, R, t)  # proj N * n_points * 2

    valids = np.all(proj >= 0, axis=2) & (proj[..., 0] < sizes[:, 0, None]) & (
                proj[..., 1] < sizes[:, 1, None]) & (depth > 0)
    depth[~valids] = np.nan

    try:
        idx_min, idx_max = np.nanargmin(depth, axis=1), np.nanargmax(depth, axis=1)
        min_point = np.take_along_axis(pts[None], idx_min[:, None, None], 1).squeeze(1)
        max_point = np.take_along_axis(pts[None], idx_max[:, None, None], 1).squeeze(1)
        return np.nanmin(depth, axis=1), np.nanmax(depth, axis=1), min_point, max_point
    except ValueError:
        return None, None, None, None

def compute_src_imgs(images, points3d, R, t, min_triangulation_angle, nsrc, nb_points_thresh):
    im_ids = list(images.keys())
    im_id_to_idx = {im_ids[i]: i for i in range(len(images))}

    adj_mat = np.zeros((len(images), len(images)), dtype=np.int)
    adj_mat_tri = np.zeros((len(images), len(images)), dtype=np.int)

    R_rel = R[None, :] @ R[:, None].transpose(0, 1, 3, 2) # N * N * 3 * 3 relative rotation matrix from i to j
    t_rel = t[None, :] - R_rel @ t[:, None] # N * N * 3 * 1 relative translation from i to j

    rel_opt_center = (np.transpose(R_rel, (0, 1, 3, 2)) @ t_rel).squeeze(3) # N * N * 3

    for p in tqdm(points3d, desc="Compute src images"):
        point = points3d[p]
        im_idx = np.array([im_id_to_idx[im_id] for im_id in point.image_ids])

        ray1 = point.xyz
        ray2 = point.xyz + rel_opt_center
        cos = np.clip(
            np.sum(ray1 * ray2, axis=-1) / np.linalg.norm(ray1, axis=-1) / np.linalg.norm(ray2, axis=-1), -1, 1)
        tri_angles = np.arccos(cos) / np.pi * 180  # N * N
        valid_mat = np.zeros((len(images), len(images)), dtype=np.bool)
        valid_mat[im_idx[None, :], im_idx[:, None]] = True

        update_mat = (tri_angles > min_triangulation_angle) & valid_mat

        adj_mat[im_idx[None, :], im_idx[:, None]] += 1
        adj_mat_tri[update_mat] += 1

    sel_idx = list()

    for i in range(len(images)):
        nb_common_points = adj_mat[i].copy()
        nb_common_points[adj_mat_tri[i] < (0.75 * adj_mat[i])] = 0

        if nb_points_thresh is None:
            sel_idx.append(np.argsort(nb_common_points)[-(nsrc):].tolist())
        else:
            idx_to_select = np.nonzero(nb_common_points > nb_points_thresh)
            if len(idx_to_select) < nsrc:
                sel_idx.append([])
            else:
                sel_idx.append(np.random.choice(idx_to_select, nsrc, replace=False).tolist())

    return sel_idx

def get_calib_from_sparse(cameras, images):
    K = np.array([compute_Kmatrix_colmap(cameras[images[idx].camera_id].params) for idx in images], dtype=np.float32)
    heights = np.array([cameras[images[idx].camera_id].height for idx in images], dtype=np.float32)
    widths = np.array([cameras[images[idx].camera_id].width for idx in images], dtype=np.float32)
    R = quat_to_rot(np.array([images[idx].qvec for idx in images])).astype(np.float32)
    t = np.array([images[idx].tvec for idx in images], dtype=np.float32)[..., None]

    return K, R, t, np.stack((widths, heights), axis=1)

def create_colmap_sparse(dataloader, args):
    # create a sparse folder for colmap using infos from dataloader
    output_path = Path(args.data_path) / "IntRes" / "colmap_sparse" / args.scene

    if output_path.exists():
        print("Sparse colmap already exists")
        return

    image_folder = output_path / "images"
    image_folder.mkdir(parents=True)

    for b in dataloader: # copy image into sparse folder (cannot shutil.copy since dataloader crops images)
        pil_im = ToPILImage()(b["imgs"][0].squeeze(0))
        pil_im.save(image_folder / (b["filename"][0] + ".jpg"))

    subprocess.call([
        str(colmap_path), "feature_extractor",
        "--database_path", str(output_path / "database.db"),
        "--image_path", str(image_folder)
    ])

    images, cameras = read_colmap_dtb_id(output_path)
    lines_cam = list()
    lines_im = list()

    for b in dataloader:
        filename = b["filename"][0] + ".jpg"
        h, w = b["imgs"][0].shape[2:]
        cam_id = cameras[filename]
        im_id = images[filename]

        K = b["K"][0, 0].numpy()
        R = b["R"][:, 0].numpy()
        q = rot_to_quat(R).squeeze(0)
        t = b["t"][0, 0].squeeze().numpy()

        lines_cam.append(
            f"{cam_id} PINHOLE {w} {h} {K[0, 0]} {K[1, 1]} {K[0, 2]} {K[1, 2]}\n"
        )

        q_str = " ".join(map(str, q))
        t_str = " ".join(map(str, t))

        lines_im.append(f"{im_id} {q_str} {t_str} {cam_id} {filename}\n\n")

    with open(output_path / "cameras.txt", 'w') as f:
        f.writelines(lines_cam)

    with open(output_path / "images.txt", "w") as f:
        f.writelines(lines_im)

    with open(output_path / "points3D.txt", "w") as f:
        pass

    # exhaustive matching
    subprocess.call([
        str(colmap_path), "exhaustive_matcher",
        "--database_path", str(output_path / "database.db")
    ])

    # point triangulations
    subprocess.call([
        str(colmap_path), "point_triangulator",
        "--database_path", str(output_path / "database.db"),
        "--image_path", str(image_folder),
        "--input_path", str(output_path),
        "--output_path", str(output_path),
        "--Mapper.ba_refine_focal_length", "0",
        "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0",
        "--Mapper.tri_ignore_two_view_tracks", "0"
    ])

    os.remove(output_path / "images.txt")
    os.remove(output_path / "cameras.txt")
    os.remove(output_path / "points3D.txt")

def read_array(path):
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()

def write_array(array, path):
    """
    see: src/mvs/mat.h
        void Mat<T>::Write(const std::string& path)
    """
    assert array.dtype == np.float32
    if len(array.shape) == 2:
        height, width = array.shape
        channels = 1
    elif len(array.shape) == 3:
        height, width, channels = array.shape
    else:
        assert False

    with open(path, "w") as fid:
        fid.write(str(width) + "&" + str(height) + "&" + str(channels) + "&")

    with open(path, "ab") as fid:
        if len(array.shape) == 2:
            array_trans = np.transpose(array, (1, 0))
        elif len(array.shape) == 3:
            array_trans = np.transpose(array, (1, 0, 2))
        else:
            assert False
        data_1d = array_trans.reshape(-1, order="F")
        data_list = data_1d.tolist()
        endian_character = "<"
        format_char_sequence = "".join(["f"] * len(data_list))
        byte_data = struct.pack(endian_character + format_char_sequence, *data_list)
        fid.write(byte_data)


def depthmap_colmap(dataloader, args):
    dense_folder = (Path(args.data_path) / "IntRes" / "colmap_dense" / args.scene)

    if dense_folder.exists():
        print("Dense colmap already computed")
        return

    else:
        dense_folder.mkdir(parents=True)

    subprocess.call([
        colmap_path,
        "image_undistorter",
        "--image_path",
        f"{args.data_path}/IntRes/colmap_sparse/{args.scene}/images",
        "--input_path",
        f"{args.data_path}/IntRes/colmap_sparse/{args.scene}",
        "--output_path",
        f"{args.data_path}/IntRes/colmap_dense/{args.scene}",
    ])

    subprocess.call([
        colmap_path,
        "patch_match_stereo",
        "--workspace_path",
        f"{args.data_path}/IntRes/colmap_dense/{args.scene}",
    ])

    for b in dataloader:
        filename = b["filename"]
        try:
            depthmap = read_array(
                f"{args.data_path}/IntRes/colmap_dense/{args.scene}/stereo/depth_maps/{filename}.jpg.geometric.bin")
        except FileNotFoundError:
            continue

        if not Path(f"{args.data_path}/IntRes/direct_depthmaps/colmap/{args.scene}").exists():
            Path(f"{args.data_path}/IntRes/direct_depthmaps/colmap/{args.scene}").mkdir(parents=True)

        np.savez(f"{args.data_path}/IntRes/direct_depthmaps/colmap/{args.scene}/{filename}_out.npz",
                 depthmap=depthmap, probability=np.ones_like(depthmap))

def colmap_fusion(dataloader, args):
    folder_name = pipeline_utils.depth_folder_name(args)

    out_path = Path(args.data_path) / "IntRes" / "colmap_fusion" / folder_name / args.scene
    if out_path.exists():
        if not args.override:
            print("Point cloud Fusion already done")
            return
        else:
            rmtree(out_path)

    out_path.mkdir(parents=True)

    input_path = f"{args.data_path}/IntRes/colmap_sparse/{args.scene}"
    subprocess.call([
        colmap_path,
        "image_undistorter",
        "--image_path",
        f"{input_path}/images",
        "--input_path",
        input_path,
        "--output_path",
        str(out_path),
    ])

    for b in dataloader:
        filename = b["filename"][0]
        depth_file = f"{args.data_path}/IntRes/depthmaps/{folder_name}/{args.scene}/{filename}_out.npz"
        try:
            npz = np.load(depth_file)
        except FileNotFoundError:
            print(f"Could not open " + depth_file)
            # do not store anything (should be ignored in colmap fusion)
            continue

        depth = npz["depthmap"]
        prob = npz["probability"]
        h, w = depth.shape

        if args.upsample:
            depth = F.interpolate(torch.from_numpy(depth).unsqueeze(0).unsqueeze(0),
                                     mode="bilinear", scale_factor=args.downscale,
                                     align_corners=False).squeeze().numpy()
            prob = F.interpolate(torch.from_numpy(prob).unsqueeze(0).unsqueeze(0),
                                     mode="bilinear", scale_factor=args.downscale,
                                     align_corners=False).squeeze().numpy()
            h *= args.downscale
            w *= args.downscale

        mask_invalid = pipeline_utils.get_mask(args, filename, prob=prob)
        depth[mask_invalid] = 0
        write_array(depth, out_path / f"stereo/depth_maps/{filename}.jpg.geometric.bin")

        if args.colmap:
            copy(
                f"{args.data_path}/IntRes/colmap_dense/{args.scene}/stereo/normal_maps/{filename}.jpg.geometric.bin",
                out_path / f"stereo/normal_maps/{filename}.jpg.geometric.bin"
            )
        else:
            normals = np.ones((h, w, 3), dtype=np.float32) / np.sqrt(3) # norm 1 for normals
            normals[mask_invalid] = 0
            write_array(normals, out_path / f"stereo/normal_maps/{filename}.jpg.geometric.bin")

    ply_out_folder =  Path(f"{args.data_path}/Points/{folder_name}")
    if not ply_out_folder.exists():
        ply_out_folder.mkdir(parents=True)

    subprocess.call([
        colmap_path,
        "stereo_fusion",
        "--workspace_path", str(out_path),
        "--output_path", str(ply_out_folder / f"{folder_name}{args.scene}.ply"),
        "--StereoFusion.max_normal_error", "10" if args.colmap else "180",
        "--StereoFusion.min_num_pixels", str(args.fusion_num_consistent),
        "--StereoFusion.max_depth_error", str(args.fusion_depth_threshold),
        "--StereoFusion.max_reproj_error", str(args.fusion_max_reproj_error)
    ])
