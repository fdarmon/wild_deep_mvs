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
import tqdm
import h5py
from utils.read_write_model_colmap import read_model
import numpy as np
from PIL import Image
from utils.colmap_utils import get_calib_from_sparse, compute_min_max_depth_visible
from utils.utils_3D import compute_triangulation_angle, quat_to_rot, relative_pose
import shutil
import argparse
import time

def getResizedSize(size, minSize):
    w, h = size
    wratio, hratio = w / minSize, h / minSize
    resizeRatio = min(wratio, hratio)
    w, h = w / resizeRatio, h / resizeRatio
    resizeW = int(w  / 32) * 32
    resizeH = int(h / 32) * 32
    return resizeW, resizeH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"])
    parser.add_argument("--out_name", default="md")
    parser.add_argument("--md_folder", default="datasets/MegaDepth_v1/")
    parser.add_argument("--md_SfM_folder", default="/media/hdd/megadepth_undistorted_sparse")
    args = parser.parse_args()

    nb_points_thresh = 100
    triangulation_angle_threshold = 5

    if args.mode == "train":
        nb_src = 4
        nb_per_scene = 150
        scenes_p = Path("data/txt/md_train.txt")
    else:
        nb_per_scene = 100
        nb_src = 9
        scenes_p = Path("test_scenes.txt")

    with open(scenes_p) as f:
        scenes_list = [s.strip() for s in f.readlines()]

    im_path = Path(args.md_folder)
    calib_path = Path(args.md_SfM_folder)
    parent_path = Path("datasets") / args.out_name / args.mode

    pbar = None
    for id_sc in tqdm.tqdm(scenes_list):
        cpt_nuplet = 0
        full_path_imgs  = im_path / id_sc / "dense0" / "imgs"
        tiic = time.time()
        print("Scene loading")
        try:
            cameras, images, points3d = read_model(calib_path / id_sc)

        except (FileNotFoundError, TypeError):
            print(f"Scene {id_sc} not found")
            continue

        out_path = parent_path / id_sc

        if not out_path.exists():
            out_path.mkdir(parents=True)

        K, R, t, sizes = get_calib_from_sparse(cameras, images)
        key_to_idx = {list(images.keys())[idx]: idx for idx in range(len(images))}
        adj_mat = dict()
        tic = time.time()
        print(f"... Done in {tic - tiic}\nCompution adj mat")

        perm = np.random.permutation(len(images))

        im_keys = list(images.keys())
        idx = 0

        if pbar is None:
            pbar = tqdm.tqdm(total=nb_per_scene, desc="Storing images")
        else:
            pbar.close()
            pbar = tqdm.tqdm(total=nb_per_scene, desc="Storing images")

        for idx in perm:
            im_ref = im_keys[idx]
            if not (full_path_imgs / images[im_ref].name).exists():
                continue

            d = dict()

            for p in images[im_ref].point3D_ids:
                if p != -1:
                    for neigh_imgs in points3d[p].image_ids:
                        if neigh_imgs != im_ref:
                            d[neigh_imgs] = d.get(neigh_imgs, 0) + 1

            nuplet_ids = []
            idx_nuplet_ids = []
            n_uplet_perm = np.random.permutation(len(images) - 1)

            im1 = images[im_ref]
            R1 = quat_to_rot(im1.qvec[None]).squeeze(0)
            t1 = im1.tvec

            # select the n images to add to ref imgs
            for n_uplet_idx in n_uplet_perm:
                im_nuplet = im_keys[n_uplet_idx]
                if not (im_nuplet in d and (full_path_imgs / images[im_nuplet].name).exists()):
                    continue

                if im_nuplet in d and d[im_nuplet] > nb_points_thresh:
                    im2 = images[im_nuplet]
                    R2 = quat_to_rot(im2.qvec[None]).squeeze(0)
                    t2 = im2.tvec

                    common_points = set(images[im_ref].point3D_ids).intersection(images[im_nuplet].point3D_ids)
                    point_cloud = np.array([points3d[p].xyz for p in common_points if p != -1])
                    R_rel, t_rel = relative_pose(R1, t1, R2, t2)
                    tri_angle = compute_triangulation_angle(point_cloud, R_rel, t_rel)
                    if (tri_angle > triangulation_angle_threshold).sum() <= nb_points_thresh:
                        continue

                    nuplet_ids.append(im_nuplet)
                    idx_nuplet_ids.append(n_uplet_idx)
                    if len(nuplet_ids) >= nb_src:
                        break

            if len(nuplet_ids) >= nb_src:
                depth_path = (full_path_imgs.parent / "depths" / images[im_ref].name).with_suffix(".h5")
                if not depth_path.exists() or depth_path.stat().st_size < 100 * 1024:
                    continue

                shutil.copy(depth_path, out_path / f"depth_{cpt_nuplet}.h5")

                imgs = [im_ref] + nuplet_ids
                idx_list = [idx] + idx_nuplet_ids
                new_K = K[idx_list].copy()
                new_sizes = list()
                for id_im, im in enumerate(imgs):
                    pil_im = Image.open(full_path_imgs / images[im].name)
                    res_size = getResizedSize(pil_im.size, minSize=512)
                    new_sizes.append(res_size)
                    new_K[id_im][0:1] *= (res_size[0] / pil_im.size[0])
                    new_K[id_im][1:2] *= (res_size[1] / pil_im.size[1])

                    pil_im = pil_im.resize(res_size, resample=Image.LANCZOS)
                    pil_im.save(out_path / f"im_{cpt_nuplet}_{id_im}.jpg")

                new_sizes = np.array(new_sizes)
                min_d, max_d, _, _ = compute_min_max_depth_visible(points3d, imgs, new_K, R[idx_list], t[idx_list], new_sizes)
                if min_d is None or np.isnan(min_d).any() or np.isnan(max_d).any():
                    # imgs have already be written but are going to be overwritten since cpt_nuplet not updated
                    print("Error computing min and max depth")
                    continue

                np.savez(out_path / f"infos_{cpt_nuplet}.npz", min_d=min_d, max_d=max_d, K=new_K, R=R[idx_list],
                         t=t[idx_list])

                cpt_nuplet += 1
                pbar.update()

            if cpt_nuplet >= nb_per_scene:
                last_cpt = cpt_nuplet
                break
