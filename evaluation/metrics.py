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
import numpy as np
from scipy.spatial import cKDTree
from scipy.io import loadmat
import sys
sys.path.append("..")
from pathlib import Path
import argparse
import pickle
from utils.utils_3D import add_hom, unproject, build_grid
import time
from utils.read_write_model_colmap import *
from pathlib import Path
import numpy as np
from utils.colmap_utils import compute_Kmatrix_colmap
import h5py
from evaluation.pipeline_utils import depth_folder_name
from utils.utils_ply import read_ply

def format_point_cloud(ply):
    pts =  np.stack((ply["x"], ply["y"], ply["z"]), axis=1)
    return pts[~(np.isnan(pts).any(axis=1))]

def reduce_pts(pts, dst, chunked=False):
    nPoints = pts.shape[0]

    indexSet = np.ones((nPoints,), dtype=np.bool)
    randOrd = np.random.permutation(nPoints)

    kdtree = cKDTree(pts)
    if chunked:
        chunks = list(range(0, nPoints, min(int(4e6), nPoints - 1)))
        chunks[-1] = nPoints

        for i in range(len(chunks) - 1):
            start_point, end_point = chunks[i:i+2]
            idx = kdtree.query_ball_point(pts[randOrd[start_point:end_point]], dst, n_jobs=8)
            for j in range(len(idx)):
                id = randOrd[start_point + j]
                if indexSet[id]:
                    indexSet[idx[j]] = False
                    indexSet[id] = True
    else:
        idx = kdtree.query_ball_tree(kdtree, dst)
        for j in range(len(idx)):
            id = randOrd[j]
            if indexSet[id]:
                indexSet[idx[id]] = False
                indexSet[id] = True
    return pts[indexSet], indexSet


def load_gt(scene_name, path):
    scene = int(scene_name[4:])
    loaded = loadmat(path / "ObsMask" / f"ObsMask{scene}_10.mat")
    bb, mask, res = loaded["BB"], loaded["ObsMask"], loaded["Res"]
    plane = loadmat(path / "ObsMask" / f"Plane{scene}.mat")["P"]
    point_cloud = format_point_cloud(read_ply(path / "Points" / "stl" / f"stl{scene:03d}_total.ply"))

    return point_cloud, mask, bb, res, plane

def eval_yfcc(pred_pts, out_path, args):
    scene = "_".join(args.scene.split("_")[:-1])
    res = np.loadtxt(Path("data/yfcc_subset_dataset/gt_resolution") / f"{scene}.txt").squeeze()
    gt_pts = format_point_cloud(read_ply(Path(args.data_path) / "Points" / "gt" / f"{scene}_gt.ply"))
    dist_gtToPred = chamfer_imw(gt_pts, pred_pts, maxdist=10*res)
    dist_predToGt = chamfer_imw(pred_pts, gt_pts, maxdist=10*res)
    res = {
        "dist_gtToPred": dist_gtToPred,
        "dist_predToGt": dist_predToGt
    }

    if not out_path.exists():
        out_path.mkdir(parents=True)

    with open(out_path / f"dists{args.scene}.pkl", "wb") as f:
        pickle.dump(res, f)


def chamfer_imw(pts_from, pts_to, maxdist=np.inf):
    kdtree = cKDTree(pts_to)
    return kdtree.query(pts_from, distance_upper_bound=maxdist, n_jobs=8)[0]


def eval_dtu(pred_pts, dst, outPath, args):
    # reimplementation of DTU evaluation matlab code
    # does not guarentee exact same results as the matlab code
    margin = 10
    maxdist = 60
    print(f"Removing duplicated points within a radius of {dst}")
    start = time.time()
    pred_pts, _ = reduce_pts(pred_pts, dst, chunked=args.chunked_eval)
    print(f"Done in {time.time() - start}s")

    gt_pts, mask, bb, res, plane = load_gt(args.scene, Path(args.data_path))

    abovePlane = (add_hom(gt_pts) @ plane) > 0
    normalized_pts = np.rint((pred_pts - bb[0:1]) / res).astype(int)

    valid1 = (normalized_pts >= 0).all(axis=1) & (normalized_pts < np.array(mask.shape)[None]).all(axis=1)
    normalized_pts = normalized_pts[valid1]

    validMask = np.zeros((pred_pts.shape[0],), dtype=np.bool)
    valid2 = mask.astype(bool)[normalized_pts[:, 0], normalized_pts[:, 1], normalized_pts[:, 2]]
    validMask[np.where(valid1)[0][valid2]] = True

    print("Computing distance from GT to Pred")
    dist_gtToPred = chamfer(gt_pts, pred_pts, bb, maxdist)
    print("Computing distance from Pred to GT")
    dist_predToGt = chamfer(pred_pts, gt_pts, bb, maxdist)

    res = {
        "margin": margin,
        "maxdist": maxdist,
        "abovePlane": abovePlane,
        "validMask": validMask,
        "dist_gtToPred": dist_gtToPred,
        "dist_predToGt": dist_predToGt
    }

    if not outPath.exists():
        outPath.mkdir(parents=True)

    with open(outPath / f"dists{args.scene}.pkl", "wb") as f:
        pickle.dump(res, f)

def chamfer(ptsFrom, ptsTo, bb, maxdist):
    rx, ry, rz = np.floor((bb[1, :] - bb[0, :]) / maxdist).astype(int)

    dist = np.ones(ptsFrom.shape[0]) * maxdist

    for x in range(rx + 1):
        for y in range(ry + 1):
            for z in range(rz + 1):
                low = bb[0, :] + np.array([x, y, z]) * maxdist
                high = low + maxdist

                validsFrom = (ptsFrom >= low[None]).all(axis=1) & (ptsFrom < high[None]).all(axis=1)

                low = low - maxdist
                high = high + maxdist

                validsTo = (ptsTo >= low[None]).all(axis=1) & (ptsTo < high[None]).all(axis=1)

                if validsTo.sum() == 0:
                    dist[validsFrom] = maxdist
                elif validsFrom.sum() == 0:
                    pass
                else:
                    kdtree = cKDTree(ptsTo[validsTo])
                    dist[validsFrom] = kdtree.query(ptsFrom[validsFrom], n_jobs=8, distance_upper_bound=maxdist)[0]

    return dist


def run(args):
    folder_name = depth_folder_name(args)

    points_path = Path(args.data_path) / "Points" / folder_name

    pred_pts = format_point_cloud(read_ply(points_path / f"{folder_name}{args.scene}.ply"))

    outPath = Path(args.data_path) / "IntRes" / "chamfer" / folder_name
    if (outPath / f"dists{args.scene}.pkl").exists() and not args.override_fusion:
        print("Chamfer already computed, continue...")
        return

    if args.dataset == "dtu":
        eval_dtu(pred_pts, 0.2, outPath, args)
    else:
        eval_yfcc(pred_pts, outPath, args)


