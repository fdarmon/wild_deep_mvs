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
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
from PIL import Image
import h5py
import data.MVSDataset
import torch
from torch.nn import functional as F


class MVSDataset(data.MVSDataset.MVSDataset):
    def __init__(self, datapath, listfile, mode, nviews, return_depth=False, **kwargs):
        super().__init__()
        assert mode in ["train", "val", "test"]

        if mode == "val":
            self.p = Path(datapath) / "test"
        else:
            self.p = Path(datapath) / mode

        self.height = 512
        self.width = 512
        self.img_idx = []
        self.scene_idx = []
        for scene in listfile:
            if (self.p / scene).exists():
                for cpt in range(1000): # hardcoded max size of dataset
                    b = True
                    for i in range(nviews):
                        b = b and (self.p / scene / f"im_{cpt}_{i}.jpg").exists()
                    b = b and (self.p / scene / f"infos_{cpt}.npz").exists()
                    if mode == "test":
                        if not (self.p / scene / f"depth_{cpt}.h5").exists():
                            for i in range(nviews):
                                b = b and (self.p / scene / f"depth_{cpt}_{i}.h5").exists()

                    else:
                        if return_depth:
                            b = b and (self.p / scene / f"depth_{cpt}.h5").exists()

                    if b:
                        self.img_idx.append(cpt)
                        self.scene_idx.append(scene)


        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.return_depth = return_depth

        nb_report = 50
        self.visu_idx = (np.arange(nb_report) * (len(self.img_idx) / nb_report)).astype(int)

        self.rands = None

    def __len__(self):
        return len(self.img_idx)

    def __getitem__(self, idx):
        imgs = []
        scene = self.scene_idx[idx]

        infos_filename = self.p / scene / f"infos_{self.img_idx[idx]}.npz"

        npz = np.load(infos_filename)
        depth_train = self.return_depth and self.mode == "train"

        if depth_train:
            with h5py.File(self.p / scene / f"depth_{self.img_idx[idx]}.h5", 'r') as f:
                depth = torch.tensor(np.array(f["depth"]))

        K, R, t = npz["K"].astype(np.float32), npz["R"].astype(np.float32), npz["t"].astype(np.float32)

        K, R, t = K[:self.nviews], R[:self.nviews], t[:self.nviews]

        proj_matrices = np.zeros((self.nviews, 4, 4), dtype=np.float32)
        proj_matrices[:, 3, 3] = 1

        for i in range(self.nviews):
            img_filename = self.p / scene / f"im_{self.img_idx[idx]}_{i}.jpg"

            im, resize_ratio = self.read_img(img_filename)
            new_K = self.rescale_calib(resize_ratio, K=K[i])
            if depth_train and i == 0:
                th, tw = depth.shape
                depth = F.interpolate(depth.view(1, 1, th, tw), size=im.shape[:-1], mode="nearest").squeeze(0)
                im, new_K, depth = self.center_crop(im, K=new_K, idx=idx, depth=depth)
            else:
                im, new_K = self.center_crop(im, K=new_K, idx=idx)

            if i == 0:
                f = img_filename

            imgs.append(im)
            K[i] = new_K

        res = {"f": str(f),
               "K": K,
               "R": R,
               "t": t,
               "depth_min": npz["min_d"].astype(np.float32),
               "depth_max": npz["max_d"].astype(np.float32)}

        if self.mode != "test":
            res["imgs"] = np.stack([im.transpose([2, 0, 1]) for im in imgs], axis=0)
            if self.return_depth:
                res["depth"] = depth
                res["mask"] = (depth >= npz["min_d"][0]) & (depth < npz["max_d"][0])

        else:
            res["imgs"] = [im.transpose([2, 0, 1]) for im in imgs]
            depths = list()
            masks = list()
            try:
                for i in range(self.nviews):
                    # not clean but i don't want to regen a full dataset
                    with h5py.File(self.p / scene / f"depth_{self.img_idx[idx]}_{i}.h5", 'r') as f:
                        depth = np.array(f["depth"])
                        depths.append(depth)
                        masks.append(depth > 0)
            except OSError:
                with h5py.File(self.p / scene / f"depth_{self.img_idx[idx]}.h5", 'r') as f:
                    depth = np.array(f["depth"]).astype(np.float)
                    depths.append(depth)
                    masks.append(depth > 0)

            res["mask"] = masks
            res["depth"] = depths

        return res
