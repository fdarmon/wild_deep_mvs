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
from .MVSDataset import MVSDataset
import numpy as np
from pathlib import Path
from utils import colmap_utils
from utils.read_write_model_colmap import read_model


class MVSDataset(MVSDataset):
    def __init__(self, datapath, listfile, mode, nviews, **kwargs):
        assert(len(listfile) == 1)
        super().__init__()
        self.datapath = Path(datapath)
        self.listfile = listfile
        self.min_triangulation_angle = 5
        self.mode = mode
        self.nviews = nviews
        self.init_calibs(self.datapath / "sparse" / self.listfile[0])

        assert self.mode == "test"

    def init_calibs(self, p):
        tmp = read_model(p)
        if tmp is None:
            print("Cannot load initial calib")
            return
        else:
            cameras, images, points3d = tmp

        self.names = [images[idx].name for idx in images]
        self.K, self.R, self.t, sizes = colmap_utils.get_calib_from_sparse(cameras, images)
        # images are cropped to multiple of 32

        self.src_imgs = colmap_utils.compute_src_imgs(images, points3d, self.R, self.t, self.min_triangulation_angle,
                                                      self.nviews - 1, None)

        self.depth_min, self.depth_max, self.pts_min, self.pts_max = self.compute_min_max_depth(points3d, images)

        N = len(self.names)

        self.imgs = []
        for n in self.names:
            im, s = self.read_img(self.datapath / "images" / self.listfile[0] / n)
            im,  = self.center_crop(im)
            self.imgs.append(im)

    def __len__(self):
        return len(self.imgs)

    def get_image_folder(self):
        assert len(self.listfile) == 1
        return self.datapath / "images" / self.listfile[0]

    def compute_min_max_depth(self, points3d, images):
        return colmap_utils.compute_min_max_depth_yao(points3d, images, self.K, self.R, self.t)

    def __getitem__(self, idx):
        # for now we feed every images
        view_ids = [idx] + self.src_imgs[idx]

        depth_min = np.stack([self.depth_min[v] for v in view_ids])
        depth_max = np.stack([self.depth_max[v] for v in view_ids])

        imgs = [self.imgs[i].transpose([2, 0, 1]) for i in view_ids]

        res =  {"imgs": imgs,
                "K": self.K[view_ids],
                "R": self.R[view_ids],
                "t": self.t[view_ids],
                "depth_min": depth_min.astype(np.float32),
                "depth_max": depth_max.astype(np.float32),
                "filename": self.names[idx].split(".")[0],
                "src_filenames": [self.names[idx_s].split(".")[0] for idx_s in self.src_imgs[idx]],
                }
        return res
