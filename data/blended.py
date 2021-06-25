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
import os
from PIL import Image
import data.MVSDataset
import torch
from torch.nn import functional as F


class MVSDataset(data.MVSDataset.MVSDataset):
    def __init__(self, datapath, listfile, mode, nviews, **kwargs):
        super().__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.resize = False  # the images are already the correct shape
        self.data_augment = True

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

        nb_report = 50
        self.visu_idx = (np.arange(nb_report) * (len(self.metas) / nb_report)).astype(int)

        self.rand_crop = False
        self.height = 576
        self.width = 768

        self.return_depth = True

    def build_list(self):
        metas = []

        # scans
        for scene in self.listfile:
            pair_file = scene + "/cams/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())

                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) < (self.nviews - 1):
                        continue
                    metas.append((scene, ref_view, src_views))

        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))

        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])
        assert float(lines[11].split()[2]) == 128
        return intrinsics, extrinsics, depth_min, depth_interval

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scene, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        klist, rlist, tlist = [], [], []
        depth_range = list()
        for i, img_id in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_id = str(img_id).zfill(8)
            img_filename = os.path.join(self.datapath,
                                        '{}/blended_images/{}.jpg'.format(scene, img_id))
            proj_mat_filename = os.path.join(self.datapath, f"{scene}/cams/{img_id}_cam.txt")

            im, r = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            if i == 0 :
                depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{}.pfm'.format(scene, img_id))
                depth = self.read_depth(depth_filename)[None]
                assert(r==1)
                # no need to resize depth since r = 1
                im, intrinsics, depth = self.center_crop(im, K=intrinsics, idx=idx, r=r, depth=depth)
            else:
                im, intrinsics = self.center_crop(im, K=intrinsics, idx=idx, r=r)

            imgs.append(im)
            klist.append(intrinsics)
            rlist.append(extrinsics[:3, :3])
            tlist.append(extrinsics[:3, 3:])

            depth_range.append((depth_min, depth_interval))


        if self.mode == "test" or self.return_depth:
            depth_max = depth_range[0][0] + 128 * depth_range[0][1]
            mask = (depth < depth_max) & (depth > depth_range[0][0])

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])

        ret = {
            "imgs": imgs,
            "K": np.stack(klist),
            "R": np.stack(rlist),
            "t": np.stack(tlist),
            "depth_min": np.stack([d[0] for d in depth_range]).astype(np.float32),
            "depth_max": np.stack([d[0] + 128 * d[1] for d in depth_range]).astype(np.float32),
        }

        if self.mode == "test" or self.return_depth:
            ret["depth"] = depth
            ret["mask"] = mask

        return ret
