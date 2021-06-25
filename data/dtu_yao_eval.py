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
import data.MVSDataset
import numpy as np
import os
from PIL import Image
from pathlib import Path


# the DTU dataset preprocessed by Yao Yao
class MVSDataset(data.MVSDataset.MVSDataset):
    def __init__(self, datapath, listfile, mode, nviews):
        super().__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews

        self.resize = False

        assert self.mode == "test"
        self.metas = self.build_list()

    def get_image_folder(self):
        assert len(self.listfile) == 1
        return Path(self.datapath) / self.listfile[0] / "images"

    def build_list(self):
        metas = []

        # scans
        for id_scan in self.listfile:
            scan = f"scan{id_scan}"
            pair_file = f"{scan}/pair.txt"
            # read the pair file
            with open(self.datapath / pair_file) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((scan, ref_view, src_views))

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
        # do like in blended mvs
        depth_interval = float(lines[11].split()[1]) * 192 / 128
        return intrinsics, extrinsics, depth_min, depth_interval

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []


        klist, rlist, tlist, dminlist, dmaxlist = [], [], [], [], []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            im, r = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            depth_max = depth_min + 128 * depth_interval

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])

            intrinsics = self.rescale_calib(r, intrinsics)
            im, intrinsics = self.center_crop(im, K=intrinsics)

            imgs.append(im)
            klist.append(intrinsics)
            rlist.append(extrinsics[:3, :3])
            tlist.append(extrinsics[:3, 3:])
            dminlist.append(depth_min)
            dmaxlist.append(depth_max)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])

        ret = {
            "imgs": imgs,
            "K": np.stack(klist),
            "R": np.stack(rlist),
            "t": np.stack(tlist),
            "depth_min": np.stack(dminlist).astype(np.float32),
            "depth_max": np.stack(dmaxlist).astype(np.float32),
            "filename": '{:0>8}'.format(view_ids[0]),
            "src_filenames": ['{:0>8}'.format(v) for v in view_ids[1:self.nviews]]
        }

        return ret
