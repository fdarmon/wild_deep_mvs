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
import data.MVSDataset


# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(data.MVSDataset.MVSDataset):
    def __init__(self, datapath, listfile, mode, nviews, **kwargs):
        super().__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.resize = False  # the images are already the correct shape

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

        if self.mode != "train":
            self.metas = [self.metas[idx] for idx in np.random.choice(len(self.metas), 1000, replace=False)]

        nb_report = 50
        self.visu_idx = (np.arange(nb_report) * (len(self.metas) / nb_report)).astype(int)

        self.rand_crop = None
        self.height = 512
        self.width = 640

        if "return_depth" in kwargs:
            self.return_depth = kwargs["return_depth"]
        else:
            self.return_depth = False

    def build_list(self):
        metas = []

        # scans
        for scan in self.listfile:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((f'scan{scan}', light_idx, ref_view, src_views))

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
        return intrinsics, extrinsics, depth_min, depth_interval

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        klist, rlist, tlist = [], [], []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths/{}_train/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            im, r = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)
            intrinsics[:2, :] *= 4  # data preprocessed by Yao, intrinsic already take into account the /4 downsampling

            depth_max = depth_min + 192 * depth_interval

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])

            im, intrinsics = self.center_crop(im, K=intrinsics, r=r)

            imgs.append(im)
            klist.append(intrinsics)
            rlist.append(extrinsics[:3, :3])
            tlist.append(extrinsics[:3, 3:])

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                         dtype=np.float32)[:self.ndepths]
                depth_values = np.stack([depth_values.copy() for k_idx in range(self.nviews)])
                if self.mode == "test" or self.return_depth:
                    mask, r = self.read_img(mask_filename)
                    mask = mask[None]
                    depth = self.read_depth(depth_filename)[None]

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])

        ret = {
            "imgs": imgs,
            "K": np.stack(klist),
            "R": np.stack(rlist),
            "t": np.stack(tlist),
            "depth_min": np.stack([depth_min for k_idx in range(self.nviews)]).astype(np.float32),
            "depth_max": np.stack([depth_max for k_idx in range(self.nviews)]).astype(np.float32),
        }

        if self.mode == "test" or self.return_depth:
            ret["depth"] = depth
            ret["mask"] = mask

        return ret


if __name__ == "__main__":
    # some testing code, just IGNORE it
    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/train.txt', 'train',
                         3, 128)
    item = dataset[50]

    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/val.txt', 'val', 3,
                         128)
    item = dataset[50]

    dataset = MVSDataset("/home/xyguo/dataset/dtu_mvs/processed/mvs_training/dtu/", '../lists/dtu/test.txt', 'test', 5,
                         128)
    item = dataset[50]

    # test homography here
    print(item.keys())
    print("imgs", item["imgs"].shape)
    print("depth", item["depth"].shape)
    print("depth_values", item["depth_values"].shape)
    print("mask", item["mask"].shape)

    ref_img = item["imgs"][0].transpose([1, 2, 0])[::4, ::4]
    src_imgs = [item["imgs"][i].transpose([1, 2, 0])[::4, ::4] for i in range(1, 5)]
    ref_proj_mat = item["proj_matrices"][0]
    src_proj_mats = [item["proj_matrices"][i] for i in range(1, 5)]
    mask = item["mask"]
    depth = item["depth"]

    height = ref_img.shape[0]
    width = ref_img.shape[1]
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    print("yy", yy.max(), yy.min())
    yy = yy.reshape([-1])
    xx = xx.reshape([-1])
    X = np.vstack((xx, yy, np.ones_like(xx)))
    D = depth.reshape([-1])
    print("X", "D", X.shape, D.shape)

    X = np.vstack((X * D, np.ones_like(xx)))
    X = np.matmul(np.linalg.inv(ref_proj_mat), X)
    X = np.matmul(src_proj_mats[0], X)
    X /= X[2]
    X = X[:2]

    yy = X[0].reshape([height, width]).astype(np.float32)
    xx = X[1].reshape([height, width]).astype(np.float32)
    # import cv2
    #
    # warped = cv2.remap(src_imgs[0], yy, xx, interpolation=cv2.INTER_LINEAR)
    # warped[mask[:, :] < 0.5] = 0
    #
    # cv2.imwrite('../tmp0.png', ref_img[:, :, ::-1] * 255)
    # cv2.imwrite('../tmp1.png', warped[:, :, ::-1] * 255)
    # cv2.imwrite('../tmp2.py.png', src_imgs[0][:, :, ::-1] * 255)