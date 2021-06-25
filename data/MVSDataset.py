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
import torch
from torch import distributed as dist
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
import re


class MVSDataset(Dataset):
    def __init__(self):
        self.multi = 32 # resolution should be a multiple of self.multi

        self.height = 512 # supposed to already be a multiple of self.multi
        self.width = 512
        self.resize = True
        self.rand_crop = False
        self.ndepths = -1
        self.interval_scale = 1

        self.load_imgs = True
        self.data_augment = False


    def set_random(self, synchronize):
        rands = torch.rand((self.__len__(), 2))

        if not synchronize:
            self.rands = rands

        else:
            if dist.get_rank() != 0:
                new_rands = torch.zeros_like(rands)
                dist.broadcast(new_rands, 0)
                self.rands = new_rands

            else:
                dist.broadcast(rands, 0)
                self.rands = rands


    def rescale_calib(self, r, K):
        scale_mat = np.eye(4, dtype=np.float32)

        scale_mat[0, 0] = 1 / r
        scale_mat[1, 1] = 1 / r

        new_K = scale_mat[:3, :3] @ K

        return new_K

    def center_crop(self, im, **kwargs):
        scale_mat = np.eye(4, dtype=np.float32)  # takes into account the / 4 of MVSNet

        if im is None:
            new_im = im
        else:
            h, w, c = im.shape

            if self.mode == "test":
                new_height = (h // self.multi) * self.multi
                new_width = (w // self.multi) * self.multi
                crop_h = 0
                crop_w = 0
            else:
                new_height = self.height
                new_width = self.width

                crop_h = (h - new_height) // 2
                crop_w = (w - new_width) // 2
            new_im = im[crop_h:crop_h + new_height, crop_w: crop_w + new_width]

            scale_mat[0, 2] = - crop_w
            scale_mat[1, 2] = - crop_h

        res = [new_im]

        if "K" in kwargs:
            res.append(scale_mat[:3, :3] @ kwargs["K"])

        if "depth" in kwargs:
            res.append(kwargs["depth"][:, crop_h:crop_h + new_height, crop_w: crop_w + new_width])

        return res

    def  read_img(self, filename):
        if not self.load_imgs:
            return None, 1

        img = Image.open(filename)
        if self.resize and self.mode == "train":
            w, h = img.size
            r = min(w / self.width, h / self.height)
            img = img.resize((int(w / r), int(h / r)), resample=Image.LANCZOS)
        else:
            r = 1

        if self.data_augment and self.mode == "train": # only for blendedmvs in train mode
            np_img = self.data_augmentation(img)
        else:
            np_img = np.array(img, dtype=np.float32) / 255.
        return np_img, r

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def motion_blur(self, img):
        max_kernel_size = 3
        # Either vertial, hozirontal or diagonal blur
        mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
        ksize = np.random.randint(0, (max_kernel_size + 1) / 2) * 2 + 1  # make sure is odd
        center = int((ksize - 1) / 2)
        kernel = np.zeros((ksize, ksize))
        if mode == 'h':
            kernel[center, :] = 1.
        elif mode == 'v':
            kernel[:, center] = 1.
        elif mode == 'diag_down':
            kernel = np.eye(ksize)
        elif mode == 'diag_up':
            kernel = np.flip(np.eye(ksize), 0)
        var = ksize * ksize / 16.
        grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
        gaussian = np.exp(-(np.square(grid - center) + np.square(grid.T - center)) / (2. * var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        img = cv2.filter2D(img, -1, kernel)
        return img

    def data_augmentation(self, im):
        im = transforms.ColorJitter(brightness=50/255, contrast=(0.3, 1.5), saturation=0, hue=0)(im)
        im = self.motion_blur(np.array(im, dtype=np.float32) / 255.)
        return im

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale
