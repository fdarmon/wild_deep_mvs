
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
from pathlib import Path
import torch
from torchvision.transforms import ToPILImage

# simple text monitor, could be changed to something that uses visdom or tensorboard
class Logger():
    def __init__(self, dir):
        if not dir.exists():
            dir.mkdir(parents=True)

        self.fname = dir / f"logs.txt"
        self.im_dirs = dir / "img_dir"
        if not self.im_dirs.exists():
            self.im_dirs.mkdir()

    def log(self, losses=None):

        with open(self.fname, 'a') as f:
            f.write(str(losses))
            f.write("\n")

    def plot_ims(self, ims):
        for im in ims:
            if not isinstance(ims[im], torch.Tensor) or len(ims[im].shape) <= 1:
                continue

            for i in range(ims[im].shape[0]):
                ToPILImage()(ims[im][i].cpu()).save(self.im_dirs / f"batch_{i}_{im}.jpg")
