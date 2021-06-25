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
from torch.utils.data import Sampler
import torch
from torch import distributed as dist

class SharedRandomSampler(Sampler):

    def __init__(self, dataset):
        super().__init__(dataset)
        self.cpt = 0
        self.len = len(dataset)
        self.perm = None
        self.set_epoch(0)

    def __len__(self):
        return self.len

    def __next__(self):
        try:
            res = self.perm[self.cpt]
            self.cpt += 1

        except IndexError:
            raise StopIteration

        return res

    def __iter__(self):
        return self

    def set_epoch(self, num_epoch):
        self.cpt = 0
        perm = torch.randperm(self.len)

        if dist.get_rank() != 0:
            new_perm = torch.zeros_like(perm)
            dist.broadcast(new_perm, 0)
            self.perm = new_perm

        else:
            dist.broadcast(perm, 0)
            self.perm = perm