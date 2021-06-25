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
import torch.distributed as dist

class Trainer:
    def __init__(self):
        self.loss_means = {}
        self.loss_running_means = {}
        self.nb_iter = 0
        self.ims = {}

    def log_epoch(self, epoch):
        res = {k: self.loss_means[k] / self.nb_iter for k in self.loss_means}
        self.loss_means = {}
        res["epoch"] = epoch
        self.nb_iter = 0
        for l in res:
            if l in ["epoch", "batch"]:
                continue
            dist.all_reduce(tensor=res[l], op=dist.ReduceOp.SUM)
            res[l] /= dist.get_world_size()
        return res

    def log_iter(self):
        res = {k: self.loss_running_means[k] / self.args.print_every for k in self.loss_running_means}
        self.loss_running_means = {}
        return res

    def keep_losses(self, losses):
        for k in losses:
            # running average only interesting for training metrics
            if k.startswith("train"):
                self.loss_running_means[k] = self.loss_running_means.get(k, 0) + losses[k]

            self.loss_means[k] = self.loss_means.get(k, 0) + losses[k]