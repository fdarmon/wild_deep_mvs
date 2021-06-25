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
import torchvision.utils as vutils
import torch
import torch.nn.functional as F


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper

# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper

# convert a function that takes 2 params into recursive style to handle nested dict/list/tuple variables
def make_recursive_func2(func):
    def wrapper(vars, param):
        if isinstance(vars, list):
            return [wrapper(x,param) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x, param) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v, param) for k, v in vars.items()}
        else:
            return func(vars, param)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tocuda".format(type(vars)))


@make_recursive_func
def add_batch(vars):
    if isinstance(vars, torch.Tensor):
        return vars.unsqueeze(0)
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for add_batch".format(type(vars)))


@make_recursive_func2
def rec_upsample(vars, size):
    if vars is None:
        return None
    if len(vars.shape) == 3:
        return F.interpolate(vars.unsqueeze(1), size=size, mode="bilinear", align_corners=False).squeeze(1)
    else:
        return F.interpolate(vars, size=size, mode="bilinear", align_corners=False)


def bayesian_version_loss(l, u, mask):
    mask_sum = torch.sum(mask)
    if mask_sum != 0:
        uncert_loss = torch.sum((l * torch.exp(-u) + u) * mask) / mask_sum
        org_loss = torch.sum(l * mask) / mask_sum
        return uncert_loss + org_loss
    else: # equivalent to return 0 but it keeps the computation graph
        uncert_loss = torch.sum((l * torch.exp(-u) + u) * mask)
        org_loss = torch.sum(l * mask)
        return uncert_loss + org_loss


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())

@make_nograd_func
@compute_metrics_for_each_image
def Rel_Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    div1, div2 = depth_est / depth_gt, depth_gt / depth_est
    err_mask = torch.max(div1, div2) > thres
    return 1 - torch.mean(err_mask.float())

# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    return torch.mean((depth_est - depth_gt).abs())

@make_nograd_func
@compute_metrics_for_each_image
def RelDepthError_metrics(depth_est, depth_gt, mask):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    return torch.mean((depth_est - depth_gt).abs() / depth_gt)

@make_nograd_func
@compute_metrics_for_each_image
def SquareRelDepthError_metrics(depth_est, depth_gt, mask):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    return torch.mean((depth_est - depth_gt) ** 2 / depth_gt)
