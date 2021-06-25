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
#import cv2
import numpy as np
import torch

#cv2.setNumThreads(0)

from matplotlib.cm import get_cmap
cmap = get_cmap("jet")

from torch.nn import Parameter
from torch import optim


def build_grid(h, w, device, normed=True):

    if device is None: # numpy
        if normed:
            gridY = np.broadcast_to(np.linspace(-1, 1, steps=h).reshape((1, -1, 1, 1)), (1, h, w, 1))
            gridX = np.broadcast_to(np.linspace(-1, 1, steps=w).reshape((1, 1, -1, 1)), (1, h, w, 1))
        else:
            gridY = np.broadcast_to(np.arange(h).reshape((1, -1, 1, 1)), (1, h, w, 1))
            gridX = np.broadcast_to(np.arange(w).reshape((1, 1, -1, 1)), (1, h, w, 1))

        return np.concatenate((gridX, gridY), axis=3)
    else:
        if normed:
            gridY = torch.linspace(-1, 1, steps=h, device=device).view(1, -1, 1, 1).expand(1, h, w, 1)
            gridX = torch.linspace(-1, 1, steps=w, device=device).view(1, 1, -1, 1).expand(1, h, w, 1)
        else:
            gridY = torch.arange(h, device=device).view(1, -1, 1, 1).expand(1, h, w, 1)
            gridX = torch.arange(w, device=device).view(1, 1, -1, 1).expand(1, h, w, 1)
        return torch.cat((gridX, gridY), dim=3)


def build_proj_matrices(K, R, t):
    """
    :param K: *,3,3
    :param R: *,3,3
    :param t: *,3,1
    :return:  *,4,4
    """
    org_shape = K.shape[:-2]
    res = torch.zeros(org_shape + (4, 4), device=K.device)
    res[..., :3, :3] = K @ R
    res[..., :3, 3:] = K @ t
    res[..., 3, 3] = 1
    return res

def project_all(coords, K, R, t):
    if isinstance(coords, torch.Tensor):
        N = K.shape[0]
        org_shape = coords.shape[:-1]
        coords = coords.reshape((-1, 3))

        unproj = (coords[None, :] @ R.transpose(2, 1) + t.transpose(2, 1)) @ K.transpose(2, 1)

        depth = unproj[:, :, 2:]
        return (unproj[:, :, :2] / torch.clamp(depth, min=1e-6)).reshape((N,) + org_shape + (2,)), depth.reshape((N,) + org_shape)
    else:  # numpy version
        N = K.shape[0]
        org_shape = coords.shape[:-1]
        coords = coords.reshape((-1, 3))

        unproj = (coords[None, :] @ R.transpose(0, 2, 1) + t.transpose(0, 2, 1)) @ K.transpose(0, 2, 1)

        depth = unproj[:, :, 2:] + 1e-6
        return (unproj[:, :, :2] / depth).reshape((N,) + org_shape + (2,)), depth.reshape((N,) + org_shape)


def add_hom(pts):
    try:
        dev = pts.device
        ones = torch.ones(pts.shape[:-1], device=dev).unsqueeze(-1)
        return torch.cat((pts, ones), dim=-1)

    except AttributeError:
        ones = np.ones((pts.shape[0], 1))
        return np.concatenate((pts, ones), axis=1)


def project(coords, Ki, Ri, ti):
    # coords array of 3D points in world coordinate, returns pixel coordinates and depth
    if isinstance(coords, torch.Tensor):
        org_shape = coords.shape[:-1]
        coords = coords.view((-1, 3))

        unproj = (coords @ Ri.t() + ti.t()) @ Ki.t()
        depth = unproj[:, 2:] + 1e-6

        return (unproj[:, :2] / depth).view(org_shape + (2, )), depth.view(org_shape)
    else:
        org_shape = coords.shape[:-1]
        coords = coords.reshape((-1, 3))

        unproj = (coords @ Ri.T + ti.T) @ Ki.T
        depth = unproj[:, 2:] + 1e-6

        return (unproj[:, :2] / depth).reshape(org_shape + (2,)), depth.reshape(org_shape)


def unproject(coords, Ki, Ri, ti, Di, invD=True, invK=False):
    # coords array of pixels in image i coordinate, returns array of 3D points in world coordinate
    try:
        coords.device
        org_shape = coords.shape[:-1]
        hom_coords = torch.cat((coords, torch.ones((org_shape + (1,)), device=coords.device)), dim=-1)

        if invD:
            mod_Di = 1 / Di
        else:
            mod_Di = Di

        if invK:
            unproj = ((hom_coords * mod_Di.unsqueeze(-1)).view(-1, 3) @ Ki.t() - ti.t()) @ Ri
        else:
            unproj = ((hom_coords * mod_Di.unsqueeze(-1)).view(-1, 3) @ torch.inverse(Ki).t() - ti.t()) @ Ri
        res = unproj.view(org_shape + (3, ))

    except AttributeError:
        org_shape = coords.shape[:-1]
        hom_coords = np.concatenate((coords, np.ones((org_shape + (1,)))), axis=-1)

        unproj = ((hom_coords * Di[..., None]).reshape(-1, 3) @ np.linalg.inv(Ki).T - ti.T) @ Ri
        res = unproj.reshape(org_shape + (3,))

    return res


def unproj_all(points, K, R, t, depth):
    """
    :param points: N * h * w * 2
    :param K: N * 3 * 3
    :param R: N * 3 * 3
    :param t: N * 3 * 1
    :param depth: N * h * w
    :return: N * h * w * 3
    """
    N, h, w, _ = points.shape
    hom_coords = torch.cat((points.view(N, -1, 2), torch.ones((N, h*w, 1), device=points.device)), dim=-1)

    tmp_k = torch.inverse(K).transpose(2, 1)
    tmp_t = t.transpose(2, 1)
    tmp_r = R

    return (((hom_coords * depth.view(N, -1, 1)) @ tmp_k - tmp_t) @ tmp_r).view((N, h, w, 3))


def flow_from_depthmaps(K, R, t, depthmaps):
    device = K.device
    N, h, w = depthmaps.shape

    grid = torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))[::-1],
                         dim=-1).float().view(1, -1, 2) # 1 * (h * w) * 2

    points3D = (add_hom(grid) * depthmaps.view(N, -1, 1) @ torch.inverse(K[0]).t() - t[0]) @ R[0]
    reprojected = (points3D @ R[1:].transpose(1, 2) + t[1:].unsqueeze(1)) @ K[1:].transpose(1, 2)
    return (reprojected[..., :2] / reprojected[..., 2:]).view(N, h, w, 2)


def inverse_proj_mat(proj_mat):
    b, N, _, _ = proj_mat.shape
    device = proj_mat.device
    tmp = torch.zeros((b, N, 4, 4), device=device)
    tmp[:, :, 3, 3] = 1
    tmp[:, :, 3, :] = proj_mat
    tmp = torch.inverse(tmp)
    return tmp[:, :, :3, :] / tmp[:, :, 3:, :]  # should be useeless since inv_prj[:, :, 3, 3] = 1


def flows_from_single_depthmap(depthmaps, proj_mat, ref_idx):
    """
    Returns flow in pixel
    :param depthmaps: b * 1 * h * w
    :param proj_mat: b * N * 3 * 4
    :param ref_idx reference index in (0,N-1)
    :return:  b * (N - 1) * h * w * 2 flows
    """
    device = proj_mat.device
    b, N, _, _ = proj_mat.shape  # B * N * 3 * 4
    b, h, w = depthmaps.shape
    inv_proj = torch.inverse(proj_mat)
    src_idx = list(range(ref_idx)) + list(range(ref_idx + 1, N))
    grid = torch.stack(torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device))[::-1],
                       dim=-1).float().view(1, 1, -1, 2) # 1 * 1 * (h * w) * 2

    points3D = add_hom(add_hom(grid) * depthmaps.view(b, 1, -1, 1)) @ inv_proj[:, ref_idx:ref_idx+1].transpose(2, 3)
    reprojected = points3D @ proj_mat[:, src_idx].transpose(2, 3)
    flow = reprojected[..., :2]
    depth = reprojected[..., 2:3]

    flow = flow / torch.clamp(depth, 1e-6)

    return flow.view(b, N - 1, h, w, 2), depth.view(b, N - 1, h, w)


def normalize_K(K, h, w):
    if len(K.shape) == 2:
        single = True
        K = K.unsqueeze(0)
    else:
        assert len(K.shape) == 3
        single = False

    b = K.shape[0]

    try:
        h.device
        h = h.float()
        w = w.float()
    except AttributeError:
        h = torch.tensor(h, device=K.device).float().unsqueeze(0)
        w = torch.tensor(w, device=K.device).float().unsqueeze(0)

    norm_mat = torch.zeros((b, 3, 3))
    norm_mat[:, 0, 0] = 2 / (w - 1)
    norm_mat[:, 1, 1] = 2 / (h - 1)
    norm_mat[:, 0, 2] = -1
    norm_mat[:, 1, 2] = -1
    norm_mat[:, 2, 2] = 1

    K = norm_mat @ K
    if single:
        K = K.squeeze(0)

    return K


def normalize(flow, h, w, clamp=None):
    try:
        h.device
    except AttributeError:
        h = torch.tensor(h, device=flow.device).float().unsqueeze(0)
        w = torch.tensor(w, device=flow.device).float().unsqueeze(0)

    if len(flow.shape) == 4:
        w = w.unsqueeze(1).unsqueeze(2)
        h = h.unsqueeze(1).unsqueeze(2)
    elif len(flow.shape) == 3:
        w = w.unsqueeze(1)
        h = h.unsqueeze(1)
    elif len(flow.shape) == 5:
        w = w.unsqueeze(0).unsqueeze(2).unsqueeze(2) # careful here !! maybe error is the unsqueeze at 0 or 1 ?
        h = h.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        # else:
        #     w = w.unsqueeze(1).unsqueeze(2).unsqueeze(2)  # careful here !! maybe error is the unsqueeze at 0 or 1 ?
        #     h = h.unsqueeze(1).unsqueeze(2).unsqueeze(2)

    res = torch.empty_like(flow)
    if res.shape[-1] == 3:
        res[..., 2] = 1

    res[..., 0] = 2 * flow[..., 0] / (w - 1) - 1
    res[..., 1] = 2 * flow[..., 1] / (h - 1) - 1

    if clamp:
        return torch.clamp(res, -clamp, clamp)
    else:
        return res


def unnormalize(flow, h, w):
    try:
        h.device
    except AttributeError:
        h = torch.tensor(h, device=flow.device).float().unsqueeze(0)
        w = torch.tensor(w, device=flow.device).float().unsqueeze(0)

    if len(flow.shape) == 4:
        w = w.unsqueeze(1).unsqueeze(2)
        h = h.unsqueeze(1).unsqueeze(2)
    elif len(flow.shape) == 3:
        w = w.unsqueeze(1)
        h = h.unsqueeze(1)

    res = torch.empty_like(flow)

    if res.shape[-1] == 3:
        res[..., 2] = 1

    res[..., 0] = (w - 1) * (flow[..., 0] + 1) / 2
    res[..., 1] = (h - 1) * (flow[..., 1] + 1) / 2

    return res

def compute_triangulation_angles(point_cloud, R, t, ref_idx=0):
    #                              hw3       k33 k31
    # compute triangulation angles between ref view and all source views for each point

    h, w, _ = point_cloud.shape
    k = R.shape[0]
    src_idx = [idx for idx in range(k) if idx != ref_idx]

    ray1 = point_cloud + (R[ref_idx].t() @ t[ref_idx]).t()  # h*w*3
    rays2 = point_cloud.view(1, h, w, 3) + (R[src_idx].transpose(1, 2) @ t[src_idx]).view(k - 1, 1, 1, 3)  # (k-1) * h * w * 3

    cos = torch.clamp(
        torch.sum(ray1.unsqueeze(0) * rays2, dim=3) / torch.clamp(torch.norm(ray1, dim=2), min=1e-12) / torch.clamp(
            torch.norm(rays2, dim=3), min=1e-12),
        -1, 1)
    return torch.acos(cos) / np.pi * 180

def compute_triangulation_angle(point_cloud, R, t):
    # given a point cloud and a relative pose, computes the triangulation of all points
    ray1 = point_cloud
    ray2 = point_cloud + (R.T @ t).T

    cos = np.clip(np.sum(ray1 * ray2, axis=1) / np.linalg.norm(ray1, axis=1) / np.linalg.norm(ray2, axis=1), -1, 1)
    res =  np.arccos(cos) / np.pi * 180
    return res

def quat_to_rot(q):
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    a2, b2, c2, d2 = a ** 2, b ** 2, c ** 2, d ** 2
    if isinstance(q, torch.Tensor):
        R = torch.empty((q.shape[0], 3, 3))
    else:
        R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = a2 + b2 - c2 - d2
    R[:, 0, 1] = 2 * b * c - 2 * a * d
    R[:, 0, 2] = 2 * a * c + 2 * b * d
    R[:, 1, 0] = 2 * a * d + 2 * b * c
    R[:, 1, 1] = a2 - b2 + c2 - d2
    R[:, 1, 2] = 2 * c * d - 2 * a * b
    R[:, 2, 0] = 2 * b * d - 2 * a * c
    R[:, 2, 1] = 2 * a * b + 2 * c * d
    R[:, 2, 2] = a2 - b2 - c2 + d2

    return R

def rot_to_quat(M):
    q = np.empty((M.shape[0], 4,))
    t = np.trace(M, axis1=1, axis2=2)

    cond1 = t > 0
    cond2 = ~cond1 & (M[:, 0, 0] > M[:, 1, 1]) & (M[:, 0, 0] > M[:, 2, 2])
    cond3 = ~cond1 & ~cond2 & (M[:, 1, 1] > M[:, 2, 2])
    cond4 = ~cond1 & ~cond2 & ~cond3

    S = 2 * np.sqrt(1.0 + t[cond1])
    q[cond1, 0] = 0.25 * S
    q[cond1, 1] = (M[cond1, 2, 1] - M[cond1, 1, 2]) / S
    q[cond1, 2] = (M[cond1, 0, 2] - M[cond1, 2, 0]) / S
    q[cond1, 3] = (M[cond1, 1, 0] - M[cond1, 0, 1]) / S

    S = np.sqrt(1.0 + M[cond2, 0, 0] - M[cond2, 1, 1] - M[cond2, 2,2]) * 2
    q[cond2, 0] = (M[cond2, 2, 1] - M[cond2, 1, 2]) / S
    q[cond2, 1] = 0.25 * S
    q[cond2, 2] = (M[cond2, 0, 1] + M[cond2, 1, 0]) / S
    q[cond2, 3] = (M[cond2, 0, 2] + M[cond2, 2, 0]) / S

    S = np.sqrt(1.0 + M[cond3, 1, 1] - M[cond3, 0, 0] - M[cond3, 2, 2]) * 2
    q[cond3, 0] = (M[cond3, 0, 2] - M[cond3, 2, 0]) / S
    q[cond3, 1] = (M[cond3, 0, 1] + M[cond3, 1, 0]) / S
    q[cond3, 2] = 0.25 * S
    q[cond3, 3] = (M[cond3, 1, 2] + M[cond3, 2, 1]) / S

    S = np.sqrt(1.0 + M[cond4, 2, 2] - M[cond4, 0, 0] - M[cond4, 1, 1]) * 2
    q[cond4, 0] = (M[cond4, 1, 0] - M[cond4, 0, 1]) / S
    q[cond4, 1] = (M[cond4, 0, 2] + M[cond4, 2, 0]) / S
    q[cond4, 2] = (M[cond4, 1, 2] + M[cond4, 2, 1]) / S
    q[cond4, 3] = 0.25 * S

    return q / np.linalg.norm(q, axis=1, keepdims=True)

def relative_pose(R1, t1, R2, t2):
    R = R2 @ R1.T
    t = t2 - R @ t1
    return R, t