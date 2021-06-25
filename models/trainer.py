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
from utils.utils_3D import flows_from_single_depthmap, normalize, build_grid, build_proj_matrices
from utils.ssimLoss import SSIM
from .utils import *
from utils.trainer import Trainer
import torch
import numpy as np
from torch.nn import functional as F
import torch.distributed as dist


class Trainer(Trainer):
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args
        self.ssim = SSIM().cuda()

        self.factors_loss = [2, 1, 0.5]

        # downsampling applied to the input before feeding it to the network
        # used when training cvp_mvsnet to enable upsampling_training
        self.input_down = 1
        if self.args.upsample_training:
            if self.args.architecture == "cvp_mvsnet":
                self.input_down = 4
            elif self.args.architecture == "vis_mvsnet":
                self.input_down = 2

        # resolution used for computing the loss is input_res / self.output_down
        # use full resolution if upsample_training, else use the output resolution of the architecture
        self.output_down = 1
        if not self.args.upsample_training:
            if self.args.architecture.startswith("mvsnet"):
                self.output_down = 4  # mvsnet architectures have output at 1/4 input res
            elif self.args.architecture == "vis_mvsnet":
                self.output_down = 2 # vis_mvsnet 1/2 input res


    def loss(self, imgs, d, proj_mat, idxs, suffix=""):
        if self.args.occ_masking:
            return self.masked_photometricloss(imgs, d, proj_mat, idxs, suffix)
        else:
            return self.photometricloss(imgs, d, proj_mat, suffix)


    def forward_network(self, cuda_sample, ref_idx):
        b, n, c, h, w = cuda_sample["imgs"].shape
        src_idx = list(range(ref_idx)) + list(range(ref_idx + 1, self.args.num_im_train))

        down_imgs = F.interpolate(
            cuda_sample["imgs"].view(-1, c, h, w), size=(h // self.input_down, w // self.input_down), mode="bilinear",
            align_corners=False
        ).view(b, n, c, h // self.input_down, w // self.input_down)

        scaled_K = cuda_sample["K"].clone()
        scaled_K[:, :, :2] /= self.input_down

        outputs = self.model(down_imgs, scaled_K, cuda_sample["R"], cuda_sample["t"],
                             cuda_sample["depth_min"],
                             cuda_sample["depth_max"],
                             reference_frame=ref_idx)

        self.ims = {
            "ref_img": cuda_sample["imgs"][:, 0]
        }

        for id_src, src_id_tmp in enumerate(src_idx):
            self.ims[f"src_img_{id_src}"] = cuda_sample["imgs"][:, src_id_tmp]
        min_d = cuda_sample["depth_min"][:, 0].unsqueeze(1).unsqueeze(2)
        max_d = cuda_sample["depth_max"][:, 0].unsqueeze(1).unsqueeze(2)
        for i, d in enumerate(outputs["depth_est_list"]):
            if d is None:
                continue
            depth_im = torch.clamp((d.detach() - min_d) / (max_d - min_d), 0, 1)
            depth_im = depth_im.detach().unsqueeze(1).expand(-1, 3, -1, -1)

            self.ims[f"scale_{i}_depth_est"] = depth_im

        return outputs

    def step(self, sample, train):
        cuda_sample = tocuda(sample)

        b, n, c, h, w = cuda_sample["imgs"].shape

        ref_idx = dist.get_rank() if self.args.occ_masking else 0
        src_idx = list(range(ref_idx)) + list(range(ref_idx + 1, self.args.num_im_train))

        outputs = self.forward_network(cuda_sample, ref_idx)

        # ==============================================================================================================
        # ====================== prepare input imgs, calib mat and outputs for loss function ===========================
        # ==============================================================================================================
        img = F.interpolate(cuda_sample["imgs"].view(-1, c, h, w),
                            size=(h // self.output_down, w // self.output_down), mode="bilinear",
                            align_corners=False).view(b, n, c, h // self.output_down, w // self.output_down)


        if self.args.supervised:
            up_depth_list = outputs["depth_est_list"]
            up_pairs_list = outputs["depth_pair_list"]
            # downsample gt to the good outputs resolution
            gt = sample["depth"].cuda()
            mask = sample["mask"].cuda()
            down_gt = list()
            down_mask = list()

            for id_d, d in enumerate(up_depth_list):
                if d is None:
                    down_gt.append(None)
                    down_mask.append(None)

                else:
                    hd, wd = d.shape[1:]
                    down_gt.append(F.interpolate(gt, size=(hd, wd), mode="bilinear", align_corners=False))
                    # we want EXACT value 1 to be sure that all 4 neighbours are valid depths
                    down_mask.append((F.interpolate(mask.float(), size=(hd, wd), mode="bilinear", align_corners=False) == 1).float())

                    # min_d = cuda_sample["depth_values"][:, 0, 0].unsqueeze(1).unsqueeze(2).unsqueeze(1)
                    # max_d = cuda_sample["depth_values"][:, 0, -1].unsqueeze(1).unsqueeze(2).unsqueeze(1)
                    #
                    # depth_im = torch.clamp((down_gt[-1].detach() - min_d) / (max_d - min_d), 0, 1)
                    # depth_im = depth_im.detach().expand(-1, 3, -1, -1)
                    #
                    # self.ims[f"scale_{id_d}_depth_gt"] = depth_im
                    # self.ims[f"scale_{id_d}_mask_gt"] = down_mask[-1].detach()
        else:
            up_depth_list = rec_upsample(outputs["depth_est_list"], (h // self.output_down, w // self.output_down))
            up_pairs_list = rec_upsample(outputs["depth_pair_list"], (h // self.output_down, w // self.output_down))

            scaled_K = cuda_sample["K"].clone()
            scaled_K[:, :, :2] /= self.output_down
            proj_mat = build_proj_matrices(scaled_K, cuda_sample["R"], cuda_sample["t"])

        # ==============================================================================================================
        # ===================================== COMPUTE THE LOSS =======================================================
        # ==============================================================================================================

        loss = 0
        for id_d, d in enumerate(up_depth_list):
            if self.args.architecture == "vis_mvsnet":
                factor = self.factors_loss[id_d]
            else:
                factor = 1

            if d is None:
                continue

            if self.args.supervised:
                depth_interval = (cuda_sample["depth_max"] - cuda_sample["depth_min"]) / 128
                l1 = torch.abs(d.unsqueeze(1) - down_gt[id_d]) / depth_interval[:, 0].view(b, 1, 1, 1)
                loss += factor * torch.sum(l1 * down_mask[id_d]) / torch.sum(down_mask[id_d])
            else:
                ssim, mask = self.loss(img, d, proj_mat, idxs=None, suffix=f"_scale{id_d}")
                mask_sum = torch.sum(mask)
                if mask_sum != 0:
                    loss += factor * torch.sum(ssim * mask) / mask_sum
                else:
                    loss += factor * torch.sum(ssim * mask)  # will be 0 but keep the computational graph

        for id_d, pairs in enumerate(up_pairs_list):
            if self.args.architecture == "vis_mvsnet":
                factor = self.factors_loss[id_d] / (n - 1)
            else:
                # This configuration should never happen (pairwise loss & architecture is not vismvsnet)
                factor = 1 / (n - 1)

            for id_pair, (d, (unc,)) in enumerate(pairs):
                pair_idx = [ref_idx, src_idx[id_pair]]

                if d is None:
                    continue

                d = d.squeeze(1)
                if self.args.supervised:
                    l1 = torch.abs(d.unsqueeze(1) - down_gt[id_d]) / depth_interval[:, 0].view(b, 1, 1, 1)
                    loss += factor * bayesian_version_loss(l1, unc, down_mask[id_d])

                else:
                    # here do not use the masking: occlusion should be visible to train the confidence network
                    ssim, mask = self.photometricloss(img[:, pair_idx], d, proj_mat[:, pair_idx],
                                           suffix=f"_scale{id_d}_pairwise{id_pair}")
                    loss += factor * bayesian_version_loss(ssim, unc, mask)

        if train:
            self.keep_losses({"train_loss": loss.detach()}) # log loss
        else:
            self.keep_losses({"val_loss": loss.detach()})  # log loss
        self.nb_iter += 1

        return loss


    def get_flow_from_depthmap(self, depth_est, proj_mat, src_size, ref_idx):
        h, w = src_size

        px_flow, depth = flows_from_single_depthmap(depth_est, proj_mat, ref_idx)
        flows = normalize(px_flow, h, w)
        # filter out depth behind the camera
        flows[depth <= 0] = -10
        # clamp flows since large values are a bug of F.grid_sample (pytorch < 1.5)
        flows = torch.clamp(flows, -10, 10)

        return flows, depth

    def photometricloss(self, imgs, depth_est, proj_mat, suffix=""):
        b, N, _, h, w = imgs.shape
        ssim = torch.zeros((b, N-1, h , w), device=imgs.device)  # B (N-1) h w

        flows, _ = self.get_flow_from_depthmap(depth_est, proj_mat, (h, w), 0)

        mask = (flows < 1).all(dim=-1) & (flows > -1).all(dim=-1)
        mask = mask.float()

        for i in range(1, N):
            warped = F.grid_sample(imgs[:, i], flows[:, i - 1], align_corners=False)
            ssim[:, i - 1] = self.ssim(imgs[:, 0], warped).mean(dim=1)

            #self.ims[f"mask{i}{scaleSuffix}"] = mask[:, i - 1].detach()
            self.ims[f"warped{i}{suffix}"] = torch.clamp(warped.clone(), 0., 1.)

        # ssim --> b * (N - 1) * 3 * h * w
        return ssim, mask

    def masked_photometricloss(self, imgs, depth_est, proj_mat, idxs=None, suffix=""):
        #                  bNchw     bhw       bN44

        b, N, c, h, w = imgs.shape


        all_depthmaps  = [torch.ones_like(depth_est) for id_view in range(N)]
        dist.all_gather(all_depthmaps, depth_est)

        i_ref = dist.get_rank()
        src_idx = list(range(i_ref)) + list(range(i_ref + 1, N))

        ssims = torch.zeros((b, len(src_idx), h, w), device=imgs.device)  # B (N-1) h w
        masks = torch.zeros((b, len(src_idx), h, w), device=imgs.device, dtype=torch.bool)  # B (N-1) h w

        ref_depthmap = depth_est.squeeze(1)
        depth_est = torch.stack(all_depthmaps, dim=1)

        self.ims[f"warped{suffix}_ref_{i_ref}src_{i_ref}"] = torch.clamp(
            (imgs[:, i_ref]).detach(), 0., 1.)

        flows, depth_src = self.get_flow_from_depthmap(ref_depthmap, proj_mat, (h, w), i_ref)

        mask = (flows < 1).all(dim=-1) & (flows > -1).all(dim=-1)

        for i in range(len(src_idx)):
            warped = F.grid_sample(imgs[:, src_idx[i]], flows[:, i], align_corners=False)
            warped_src_depth = F.grid_sample(depth_est[:, src_idx[i]].unsqueeze(1), flows[:, i], align_corners=False).squeeze(1)

            reproj_diff = torch.abs(depth_src[:, i] - warped_src_depth) / torch.clamp(warped_src_depth, 1e-8).detach()
            mask_inside = mask[:, i].float()

            ssims[:, i] = self.ssim(imgs[:, i_ref], warped).mean(dim=1)
            diff_mask = mask_inside * (reproj_diff < self.args.geom_clamping)
            masks[:, i] = diff_mask

            self.ims[f"warped{suffix}_ref_{i_ref}src_{src_idx[i]}_masked"] = torch.clamp((diff_mask.unsqueeze(1) * warped).detach(), 0., 1.)

        return ssims, masks

    def test(self, sample):
        cuda_sample = tocuda(sample)
        if isinstance(sample["imgs"], torch.Tensor):
            mask = cuda_sample["mask"][:, 0]
            depth_gt = cuda_sample["depth"][:, 0]
        else:
            mask = cuda_sample["mask"][0]
            depth_gt = cuda_sample["depth"][0]

        with torch.no_grad():
            if self.args.architecture == "vis_mvsnet":
                # test with twice as much depth samples
                outputs = self.model(cuda_sample["imgs"], cuda_sample["K"], cuda_sample["R"], cuda_sample["t"],
                                     cuda_sample["depth_min"], cuda_sample["depth_max"],
                                     depth_nums = [64, 32, 16], scales=[2, 1, 0.5])

            elif self.args.architecture == "cvp_mvsnet" and self.args.dataset != "dtu_yao":
                # test with 4 scales instead of 2 for md and blended
                outputs = self.model(cuda_sample["imgs"], cuda_sample["K"], cuda_sample["R"], cuda_sample["t"],
                                     cuda_sample["depth_min"], cuda_sample["depth_max"], nscale=4)

            else:
                # test with the same config as training
                outputs = self.model(cuda_sample["imgs"], cuda_sample["K"], cuda_sample["R"], cuda_sample["t"],
                                     cuda_sample["depth_min"], cuda_sample["depth_max"])

            depth_est = outputs["depth"]
            _, h, w = mask.shape

            depth_est_up = F.interpolate(depth_est.unsqueeze(1), (h, w), mode="bilinear", align_corners=False).squeeze(
                1)
            step_size = (cuda_sample["depth_max"] - cuda_sample["depth_min"]) / 128

            depth_est_up /= step_size[:, 0]
            depth_gt /= step_size[:, 0]

        self.keep_losses({
                "EPE": AbsDepthError_metrics(depth_est_up, depth_gt, mask > 0.5).detach(),
                "1pxError": Thres_metrics(depth_est_up, depth_gt, mask > 0.5, 1).detach(),
                "3pxError": Thres_metrics(depth_est_up, depth_gt, mask > 0.5, 3).detach(),
            })
        self.nb_iter += 1
