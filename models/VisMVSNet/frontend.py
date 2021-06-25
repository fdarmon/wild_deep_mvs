from .model_cas import Model
from torch import nn
import torch
from torch.nn import functional as F

class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model()
        self.depth_nums = [32, 16, 8]
        self.interval_scales = [4, 2, 1]
        # 32 * 4 = 128 depth range at train time --> NEED FOR INTERVAL SCALING

    def fill_cam_array(self, K, R, t, start_depth, depth_interval):
        b = K.shape[0]
        res = torch.zeros((b, 2, 4, 4), device=K.device)

        res[:, 0, :3, :3] = R
        res[:, 0, :3, 3:4] = t
        res[:, 1, :3, :3] = K

        res[:, 1, 3, 0] = start_depth
        res[:, 1, 3, 1] = depth_interval
        return res

    def forward(self, imgs, K, R, t, depth_min, depth_max, reference_frame=0, **kwargs):
        depth_interval = (depth_max - depth_min) / 128
        mem = False
        mode = "soft"
        upsample = False

        # possibility to override training parameters for testing
        if "interval_scales" in kwargs:
            interval_scales = kwargs["interval_scales"]
        else:
            interval_scales = self.interval_scales

        if "depth_nums" in kwargs:
            depth_nums = kwargs["depth_nums"]
        else:
            depth_nums = self.depth_nums

        if not isinstance(imgs, list):
            imgs = torch.unbind(imgs, dim=1)

        v = len(imgs)
        ref, src = imgs[reference_frame], imgs[:reference_frame] + imgs[reference_frame + 1:]
        src_idx = list(range(reference_frame)) + list(range(reference_frame + 1, v))

        n = ref.shape[0]

        ref_cam = self.fill_cam_array(K[:, reference_frame], R[:, reference_frame], t[:, reference_frame],
                                      depth_min[:, reference_frame], depth_interval[:, reference_frame])
        srcs_cam = [
            self.fill_cam_array(K[:, i], R[:, i], t[:, i],
                                depth_min[:, i], depth_interval[:, i]) for i in src_idx
        ]

        ref_feat_1, ref_feat_2, ref_feat_3 = self.model.feat_ext(ref)

        feat_packs = [self.model.feat_ext(s) for s in src]
        srcs_feat_1, srcs_feat_2, srcs_feat_3 = [[f[i] for f in feat_packs] for i in range(3)]

        depth_interval = depth_interval[:, reference_frame].view(n, 1, 1, 1)

        est_depth_1, prob_map_1, pair_results_1 = self.model.stage1([ref_feat_1, ref_cam, srcs_feat_1, srcs_cam],
                                                                    depth_num=depth_nums[0], upsample=False,
                                                                    mem=mem,
                                                                    mode=mode, depth_start_override=None,
                                                                    depth_interval_override=depth_interval *
                                                                                            interval_scales[
                                                                                                0], s_scale=8)

        prob_map_1_up = F.interpolate(prob_map_1, scale_factor=4, mode='bilinear', align_corners=False)

        depth_start_2 = F.interpolate(est_depth_1.detach(), size=(ref_feat_2.size()[2], ref_feat_2.size()[3]),
                                      mode='bilinear', align_corners=False) - depth_nums[1] * depth_interval * \
                        self.interval_scales[1] / 2
        est_depth_2, prob_map_2, pair_results_2 = self.model.stage2([ref_feat_2, ref_cam, srcs_feat_2, srcs_cam],
                                                                    depth_num=depth_nums[1], upsample=False,
                                                                    mem=mem,
                                                                    mode=mode, depth_start_override=depth_start_2,
                                                                    depth_interval_override=depth_interval *
                                                                                            interval_scales[
                                                                                                1], s_scale=4)

        prob_map_2_up = F.interpolate(prob_map_2, scale_factor=2, mode='bilinear', align_corners=False)

        depth_start_3 = F.interpolate(est_depth_2.detach(), size=(ref_feat_3.size()[2], ref_feat_3.size()[3]),
                                      mode='bilinear', align_corners=False) - depth_nums[2] * depth_interval * \
                        self.interval_scales[2] / 2
        est_depth_3, prob_map_3, pair_results_3 = self.model.stage3([ref_feat_3, ref_cam, srcs_feat_3, srcs_cam],
                                                                    depth_num=depth_nums[2], upsample=upsample,
                                                                    mem=mem,
                                                                    mode=mode, depth_start_override=depth_start_3,
                                                                    depth_interval_override=depth_interval *
                                                                                            interval_scales[
                                                                                                2], s_scale=2)

        # inverted scales !!!
        depth_est_list = [est_depth_3.squeeze(1), est_depth_2.squeeze(1), est_depth_1.squeeze(1)]
        depth_pair_list = [pair_results_3, pair_results_2, pair_results_1]

        return {
            "depth": est_depth_3.squeeze(1),
            "depth_est_list": depth_est_list,
            "depth_pair_list": depth_pair_list,
            "photometric_confidence": torch.cat([prob_map_1_up, prob_map_2_up, prob_map_3], dim=1)
        }
