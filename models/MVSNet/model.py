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
from .module import *
from utils.utils_3D import build_proj_matrices
import os
from torch import distributed as dist

class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.inplanes = 32

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))
        return x

class CostRegNet(nn.Module):
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x, down_ft=None):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)

        return x


class MVSNet(nn.Module):
    def __init__(self, aggregation="variance"):
        super(MVSNet, self).__init__()

        self.feature = FeatureNet()
        self.cost_regularization = CostRegNet()
        # initialize temperature
        if aggregation == "softmin":
            self.register_parameter("temp", torch.nn.Parameter(torch.ones((1))))

        self.aggregation=aggregation
        self.num_depth = 192 # never changes


    def extract_features(self, imgs):
        if self.aggregation.startswith("norm"):
            features = [F.normalize(self.feature(img), dim=1) for img in imgs]
        else:
            features = [self.feature(img) for img in imgs]

        return features

    def build_cost_volume(self, ref_feature, src_features, ref_proj, src_projs, depth_values):
        b, c, ftH, ftW = ref_feature.shape
        num_views = len(src_features) + 1

        if self.aggregation == "variance":
            ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, self.num_depth, 1, 1)

            volume_sum = ref_volume
            volume_sq_sum = ref_volume ** 2
            del ref_volume

            for src_fea, src_proj in zip(src_features, src_projs):
                # warpped features
                warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values, ref_feature.shape[-2:])

                if self.training:
                    volume_sum = volume_sum + warped_volume
                    volume_sq_sum = volume_sq_sum + warped_volume ** 2
                else:
                    # save memory when testing with inline operations
                    volume_sum += warped_volume
                    volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
                del warped_volume
            # aggregate multiple feature volumes by variance

            cost_volume = volume_sq_sum.div_(num_views).sub_(volume_sum.pow_(2).div_(num_views ** 2))

            del volume_sq_sum
            del volume_sum

            return cost_volume

        elif self.aggregation == "softmin":
            ref_volume = ref_feature.unsqueeze(2)

            sum_exp = torch.zeros((b, 1, self.num_depth, ftH, ftW), device=ref_volume.device)
            sum_val = torch.zeros((b, c, self.num_depth, ftH, ftW), device=ref_volume.device)

            for src_fea, src_proj in zip(src_features, src_projs):
                # warpped features
                warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values,
                                                           ref_feature.shape[-2:])

                if self.training:
                    diff = (ref_volume - warped_volume) ** 2
                    exp = torch.exp(- self.temp * diff.sum(dim=1, keepdim=True))

                    sum_exp += exp
                    sum_val += exp * diff

                else:
                    warped_volume.sub_(ref_volume).pow_(2)
                    exp = torch.exp(- self.temp * warped_volume.sum(dim=1, keepdim=True))
                    sum_exp.add_(exp)
                    warped_volume.mul_(exp)
                    sum_val.add_(warped_volume)

                    del exp

                del warped_volume


            cost_volume = sum_val.div_(sum_exp + 1e-6)

            return cost_volume

        else:
            raise NotImplementedError("Aggregation: " + self.aggregation)

    def forward(self, imgs, K, R, t, depth_min, depth_max, reference_frame=0, **kwargs):
        try:
            imgs = torch.unbind(imgs, 1)
        except TypeError: # imgs is already a list
            pass
        scaled_K = K.clone()
        scaled_K[:, :, :2] /= 4
        proj_matrices = build_proj_matrices(scaled_K, R, t)
        proj_matrices = torch.unbind(proj_matrices, 1)
        tmp_range = torch.arange(self.num_depth, device=depth_min.device).view(1, 1, -1)
        depth_range = (depth_max - depth_min) / (self.num_depth - 1)
        depth_values = depth_min.unsqueeze(-1) + depth_range.unsqueeze(-1) * tmp_range

        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = self.extract_features(imgs)

        ref_feature, src_features = features[reference_frame], features[:reference_frame] + features[reference_frame + 1:]
        b, c, ftH, ftW = ref_feature.shape
        ref_proj, src_projs = proj_matrices[reference_frame], proj_matrices[:reference_frame] + proj_matrices[reference_frame + 1:]

        # step 2. differentiable homograph, build cost volume
        cost_volume = self.build_cost_volume(ref_feature, src_features, ref_proj, src_projs, depth_values[:, reference_frame])

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(cost_volume)

        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values[:, reference_frame])

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(self.num_depth, device=prob_volume.device, dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        return {"depth": depth, "depth_est_list": [depth,], "depth_pair_list": [],
                "photometric_confidence": photometric_confidence}
