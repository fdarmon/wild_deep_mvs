from torch import nn
import torch
from .models.net import network

class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = network()

    def forward(self, imgs, K, R, t, depth_min, depth_max, reference_frame=0, **kwargs):

        src_idx = list(range(reference_frame)) + list(range(reference_frame + 1, K.shape[1]))

        if isinstance(imgs, torch.Tensor):
            ref_img = imgs[:, reference_frame]
            src_imgs = torch.unbind(imgs[:, src_idx], dim=1)
        else:
            ref_img = imgs[reference_frame]
            src_imgs = imgs[:reference_frame] + imgs[reference_frame + 1:]

        b = ref_img.shape[0]
        N = len(src_imgs)
        ref_in = K[:, reference_frame]
        src_in = K[:, src_idx]
        ref_ex = torch.cat((R[:, reference_frame], t[:, reference_frame]), dim=2)
        ref_ex = torch.cat((ref_ex, torch.tensor([0., 0., 0., 1.]).cuda().view(1, 1, 4).expand(b, 1, 4)), dim=1)

        src_ex = torch.cat((R[:, src_idx], t[:, src_idx]), dim=3)  # B N 3 4
        src_ex = torch.cat((src_ex, torch.tensor([0., 0., 0., 1.]).cuda().view(1, 1, 1, -1).expand(b, N, 1, 4)), dim=2)
        output = self.model(ref_img, src_imgs, ref_in, src_in, ref_ex, src_ex, depth_min[:, reference_frame],
                            depth_max[:, reference_frame], **kwargs)

        return {
            "depth": output["depth_est_list"][0].squeeze(1),
            "depth_est_list": output["depth_est_list"],
            "depth_pair_list": [],
            "photometric_confidence": output["prob_confidence"].unsqueeze(1)
        }




