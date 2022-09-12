# -----------------------------------------------------
# Written by Haoyi Zhu (zhuhaoyi@sjtu.edu.cn). All rights reserved.
# -----------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import LOSS
from .lpips import LPIPS


@LOSS.register_module
class Criterion(nn.Module):
    """
    Class for X-NeRF criterion loss functions.

    Args:
        use_perceptual_loss (``bool``): Whether to use perceptual loss. Default: ``True``.
        use_tv_loss (``bool``): Whether to use total variation loss. Default: ``False``.
        use_sn_loss (``bool``): Whether to use surface normal loss. Default: ``False``.
        loss_weight (``dict``): Loss weight for each loss type.
    """

    def __init__(
        self,
        use_perceptual_loss: bool = True,
        use_tv_loss: bool = False,
        use_sn_loss: bool = False,
        **loss_weight
    ) -> None:
        super(Criterion, self).__init__()
        self.depth_loss_weight = loss_weight.get("depth", 0.1)
        self.use_perceptual_loss = use_perceptual_loss
        self.use_tv_loss = use_tv_loss
        self.use_sn_loss = use_sn_loss

        if self.use_perceptual_loss:
            self.perceptual_loss = LPIPS().eval().cuda()
            self.perceptual_loss.requires_grad = False
            self.perceptual_loss_weight = loss_weight.get("perceptual", 1.0)
        if self.use_tv_loss:
            self.tv_loss_weight = loss_weight.get("tv", 5e-2)
        if self.use_sn_loss:
            self.sn_loss_weight = loss_weight.get("sn", 1e-3)

    def _safe_mean(self, data, mask, default_res=0.0):
        masked_data = data[mask]
        return (
            torch.tensor(default_res).to(masked_data.device)
            if masked_data.numel() == 0
            else masked_data.mean()
        )

    def _l1(self, pred, gt):
        return torch.abs(pred - gt)

    def _l2(self, pred, gt):
        return (pred - gt) ** 2

    def masked_l1_loss(self, pred, gt, mask):
        return self._safe_mean(self._l1(pred, gt), mask)

    def masked_mse_loss(self, pred, gt, mask):
        return self._safe_mean(self._l2(pred, gt), mask)

    def tv_loss(self, x):
        batch_size = x.size()[0]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :-1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w)

    def _gradient(self, x):
        left = x
        right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        top = x
        bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]
        dx, dy = right - left, bottom - top
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0
        return dx, dy

    def _get_surface_normal_from_xyz(self, x, epsilon=1e-8):
        dx, dy = self._gradient(x)
        surface_normal = torch.cross(dx, dy, dim=1)
        surface_normal = surface_normal / (
            torch.norm(surface_normal, dim=1, keepdim=True) + epsilon
        )
        return surface_normal

    def sn_loss(self, pred_depth, gt_depth, rays_o, rays_d):
        pred_xyz = rays_o + pred_depth * rays_d
        gt_xyz = rays_o + gt_depth * rays_d
        pred_sn = self._get_surface_normal_from_xyz(pred_xyz)
        gt_sn = self._get_surface_normal_from_xyz(gt_xyz)
        sn_loss = 1 - F.cosine_similarity(pred_sn, gt_sn, dim=1)

        padded_gt_depth = F.pad(gt_depth, [1, 1, 1, 1])
        mask = (
            (gt_depth > 0)
            + (padded_gt_depth[:, :, 1:-1, 2:] > 0)
            + (padded_gt_depth[:, :, 2:, 1:-1] > 0)
            + (padded_gt_depth[:, :, 1:-1, :-2] > 0)
            + (padded_gt_depth[:, :, :-2, 1:-1] > 0)
        ).squeeze(1)

        return self._safe_mean(sn_loss, mask)

    def _to_patch(self, data, patch_h, patch_w, channel=3):
        return (
            data.reshape(-1, patch_h, patch_w, channel)
            .contiguous()
            .transpose(2, 3)
            .transpose(1, 2)
        )

    def forward(self, pred_rgb, pred_depth, gt_rgb, gt_depth, **kwargs):
        loss_dict = dict()

        rgb_loss = F.mse_loss(pred_rgb, gt_rgb)
        depth_mask = gt_depth > 0
        depth_loss = self.masked_mse_loss(pred_depth, gt_depth, depth_mask)

        total_loss = rgb_loss + self.depth_loss_weight * depth_loss

        if self.use_perceptual_loss:
            patch_h, patch_w = kwargs["patch_h"], kwargs["patch_w"]
            percept_loss = self.perceptual_loss(
                self._to_patch(pred_rgb, patch_h, patch_w),
                self._to_patch(gt_rgb, patch_h, patch_w),
            )
            percept_loss = torch.mean(percept_loss)
            total_loss += self.perceptual_loss_weight * percept_loss
            loss_dict["perceptual"] = percept_loss

        if self.use_tv_loss:
            tv_loss = torch.mean(
                self.tv_loss(self._to_patch(pred_rgb, patch_h, patch_w))
            )
            total_loss += self.tv_loss_weight * tv_loss
            loss_dict["tv"] = tv_loss

        if self.use_sn_loss:
            rays_o, rays_d = kwargs["rays_o"], kwargs["rays_d"]
            sn_loss = self.sn_loss(
                self._to_patch(pred_depth, patch_h, patch_w, channel=1),
                self._to_patch(gt_depth, patch_h, patch_w, channel=1),
                self._to_patch(rays_o, patch_h, patch_w),
                self._to_patch(rays_d, patch_h, patch_w),
            )
            total_loss += self.sn_loss_weight * sn_loss
            loss_dict["sn"] = sn_loss

        loss_dict["rgb"] = rgb_loss
        loss_dict["depth"] = depth_loss
        loss_dict["total"] = total_loss

        return loss_dict
