"""
Utils related to rendering.
"""

import os

import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.cpp_extension import load
from torch_scatter import segment_coo


def render_outputs(
    output,
    ray_pts,
    ray_id,
    step_id,
    b_size,
    step_size,
    voxel_size,
    alpha_init,
    alpha_thr=5e-4,
    discrete_coords=None,
    rm=None,
    viewdirs=None,
    sh_deg=0,
):
    interp = ME.MinkowskiInterpolation()

    ori_loc_out = None
    if discrete_coords is not None:
        ori_loc_out = interp(output, discrete_coords.float())

    if rm is not None and rm.any():
        rm = rm.cuda()
        ray_pts = (
            rm
            @ torch.cat([ray_pts, torch.ones((ray_pts.shape[0], 1)).cuda()], 1)[
                :, :, None
            ]
        )
        ray_pts = ray_pts / ray_pts[:, -1:, :]
        ray_pts = ray_pts[:, :3, 0]

    inter_coords = ME.utils.batched_coordinates([ray_pts / voxel_size]).cuda().float()
    inter_out = interp(output, inter_coords)

    valid = inter_out.detach().any(-1)

    density = inter_out[:, 0].contiguous().flatten()
    alpha = Raw2Alpha.apply(density, np.log(1 / (1 - alpha_init) - 1), step_size)

    mask = (alpha > alpha_thr) * valid
    ray_id = ray_id[mask]
    step_id = step_id[mask]
    alpha = alpha[mask]
    inter_out = inter_out[mask]
    valid = valid[mask]

    weights, _ = Alphas2Weights.apply(alpha, ray_id, b_size)

    mask = (weights > alpha_thr) * valid
    weights = weights[mask]
    ray_id = ray_id[mask]
    step_id = step_id[mask]
    inter_out = inter_out[mask]
    valid = valid[mask]

    k0 = inter_out[:, 1:]
    if sh_deg > 0:
        assert viewdirs is not None
        k0 = eval_sh(
            sh_deg, k0.reshape(*k0.shape[:-1], -1, (sh_deg + 1) ** 2), viewdirs[ray_id]
        )

    rgb = torch.sigmoid(k0).squeeze(0)

    rgb_output = segment_coo(
        src=(weights.unsqueeze(-1) * rgb),
        index=ray_id,
        out=torch.zeros([b_size, 3]).float().cuda(),
        reduce="sum",
    )

    depth_output = segment_coo(
        src=(weights * (step_id * step_size * voxel_size)),
        index=ray_id,
        out=torch.zeros([b_size]).float().cuda(),
        reduce="sum",
    )

    torch.cuda.empty_cache()

    return rgb_output, depth_output, ori_loc_out


"""
The following classes and functions are modified from DVGO.
Reference: https://github.com/sunset1995/DirectVoxGO
"""

parent_dir = os.path.dirname(os.path.abspath(__file__))
render_utils_cuda = load(
    name="render_utils_cuda",
    sources=[
        os.path.join(parent_dir, path)
        for path in ["cuda/render_utils.cpp", "cuda/render_utils_kernel.cu"]
    ],
    verbose=True,
)


mse2psnr = lambda x: -10.0 * torch.log10(x)
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


def get_rays(H, W, K, c2w, mode="center", padding=0):
    H, W = H + padding * 2, W + padding * 2
    i, j = torch.meshgrid(
        torch.linspace(0, W - 1, W, device=c2w.device) - padding,
        torch.linspace(0, H - 1, H, device=c2w.device) - padding,
    )  # pytorch's meshgrid has indexing='ij'
    i = i.t().float()
    j = j.t().float()
    if mode == "lefttop":
        pass
    elif mode == "center":
        i, j = i + 0.5, j + 0.5
    elif mode == "random":
        i = i + torch.rand_like(i)
        j = j + torch.rand_like(j)
    else:
        raise NotImplementedError

    dirs = torch.stack(
        [(i - K[0][2]) / K[0][0], (j - K[1][2]) / K[1][1], torch.ones_like(i)], -1
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d


def get_rays_np(H, W, K, c2w, padding=0):
    i, j = np.meshgrid(
        np.arange(W + padding * 2, dtype=np.float32) - padding,
        np.arange(H + padding * 2, dtype=np.float32) - padding,
        indexing="xy",
    )
    dirs = np.stack(
        [(i - K[0][2]) / K[0][0], -(j - K[1][2]) / K[1][1], -np.ones_like(i)], -1
    )
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(
        dirs[..., np.newaxis, :] * c2w[:3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, 3], np.shape(rays_d))
    return rays_o, rays_d


def get_rays_of_a_view(H, W, K, c2w, mode="center", padding=0):
    rays_o, rays_d = get_rays(H, W, K, c2w, mode=mode, padding=padding)
    viewdirs = rays_d / rays_d.norm(dim=-1, keepdim=True)

    return rays_o, rays_d, viewdirs


def sample_ray(rays_o, rays_d, near, far, stepsize, xyz_min, xyz_max, voxel_size=0.005):
    rays_o = rays_o.contiguous()
    rays_d = rays_d.contiguous()
    stepdist = stepsize * voxel_size

    N_steps = int((far - near) / stepdist)
    t_vals = torch.linspace(0.0, 1.0, steps=N_steps).cuda()
    near, far = (
        near * torch.ones_like(rays_d[..., :1]).cuda(),
        far * torch.ones_like(rays_d[..., :1]).cuda(),
    )
    z_vals = near * (1.0 - t_vals) + far * (t_vals)

    ray_pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    ray_id = torch.arange(ray_pts.shape[0])[:, None].repeat(1, ray_pts.shape[1]).cuda()
    step_id = torch.arange(ray_pts.shape[1])[None, :].repeat(ray_pts.shape[0], 1).cuda()

    ray_pts = ray_pts.reshape(-1, 3)
    ray_id = ray_id.reshape(-1)
    step_id = step_id.reshape(-1)

    mask_outbbox = (
        (ray_pts[:, 0] > xyz_max[0])
        + (ray_pts[:, 1] > xyz_max[1])
        + (ray_pts[:, 2] > xyz_max[2])
        + (ray_pts[:, 0] < xyz_min[0])
        + (ray_pts[:, 1] < xyz_min[1])
        + (ray_pts[:, 2] < xyz_min[2])
    )

    mask_inbbox = ~mask_outbbox
    ray_pts = ray_pts[mask_inbbox]
    ray_id = ray_id[mask_inbbox]
    step_id = step_id[mask_inbbox]

    return ray_pts, ray_id, step_id


class Raw2Alpha(torch.autograd.Function):
    @staticmethod
    def forward(ctx, density, shift, interval):
        """
        alpha = 1 - exp(-softplus(density + shift) * interval)
              = 1 - exp(-log(1 + exp(density + shift)) * interval)
              = 1 - exp(log(1 + exp(density + shift)) ^ (-interval))
              = 1 - (1 + exp(density + shift)) ^ (-interval)
        """
        exp, alpha = render_utils_cuda.raw2alpha(density, shift, interval)
        if density.requires_grad:
            ctx.save_for_backward(exp)
            ctx.interval = interval
        return alpha

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_back):
        """
        alpha' = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)'
               = interval * ((1 + exp(density + shift)) ^ (-interval-1)) * exp(density + shift)
        """
        exp = ctx.saved_tensors[0]
        interval = ctx.interval
        return (
            render_utils_cuda.raw2alpha_backward(exp, grad_back.contiguous(), interval),
            None,
            None,
        )


class Alphas2Weights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, alpha, ray_id, N):
        weights, T, alphainv_last, i_start, i_end = render_utils_cuda.alpha2weight(
            alpha, ray_id, N
        )
        if alpha.requires_grad:
            ctx.save_for_backward(alpha, weights, T, alphainv_last, i_start, i_end)
            ctx.n_rays = N
        return weights, alphainv_last

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_weights, grad_last):
        alpha, weights, T, alphainv_last, i_start, i_end = ctx.saved_tensors
        grad = render_utils_cuda.alpha2weight_backward(
            alpha,
            weights,
            T,
            alphainv_last,
            i_start,
            i_end,
            ctx.n_rays,
            grad_weights,
            grad_last,
        )
        return grad, None, None


"""
The following functions are modified from PlenOctree.
Reference: https://github.com/sxyu/plenoctree/blob/master/nerf_sh/nerf/sh.py
"""

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]


def eval_sh(deg, sh, dirs):
    assert deg <= 4 and deg >= 0
    assert (deg + 1) ** 2 == sh.shape[-1]
    C = sh.shape[-2]

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (
            result - C1 * y * sh[..., 1] + C1 * z * sh[..., 2] - C1 * x * sh[..., 3]
        )
        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (
                result
                + C2[0] * xy * sh[..., 4]
                + C2[1] * yz * sh[..., 5]
                + C2[2] * (2.0 * zz - xx - yy) * sh[..., 6]
                + C2[3] * xz * sh[..., 7]
                + C2[4] * (xx - yy) * sh[..., 8]
            )

            if deg > 2:
                result = (
                    result
                    + C3[0] * y * (3 * xx - yy) * sh[..., 9]
                    + C3[1] * xy * z * sh[..., 10]
                    + C3[2] * y * (4 * zz - xx - yy) * sh[..., 11]
                    + C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12]
                    + C3[4] * x * (4 * zz - xx - yy) * sh[..., 13]
                    + C3[5] * z * (xx - yy) * sh[..., 14]
                    + C3[6] * x * (xx - 3 * yy) * sh[..., 15]
                )
                if deg > 3:
                    result = (
                        result
                        + C4[0] * xy * (xx - yy) * sh[..., 16]
                        + C4[1] * yz * (3 * xx - yy) * sh[..., 17]
                        + C4[2] * xy * (7 * zz - 1) * sh[..., 18]
                        + C4[3] * yz * (7 * zz - 3) * sh[..., 19]
                        + C4[4] * (zz * (35 * zz - 30) + 3) * sh[..., 20]
                        + C4[5] * xz * (7 * zz - 3) * sh[..., 21]
                        + C4[6] * (xx - yy) * (7 * zz - 1) * sh[..., 22]
                        + C4[7] * xz * (xx - 3 * yy) * sh[..., 23]
                        + C4[8]
                        * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))
                        * sh[..., 24]
                    )
    return result
