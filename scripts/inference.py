"""Demo script to inference spiral view videos."""
import os

import hydra
import imageio
import MinkowskiEngine as ME
import numpy as np
import torch
from tqdm import tqdm

from xnerf.models import builder
from xnerf.utils.config import init_exp
from xnerf.utils.metrics import depth_err, rgb_lpips, rgb_psnr, rgb_ssim
from xnerf.utils.render import mse2psnr, render_outputs, sample_ray, to8b


def inference(cfg, m):
    is_multiscene = cfg.dataset.name == "multi_scene"
    if is_multiscene:
        backup_scene_id = cfg.dataset.scene_id
        cfg.dataset.scene_id = backup_scene_id + cfg.dataset.novel_scene_id

    inference_datasets = builder.build_dataset(
        cfg.dataset,
        preset_cfg=cfg.data_preset,
        train=False,
        ray_batch_size=cfg.inference.ray_batch_size,
        patch_size=None,
        spiral=True,
        seen_view=False,
        lazy_import=cfg.inference.get("lazy_import", False),
        no_pcds=(m is None),
    )
    inference_datasets = (
        inference_datasets.sub_datasets if is_multiscene else [inference_datasets]
    )

    if m:
        m.eval()
        radiance_field = {}
    else:
        radiance_field = np.load(
            os.path.join(cfg.work_dir, "radiance_field.npy"), allow_pickle=True
        )[()]

    sh_deg = cfg.model.sh_deg
    step_size = cfg.data_preset.step_size
    voxel_size = cfg.data_preset.voxel_size
    alpha_init = cfg.data_preset.alpha_init
    alpha_thr = cfg.data_preset.alpha_thr
    near, far = cfg.data_preset.min_depth * 0.9, cfg.data_preset.max_depth * 1.1

    for inference_dataset in inference_datasets:
        scene_id = inference_dataset._scene_id
        inference_loader = torch.utils.data.DataLoader(
            inference_dataset, batch_size=1, shuffle=False, num_workers=32
        )

        if m:
            keep_region = ME.utils.batched_coordinates(
                [inference_dataset.keep_region]
            ).cuda()
            discrete_coords = ME.utils.batched_coordinates(
                [inference_dataset.discrete_coords]
            ).cuda()
            discrete_feats = inference_dataset.discrete_feats.float().cuda()

            inputs = ME.SparseTensor(
                features=discrete_feats, coordinates=discrete_coords, device="cuda:0"
            )
            cm = inputs.coordinate_manager
            target_key, _ = cm.insert_and_map(keep_region, string_id="target")
            output = m(inputs, target_key)[-1]

            del keep_region, discrete_coords, discrete_feats, inputs
            torch.cuda.empty_cache()

            radiance_field[scene_id] = {
                "coords": output.C.cpu().numpy(),
                "feats": output.F.cpu().numpy(),
            }
        else:
            coords = torch.from_numpy(radiance_field[scene_id]["coords"]).cuda()
            feats = torch.from_numpy(radiance_field[scene_id]["feats"]).float().cuda()
            output = ME.SparseTensor(
                features=feats, coordinates=coords, device="cuda:0"
            )

        pred_rgbs = []
        pred_depths = []
        gt_rgbs = []
        gt_depths = []
        for all_tensors in tqdm(inference_loader, dynamic_ncols=True):
            all_tensors = [t.squeeze(0).cuda() for t in all_tensors]
            rays_o, rays_d, viewdirs, rgb_target, depth_target = all_tensors

            b_size = rays_o.shape[0]

            ray_pts, ray_id, step_id = sample_ray(
                rays_o,
                rays_d,
                near=near,
                far=far,
                stepsize=step_size,
                xyz_min=torch.Tensor(
                    [
                        cfg.data_preset.x_range[0],
                        cfg.data_preset.y_range[0],
                        cfg.data_preset.z_range[0],
                    ]
                ).cuda(),
                xyz_max=torch.Tensor(
                    [
                        cfg.data_preset.x_range[1],
                        cfg.data_preset.y_range[1],
                        cfg.data_preset.z_range[1],
                    ]
                ).cuda(),
                voxel_size=voxel_size,
            )

            rgb_output, depth_output, _ = render_outputs(
                output=output,
                discrete_coords=None,
                ray_pts=ray_pts,
                ray_id=ray_id,
                step_id=step_id,
                b_size=b_size,
                step_size=step_size,
                voxel_size=voxel_size,
                alpha_init=alpha_init,
                alpha_thr=alpha_thr,
                rm=None,
                sh_deg=sh_deg,
                viewdirs=viewdirs,
            )

            pred_rgbs.append(rgb_output.cpu().data.numpy())
            pred_depths.append(
                depth_output.cpu().data.numpy()
                / np.max(depth_output.cpu().data.numpy())
            )
            gt_rgbs.append(rgb_target.cpu().data.numpy())
            gt_depths.append(depth_target.cpu().data.numpy())

        del rays_o, rays_d, rgb_target, depth_target
        del ray_pts, ray_id, step_id, rgb_output, depth_output, output
        torch.cuda.empty_cache()

        img_shape = (-1, inference_dataset._height, inference_dataset._width)
        pred_rgbs = np.concatenate(pred_rgbs).reshape(*img_shape, 3)
        pred_depths = np.concatenate(pred_depths).reshape(*img_shape)
        gt_rgbs = np.concatenate(gt_rgbs).reshape(*img_shape, 3)
        gt_depths = np.concatenate(gt_depths).reshape(*img_shape)

        prefix = f"inference_results/" if is_multiscene else ""
        if is_multiscene and not os.path.exists(os.path.join(cfg.work_dir, prefix)):
            os.makedirs(os.path.join(cfg.work_dir, prefix))

        imageio.mimwrite(
            os.path.join(cfg.work_dir, f"{prefix}scene_{scene_id}_rgb_video.mp4"),
            to8b(pred_rgbs),
            fps=30,
            quality=8,
        )
        imageio.mimwrite(
            os.path.join(cfg.work_dir, f"{prefix}scene_{scene_id}_depth_video.mp4"),
            to8b(pred_depths),
            fps=30,
            quality=8,
        )

    if m:
        np.save(os.path.join(cfg.work_dir, "radiance_field.npy"), radiance_field)

    return


@hydra.main(config_path="../configs", config_name="inference")
def main(cfg):
    cfg, logger = init_exp(cfg, "inference.log")
    logger.info("******************************")
    logger.info(cfg)
    logger.info("******************************")

    torch.backends.cudnn.benchmark = True

    if os.path.exists(os.path.join(cfg.work_dir, "radiance_field.npy")):
        m = None
    else:
        print("No radiance field found.")
        m = preset_model(cfg, logger)
        m.cuda()

    with torch.no_grad():
        inference(cfg, m)


def preset_model(cfg, logger):
    model = builder.build_model(cfg.model, preset_cfg=cfg.data_preset)

    assert os.path.exists(cfg.ckpt_path), f"{cfg.ckpt_path} does not exist"

    logger.info(f"Loading model from {cfg.ckpt_path}...")
    model.load_state_dict(torch.load(cfg.ckpt_path))

    return model


if __name__ == "__main__":
    main()
