import math
import os
import sys

import cv2
import hydra
import imageio
import MinkowskiEngine as ME
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from tqdm import tqdm

from xnerf.models import builder
from xnerf.utils.config import init_exp
from xnerf.utils.logger import DataLogger, board_writing
from xnerf.utils.render import mse2psnr, render_outputs, sample_ray, to8b


def train(cfg, train_loader, m, optimizer, criterion, writer):
    step_size = cfg.data_preset.step_size
    voxel_size = cfg.data_preset.voxel_size
    alpha_init = cfg.data_preset.alpha_init
    alpha_thr = cfg.data_preset.alpha_thr
    patch_sample = cfg.data_preset.patch_sample
    near, far = cfg.data_preset.min_depth * 0.9, cfg.data_preset.max_depth * 1.1

    sh_deg = cfg.model.sh_deg
    use_perceptual_loss = cfg.loss.use_perceptual_loss
    use_tv_loss = cfg.loss.use_tv_loss
    use_sn_loss = cfg.loss.use_sn_loss
    if use_perceptual_loss:
        assert patch_sample, "Perceptual loss must use patch sample."
        perceptual_loss_logger = DataLogger()
    if use_tv_loss:
        assert patch_sample, "TV loss must use patch sample."
        tv_loss_logger = DataLogger()
    if use_sn_loss:
        assert patch_sample, "SN loss must use patch sample."
        sn_loss_logger = DataLogger()

    rgb_loss_logger = DataLogger()
    total_loss_logger = DataLogger()
    depth_loss_logger = DataLogger()
    psnr_logger = DataLogger()

    m.train()

    train_loader = tqdm(train_loader, dynamic_ncols=True)

    for i, all_tensors in enumerate(train_loader):
        torch.cuda.empty_cache()
        all_tensors = [t.squeeze(0) for t in all_tensors]

        (
            discrete_coords,
            discrete_feats,
            rays_o,
            rays_d,
            viewdirs,
            rgb_target,
            depth_target,
            keep_region,
            rm,
        ) = all_tensors
        rays_o, rays_d, viewdirs, rgb_target, depth_target = (
            rays_o.cuda(),
            rays_d.cuda(),
            viewdirs.cuda(),
            rgb_target.cuda(),
            depth_target.cuda(),
        )
        if patch_sample:
            patch_h, patch_w = rays_o.shape[-3], rays_o.shape[-2]
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            viewdirs = viewdirs.reshape(-1, 3)
            rgb_target = rgb_target.reshape(-1, 3)
            depth_target = depth_target.reshape(-1)

        if rm is not None and rm.any():
            rm = rm.cuda()
            if viewdirs is not None:
                viewdirs = (
                    rm
                    @ torch.cat(
                        [viewdirs, torch.ones((viewdirs.shape[0], 1)).cuda()], 1
                    )[:, :, None]
                )
                viewdirs = viewdirs / viewdirs[:, -1:, :]
                viewdirs = viewdirs[:, :3, 0]

        keep_region = ME.utils.batched_coordinates([keep_region]).cuda()
        discrete_coords = ME.utils.batched_coordinates([discrete_coords]).cuda()
        discrete_feats = discrete_feats.float().cuda()

        inputs = ME.SparseTensor(discrete_feats, discrete_coords, device="cuda:0")
        cm = inputs.coordinate_manager
        target_key, _ = cm.insert_and_map(keep_region, string_id="target")
        outputs = m(inputs, target_key)

        del inputs, keep_region, target_key, discrete_coords, discrete_feats
        torch.cuda.empty_cache()
        for layer_idx, output in enumerate(reversed(outputs)):
            torch.cuda.empty_cache()

            b_size = rays_o.shape[0]
            _voxel_size = voxel_size * (2**layer_idx)

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
                voxel_size=_voxel_size,
            )

            rgb_output, depth_output, _ = render_outputs(
                output=output,
                ray_pts=ray_pts,
                ray_id=ray_id,
                step_id=step_id,
                b_size=b_size,
                step_size=step_size,
                voxel_size=_voxel_size,
                alpha_init=alpha_init,
                alpha_thr=alpha_thr,
                rm=rm,
                viewdirs=viewdirs,
                sh_deg=sh_deg,
            )

            if not use_perceptual_loss:
                loss_dict = criterion(
                    rgb_output, depth_output, rgb_target, depth_target
                )
            elif not use_sn_loss:
                loss_dict = criterion(
                    rgb_output,
                    depth_output,
                    rgb_target,
                    depth_target,
                    patch_h=patch_h,
                    patch_w=patch_w,
                )
            else:
                loss_dict = criterion(
                    rgb_output,
                    depth_output,
                    rgb_target,
                    depth_target,
                    patch_h=patch_h,
                    patch_w=patch_w,
                    rays_o=rays_o,
                    rays_d=rays_d,
                )
            psnr = mse2psnr(loss_dict["rgb"].detach().cpu())

            if layer_idx == 0:
                if use_perceptual_loss:
                    perceptual_loss_logger.update(
                        loss_dict["perceptual"].item(), b_size
                    )
                if use_tv_loss:
                    tv_loss_logger.update(loss_dict["tv"].item(), b_size)
                if use_sn_loss:
                    sn_loss_logger.update(loss_dict["sn"].item(), b_size)
                rgb_loss_logger.update(loss_dict["rgb"].item(), b_size)
                depth_loss_logger.update(loss_dict["depth"].item(), b_size)
                total_loss_logger.update(loss_dict["total"].item(), b_size)
                psnr_logger.update(psnr.item(), b_size)

                cfg.trainIters += 1

                info_dict = {}
                if use_perceptual_loss:
                    info_dict["percep_loss"] = perceptual_loss_logger.avg
                if use_tv_loss:
                    info_dict["tv_loss"] = tv_loss_logger.avg
                if use_sn_loss:
                    info_dict["sn_loss"] = sn_loss_logger.avg

                board_writing(
                    writer,
                    rgb_loss_logger.avg,
                    psnr_logger.avg,
                    depth_loss_logger.avg,
                    total_loss_logger.avg,
                    cfg.trainIters,
                    dataset="Train",
                    **info_dict,
                )
                loss = loss_dict["total"]
            else:
                loss += loss_dict["total"] / (4**layer_idx)

            del output
            torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        del rays_o, rays_d, rgb_target, depth_target
        del ray_pts, ray_id, step_id, rgb_output, depth_output, outputs, viewdirs
        del loss_dict, loss
        torch.cuda.empty_cache()

        tqdm_description = (
            f"total: {total_loss_logger.avg:.6f} | rgb: {rgb_loss_logger.avg:.6f} "
            f"| psnr: {psnr_logger.avg:.4f} | depth: {depth_loss_logger.avg:.6f}"
        )
        if use_perceptual_loss:
            tqdm_description += f" | percep: {perceptual_loss_logger.avg:.6f}"
        if use_tv_loss:
            tqdm_description += f" | tv: {tv_loss_logger.avg:.6f}"
        if use_sn_loss:
            tqdm_description += f" | sn: {sn_loss_logger.avg:.6f}"
        train_loader.set_description(tqdm_description)

    train_loader.close()

    return psnr_logger.avg, depth_loss_logger.avg


def validate(cfg, m, seen_view=False):
    is_multiscene = cfg.dataset.name == "multi_scene"
    if is_multiscene:
        backup_scene_id = cfg.dataset.scene_id
        cfg.dataset.scene_id = backup_scene_id + cfg.dataset.novel_scene_id

    val_datasets = builder.build_dataset(
        cfg.dataset,
        preset_cfg=cfg.data_preset,
        train=False,
        ray_batch_size=cfg.val.ray_batch_size,
        patch_size=cfg.val.get("patch_size", None),
        spiral=False,
        seen_view=seen_view,
        lazy_import=cfg.val.get("lazy_import", False),
    )
    val_datasets = val_datasets.sub_datasets if is_multiscene else [val_datasets]

    m.eval()

    sh_deg = cfg.model.sh_deg
    step_size = cfg.data_preset.step_size
    voxel_size = cfg.data_preset.voxel_size
    alpha_init = cfg.data_preset.alpha_init
    alpha_thr = cfg.data_preset.alpha_thr
    patch_sample = cfg.data_preset.patch_sample
    near, far = cfg.data_preset.min_depth * 0.9, cfg.data_preset.max_depth * 1.1

    if is_multiscene:
        scene_psnrs = []
        scene_depth_errs = []

    for scene_id, val_dataset in enumerate(val_datasets):
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, shuffle=False, num_workers=32
        )

        keep_region = ME.utils.batched_coordinates([val_dataset.keep_region]).cuda()
        discrete_coords = ME.utils.batched_coordinates(
            [val_dataset.discrete_coords]
        ).cuda()
        discrete_feats = val_dataset.discrete_feats.float().cuda()

        inputs = ME.SparseTensor(discrete_feats, discrete_coords, device="cuda:0")
        cm = inputs.coordinate_manager
        target_key, _ = cm.insert_and_map(keep_region, string_id="target")
        output = m(inputs, target_key)[-1]

        del keep_region, discrete_coords, discrete_feats, inputs
        torch.cuda.empty_cache()

        pred_rgbs = []
        pred_depths = []
        gt_rgbs = []
        gt_depths = []
        for all_tensors in tqdm(val_loader, dynamic_ncols=True):
            all_tensors = [t.squeeze(0).cuda() for t in all_tensors]
            rays_o, rays_d, viewdirs, rgb_target, depth_target = all_tensors
            if patch_sample:
                patch_h, patch_w = rays_o.shape[0], rays_o.shape[1]
                rays_o = rays_o.reshape(-1, 3)
                rays_d = rays_d.reshape(-1, 3)
                viewdirs = viewdirs.reshape(-1, 3)
                rgb_target = rgb_target.reshape(-1, 3)
                depth_target = depth_target.reshape(-1)

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
            pred_depths.append(depth_output.cpu().data.numpy())
            gt_rgbs.append(rgb_target.cpu().data.numpy())
            gt_depths.append(depth_target.cpu().data.numpy())

        del rays_o, rays_d, rgb_target, depth_target
        del ray_pts, ray_id, step_id, rgb_output, depth_output, output
        torch.cuda.empty_cache()

        img_shape = (-1, val_dataset._height, val_dataset._width)
        pred_rgbs = np.concatenate(pred_rgbs).reshape(*img_shape, 3)
        pred_depths = np.concatenate(pred_depths).reshape(*img_shape)
        gt_rgbs = np.concatenate(gt_rgbs).reshape(*img_shape, 3)
        gt_depths = np.concatenate(gt_depths).reshape(*img_shape)

        psnrs = []
        depth_errs = []

        prefix = f"scene_{scene_id}/" if is_multiscene else ""
        if is_multiscene and not os.path.exists(os.path.join(cfg.work_dir, prefix)):
            os.makedirs(os.path.join(cfg.work_dir, prefix))

        suffix = "seen_view" if seen_view else "unseen_view"
        for i in range(pred_rgbs.shape[0]):
            p = -10.0 * np.log10(np.mean(np.square(pred_rgbs[i] - gt_rgbs[i])))
            psnrs.append(p)

            gt_depth = gt_depths[i]
            depth_errs.append(
                np.mean(
                    np.square(pred_depths[i][gt_depth > 0] - gt_depth[gt_depth > 0])
                )
            )

            imageio.imwrite(
                os.path.join(cfg.work_dir, f"{prefix}rgb_{suffix}_{str(i)}.png"),
                to8b(pred_rgbs[i]),
            )
            imageio.imwrite(
                os.path.join(cfg.work_dir, f"{prefix}depth_{suffix}_{str(i)}.png"),
                to8b(pred_depths[i] / np.max(pred_depths[i])),
            )
            imageio.imwrite(
                os.path.join(cfg.work_dir, f"{prefix}gt_rgb_{suffix}_{str(i)}.png"),
                to8b(gt_rgbs[i]),
            )
            imageio.imwrite(
                os.path.join(cfg.work_dir, f"{prefix}gt_depth_{suffix}_{str(i)}.png"),
                to8b(gt_depths[i] / np.max(gt_depths[i])),
            )

        if not is_multiscene:
            return np.mean(psnrs), np.mean(depth_errs)
        else:
            scene_psnrs.append(np.mean(psnrs))
            scene_depth_errs.append(np.mean(depth_errs))
    if is_multiscene:
        cfg.dataset.scene_id = backup_scene_id

    return np.mean(scene_psnrs), np.mean(scene_depth_errs)


@hydra.main(config_path="../configs", config_name="train")
def main(cfg):
    cfg, logger = init_exp(cfg)
    logger.info("******************************")
    logger.info(cfg)
    logger.info("******************************")

    train_dataset = builder.build_dataset(
        cfg.dataset,
        preset_cfg=cfg.data_preset,
        train=True,
        ray_batch_size=cfg.train.ray_batch_size,
        patch_size=cfg.train.get("patch_size", None),
        spiral=False,
        lazy_import=cfg.train.get("lazy_import", False),
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=32
    )

    torch.backends.cudnn.benchmark = True

    # Model Initialize
    m = preset_model(cfg, logger)
    m.cuda()

    criterion = builder.build_loss(cfg.loss).cuda()

    if cfg.train.optimizer == "adam":
        optimizer = torch.optim.Adam(m.parameters(), cfg.train.lr)
    elif cfg.train.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            m.parameters(), cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
    elif cfg.train.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            m.parameters(),
            lr=cfg.train.lr,
            momentum=0.9,
            weight_decay=1e-4,
        )
    else:
        raise NotImplementedError

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.train.lr_step, gamma=cfg.train.lr_factor
    )

    writer = SummaryWriter(os.path.join(cfg.work_dir, "tensorboard"))

    for i in range(cfg.train.begin_epoch, cfg.train.end_epoch):
        train_dataset._shuffle()
        current_lr = optimizer.state_dict()["param_groups"][0]["lr"]

        logger.info(
            f"############# Starting Epoch {i} | lr: {current_lr} #############"
        )

        # Training
        psnr, depth_loss = train(cfg, train_loader, m, optimizer, criterion, writer)
        logger.epochInfo("Train", i, psnr, depth_loss)
        torch.cuda.empty_cache()

        lr_scheduler.step()

        if (i + 1) % cfg.snapshot == 0:
            # Save checkpoint
            torch.save(m.state_dict(), os.path.join(cfg.work_dir, "ckpt.pth"))
            # Prediction Test
            with torch.no_grad():
                unseen_psnr, unseen_depth_error = validate(cfg, m, seen_view=False)
                seen_psnr, seen_depth_error = validate(cfg, m, seen_view=True)
                logger.info(
                    f"##### Epoch {i} | unseen psnr: {unseen_psnr:.6f} | unseen depth error: {unseen_depth_error:.6f} | seen psnr: {seen_psnr:.6f} | seen depth error: {seen_depth_error:.6f} #####"
                )

    torch.save(m.state_dict(), os.path.join(cfg.work_dir, "ckpt.pth"))


def preset_model(cfg, logger):
    model = builder.build_model(cfg.model, preset_cfg=cfg.data_preset)

    if cfg.model.pretrained:
        logger.info(f"Loading model from {cfg.model.pretrained}...")
        model.load_state_dict(torch.load(cfg.model.pretrained))
    elif cfg.model.try_load:
        logger.info(f"Loading model from {cfg.model.try_load}...")
        pretrained_state = torch.load(cfg.model.try_load)
        model_state = model.state_dict()
        pretrained_state = {
            k: v
            for k, v in pretrained_state.items()
            if k in model_state and v.size() == model_state[k].size()
        }

        model_state.update(pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info("Create new model")
        logger.info("=> init weights")
        # model._initialize()

    return model


if __name__ == "__main__":
    main()
