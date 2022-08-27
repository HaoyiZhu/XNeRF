# -----------------------------------------------------
# Written by Haoyi Zhu (zhuhaoyi@sjtu.edu.cn). All rights reserved.
# -----------------------------------------------------
"""XNeRF single scene dataset."""
import math
import os
import random

import cv2
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from xnerf.models.builder import DATASET
from xnerf.utils.render import get_rays_of_a_view


@DATASET.register_module
class XNeRFSingleScene(torch.utils.data.Dataset):
    """XNeRF single scene dataset."""

    _width = 640
    _height = 360

    def __init__(
        self,
        train,
        ray_batch_size=10240,
        patch_size=None,
        spiral=False,
        seen_view=False,
        lazy_import=False,
        no_pcds=False,
        **cfg,
    ):
        super(XNeRFSingleScene, self).__init__()
        self.train = train
        self.ray_batch_size = ray_batch_size  # num of rays per batch
        self.spiral = spiral
        self.seen_view = seen_view  # test on seen views
        self.lazy_import = lazy_import
        self.no_pcds = no_pcds and (not self.train)
        if self.spiral:
            assert not self.train, "Only when testing is spiral supported."

        self._cfg = cfg
        self._preset_cfg = cfg["preset"]
        self._root = cfg["root"]
        self._scene_id = cfg["scene_id"]
        self.test_view_indexes = cfg["novel_view_indexes"]

        self.calib = np.load(
            os.path.join(self._root, f"scene{self._scene_id}", "calib.npy"),
            allow_pickle=True,
        ).item()
        if self.train or self.spiral or self.seen_view:
            self.n_view = len(self.calib.keys()) - len(self.test_view_indexes)
        else:
            self.n_view = len(self.test_view_indexes)
        # print(f'{self.n_view} views')

        self._downsample_voxel_size_m = self._preset_cfg.downsample_voxel_size_m
        self._min_depth = self._preset_cfg.min_depth
        self._max_depth = self._preset_cfg.max_depth
        self._smooth_pcd = self._preset_cfg.smooth_pcd.apply
        if self._smooth_pcd:
            self._filter_num_neighbor = self._preset_cfg.smooth_pcd.filter_num_neighbor
            self._filter_std_ratio = self._preset_cfg.smooth_pcd.filter_std_ratio
            self._filter_radius_m = self._preset_cfg.smooth_pcd.filter_radius_m

        self._x_range = self._preset_cfg.x_range
        self._y_range = self._preset_cfg.y_range
        self._z_range = self._preset_cfg.z_range
        self._voxel_size = self._preset_cfg.voxel_size
        self._step_size = self._preset_cfg.step_size

        self._rot_p = self._preset_cfg.get("rot_p", 0)
        self.set_keep_region = self._preset_cfg.keep_region.apply
        if self.set_keep_region:
            self._margin = self._preset_cfg.keep_region.margin
            if self._rot_p <= 0:
                self.keep_region = None

        self.patch_sample = self._preset_cfg.get("patch_sample", False) and self.train
        if self.patch_sample:
            self.patch_size = patch_size
            self.num_patch_row = math.ceil(self._width / self.patch_size[1])
            self.num_patch_col = math.ceil(self._height / self.patch_size[0])
            self.num_patch_per_view = self.num_patch_col * self.num_patch_row
            self.patch_index = [
                np.arange(self.num_patch_per_view) for _ in range(self.n_view)
            ]

        self._load_items()

        if self.spiral:
            self.n_view = len(self.items["intrinsics"])
            self.N_views = self._preset_cfg.get("N_views", 20)
            self.N_rots = self._preset_cfg.get("N_rots", 1)
            self.rads = self._preset_cfg.get("rads", [0.05, 0.05, 0.05])

        if (not self.train) and (not self.no_pcds):
            coords, feats = self._get_pcd()
            (
                self.discrete_coords,
                self.discrete_feats,
                self.keep_region,
            ) = self._get_inputs(coords, feats)

    def __getitem__(self, idx):
        if not self.patch_sample:
            rays_o = self.items["rays_o"][
                idx * self.ray_batch_size : (idx + 1) * self.ray_batch_size
            ]
            rays_d = self.items["rays_d"][
                idx * self.ray_batch_size : (idx + 1) * self.ray_batch_size
            ]
            viewdirs = self.items["viewdirs"][
                idx * self.ray_batch_size : (idx + 1) * self.ray_batch_size
            ]

            rgb_target = self.items["rgbs"][
                idx * self.ray_batch_size : (idx + 1) * self.ray_batch_size
            ]
            depth_target = self.items["depths"][
                idx * self.ray_batch_size : (idx + 1) * self.ray_batch_size
            ]
        else:
            patch_start, patch_end = [], []
            for p_idx in self.patch_index:
                patch_idx_row = math.floor(p_idx[idx] // self.num_patch_row)
                patch_idx_col = p_idx[idx] - self.num_patch_row * patch_idx_row
                # random shift patch
                shift_row = np.random.rand() - 0.5
                shift_col = np.random.rand() - 0.5
                patch_idx_row = np.clip(
                    patch_idx_row + shift_row, 0, self.num_patch_col - 1
                )
                patch_idx_col = np.clip(
                    patch_idx_col + shift_col, 0, self.num_patch_row - 1
                )

                patch_start.append(
                    (
                        int(patch_idx_row * self.patch_size[0]),
                        int(patch_idx_col * self.patch_size[1]),
                    )
                )
                patch_end.append(
                    (
                        int((patch_idx_row + 1) * self.patch_size[0]),
                        int((patch_idx_col + 1) * self.patch_size[1]),
                    )
                )

            rays_o = [
                self.items["rays_o"][
                    i,
                    patch_start[i][0] : patch_end[i][0],
                    patch_start[i][1] : patch_end[i][1],
                    :,
                ]
                for i in range(self.n_view)
            ]
            rays_d = [
                self.items["rays_d"][
                    i,
                    patch_start[i][0] : patch_end[i][0],
                    patch_start[i][1] : patch_end[i][1],
                    :,
                ]
                for i in range(self.n_view)
            ]
            viewdirs = [
                self.items["viewdirs"][
                    i,
                    patch_start[i][0] : patch_end[i][0],
                    patch_start[i][1] : patch_end[i][1],
                    :,
                ]
                for i in range(self.n_view)
            ]
            rgb_target = [
                self.items["rgbs"][
                    i,
                    patch_start[i][0] : patch_end[i][0],
                    patch_start[i][1] : patch_end[i][1],
                    :,
                ]
                for i in range(self.n_view)
            ]
            depth_target = [
                self.items["depths"][
                    i,
                    patch_start[i][0] : patch_end[i][0],
                    patch_start[i][1] : patch_end[i][1],
                ]
                for i in range(self.n_view)
            ]

            idx2 = random.choice(range(self.num_patch_per_view))
            while idx2 == idx:
                idx2 = random.choice(range(self.num_patch_per_view))
            idx = idx2
            patch_start, patch_end = [], []
            for p_idx in self.patch_index:
                patch_idx_row = math.floor(p_idx[idx] // self.num_patch_row)
                patch_idx_col = p_idx[idx] - self.num_patch_row * patch_idx_row
                # random shift patch
                shift_row = np.random.rand() - 0.5
                shift_col = np.random.rand() - 0.5
                patch_idx_row = np.clip(
                    patch_idx_row + shift_row, 0, self.num_patch_col - 1
                )
                patch_idx_col = np.clip(
                    patch_idx_col + shift_col, 0, self.num_patch_row - 1
                )

                patch_start.append(
                    (
                        int(patch_idx_row * self.patch_size[0]),
                        int(patch_idx_col * self.patch_size[1]),
                    )
                )
                patch_end.append(
                    (
                        int((patch_idx_row + 1) * self.patch_size[0]),
                        int((patch_idx_col + 1) * self.patch_size[1]),
                    )
                )

            rays_o = [
                self.items["rays_o"][
                    i,
                    patch_start[i][0] : patch_end[i][0],
                    patch_start[i][1] : patch_end[i][1],
                    :,
                ]
                for i in range(self.n_view)
            ]
            rays_d = [
                self.items["rays_d"][
                    i,
                    patch_start[i][0] : patch_end[i][0],
                    patch_start[i][1] : patch_end[i][1],
                    :,
                ]
                for i in range(self.n_view)
            ]
            viewdirs = [
                self.items["viewdirs"][
                    i,
                    patch_start[i][0] : patch_end[i][0],
                    patch_start[i][1] : patch_end[i][1],
                    :,
                ]
                for i in range(self.n_view)
            ]
            rgb_target = [
                self.items["rgbs"][
                    i,
                    patch_start[i][0] : patch_end[i][0],
                    patch_start[i][1] : patch_end[i][1],
                    :,
                ]
                for i in range(self.n_view)
            ]
            depth_target = [
                self.items["depths"][
                    i,
                    patch_start[i][0] : patch_end[i][0],
                    patch_start[i][1] : patch_end[i][1],
                ]
                for i in range(self.n_view)
            ]

            rays_o = torch.stack(rays_o)
            rays_d = torch.stack(rays_d)
            viewdirs = torch.stack(viewdirs)
            rgb_target = torch.stack(rgb_target)
            depth_target = torch.stack(depth_target)

        if self.train:
            coords, feats = self._get_pcd()

            rm = torch.tensor([0])
            if np.random.rand() < self._rot_p:
                random_theta = (
                    np.random.rand(3) - np.array([0.5, 0.5, 0.5])
                ) * np.array([1.0, 0.05, 0.05]) + np.array([0.5, 0.5, 0.5])
                rm = self._rotation_matrix_zyz(random_theta)

                coords = (
                    rm
                    @ np.concatenate([coords, np.ones((coords.shape[0], 1))], 1)[
                        :, :, None
                    ]
                )
                coords = coords / coords[:, -1:, :]
                coords = coords[:, :3, 0]

                rm = torch.FloatTensor(rm)

            discrete_coords, discrete_feats, keep_region = self._get_inputs(
                coords, feats
            )

            return (
                discrete_coords,
                discrete_feats,
                rays_o,
                rays_d,
                viewdirs,
                rgb_target,
                depth_target,
                keep_region,
                rm,
            )
        else:
            return rays_o, rays_d, viewdirs, rgb_target, depth_target

    def __len__(self):
        if not self.patch_sample:
            total_len = self._width * self._height * self.n_view
            return math.ceil(total_len / self.ray_batch_size)
        else:
            return self.num_patch_per_view

    def _shuffle(self):
        if self.train:
            if not self.patch_sample:
                index = np.arange(self._width * self._height * self.n_view)
                np.random.shuffle(index)
                self.items["rays_o"] = self.items["rays_o"][index]
                self.items["rays_d"] = self.items["rays_d"][index]
                self.items["viewdirs"] = self.items["viewdirs"][index]
                self.items["rgbs"] = self.items["rgbs"][index]
                self.items["depths"] = self.items["depths"][index]
            else:
                for p_idx in range(self.n_view):
                    np.random.shuffle(self.patch_index[p_idx])

    def _get_inputs(self, coords, feats):
        discrete_coords, discrete_feats = ME.utils.sparse_quantize(
            coordinates=coords,
            features=feats,
            quantization_size=self._voxel_size,
            ignore_label=-100,
        )
        discrete_feats = torch.from_numpy(discrete_feats)

        keep_region = self._set_keep_region(discrete_coords)

        return discrete_coords, discrete_feats, keep_region

    def _get_pcd(self):
        pcds = self.items["pcds"]

        if self.train:
            np.random.shuffle(pcds)

        full_pcd = pcds[0]
        for i in range(1, len(pcds)):
            full_pcd = self._merge_pointclouds(pcds[i], full_pcd)

        coords = np.asarray(full_pcd.points)
        feats = np.asarray(full_pcd.colors)

        return coords, feats

    def _set_keep_region(self, discrete_coords):
        keep_region = torch.LongTensor([[0, 0, 0]])
        if self.set_keep_region:
            if self._rot_p <= 0 and self.keep_region is not None:
                return self.keep_region
            grid_w = (
                int(
                    (self._x_range[1] - self._x_range[0] + self._margin * 2)
                    / self._voxel_size
                )
                + 2
            )
            grid_h = (
                int(
                    (self._y_range[1] - self._y_range[0] + self._margin * 2)
                    / self._voxel_size
                )
                + 2
            )
            grid_d = (
                int(
                    (self._z_range[1] - self._z_range[0] + self._margin * 2)
                    / self._voxel_size
                )
                + 2
            )
            grid = torch.zeros((grid_w, grid_h, grid_d))
            origin = torch.LongTensor(
                [
                    (self._x_range[0] - self._margin) / self._voxel_size - 1,
                    (self._y_range[0] - self._margin) / self._voxel_size - 1,
                    (self._z_range[0] - self._margin) / self._voxel_size - 1,
                ]
            )
            grid_margin = int(self._margin / self._voxel_size + 0.5)
            for i in range(discrete_coords.shape[0]):
                x, y, z = discrete_coords[i] - origin
                grid[
                    x - grid_margin : x + grid_margin + 1,
                    y - grid_margin : y + grid_margin + 1,
                    z - grid_margin : z + grid_margin + 1,
                ] = 1
            keep_region = (grid > 0).nonzero() + origin

            if self._rot_p <= 0 and self.keep_region is None:
                self.keep_region = keep_region

        return keep_region

    def _merge_pointclouds(self, pcd1, pcd2):
        merged_points = np.vstack((np.asarray(pcd1.points), np.asarray(pcd2.points)))
        merged_colors = np.vstack((np.asarray(pcd1.colors), np.asarray(pcd2.colors)))

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

        if self._smooth_pcd:
            _, ind = merged_pcd.remove_statistical_outlier(
                nb_neighbors=self._filter_num_neighbor, std_ratio=self._filter_std_ratio
            )
            merged_pcd = merged_pcd.select_by_index(ind)

        return merged_pcd

    def _rgbd_to_pointcloud(
        self,
        color_image_name,
        depth_image_name,
        width,
        height,
        camera_matrix,
        extrinsic=np.eye(4),
        depth_scale=1000,
        use_o3d=False,
    ):
        color = cv2.cvtColor(cv2.imread(color_image_name), cv2.COLOR_BGR2RGB)
        depths = cv2.imread(depth_image_name, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depths[np.isnan(depths)] = 0
        depths /= depth_scale  # from millimeters to meters

        assert color.shape[:2] == depths.shape[:2], self._scene_id

        if use_o3d:
            depths[depths < self._min_depth] = 0
            depths[depths > self._max_depth] = 0

            rgbd_image = o3d.geometry.RGBDImage()
            rgbd_image = rgbd_image.create_from_color_and_depth(
                o3d.geometry.Image(np.copy(color)),
                o3d.geometry.Image(np.copy(depths)),
                depth_scale=1.0,
                convert_rgb_to_intensity=False,
            )

            intrinsic = o3d.camera.PinholeCameraIntrinsic()
            intrinsic.set_intrinsics(
                int(width),
                int(height),
                camera_matrix[0, 0],
                camera_matrix[1, 1],
                camera_matrix[0, 2],
                camera_matrix[1, 2],
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                intrinsic,
                extrinsic=extrinsic,
                project_valid_depth_only=True,
            )
            return pcd, color, depths
        else:
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

            xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
            xmap, ymap = np.meshgrid(xmap, ymap)

            points_z = depths
            points_x = (xmap - cx) / fx * points_z
            points_y = (ymap - cy) / fy * points_z

            mask = (points_z > self._min_depth) & (points_z < self._max_depth)
            points = np.stack([points_x, points_y, points_z], axis=-1)
            points = points[mask].astype(np.float32).reshape(-1, 3)

            colors = np.copy(color[mask]).astype(np.float32).reshape(-1, 3)

            points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
            points = np.linalg.inv(extrinsic) @ points[:, :, None]
            points = points / points[:, -1:, :]
            points = points[:, :3]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255)

        return pcd, color, points_z * mask

    def _filter_pointclouds(self, pcds, num_neighbors, std_ratio, radius):
        for i in range(len(pcds)):
            _, ind = pcds[i].remove_statistical_outlier(
                nb_neighbors=num_neighbors, std_ratio=std_ratio
            )
            pcds[i] = pcds[i].select_by_index(ind)
            if radius > 0:
                _, ind = pcds[i].remove_radius_outlier(
                    nb_points=num_neighbors, radius=radius
                )
                pcds[i] = pcds[i].select_by_index(ind)

        return pcds

    def _load_items(self):
        """loading training and testing data, including points clouds, rgbs, depths, etc."""
        if self.lazy_import and os.path.exists(
            os.path.join(self._root, f"scene{self._scene_id}", "items.npy")
        ):
            items = np.load(
                os.path.join(
                    self._root,
                    f"scene{self._scene_id}",
                    f"{'' if self.seen_view else 'un'}seen_view_items.npy",
                ),
                allow_pickle=True,
            ).item()
            self.items = {}
            self.items["pcds"] = [
                self._construct_pcd(items["pcd_points"][i], items["pcd_colors"][i])
                for i in range(len(items["pcd_points"]))
            ]
            self.items["rgbs"] = torch.from_numpy(items["rgbs"])
            self.items["depths"] = torch.from_numpy(items["depths"])
            self.items["intrinsics"] = [
                torch.from_numpy(intrinsic) for intrinsic in items["intrinsics"]
            ]
            self.items["extrinsics"] = [
                torch.from_numpy(extrinsic) for extrinsic in items["extrinsics"]
            ]
            self.items["poses"] = torch.from_numpy(items["poses"])
            self.items["rays_o"] = torch.from_numpy(items["rays_o"])
            self.items["rays_d"] = torch.from_numpy(items["rays_d"])
            self.items["viewdirs"] = torch.from_numpy(items["viewdirs"])
            return

        pcds = []
        rgbs = []
        depths = []
        intrinsics = []
        extrinsics = []
        poses = []

        for view_id in self.calib:
            idx = int(view_id[-1])

            rgb_path = os.path.join(
                self._root, f"scene{self._scene_id}", view_id, "color.png"
            )
            depth_path = os.path.join(
                self._root, f"scene{self._scene_id}", view_id, "depth.png"
            )

            pcd, rgb, depth = self._rgbd_to_pointcloud(
                rgb_path,
                depth_path,
                self._width,
                self._height,
                self.calib[view_id]["intrinsic"][0:2][0:2],
                self.calib[view_id]["extrinsic"][0],
                depth_scale=self.calib[view_id]["depth_scale"],
            )

            if (
                idx not in self.test_view_indexes
            ):  # input pointcloud cannot contain test view
                pcds.append(pcd)

            if ((self.train or self.spiral) and idx not in self.test_view_indexes) or (
                (not self.train and not self.spiral)
                and (
                    (self.seen_view and idx not in self.test_view_indexes)
                    or (not self.seen_view and idx in self.test_view_indexes)
                )
            ):
                rgbs.append(torch.from_numpy(rgb / 255).float())
                depths.append(torch.from_numpy(depth))

                intrinsics.append(torch.from_numpy(self.calib[view_id]["intrinsic"]))
                extrinsics.append(torch.from_numpy(self.calib[view_id]["extrinsic"][0]))
                poses.append(
                    torch.from_numpy(
                        self._extrinsic_to_camera_pose(
                            self.calib[view_id]["extrinsic"][0]
                        )
                    )
                )

            if self.spiral:
                c2w = self._extrinsic_to_camera_pose(
                    self.calib[view_id]["extrinsic"][0]
                )
                focal = (
                    self.calib[view_id]["intrinsic"][0, 0]
                    + self.calib[view_id]["intrinsic"][1, 1]
                ) / 2
                up = c2w[None, :3, 1].sum(0)
                render_poses = self._render_path_spiral(
                    c2w,
                    up,
                    self.rads,
                    focal,
                    zrate=0.5,
                    rots=self.N_rots,
                    N=self.N_views,
                )

                for j in range(len(render_poses)):
                    rgbs.append(torch.zeros(rgb.shape).float())
                    depths.append(torch.zeros(depth.shape).float())
                    intrinsics.append(self.calib[view_id]["intrinsic"])
                    poses.append(torch.from_numpy(render_poses[j]).float())

        # pre-process point clouds
        pcds = [pcd.voxel_down_sample(self._downsample_voxel_size_m) for pcd in pcds]
        if self._smooth_pcd:
            pcds = self._filter_pointclouds(
                pcds,
                self._filter_num_neighbor,
                self._filter_std_ratio,
                self._filter_radius_m,
            )

        all_rays_o, all_rays_d, all_viewdirs = [], [], []
        for i in range(len(poses)):
            rays_o, rays_d, viewdirs = get_rays_of_a_view(
                self._height,
                self._width,
                intrinsics[i][:3, :3],
                poses[i],
                mode="lefttop",
                padding=0,
            )
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)
            all_viewdirs.append(viewdirs)

        if self.lazy_import:
            items = {
                "pcd_points": [np.asarray(pcd.points) for pcd in pcds],
                "pcd_colors": [np.asarray(pcd.colors) for pcd in pcds],
                "rgbs": np.stack(rgbs).reshape(-1, 3)
                if not self.patch_sample
                else np.stack(rgbs),
                "depths": np.stack(depths).reshape(-1)
                if not self.patch_sample
                else np.stack(depths),
                "intrinsics": [intrinsic.numpy() for intrinsic in intrinsics],
                "extrinsics": [extrinsic.numpy() for extrinsic in extrinsics],
                "poses": np.stack(poses),
                "rays_o": np.stack(all_rays_o).reshape(-1, 3)
                if not self.patch_sample
                else np.stack(all_rays_o),
                "rays_d": np.stack(all_rays_d).reshape(-1, 3)
                if not self.patch_sample
                else np.stack(all_rays_d),
                "viewdirs": np.stack(all_viewdirs).reshape(-1, 3)
                if not self.patch_sample
                else np.stack(all_viewdirs),
            }
            np.save(
                os.path.join(
                    self._root,
                    f"scene{self._scene_id}",
                    f"{'' if self.seen_view else 'un'}seen_view_items.npy",
                ),
                items,
                allow_pickle=True,
            )

        self.items = {
            "pcds": pcds,
            "rgbs": torch.stack(rgbs).reshape(-1, 3)
            if not self.patch_sample
            else torch.stack(rgbs),
            "depths": torch.stack(depths).reshape(-1)
            if not self.patch_sample
            else torch.stack(depths),
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "poses": torch.stack(poses),
            "rays_o": torch.stack(all_rays_o).reshape(-1, 3)
            if not self.patch_sample
            else torch.stack(all_rays_o),
            "rays_d": torch.stack(all_rays_d).reshape(-1, 3)
            if not self.patch_sample
            else torch.stack(all_rays_d),
            "viewdirs": torch.stack(all_viewdirs).reshape(-1, 3)
            if not self.patch_sample
            else torch.stack(all_viewdirs),
        }

    def _construct_pcd(self, points, colors):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd

    def _multi_resolution_pcd_downsample(self, pcd):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.points)

        inner_area = (
            (points[:, 0] > -1.0)
            * (points[:, 0] < 1.0)
            * (points[:, 1] > -1.0)
            * (points[:, 1] < 1.0)
            * (points[:, 2] > -0.1)
            * (points[:, 2] < 1.0)
        )
        inner_pcd = o3d.geometry.PointCloud()
        inner_pcd.points = o3d.utility.Vector3dVector(points[inner_area])
        inner_pcd.colors = o3d.utility.Vector3dVector(colors[inner_area])

        outer_pcd = o3d.geometry.PointCloud()
        outer_pcd.points = o3d.utility.Vector3dVector(points[~inner_area])
        outer_pcd.colors = o3d.utility.Vector3dVector(colors[~inner_area])

        inner_pcd = inner_pcd.voxel_down_sample(self._downsample_voxel_size_m)
        outer_pcd = outer_pcd.voxel_down_sample(self._downsample_voxel_size_m * 2)

        final_pcd = o3d.geometry.PointCloud()
        final_pcd.points = o3d.utility.Vector3dVector(
            np.concatenate([np.asarray(inner_pcd.points), np.asarray(outer_pcd.points)])
        )
        final_pcd.colors = o3d.utility.Vector3dVector(
            np.concatenate([np.asarray(inner_pcd.colors), np.asarray(outer_pcd.colors)])
        )

        return final_pcd

    def _extrinsic_to_camera_pose(self, extrinsic):
        return np.linalg.inv(extrinsic)[:3, :4].astype(np.float32)

    def _rotation_matrix_axis(self, dim, theta):
        # x-axis
        if dim == 0:
            rm = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, np.cos(theta), -np.sin(theta)],
                    [0.0, np.sin(theta), np.cos(theta)],
                ]
            )
        # y-axis
        elif dim == 1:
            rm = np.array(
                [
                    [np.cos(theta), 0.0, np.sin(theta)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(theta), 0.0, np.cos(theta)],
                ]
            )
        # z-axis
        elif dim == 2:
            rm = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta), np.cos(theta), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
        else:
            raise

        return rm

    def _rotation_matrix_zyz(self, params):
        phi = params[0] * 2 * np.pi - np.pi
        theta = params[1] * 2 * np.pi - np.pi
        psi_t = params[2] * 2 * np.pi - np.pi

        # loc_r = params[3:6] * 2 - 1
        loc_r = np.array([0, 0, 0])

        # first rotate about z axis for angle psi_t
        a1 = self._rotation_matrix_axis(2, psi_t)
        a2 = self._rotation_matrix_axis(1, theta)
        a3 = self._rotation_matrix_axis(0, phi)
        rm = np.dot(np.dot(a3, a2), a1)

        rm = np.transpose(rm)

        c = np.dot(-rm, loc_r[:, None])

        rm = rm.reshape(-1)

        theta = np.concatenate(
            [rm[:3], c[0], rm[3:6], c[1], rm[6:9], c[2], [0, 0, 0, 1]]
        ).reshape(-1, 4)

        return theta

    def _normalize(self, x):
        return x / np.linalg.norm(x)

    def _viewmatrix(self, z, up, pos):
        vec2 = self._normalize(z)
        vec1_avg = up
        vec0 = self._normalize(np.cross(vec1_avg, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def _render_path_spiral(self, c2w, up, rads, focal, zrate, rots, N):
        render_poses = []
        rads = np.array(list(rads) + [1.0])
        hwf = c2w[:, 4:5]

        for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
            c = np.dot(
                c2w[:3, :4],
                np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
                * rads,
            )
            z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
            render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
        return render_poses
