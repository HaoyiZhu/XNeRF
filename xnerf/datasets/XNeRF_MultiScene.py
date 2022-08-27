# -----------------------------------------------------
# Written by Haoyi Zhu (zhuhaoyi@sjtu.edu.cn). All rights reserved.
# -----------------------------------------------------
"""XNeRF multi scene dataset."""
import copy
import os

import cv2
import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

from xnerf.datasets.XNeRF_SingleScene import XNeRFSingleScene
from xnerf.models.builder import DATASET
from xnerf.utils.render import get_rays_of_a_view


@DATASET.register_module
class XNeRFMultiScene(torch.utils.data.Dataset):
    """XNeRF multi scene dataset."""

    def __init__(
        self,
        train,
        ray_batch_size=10240,
        patch_size=None,
        spiral=False,
        seen_view=False,
        lazy_import=False,
        **cfg,
    ):
        self.train = train

        self._cfg = cfg
        self.scene_ids = cfg["scene_id"]

        self.sub_datasets = []
        for s_id in self.scene_ids:
            # create single scene config
            ss_cfg = copy.deepcopy(self._cfg)
            ss_cfg["scene_id"] = s_id
            self.sub_datasets.append(
                XNeRFSingleScene(
                    train=train,
                    ray_batch_size=ray_batch_size,
                    patch_size=patch_size,
                    spiral=spiral,
                    seen_view=seen_view,
                    **ss_cfg,
                )
            )

        self.sub_len = len(self.sub_datasets[0])
        self.total_len = self.sub_len * len(self.sub_datasets)
        for ss_idx, sub_dataset in enumerate(self.sub_datasets):
            assert (
                len(sub_dataset) == self.sub_len
            ), f"{ss_idx}th scene length {len(sub_dataset)} != {self.sub_len}"

        self.idx_mapping = np.arange(self.total_len)

    def __getitem__(self, idx):
        mapped_idx = self.idx_mapping[idx]

        return self.sub_datasets[mapped_idx // self.sub_len][mapped_idx % self.sub_len]

    def __len__(self):
        return self.total_len

    def _shuffle(self):
        if self.train:
            np.random.shuffle(self.idx_mapping)
