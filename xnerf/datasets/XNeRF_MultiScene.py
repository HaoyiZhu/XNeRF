# -----------------------------------------------------
# Written by Haoyi Zhu (zhuhaoyi@sjtu.edu.cn). All rights reserved.
# -----------------------------------------------------
"""X-NeRF multi scene dataset.
"""

import copy
import os
from typing import Any, Dict, List, Optional

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
    """
    Class for X-NeRF multi scene dataset.
    This class is implemented by concatenating multiple single scene datasets.

    Args:
        train (bool): Whether to use train split or not.
        ray_batch_size (int): Number of rays in each batch.
                Will be ignored when using patch sampling. Default: ``10240``.
        patch_size (List[int]): Size of patch when using patch sampling.
                Only useful during training. Default: ``None``.
        spiral (bool): Whether to use spiral sampling.
                Only useful during inference. Default: ``False``.
        seen_view (bool): Whether to use seen views. Default: ``False``.
        lazy_import (bool): Whether to save a backup for loaded items to import data lazily.
                Be careful to use this option when customizing your settings. Default: ``False``.
        no_pcds (bool): Whether to load point clouds.
                Useful after radiance field has been saved. Default: ``False``.
        cfg (dict): Config dict for dataset settings.
    """

    def __init__(
        self,
        train: bool,
        ray_batch_size: Optional[int] = 10240,
        patch_size: Optional[List[int]] = None,
        spiral: Optional[bool] = False,
        seen_view: Optional[bool] = False,
        lazy_import: Optional[bool] = False,
        no_pcds: Optional[bool] = False,
        **cfg: Dict[str, Any],
    ) -> None:
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
                    lazy_import=lazy_import,
                    no_pcds=no_pcds,
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
        """
        Shuffle the dataset. This method should be called before each iteration during training.
        """

        if self.train:
            np.random.shuffle(self.idx_mapping)
