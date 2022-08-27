# -----------------------------------------------------
# Written by Haoyi Zhu (zhuhaoyi@sjtu.edu.cn). All rights reserved.
# -----------------------------------------------------

import torch
import torch.nn.functional as F


class DataLogger(object):
    """Average data logger."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.value = 0
        self.sum = 0
        self.cnt = 0
        self.avg = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.cnt += n
        self._cal_avg()

    def _cal_avg(self):
        self.avg = self.sum / self.cnt


def board_writing(
    writer,
    rgb_loss,
    psnr,
    depth_loss,
    total_loss,
    iterations,
    dataset="Train",
    **losses,
):
    writer.add_scalar(f"{dataset}/rgb_loss", rgb_loss, iterations)
    writer.add_scalar(f"{dataset}/psnr", psnr, iterations)
    writer.add_scalar(f"{dataset}/depth_loss", depth_loss, iterations)
    writer.add_scalar(f"{dataset}/total_loss", total_loss, iterations)
    for key, value in losses.items():
        writer.add_scalar(f"{dataset}/{key}", value, iterations)
