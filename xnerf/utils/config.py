import logging
import os
from types import MethodType


def epochInfo(self, set, idx, psnr, depth_loss):
    self.info(
        "{set}-{idx:d} epoch | psnr:{psnr:.8f} | depth_loss:{depth_loss:.4f}".format(
            set=set, idx=idx, psnr=psnr, depth_loss=depth_loss
        )
    )


def init_exp(cfg, logfile="training.log"):
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)

    filehandler = logging.FileHandler(os.path.join(cfg.work_dir, logfile))
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    logger.epochInfo = MethodType(epochInfo, logger)

    return cfg, logger
