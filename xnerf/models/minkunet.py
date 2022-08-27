# -----------------------------------------------------
# Written by Haoyi Zhu (zhuhaoyi@sjtu.edu.cn). All rights reserved.
# Modified from https://github.com/NVIDIA/MinkowskiEngine/blob/master/examples/unet.py
# ------------------------------------------------------------------------------
import MinkowskiEngine as ME
import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import MODEL
from .layers.Resnet import BasicBlock, Bottleneck, ResNetBase


class MinkUNetBase(ResNetBase):
    BLOCK = None
    PLANES = None
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    PLANES = (96, 128, 256, 512, 256, 128, 96, 64)
    INIT_DIM = 64
    OUT_TENSOR_STRIDE = 1

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling
    # initialize_coords
    def __init__(self, **cfg):
        self.is_prune = cfg["is_prune"]
        self.sh_deg = cfg["sh_deg"]
        self.num_stage = cfg["num_stage"]
        ResNetBase.__init__(
            self,
            cfg["in_channels"],
            cfg["out_channels"],
            expand_coordinates=cfg.get("expand_coordinates", False),
            D=3,
        )
        self._preset_cfg = cfg["preset"]
        self.alpha_init = torch.log(
            1 / (1 - torch.Tensor([self._preset_cfg.alpha_init])) - 1
        ).cuda()
        self.interval = torch.Tensor([self._preset_cfg.step_size]).cuda()
        self.alpha_thr = torch.Tensor([self._preset_cfg.alpha_thr * 0.1]).cuda()

    def network_initialization(self, in_channels, out_channels, expand_coordinates, D):
        if self.sh_deg > 0:
            out_channels = 1 + 3 * (self.sh_deg + 1) ** 2

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = ME.MinkowskiConvolution(
            in_channels,
            self.inplanes,
            kernel_size=5,
            expand_coordinates=expand_coordinates,
            dimension=D,
        )

        self.bn0 = ME.MinkowskiBatchNorm(self.inplanes)

        self.conv1p1s2 = ME.MinkowskiConvolution(
            self.inplanes,
            self.PLANES[0],
            kernel_size=2,
            stride=2,
            expand_coordinates=expand_coordinates,
            dimension=D,
        )
        self.bn1 = ME.MinkowskiBatchNorm(self.PLANES[0])

        self.inplanes = self.PLANES[0]

        self.block1 = self._make_layer(self.BLOCK, self.PLANES[0], self.LAYERS[0])

        self.conv2p2s2 = ME.MinkowskiConvolution(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            expand_coordinates=expand_coordinates,
            dimension=D,
        )
        self.bn2 = ME.MinkowskiBatchNorm(self.inplanes)

        self.block2 = self._make_layer(self.BLOCK, self.PLANES[1], self.LAYERS[1])

        self.conv3p4s2 = ME.MinkowskiConvolution(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            expand_coordinates=expand_coordinates,
            dimension=D,
        )

        self.bn3 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block3 = self._make_layer(self.BLOCK, self.PLANES[2], self.LAYERS[2])

        self.conv4p8s2 = ME.MinkowskiConvolution(
            self.inplanes,
            self.inplanes,
            kernel_size=2,
            stride=2,
            expand_coordinates=expand_coordinates,
            dimension=D,
        )
        self.bn4 = ME.MinkowskiBatchNorm(self.inplanes)
        self.block4 = self._make_layer(self.BLOCK, self.PLANES[3], self.LAYERS[3])

        self.convtr4p16s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            self.inplanes, self.PLANES[4], kernel_size=2, stride=2, dimension=D
        )
        self.bntr4 = ME.MinkowskiBatchNorm(self.PLANES[4])

        self.inplanes = self.PLANES[4] * self.BLOCK.expansion
        self.block5 = self._make_layer(self.BLOCK, self.PLANES[4], self.LAYERS[4])
        self.convtr5p8s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            self.inplanes, self.PLANES[5], kernel_size=2, stride=2, dimension=D
        )
        self.bntr5 = ME.MinkowskiBatchNorm(self.PLANES[5])

        self.inplanes = self.PLANES[5] * self.BLOCK.expansion
        self.block6 = self._make_layer(self.BLOCK, self.PLANES[5], self.LAYERS[5])
        self.convtr6p4s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            self.inplanes, self.PLANES[6], kernel_size=2, stride=2, dimension=D
        )
        self.bntr6 = ME.MinkowskiBatchNorm(self.PLANES[6])

        self.inplanes = self.PLANES[6] * self.BLOCK.expansion
        self.block7 = self._make_layer(self.BLOCK, self.PLANES[6], self.LAYERS[6])
        self.convtr7p2s2 = ME.MinkowskiGenerativeConvolutionTranspose(
            self.inplanes, self.PLANES[7], kernel_size=2, stride=2, dimension=D
        )
        self.bntr7 = ME.MinkowskiBatchNorm(self.PLANES[7])

        self.inplanes = self.PLANES[7]
        self.block8 = self._make_layer(self.BLOCK, self.PLANES[7], self.LAYERS[7])

        self.out_final = ME.MinkowskiConvolution(
            self.PLANES[7] * self.BLOCK.expansion,
            out_channels,
            kernel_size=1,
            bias=True,
            expand_coordinates=expand_coordinates,
            dimension=D,
        )

        if self.is_prune:
            self.pruning = ME.MinkowskiPruning()

        if self.num_stage >= 4:
            self.out_1 = ME.MinkowskiConvolution(
                self.PLANES[4] * self.BLOCK.expansion,
                out_channels,
                kernel_size=1,
                bias=True,
                expand_coordinates=expand_coordinates,
                dimension=D,
            )
        if self.num_stage >= 3:
            self.out_2 = ME.MinkowskiConvolution(
                self.PLANES[5] * self.BLOCK.expansion,
                out_channels,
                kernel_size=1,
                bias=True,
                expand_coordinates=expand_coordinates,
                dimension=D,
            )
        if self.num_stage >= 2:
            self.out_3 = ME.MinkowskiConvolution(
                self.PLANES[6] * self.BLOCK.expansion,
                out_channels,
                kernel_size=1,
                bias=True,
                expand_coordinates=expand_coordinates,
                dimension=D,
            )

        self.elu = ME.MinkowskiELU()

    def get_target(self, out, target_key, kernel_size=1):
        with torch.no_grad():
            target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
            cm = out.coordinate_manager
            strided_target_key = cm.stride(
                target_key,
                out.tensor_stride[0],
            )
            kernel_map = cm.kernel_map(
                out.coordinate_map_key,
                strided_target_key,
                kernel_size=kernel_size,
                region_type=1,
            )
            for k, curr_in in kernel_map.items():
                target[curr_in[0].long()] = 1
        return target

    def forward(self, x, target_key):
        outputs = []

        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.elu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.elu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.elu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.elu(out)
        out_b3p8 = self.block3(out)

        # tensor_stride=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.elu(out)
        out = self.block4(out)

        # tensor_stride=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.elu(out)

        out = out + out_b3p8
        out = self.block5(out)

        if self.num_stage >= 4:
            if self.training or self.is_prune:
                hidden_output1 = self.out_1(out)
                outputs.append(hidden_output1)

            if self.is_prune:
                keep1 = (
                    1
                    - (
                        (1 + torch.exp(hidden_output1.F[:, 0] + self.alpha_init))
                        ** (-self.interval)
                    )
                    > self.alpha_thr
                ).squeeze()
                target = self.get_target(out, target_key)
                if True or self.training:
                    keep1 += target
                out = self.pruning(out, keep1)

        # tensor_stride=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.elu(out)
        out = out + out_b2p4
        out = self.block6(out)

        if self.num_stage >= 3:
            if self.training or self.is_prune:
                hidden_output2 = self.out_2(out)
                outputs.append(hidden_output2)

            if self.is_prune:
                keep2 = (
                    1
                    - (
                        (1 + torch.exp(hidden_output2.F[:, 0] + self.alpha_init))
                        ** (-self.interval)
                    )
                    > self.alpha_thr
                ).squeeze()
                target = self.get_target(out, target_key)

                if True or self.training:
                    keep2 += target
                out = self.pruning(out, keep2)

        # tensor_stride=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.elu(out)
        out = out + out_b1p2
        out = self.block7(out)

        if self.num_stage >= 2:
            if self.training or self.is_prune:
                hidden_output3 = self.out_3(out)
                outputs.append(hidden_output3)
            if self.is_prune:
                keep3 = (
                    1
                    - (
                        (1 + torch.exp(hidden_output3.F[:, 0] + self.alpha_init))
                        ** (-self.interval)
                    )
                    > self.alpha_thr
                ).squeeze()
                target = self.get_target(out, target_key)

                if True or self.training:
                    keep3 += target
                out = self.pruning(out, keep3)

        # tensor_stride=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.elu(out)
        out = out + out_p1
        out = self.block8(out)

        hidden_output4 = self.out_final(out)
        outputs.append(hidden_output4)

        return outputs

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)
        # return


@MODEL.register_module
class MinkUNet14(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)


@MODEL.register_module
class MinkUNet14Wide(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (1, 1, 1, 1, 1, 1, 1, 1)
    PLANES = (128, 256, 512, 1024, 512, 256, 128, 64)


@MODEL.register_module
class MinkUNet18(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)


@MODEL.register_module
class MinkUNet34(MinkUNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


@MODEL.register_module
class MinkUNet50(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)


@MODEL.register_module
class MinkUNet101(MinkUNetBase):
    BLOCK = Bottleneck
    LAYERS = (2, 3, 4, 23, 2, 2, 2, 2)


@MODEL.register_module
class MinkUNet14A(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


@MODEL.register_module
class MinkUNet14B(MinkUNet14):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


@MODEL.register_module
class MinkUNet14C(MinkUNet14):
    PLANES = (32, 64, 128, 256, 192, 192, 128, 128)


@MODEL.register_module
class MinkUNet14D(MinkUNet14):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


@MODEL.register_module
class MinkUNet18A(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 96, 96)


@MODEL.register_module
class MinkUNet18B(MinkUNet18):
    PLANES = (32, 64, 128, 256, 128, 128, 128, 128)


@MODEL.register_module
class MinkUNet18D(MinkUNet18):
    PLANES = (32, 64, 128, 256, 384, 384, 384, 384)


@MODEL.register_module
class MinkUNet34A(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 64)


@MODEL.register_module
class MinkUNet34B(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 64, 32)


@MODEL.register_module
class MinkUNet34C(MinkUNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)
