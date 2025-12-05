from __future__ import annotations

import torch
import torch.nn as nn


def init_weights(net: nn.Module, init_type: str = "normal", gain: float = 0.02) -> None:
    def init_func(m: nn.Module) -> None:
        classname = m.__class__.__name__
        if hasattr(m, "weight") and ("Conv" in classname or "Linear" in classname):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            else:
                raise NotImplementedError(f"init method {init_type} not implemented")
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm2d" in classname:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


class ResnetBlock(nn.Module):
    def __init__(self, dim: int, padding_type: str = "reflect") -> None:
        super().__init__()
        p = 0
        if padding_type == "reflect":
            self.pad = nn.ReflectionPad2d(1)
        elif padding_type == "replicate":
            self.pad = nn.ReplicationPad2d(1)
        elif padding_type == "zero":
            self.pad = nn.ZeroPad2d(1)
        else:
            raise NotImplementedError

        self.conv_block = nn.Sequential(
            self.pad,
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),
            self.pad,
            nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """
    ResNet-based generator from CycleGAN (Johnson et al. style).
    """

    def __init__(self, input_nc: int = 1, output_nc: int = 1, ngf: int = 64, n_blocks: int = 6):
        super().__init__()
        assert n_blocks >= 0
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=False),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
        ]

        # Downsample
        mult = 1
        for _ in range(2):
            model += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult * 2),
                nn.ReLU(True),
            ]
            mult *= 2

        # Residual blocks
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        # Upsample
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(ngf * mult, ngf * mult // 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(ngf * mult // 2),
                nn.ReLU(True),
            ]
            mult //= 2

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator from CycleGAN/Pix2Pix.
    """

    def __init__(self, input_nc: int = 1, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)
