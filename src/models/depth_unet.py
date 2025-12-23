import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models
except ImportError as exc:  # pragma: no cover
    raise ImportError("torchvision is required for DepthUNet") from exc


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DepthUNet(nn.Module):
    """
    U-Net with a ResNet encoder and 2-channel output (mean, variance).
    Applies softplus + offset, then clamps to [min, max].
    """

    def __init__(
        self,
        encoder: str = "resnet50",
        pretrained: bool = True,
        min_depth: float = 1e-3,
        min_var: float = 1e-6,
        max_val: float = 10.0,
    ):
        super().__init__()
        self.min_depth = min_depth
        self.min_var = min_var
        self.max_val = max_val

        encoder_fn = getattr(models, encoder, None)
        if encoder_fn is None:
            raise ValueError(f"Unsupported encoder: {encoder}")
        weights = "DEFAULT" if pretrained else None
        backbone = encoder_fn(weights=weights)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.up4 = UpBlock(2048, 1024, 512)
        self.up3 = UpBlock(512, 512, 256)
        self.up2 = UpBlock(256, 256, 128)
        self.up1 = UpBlock(128, 64, 64)
        self.conv_out = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x: torch.Tensor):
        x0 = self.layer0(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)

        out = self.conv_out(d1)
        out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=True)

        mean = F.softplus(out[:, :1]) + self.min_depth
        var = F.softplus(out[:, 1:2]) + self.min_var
        mean = torch.clamp(mean, self.min_depth, self.max_val)
        var = torch.clamp(var, self.min_var, self.max_val)
        return mean, var
