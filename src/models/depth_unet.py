import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError as exc:  # pragma: no cover
    raise ImportError("timm is required for DepthUNet") from exc


class UpSampleBN(nn.Module):
    def __init__(self, skip_input: int, output_features: int):
        super().__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x: torch.Tensor, concat_with: torch.Tensor) -> torch.Tensor:
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode="bilinear", align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features: int = 2048, num_classes: int = 1, bottleneck_features: int = 2048):
        super().__init__()
        features = int(num_features)
        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )
        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        out = self.conv3(x_d4)
        return out


class Encoder(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.original_model = backbone

    def forward(self, x: torch.Tensor):
        feats = self.original_model(x)
        features = [None] * 12
        features[4] = feats[0]
        features[5] = feats[1]
        features[6] = feats[2]
        features[8] = feats[3]
        features[11] = feats[4]
        return features


def softmax_inverse(x: float) -> float:
    return math.log(math.exp(x) - 1)


class DepthUNet(nn.Module):
    def __init__(
        self,
        encoder: str = "tf_efficientnet_b5_ap",
        pretrained: bool = True,
        min_depth: float = 1e-3,
        max_val: float = 10.0,
    ):
        super().__init__()
        self.min_val = min_depth
        self.max_val = max_val
        initial_mean = 1.0
        self.init_mean_offset = softmax_inverse(initial_mean - self.min_val)

        self.min_var = 1e-6
        initial_var = 1.0
        self.init_var_offset = softmax_inverse(initial_var - self.min_var)
        self.max_var = 10.0

        backbone = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        self.encoder = Encoder(backbone)
        self.decoder = DecoderBN(num_classes=128)
        self.conv_out = nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor):
        out = self.decoder(self.encoder(x))
        out = self.conv_out(out)

        mean = out[:, :1]
        mean = F.softplus(mean + self.init_mean_offset) + self.min_val
        mean = torch.clamp(mean, self.min_val, self.max_val)

        var = out[:, 1:2]
        var = F.softplus(var + self.init_var_offset) + self.min_var
        var = torch.clamp(var, self.min_var, self.max_var)
        return mean, var
