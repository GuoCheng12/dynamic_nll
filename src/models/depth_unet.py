import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
geffnet_path = os.path.join(project_root, 'third_party', 'gen-efficientnet-pytorch')

if geffnet_path not in sys.path:
    sys.path.append(geffnet_path)

try:
    import geffnet
    print(f"[DepthUNet] Successfully imported local geffnet from: {geffnet_path}")
except ImportError:
    print(f"\n[DepthUNet] CRITICAL ERROR: Could not import 'geffnet'.")
    print(f"Please ensure you have cloned the repo into: {geffnet_path}")
    print("Command: git clone https://github.com/rwightman/gen-efficientnet-pytorch.git third_party/gen-efficientnet-pytorch\n")
    raise

class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU()
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=2048):
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)

        self.up1 = UpSampleBN(skip_input=features // 1 + 112 + 64, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 40 + 24, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 24 + 16, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 16 + 8, output_features=features // 16)

        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        # [Correct Indices for geffnet / Original Author]
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[4], features[5], features[6], features[8], features[11]

        x_d0 = self.conv2(x_block4)
        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        out = self.conv3(x_d4)
        return out


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


# ==============================================================================
# 3. DepthUNet 主类 (支持本地权重加载 + 差分学习率)
# ==============================================================================

class DepthUNet(nn.Module):
    def __init__(self, encoder='tf_efficientnet_b5_ap', pretrained=True, min_depth=1e-3, max_val=10.0):
        super(DepthUNet, self).__init__()
        self.min_val = min_depth
        self.max_val = max_val

        initial_mean = 1.0
        self.init_mean_offset = math.log(math.exp(initial_mean - self.min_val) - 1)
        
        self.min_var = 1e-6
        initial_var = 1.0
        self.init_var_offset = math.log(math.exp(initial_var - self.min_var) - 1)
        self.max_var = 10.0 

        print(f"[DepthUNet] Initializing backbone: {encoder}")
        
        basemodel = geffnet.create_model(encoder, pretrained=False)
        
        if pretrained:
            weights_filename = "tf_efficientnet_b5_ap-9e82fae8.pth"
            weights_path = os.path.join(project_root, 'pretrained_weights', weights_filename)
            
            print(f"[DepthUNet] Attempting to load weights from: {os.path.abspath(weights_path)}")
            
            if os.path.exists(weights_path):
                print(f"[DepthUNet] ✅ Found file. Loading state_dict...")
                state_dict = torch.load(weights_path, map_location='cpu')
                basemodel.load_state_dict(state_dict, strict=True)
                print(f"[DepthUNet] ✅ Weights loaded successfully.")
            else:
                raise FileNotFoundError(f"❌ CRITICAL ERROR: Weight file not found at: {weights_path}")

        # Remove last layers
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        
        self.encoder = Encoder(basemodel)
        self.decoder = DecoderBN(num_classes=128)
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        enc_feats = self.encoder(x)
        unet_out = self.decoder(enc_feats)
        out = self.conv_out(unet_out)
        
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
            
        mean = out[:, :1]
        mean = F.softplus(mean + self.init_mean_offset) + self.min_val
        mean = torch.clamp(mean, self.min_val, self.max_val)

        var = out[:, 1:2]
        var = F.softplus(var + self.init_var_offset) + self.min_var
        var = torch.clamp(var, self.min_val, self.max_var)

        return mean, var

    def get_1x_lr_params(self):
        return self.encoder.parameters()

    def get_10x_lr_params(self):
        modules = [self.decoder, self.conv_out]
        for m in modules:
            yield from m.parameters()