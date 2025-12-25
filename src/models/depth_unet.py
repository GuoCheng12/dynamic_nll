import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# ==============================================================================
# 1. 环境配置：加载本地 geffnet 库
# ==============================================================================

# 获取当前文件 (src/models/depth_unet.py) 的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 推导项目根目录 (假设当前在 src/models，向上两级到达项目根目录)
project_root = os.path.dirname(os.path.dirname(current_dir))
# 拼接 third_party/gen-efficientnet-pytorch 的路径
geffnet_path = os.path.join(project_root, 'third_party', 'gen-efficientnet-pytorch')

# 将该路径加入 python 搜索路径，以便 import geffnet
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

# ==============================================================================
# 2. 原作者的辅助类 (完全复刻，保证结构一致)
# ==============================================================================

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
        # 这对应的是 gen-efficientnet-pytorch 的层级结构
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
        
        # --- Init Offsets for Softplus ---
        # 确保初始输出均值在 1.5 左右，方差在 1.0 左右
        initial_mean = 1.0
        self.init_mean_offset = math.log(math.exp(initial_mean - self.min_val) - 1)
        
        self.min_var = 1e-6
        initial_var = 1.0
        self.init_var_offset = math.log(math.exp(initial_var - self.min_var) - 1)
        self.max_var = 10.0 

        # --- Load Backbone (Offline) ---
        print(f"[DepthUNet] Initializing backbone: {encoder}")
        
        # 1. 创建模型结构 (pretrained=False, 因为我们要手动加载文件)
        # 注意：这里使用的是本地 import 的 geffnet
        basemodel = geffnet.create_model(encoder, pretrained=False)
        
        # 2. 手动加载本地权重
        if pretrained:
            # 权重文件硬编码，或者你可以改为传参
            weights_filename = "tf_efficientnet_b5_ap-9e82e2b5.pth"
            weights_path = os.path.join(project_root, 'pretrained_weights', weights_filename)
            
            if os.path.exists(weights_path):
                print(f"[DepthUNet] Loading local weights from: {weights_path}")
                state_dict = torch.load(weights_path, map_location='cpu')
                # strict=True 保证所有 key 完美匹配，如果不匹配会报错，帮我们发现问题
                basemodel.load_state_dict(state_dict, strict=True)
            else:
                print(f"[DepthUNet] WARNING: Weights file not found at: {weights_path}")
                print("[DepthUNet] Initializing with RANDOM weights. Metrics will be bad initially.")
                # 如果你想强制必须有权重，这里可以 raise FileNotFoundError

        # Remove last layers (Classification Head)
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        
        # Wrap with Author's Encoder
        self.encoder = Encoder(basemodel)
        
        # Decoder (output 128 channels)
        self.decoder = DecoderBN(num_classes=128)
        
        # Prediction Head (128 -> 2 channels: Mean, Variance)
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        # 1. Encoder Pass
        enc_feats = self.encoder(x)
        
        # 2. Decoder Pass
        unet_out = self.decoder(enc_feats)
        
        # 3. Head Pass
        out = self.conv_out(unet_out)
        
        # 4. Upsample to Input Resolution (if needed)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
            
        # 5. Split Mean/Var and Apply Constraints
        mean = out[:, :1]
        mean = F.softplus(mean + self.init_mean_offset) + self.min_val
        mean = torch.clamp(mean, self.min_val, self.max_val)

        var = out[:, 1:2]
        var = F.softplus(var + self.init_var_offset) + self.min_var
        var = torch.clamp(var, self.min_val, self.max_var)

        return mean, var

    def get_1x_lr_params(self):
        """返回 Backbone 的参数迭代器 (用于 0.1x LR)"""
        return self.encoder.parameters()

    def get_10x_lr_params(self):
        """返回 Decoder 和 Head 的参数迭代器 (用于 1.0x LR)"""
        modules = [self.decoder, self.conv_out]
        for m in modules:
            yield from m.parameters()