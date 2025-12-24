import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 1. Original Author's Helper Classes (EXACT COPY) ---

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
        # [REVERTED TO ORIGINAL INDICES]
        # Since we use gen-efficientnet-pytorch, these are correct again.
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


# --- 2. DepthUNet Wrapper (Modified for Hub Load) ---

class DepthUNet(nn.Module):
    def __init__(self, encoder='tf_efficientnet_b5_ap', pretrained=True, min_depth=1e-3, max_val=10.0):
        super(DepthUNet, self).__init__()
        self.min_val = min_depth
        self.max_val = max_val
        
        # Init Offsets (Critical for NLL convergence)
        # Using Log(Exp(x)-1) inverse to match softplus
        initial_mean = 1.5 # Start slightly higher than 1.0 to be safe
        self.init_mean_offset = math.log(math.exp(initial_mean - self.min_val) - 1)
        
        self.min_var = 1e-6
        initial_var = 1.0
        self.init_var_offset = math.log(math.exp(initial_var - self.min_var) - 1)
        self.max_var = 10.0 

        # --- Load Backbone via Torch Hub (Original Way) ---
        print(f"Loading backbone from torch.hub: rwightman/gen-efficientnet-pytorch, model: {encoder}")
        # Note: This requires internet access on the first run to download the code and weights
        try:
            basemodel = torch.hub.load('rwightman/gen-efficientnet-pytorch', encoder, pretrained=pretrained)
        except Exception as e:
            print("Failed to load via torch.hub. Please ensure you have internet access or the repo is cached.")
            raise e

        # Remove last layers (Original Author Logic)
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        
        # Wrap with Author's Encoder
        self.encoder = Encoder(basemodel)
        
        # Decoder (output 128 channels)
        self.decoder = DecoderBN(num_classes=128)
        
        # Prediction Head (128 -> 2 channels: Mean, Var)
        # We replace the author's Bin head with our Gaussian head
        self.conv_out = nn.Sequential(
            nn.Conv2d(128, 2, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        # Encoder -> Decoder
        enc_feats = self.encoder(x)
        unet_out = self.decoder(enc_feats)
        
        # Prediction Head
        out = self.conv_out(unet_out)
        
        # Resolution Fix: Interpolate to input size if needed
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode='bilinear', align_corners=True)
            
        # Split and Post-process (Softplus + Offset)
        mean = out[:, :1]
        mean = F.softplus(mean + self.init_mean_offset) + self.min_val
        mean = torch.clamp(mean, self.min_val, self.max_val)

        var = out[:, 1:2]
        var = F.softplus(var + self.init_var_offset) + self.min_var
        var = torch.clamp(var, self.min_val, self.max_var)

        # Return order: Depth, Variance (Matches your train.py expectation)
        return mean, var

    def get_1x_lr_params(self):
        """Returns generator for backbone parameters (lr/10)"""
        return self.encoder.parameters()

    def get_10x_lr_params(self):
        """Returns generator for decoder/head parameters (lr)"""
        modules = [self.decoder, self.conv_out]
        for m in modules:
            yield from m.parameters()