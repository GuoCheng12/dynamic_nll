import torch
import os
import sys
import numpy as np

# 1. 设置路径，确保能 import 你的 src
sys.path.append(os.getcwd())

# 2. 引入你的模型类
from src.models.depth_unet import DepthUNet

def get_layer_stats(model):
    """获取第一层卷积的均值和标准差，作为指纹"""
    # 获取 Encoder 的第一个卷积层
    # 在 geffnet/efficientnet 中，通常是 conv_stem
    first_conv = model.encoder.original_model.conv_stem
    weight = first_conv.weight.data
    return weight.mean().item(), weight.std().item(), weight

def check_pretrain_loading():
    print("--- 🕵️‍♀️ Pretrain Weight Investigation ---")
    
    # Path to your .pth file
    weights_path = "/datasets/workspace/dynamic_nll/pretrained_weights/tf_efficientnet_b5_ap-9e82fae8.pth"
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weight file not found at {weights_path}")
        return

    # A. 加载 .pth 文件本身看看它的指纹
    print(f"1. Inspecting raw .pth file: {weights_path}")
    state_dict = torch.load(weights_path, map_location='cpu')
    # 找到 conv_stem 的权重
    if 'conv_stem.weight' in state_dict:
        pth_weight = state_dict['conv_stem.weight']
        print(f"   Ref (.pth) Conv Stem: Mean={pth_weight.mean():.6f}, Std={pth_weight.std():.6f}")
    else:
        print("   ❓ Could not find 'conv_stem.weight' in .pth file keys. Listing first 5 keys:")
        print(list(state_dict.keys())[:5])
        return

    # B. 初始化一个【不加载权重】的模型 (Random Init)
    print("\n2. Initializing Dummy Model (pretrained=False)...")
    model_random = DepthUNet(encoder='tf_efficientnet_b5_ap', pretrained=False)
    rand_mean, rand_std, rand_w = get_layer_stats(model_random)
    print(f"   Random Init Conv Stem: Mean={rand_mean:.6f}, Std={rand_std:.6f}")

    # C. 初始化你的【加载权重】的模型 (Your Logic)
    print("\n3. Initializing Your Model (pretrained=True)...")
    try:
        model_loaded = DepthUNet(encoder='tf_efficientnet_b5_ap', pretrained=True)
        load_mean, load_std, load_w = get_layer_stats(model_loaded)
        print(f"   Loaded Model Conv Stem: Mean={load_mean:.6f}, Std={load_std:.6f}")
    except Exception as e:
        print(f"❌ Crash during loading: {e}")
        return

    # D. 最终判决
    print("\n--- ⚖️ Verdict ---")
    
    # 比较 Random vs Loaded
    # ... (前面的代码不变)

    # D. 最终判决 (修改这部分)
    print("\n--- ⚖️ Verdict ---")
    
    diff = abs(load_mean - pth_weight.mean().item())
    print(f"   [DEBUG] Difference: {diff:.9f}")
    
    if diff < 1e-5: # 放宽一点点标准，如果是 0.000001 这种级别，就是通过
        print("✅ [PASS] Loaded model matches .pth file (within precision tolerance).")
        print("   👉 Conclusion: Weight Loading is SUCCESSFUL. The problem is elsewhere.")
    else:
        print("❌ [FAIL] Weights are significantly different!")
        print(f"   Values -> Pth: {pth_weight.mean().item():.6f} | Loaded: {load_mean:.6f}")

if __name__ == "__main__":
    check_pretrain_loading()