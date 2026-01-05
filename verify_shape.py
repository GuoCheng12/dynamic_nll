import torch
import sys
import os

# 1. Setup paths
sys.path.append(os.getcwd())
geffnet_path = os.path.join(os.getcwd(), 'third_party', 'gen-efficientnet-pytorch')
sys.path.append(geffnet_path)

# 2. Import
try:
    from src.models.depth_unet import DepthUNet
except ImportError as e:
    print(f"Import Error: {e}")
    exit(1)

def verify_shapes():
    print("--- 📐 Architecture Verification (Physical Scale Check) ---")
    # Initialize model (don't care about weights here, just shape)
    model = DepthUNet(encoder='tf_efficientnet_b5_ap', pretrained=False)
    model.eval()
    
    # Mock Input (NYU Size)
    x = torch.randn(1, 3, 480, 640)
    print(f"Input Shape: {x.shape}")
    
    with torch.no_grad():
        # Run Encoder Manually
        features = model.encoder(x)
        
    # We expect these indices to match specific scales
    # Indices used in DecoderBN: [4, 5, 6, 8, 11]
    
    expectations = {
        4:  {"name": "Block 0", "scale": "1/2",  "h": 240, "w": 320, "ch": 24},
        5:  {"name": "Block 1", "scale": "1/4",  "h": 120, "w": 160, "ch": 40},
        6:  {"name": "Block 2", "scale": "1/8",  "h": 60,  "w": 80,  "ch": 64},
        8:  {"name": "Block 4", "scale": "1/16", "h": 30,  "w": 40,  "ch": 176},
        11: {"name": "ConvHead", "scale": "1/32", "h": 15,  "w": 20,  "ch": 2048},
    }
    
    all_pass = True
    
    print(f"\n{'Index':<6} | {'Expected':<15} | {'Actual Shape':<20} | {'Status':<10}")
    print("-" * 60)
    
    for i, feat in enumerate(features):
        shape = list(feat.shape)
        h, w = shape[2], shape[3]
        ch = shape[1]
        
        status = ""
        if i in expectations:
            exp = expectations[i]
            # Check Scale (Tolerance +/- 1 pixel due to padding)
            scale_ok = (abs(h - exp['h']) <= 1) and (abs(w - exp['w']) <= 1)
            # Check Channels (Must be exact)
            ch_ok = (ch == exp['ch'])
            
            if scale_ok and ch_ok:
                status = "✅ OK"
            else:
                status = f"❌ MISMATCH! (Want {exp['h']}x{exp['w']}, {exp['ch']}ch)"
                all_pass = False
            
            print(f"{i:<6} | {exp['scale']:<15} | {str(shape):<20} | {status}")
        else:
            # Unused layers
            pass
            # print(f"{i:<6} | {'(Unused)':<15} | {str(shape):<20} | -")

    print("-" * 60)
    if all_pass:
        print("\n🚀 CONCLUSION: Architecture is 100% CORRECT.")
    else:
        print("\n💀 CONCLUSION: Architecture is BROKEN. Indices are wrong.")

if __name__ == "__main__":
    verify_shapes()