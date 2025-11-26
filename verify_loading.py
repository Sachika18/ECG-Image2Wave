import torch
from app import ResNetUNetDigitizer
import numpy as np

print("=" * 80)
print("CHECKPOINT LOADING VERIFICATION")
print("=" * 80)

# Create model WITHOUT loading checkpoint
print("\n1. Model with RANDOM initialization:")
model_random = ResNetUNetDigitizer(out_T=10250, enc_dim=512, fuse_dim=256)
first_weight_random = model_random.encoder.proj.weight.data.clone()
print(f"   encoder.proj.weight[0,0:5] = {first_weight_random[0, :5]}")
print(f"   encoder.proj.weight mean = {first_weight_random.mean():.6f}")
print(f"   encoder.proj.weight std  = {first_weight_random.std():.6f}")

# Now load checkpoint
print("\n2. Loading checkpoint 'resnet_unet_best.pth'...")
try:
    ckpt = torch.load('resnet_unet_best.pth', map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    print(f"   Checkpoint has {len(ckpt)} keys")
    
    # Load into model
    model_random.load_state_dict(ckpt, strict=False)
    print("   Checkpoint loaded successfully with strict=False")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()

# Check if weights changed
print("\n3. Model AFTER loading checkpoint:")
first_weight_loaded = model_random.encoder.proj.weight.data.clone()
print(f"   encoder.proj.weight[0,0:5] = {first_weight_loaded[0, :5]}")
print(f"   encoder.proj.weight mean = {first_weight_loaded.mean():.6f}")
print(f"   encoder.proj.weight std  = {first_weight_loaded.std():.6f}")

# Check if weights actually changed
weight_diff = (first_weight_random - first_weight_loaded).abs().sum().item()
print(f"\n4. Weight difference (abs sum): {weight_diff:.6f}")
if weight_diff < 1e-6:
    print("   ⚠️  WARNING: Weights did NOT change after loading checkpoint!")
    print("   This suggests the checkpoint is NOT being loaded.")
else:
    print("   ✓ Weights CHANGED after loading checkpoint (good)")

# Also check decoder weights
print("\n5. Checking decoder.final weights:")
decoder_final_weight = model_random.decoder.final.weight.data.clone()
print(f"   decoder.final.weight[0,0:5] = {decoder_final_weight[0, :5]}")
print(f"   decoder.final.weight mean = {decoder_final_weight.mean():.6f}")
print(f"   decoder.final.weight std  = {decoder_final_weight.std():.6f}")

# Test forward pass with random input
print("\n6. Testing forward pass:")
model_random.eval()
test_input = torch.randn(1, 12, 3, 128, 128)
with torch.no_grad():
    output = model_random(test_input)
print(f"   Output shape: {output.shape}")
print(f"   Output[0,0,:10] = {output[0, 0, :10]}")
print(f"   Output[0,1,:10] = {output[0, 1, :10]}")
print(f"   Output mean = {output.mean():.6f}, std = {output.std():.6f}")

# Test with same input twice
print("\n7. Testing determinism:")
test_input2 = torch.randn(1, 12, 3, 128, 128)
with torch.no_grad():
    output2a = model_random(test_input2)
    output2b = model_random(test_input2)
diff = (output2a - output2b).abs().sum().item()
print(f"   Difference between two runs with same input: {diff:.8f}")
if diff < 1e-6:
    print("   ✓ Model is deterministic (good)")
else:
    print(f"   ⚠️  Model is non-deterministic! Diff: {diff}")
