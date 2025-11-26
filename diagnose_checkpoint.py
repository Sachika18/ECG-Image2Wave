import torch
from app import ResNetUNetDigitizer

print("=" * 80)
print("CHECKPOINT INSPECTION")
print("=" * 80)

# Load checkpoint
ckpt_path = 'resnet_unet_best.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']

print(f"\nCheckpoint keys ({len(ckpt)} total):")
for i, (k, v) in enumerate(list(ckpt.items())[:50]):
    print(f"  {i:3d}. {k:60s} {v.shape}")
if len(ckpt) > 50:
    print(f"  ... and {len(ckpt) - 50} more")

print("\n" + "=" * 80)
print("MODEL STRUCTURE")
print("=" * 80)

# Create model
model = ResNetUNetDigitizer(out_T=10250, enc_dim=512, fuse_dim=256)
model_state = model.state_dict()

print(f"\nModel keys ({len(model_state)} total):")
for i, (k, v) in enumerate(list(model_state.items())[:50]):
    print(f"  {i:3d}. {k:60s} {v.shape}")
if len(model_state) > 50:
    print(f"  ... and {len(model_state) - 50} more")

print("\n" + "=" * 80)
print("KEY MATCHING ANALYSIS")
print("=" * 80)

# Find matches
ckpt_keys = set(ckpt.keys())
model_keys = set(model_state.keys())

matched = ckpt_keys & model_keys
only_in_ckpt = ckpt_keys - model_keys
only_in_model = model_keys - ckpt_keys

print(f"\nMatched keys: {len(matched)}")
for k in sorted(matched)[:10]:
    print(f"  {k}")
if len(matched) > 10:
    print(f"  ... and {len(matched) - 10} more")

print(f"\nOnly in checkpoint ({len(only_in_ckpt)}):")
for k in sorted(only_in_ckpt)[:20]:
    print(f"  {k}")
if len(only_in_ckpt) > 20:
    print(f"  ... and {len(only_in_ckpt) - 20} more")

print(f"\nOnly in model ({len(only_in_model)}):")
for k in sorted(only_in_model)[:20]:
    print(f"  {k}")
if len(only_in_model) > 20:
    print(f"  ... and {len(only_in_model) - 20} more")

# Check shape mismatches for matched keys
shape_mismatches = []
for k in matched:
    if ckpt[k].shape != model_state[k].shape:
        shape_mismatches.append((k, ckpt[k].shape, model_state[k].shape))

print(f"\nShape mismatches in matched keys: {len(shape_mismatches)}")
for k, ck_shape, m_shape in shape_mismatches[:10]:
    print(f"  {k}: checkpoint {ck_shape} vs model {m_shape}")
if len(shape_mismatches) > 10:
    print(f"  ... and {len(shape_mismatches) - 10} more")
