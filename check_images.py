import torch
from app import ECGPredictor, ResNetUNetDigitizer
from PIL import Image
import torchvision.transforms as T
import numpy as np

print("=" * 80)
print("INVESTIGATING: WHY IDENTICAL IMAGES PRODUCE IDENTICAL OUTPUTS")
print("=" * 80)

# Load the two images
img1_path = 'input_ecg/360_F_584469054_Frq9dABbkYVTyU40SXOpfjvry36bPqJC.jpg'
img2_path = 'input_ecg/upload.jpg'

img1 = Image.open(img1_path).convert('RGB')
img2 = Image.open(img2_path).convert('RGB')

print(f"\n1. Image shapes:")
print(f"   Image 1: {img1.size}")
print(f"   Image 2: {img2.size}")

# Check if they are identical
img1_array = np.array(img1)
img2_array = np.array(img2)
print(f"\n2. Are the images identical?")
print(f"   Image 1 shape: {img1_array.shape}")
print(f"   Image 2 shape: {img2_array.shape}")
if np.array_equal(img1_array, img2_array):
    print("   ⚠️  YES - The two images are IDENTICAL!")
    print("   This explains why the model produces the same output.")
else:
    print(f"   No - Images are different")
    diff = np.abs(img1_array.astype(float) - img2_array.astype(float)).mean()
    print(f"   Mean pixel difference: {diff:.2f}")

# Now test: what if we use DIFFERENT images for each panel?
print("\n3. Testing with DIFFERENT images per panel:")

transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

# Create a batch where each panel is a different image (cycle between img1 and img2)
model = ResNetUNetDigitizer(out_T=10250, enc_dim=512, fuse_dim=256)
ckpt = torch.load('resnet_unet_best.pth', map_location='cpu')
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']
model.load_state_dict(ckpt, strict=False)
model.eval()

# Create 12 panels alternating between img1 and img2
panels_alt = []
for i in range(12):
    img = img1 if i % 2 == 0 else img2
    panels_alt.append(transform(img))

panels_alt = torch.stack(panels_alt, dim=0).unsqueeze(0)  # (1, 12, 3, 128, 128)

print(f"   Input shape: {panels_alt.shape}")
print(f"   Panel 0 and Panel 1 are different: {not torch.allclose(panels_alt[0,0], panels_alt[0,1])}")
print(f"   Panel 0 and Panel 2 are same image: {torch.allclose(panels_alt[0,0], panels_alt[0,2])}")

with torch.no_grad():
    output_alt = model(panels_alt).numpy()

print(f"   Output shape: {output_alt.shape}")
print(f"\n   Lead 0 (first 10 points): {output_alt[0, :10]}")
print(f"   Lead 1 (first 10 points): {output_alt[1, :10]}")
print(f"   Lead 0 mean: {output_alt[0].mean():.4f}, std: {output_alt[0].std():.4f}")
print(f"   Lead 1 mean: {output_alt[1].mean():.4f}, std: {output_alt[1].std():.4f}")

# Check per-lead means
means = output_alt.mean(axis=1)
print(f"\n   Per-lead means: {means}")
print(f"   Per-lead mean std: {means.std():.4f}")

# Conclusion
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
if np.array_equal(img1_array, img2_array):
    print("\n✗ PROBLEM FOUND:")
    print("  Both test images (360_F_584469054... and upload.jpg) are IDENTICAL.")
    print("  The model correctly produces identical outputs for identical inputs!")
    print("\n  To test if the model responds to different images:")
    print("  1. Use truly different ECG panel images")
    print("  2. Or artificially modify one of the test images")
    print("  3. The app is likely working correctly.")
else:
    print("\n✓ Images are different, but model produced identical outputs.")
    print("  This suggests the model is not responding to image content.")
