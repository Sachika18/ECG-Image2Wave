import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

print("Generating synthetic ECG panel test images with STRONG differences...")

# Create output directory if needed
os.makedirs('input_ecg', exist_ok=True)

# Image parameters
width, height = 558, 360

# ============================================================================
# Test Image 1: RED background with LOW frequency waves
# ============================================================================
img1_array = np.ones((height, width, 3), dtype=np.uint8) * 200  # light gray
y_center = height // 2

# Draw multiple low-frequency, high-amplitude waves in RED
for x in range(width):
    # Low frequency, large amplitude
    y1 = y_center + 100 * np.sin(0.01 * x)
    y1 = max(0, min(height-1, int(y1)))
    img1_array[y1, x, :] = [255, 50, 50]  # RED
    
    # Add some secondary oscillations
    y2 = y_center + 60 * np.sin(0.02 * x + 1)
    y2 = max(0, min(height-1, int(y2)))
    img1_array[y2, x, :] = [200, 30, 30]  # darker red

img1 = Image.fromarray(img1_array)
draw1 = ImageDraw.Draw(img1)
draw1.text((20, 20), "IMAGE 1: RED + LOW FREQ", fill=(0, 0, 0))

img1_path = 'input_ecg/test_ecg_1.jpg'
img1.save(img1_path)
print(f"✓ Saved {img1_path}")

# ============================================================================
# Test Image 2: BLUE background with HIGH frequency spikes
# ============================================================================
img2_array = np.ones((height, width, 3), dtype=np.uint8) * 200  # light gray
y_center = height // 2

# Draw multiple high-frequency, sharp spikes in BLUE
for x in range(width):
    # High frequency sharp peaks
    peak_intensity = 80 * max(0, np.sin(0.1 * x))
    y = y_center - peak_intensity
    y = max(0, min(height-1, int(y)))
    img2_array[y, x, :] = [50, 50, 255]  # BLUE
    
    # Very sharp QRS-like complexes every 100 pixels
    if x % 100 < 15:
        sharp = 120 * (1 - abs((x % 100 - 7.5) / 7.5))
        y = y_center - sharp
        y = max(0, min(height-1, int(y)))
        img2_array[y, x, :] = [30, 30, 200]  # darker blue

img2 = Image.fromarray(img2_array)
draw2 = ImageDraw.Draw(img2)
draw2.text((20, 20), "IMAGE 2: BLUE + HIGH FREQ", fill=(255, 255, 255))

img2_path = 'input_ecg/test_ecg_2.jpg'
img2.save(img2_path)
print(f"✓ Saved {img2_path}")

print("\n" + "=" * 80)
print("Synthetic test images generated with STRONG visual differences!")
print("=" * 80)
print(f"\nImage 1: RED + LOW frequency waves")
print(f"Image 2: BLUE + HIGH frequency spikes")
print(f"\nNow test the model with these visually distinct images:")
print(f"\npython test_predict.py input_ecg/test_ecg_1.jpg input_ecg/test_ecg_2.jpg\n")
print("Expected results (if model responds to image content):")
print("  - Different per-lead means and stds for each image")
print("  - Pairwise correlation much lower than 1.0 (should be < 0.8)")
print("  - Higher L2 distances between predictions\n")
