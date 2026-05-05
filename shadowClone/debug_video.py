"""Debug script to analyze video and find optimal green screen parameters."""
import cv2
import numpy as np
from pathlib import Path
from config import GREEN_HSV_LOWER, GREEN_HSV_UPPER

VFX_VIDEO_PATH = str(Path('assets') / 'cursed_flame.mp4')

print(f"Analyzing video: {VFX_VIDEO_PATH}")
print("=" * 60)

cap = cv2.VideoCapture(VFX_VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video file: {VFX_VIDEO_PATH}")
    exit(1)

success, frame = cap.read()
if not success:
    print("ERROR: Could not read first frame from video")
    exit(1)

print(f"Video resolution: {frame.shape[1]}x{frame.shape[0]}")
print(f"Frame channels: {frame.shape[2]}")

# Analyze the frame
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
bgr = frame

print("\nFrame statistics:")
print(f"  BGR - Min: {bgr.min()}, Max: {bgr.max()}, Mean: {bgr.mean():.2f}")
print(f"  HSV - Min: {hsv.min()}, Max: {hsv.max()}")

# Sample from different regions to understand color distribution
h, w = frame.shape[:2]
regions = {
    'top-left': (w//4, h//4),
    'center': (w//2, h//2),
    'top-right': (3*w//4, h//4),
    'bottom-left': (w//4, 3*h//4),
    'bottom-right': (3*w//4, 3*h//4),
}

print("\nColor samples from different regions (BGR, HSV):")
for region_name, (x, y) in regions.items():
    bgr_sample = frame[y, x]
    hsv_sample = hsv[y, x]
    print(f"  {region_name:12s}: BGR{tuple(bgr_sample)}, HSV{tuple(hsv_sample)}")

# Check for predominant colors
print("\nHue distribution (excluding very low saturation):")
saturated_hsv = hsv[hsv[:, :, 1] > 50]  # Only saturated colors
if len(saturated_hsv) > 0:
    hue_values = saturated_hsv[:, 0]
    print(f"  Hue range: {hue_values.min()}-{hue_values.max()}")
    print(f"  Hue mean: {hue_values.mean():.1f}")
    print(f"  Hue std: {hue_values.std():.1f}")
    
    # Find hue peaks
    hue_hist = np.histogram(hue_values, bins=180, range=(0, 180))[0]
    top_hues = np.argsort(hue_hist)[-3:][::-1]
    print(f"  Top hue bins: {top_hues} (these represent predominant colors)")

print("\nTesting green screen detection with current settings:")

mask = cv2.inRange(hsv, np.array(GREEN_HSV_LOWER, dtype=np.uint8), np.array(GREEN_HSV_UPPER, dtype=np.uint8))
green_pixels = np.count_nonzero(mask)
total_pixels = frame.shape[0] * frame.shape[1]
green_percent = (green_pixels / total_pixels) * 100

print(f"  Current GREEN_HSV_LOWER: {GREEN_HSV_LOWER}")
print(f"  Current GREEN_HSV_UPPER: {GREEN_HSV_UPPER}")
print(f"  Green pixels detected: {green_pixels} ({green_percent:.2f}%)")

if green_percent < 5:
    print("  ⚠️  WARNING: Very few green pixels detected. The green screen may not match expected color.")
    print("     Consider widening the HSV range or checking if the video has a green screen.")
elif green_percent > 80:
    print("  ⚠️  WARNING: Most of the frame is detected as green. The color range may be too broad.")
else:
    print("  ✓ Green detection seems reasonable.")

print("\n" + "=" * 60)
print("Display settings saved to 'debug_frame.png' and 'debug_mask.png'")

# Save visualization
cv2.imwrite('debug_frame.png', frame)
cv2.imwrite('debug_mask.png', mask)
print("Frame saved as 'debug_frame.png'")
print("Mask saved as 'debug_mask.png'")

cap.release()
