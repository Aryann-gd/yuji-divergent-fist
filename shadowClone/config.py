"""Application configuration and tunable parameters."""

from pathlib import Path

# Hand / fist detection tuning.
# Lower values make fist detection more sensitive; raise slightly if the effect triggers too early.
FIST_CLOSE_THRESHOLD = 0.6
FIST_RELEASE_THRESHOLD = 0.8
FIST_HOLD_FRAMES = 15

# Green-screen VFX tuning.
VFX_VIDEO_PATH = str(Path('assets') / 'cursed_flame.mp4')
# Narrow range to catch pure green (hue 60) but NOT the flame (hue 75-76)
# The flame effect uses yellow-green, so excluding hue 70+ preserves the flame
GREEN_HSV_LOWER = (50, 100, 80)
GREEN_HSV_UPPER = (70, 255, 255)
MASK_OPEN_ITER = 0
MASK_CLOSE_ITER = 0
MASK_BLUR_SIZE = (3, 3)
EFFECT_SCALE_FACTOR = 1.3
EFFECT_SIZE_MIN = 0.3
EFFECT_SIZE_MAX = 3.0
EFFECT_VERTICAL_OFFSET = -25
EFFECT_ALPHA_MULTIPLIER = 0.6

# Video capture settings.
CAMERA_ID = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30
SHOW_LANDMARKS = False
MIRROR_FRAME = True

# Key bindings.
KEY_QUIT = ord('q')
KEY_SCREENSHOT = ord('s')
