"""Render cursed energy visuals using OpenCV and NumPy with chroma-keyed video overlay."""
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from config import (
    EFFECT_SCALE_FACTOR,
    EFFECT_SIZE_MAX,
    EFFECT_SIZE_MIN,
    EFFECT_VERTICAL_OFFSET,
    EFFECT_ALPHA_MULTIPLIER,
    GREEN_HSV_LOWER,
    GREEN_HSV_UPPER,
    MASK_BLUR_SIZE,
    MASK_CLOSE_ITER,
    MASK_OPEN_ITER,
    VFX_VIDEO_PATH,
)


def blend_roi(base_roi: np.ndarray, overlay: np.ndarray, alpha) -> np.ndarray:
    """Blend an RGBA-like overlay into the target ROI using alpha values."""
    alpha_arr = np.asarray(alpha, dtype=np.float32)
    if alpha_arr.ndim == 0:
        alpha_arr = alpha_arr[np.newaxis]
    if alpha_arr.ndim == 2:
        alpha_arr = alpha_arr[..., None]

    base_f = base_roi.astype(np.float32)
    overlay_f = overlay.astype(np.float32)
    blended = base_f * (1.0 - alpha_arr) + overlay_f * alpha_arr
    return np.clip(blended, 0, 255).astype(np.uint8)


class CursedEnergyEffect:
    def __init__(self) -> None:
        self.video_cap = cv2.VideoCapture(VFX_VIDEO_PATH)
        self.vfx_frame = None
        self.video_open_failed = False
        self.source_size = (0, 0)
        self.frame_count = 0

        if not self.video_cap.isOpened():
            print(f'Warning: Unable to open VFX video at {VFX_VIDEO_PATH}. Overlay will be disabled.')
            self.video_open_failed = True
        else:
            success, frame = self.video_cap.read()
            if success and frame is not None:
                self.vfx_frame = frame
                self.source_size = frame.shape[1], frame.shape[0]
                print(f'Video loaded successfully. Size: {self.source_size}')
            else:
                self.video_open_failed = True
                print('Warning: VFX video opened but first frame read failed.')

        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _advance_vfx(self) -> None:
        if self.video_open_failed:
            return

        success, frame = self.video_cap.read()
        if not success or frame is None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.video_cap.read()

        if success and frame is not None:
            self.vfx_frame = frame
        else:
            self.video_open_failed = True
            self.vfx_frame = None

    def _make_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create a mask by detecting green screen. Inverted to keep content (non-green areas)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for green screen areas - using very selective range
        mask_green = cv2.inRange(hsv, np.array(GREEN_HSV_LOWER, dtype=np.uint8), np.array(GREEN_HSV_UPPER, dtype=np.uint8))
        
        # Apply morphological operations to clean up the mask
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, self.open_kernel, iterations=MASK_OPEN_ITER)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_CLOSE, self.close_kernel, iterations=MASK_CLOSE_ITER)
        mask_green = cv2.GaussianBlur(mask_green, MASK_BLUR_SIZE, 0)
        
        # Invert: where green=255 (green screen), we want 0 (transparent)
        # where green=0 (content), we want 255 (opaque)
        mask_content = 255 - mask_green
        
        return mask_content

    def _prepare_overlay(self, target_width: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if self.vfx_frame is None or self.video_open_failed:
            return None

        frame = self.vfx_frame
        aspect_ratio = frame.shape[0] / frame.shape[1]
        target_width = max(int(self.source_size[0] * EFFECT_SIZE_MIN), min(int(self.source_size[0] * EFFECT_SIZE_MAX), target_width))
        target_height = int(target_width * aspect_ratio)

        resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        mask = self._make_mask(resized)
        
        # mask is already inverted: 255 where content is, 0 where green screen is
        alpha = mask.astype(np.float32) / 255.0
        
        return resized, alpha

    def update(self, center: Tuple[int, int], active: bool, hand_size_px: int) -> None:
        self._advance_vfx()

    def render(self, frame: np.ndarray, center: Tuple[int, int], hand_size_px: Optional[int], active: bool, frame_counter: int) -> np.ndarray:
        if self.video_open_failed:
            return frame

        output = frame
        height, width = frame.shape[:2]
        cx, cy = center

        if active and self.vfx_frame is not None:
            target_width = int(max(1, (hand_size_px or int(frame.shape[1] * 0.14)) * EFFECT_SCALE_FACTOR))
            prepared = self._prepare_overlay(target_width)
            if prepared is not None:
                vfx_resized, alpha_mask = prepared
                overlay_h, overlay_w = vfx_resized.shape[:2]
                x0 = int(cx - overlay_w / 2)
                y0 = int(cy - overlay_h / 2 + EFFECT_VERTICAL_OFFSET)
                x1 = x0 + overlay_w
                y1 = y0 + overlay_h

                x0_clamped = max(0, x0)
                y0_clamped = max(0, y0)
                x1_clamped = min(width, x1)
                y1_clamped = min(height, y1)

                if x0_clamped < x1_clamped and y0_clamped < y1_clamped:
                    ox0 = x0_clamped - x0
                    oy0 = y0_clamped - y0
                    ox1 = ox0 + (x1_clamped - x0_clamped)
                    oy1 = oy0 + (y1_clamped - y0_clamped)

                    overlay_crop = vfx_resized[oy0:oy1, ox0:ox1]
                    alpha_crop = alpha_mask[oy0:oy1, ox0:ox1]
                    alpha_crop = alpha_crop * EFFECT_ALPHA_MULTIPLIER
                    roi = output[y0_clamped:y1_clamped, x0_clamped:x1_clamped]
                    output[y0_clamped:y1_clamped, x0_clamped:x1_clamped] = blend_roi(roi, overlay_crop, alpha_crop)

        return output
