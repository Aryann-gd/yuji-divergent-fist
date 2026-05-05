import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pygame
from mediapipe.python.solutions import hands as mp_hands

mp_drawing_styles = None
mp_drawing = None

from config import (
    CAMERA_ID,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
    SHOW_LANDMARKS,
    MIRROR_FRAME,
    KEY_QUIT,
    KEY_SCREENSHOT,
    FIST_HOLD_FRAMES,
    FIST_CLOSE_THRESHOLD,
    FIST_RELEASE_THRESHOLD,
)
from cursed_energy import CursedEnergyEffect
from fist_detector import fist_center_px, fist_size_px, is_fist

pygame.mixer.init()
flame_sound = pygame.mixer.Sound('assets/flame_sound.wav')

if SHOW_LANDMARKS:
    try:
        from mediapipe.python.solutions import drawing_styles as mp_drawing_styles
        from mediapipe.python.solutions import drawing_utils as mp_drawing
    except ImportError:
        mp_drawing_styles = None
        mp_drawing = None
        print('Warning: Mediapipe drawing modules are unavailable. Landmark rendering disabled.')


def initialize_camera() -> cv2.VideoCapture:
    capture = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, FPS)
    if not capture.isOpened():
        raise RuntimeError(
            'Unable to open webcam. On Windows, make sure camera permissions are available.'
        )
    return capture


def draw_landmarks(frame: np.ndarray, hand_landmarks) -> None:
    if mp_drawing is None or mp_drawing_styles is None:
        return
    mp_drawing.draw_landmarks(
        frame,
        hand_landmarks,
        list(mp_hands.HAND_CONNECTIONS),
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )


def save_screenshot(frame: np.ndarray) -> None:
    output_dir = Path('assets')
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = output_dir / f'screenshot_{timestamp}.png'
    cv2.imwrite(str(filename), frame)
    print(f'Screenshot saved: {filename}')


def main() -> None:
    capture = initialize_camera()
    energy_effect = CursedEnergyEffect()

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    ) as hands:
        # Track state for each hand independently
        hand_states = {}  # hand_idx -> is_active
        consecutive_frames_dict = {}  # hand_idx -> frame_count
        frame_counter = 0
        last_time = time.time()

        while True:
            success, frame = capture.read()
            if not success:
                print('Frame read failed, stopping.')
                break

            if MIRROR_FRAME:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results: Any = hands.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Get all detected hands
            all_hands = results.multi_hand_landmarks if results.multi_hand_landmarks else []
            current_frame = frame
            gap_text = '0.000'
            any_hand_active = False

            # Process each detected hand independently
            for hand_idx, hand_landmarks_obj in enumerate(all_hands):
                hand_landmarks = hand_landmarks_obj.landmark

                # Initialize state for this hand if needed
                if hand_idx not in hand_states:
                    hand_states[hand_idx] = False
                    consecutive_frames_dict[hand_idx] = 0

                previous_state = hand_states[hand_idx]

                # Compute fist state for this hand
                fist_condition, gap = is_fist(hand_landmarks, FIST_CLOSE_THRESHOLD, FIST_RELEASE_THRESHOLD, previous_state)
                gap_text = f'{gap:.3f}'

                # Update consecutive frames counter
                if fist_condition:
                    consecutive_frames_dict[hand_idx] += 1
                else:
                    consecutive_frames_dict[hand_idx] = 0

                # Update active state based on hold duration
                if consecutive_frames_dict[hand_idx] >= FIST_HOLD_FRAMES:
                    fist_active_hand = True
                elif not fist_condition:
                    fist_active_hand = False
                else:
                    fist_active_hand = previous_state

                # Play sound only on first activation of this hand
                if fist_active_hand and not previous_state:
                    flame_sound.play()
                    print(f'[DEBUG] Hand {hand_idx} activated! VFX frame present: {energy_effect.vfx_frame is not None}')

                # Update state
                hand_states[hand_idx] = fist_active_hand
                any_hand_active = any_hand_active or fist_active_hand

                # Render overlay for active hands
                if fist_active_hand:
                    fist_center = fist_center_px(hand_landmarks, frame.shape[1], frame.shape[0])
                    hand_size = fist_size_px(hand_landmarks, frame.shape[1], frame.shape[0])
                    if fist_center is not None and hand_size is not None:
                        energy_effect.update(fist_center, True, hand_size)
                        current_frame = energy_effect.render(current_frame, fist_center, hand_size, True, frame_counter)

            # Clean up state for hands that are no longer detected
            for hand_idx in list(hand_states.keys()):
                if hand_idx >= len(all_hands):
                    del hand_states[hand_idx]
                    del consecutive_frames_dict[hand_idx]

            frame = current_frame

            # Draw landmarks if enabled
            if SHOW_LANDMARKS and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    draw_landmarks(frame, hand_landmarks)

            # Update FPS
            current_time = time.time()
            fps = 1.0 / max(1e-6, current_time - last_time)
            last_time = current_time
            frame_counter += 1

            # Draw UI text
            cv2.putText(
                frame,
                f'FPS: {fps:.1f}',
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (240, 240, 240),
                2,
            )
            cv2.putText(
                frame,
                f'Hands: {len(all_hands)} | Active: {sum(hand_states.values())}',
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f'Gap: {gap_text}',
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 220, 255),
                2,
            )

            cv2.imshow('Cursed Fist', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_QUIT:
                break
            if key == KEY_SCREENSHOT:
                save_screenshot(frame)

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
