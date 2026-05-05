"""Fist detection logic based on MediaPipe hand landmarks."""
from typing import Optional, Tuple

import numpy as np

# MediaPipe Hands returns a list of 21 normalized landmarks per detected hand.
# Each landmark is a structure with attributes x, y, z in the [0, 1] range.
FINGER_TIP_INDICES = [8, 12, 16, 20]
FINGER_PIP_INDICES = [7, 11, 15, 19]
FINGER_MCP_INDICES = [5, 9, 13, 17]
WRIST_INDEX = 0
PALM_INDICES = [0, 1, 5, 9, 13, 17]


def normalized_landmarks_to_array(landmarks) -> np.ndarray:
    """Convert MediaPipe normalized landmarks to a NumPy array of shape (21, 3)."""
    array = np.zeros((21, 3), dtype=np.float32)
    for index, landmark in enumerate(landmarks):
        array[index, 0] = landmark.x
        array[index, 1] = landmark.y
        array[index, 2] = landmark.z
    return array


def palm_center(landmark_array: np.ndarray) -> np.ndarray:
    """Compute a stable palm center from palm landmarks."""
    return np.mean(landmark_array[PALM_INDICES, :2], axis=0)


def fingertip_distances(landmark_array: np.ndarray) -> np.ndarray:
    """Return distances from index/middle/ring/pinky fingertips to the palm center."""
    palm = palm_center(landmark_array)
    tips = landmark_array[FINGER_TIP_INDICES, :2]
    return np.linalg.norm(tips - palm, axis=1)


def finger_curl_ratios(landmark_array: np.ndarray) -> np.ndarray:
    """Return normalized curl ratios for each finger tip relative to its PIP and MCP joints."""
    ratios = []
    for tip_idx, pip_idx, mcp_idx in zip(FINGER_TIP_INDICES, FINGER_PIP_INDICES, FINGER_MCP_INDICES):
        tip = landmark_array[tip_idx, :2]
        pip = landmark_array[pip_idx, :2]
        mcp = landmark_array[mcp_idx, :2]
        tip_to_pip = np.linalg.norm(tip - pip)
        pip_to_mcp = np.linalg.norm(pip - mcp)
        ratios.append(float(tip_to_pip / max(pip_to_mcp, 1e-6)))
    return np.clip(np.array(ratios, dtype=np.float32), 0.0, 2.5)


def hand_reference_size(landmark_array: np.ndarray) -> float:
    """Return a normalized size reference for the hand from wrist to middle finger MCP."""
    return float(np.linalg.norm(landmark_array[WRIST_INDEX, :2] - landmark_array[9, :2]))


def hand_bbox_size(landmark_array: np.ndarray) -> float:
    """Return a normalized approximate hand size using the 2D landmark bounding box max dimension."""
    xy = landmark_array[:, :2]
    min_xy = np.min(xy, axis=0)
    max_xy = np.max(xy, axis=0)
    return float(np.max(max_xy - min_xy))


def is_fist(landmarks, close_threshold: float, release_threshold: float, previous_state: bool) -> Tuple[bool, float]:
    """Detect a closed fist using tip-to-palm spacing and finger curl scores."""
    if landmarks is None:
        return False, 0.0

    landmark_array = normalized_landmarks_to_array(landmarks)
    distances = fingertip_distances(landmark_array)
    reference_size = hand_reference_size(landmark_array)
    normalized_gap = float(np.mean(distances) / max(reference_size, 1e-6))
    normalized_curl = float(np.mean(finger_curl_ratios(landmark_array)))

    # Lower score means the hand is more closed.
    combined_score = normalized_gap * 0.55 + normalized_curl * 0.45
    threshold = close_threshold if not previous_state else release_threshold
    return combined_score < threshold, combined_score


def fist_center_px(landmarks, frame_width: int, frame_height: int) -> Optional[Tuple[int, int]]:
    """Return the pixel coordinates of the effect center for the fist."""
    if landmarks is None:
        return None

    landmark_array = normalized_landmarks_to_array(landmarks)
    palm = palm_center(landmark_array)
    xy = landmark_array[:, :2]
    bbox_center = (np.min(xy, axis=0) + np.max(xy, axis=0)) / 2.0
    center_norm = (palm * 0.7 + bbox_center * 0.3)
    return int(center_norm[0] * frame_width), int(center_norm[1] * frame_height)


def fist_size_px(landmarks, frame_width: int, frame_height: int) -> Optional[int]:
    """Return a pixel-based hand size estimate used for scaling the overlay."""
    if landmarks is None:
        return None

    landmark_array = normalized_landmarks_to_array(landmarks)
    bbox_size = hand_bbox_size(landmark_array)
    return int(bbox_size * max(frame_width, frame_height))
