from __future__ import annotations

import numpy as np


NUM_SLOTS = 2
POINTS_PER_SLOT = 27
NUM_POINTS = NUM_SLOTS * POINTS_PER_SLOT
NUM_CHANNELS = 4
TARGET_LENGTH = 64
INPUT_SHAPE = (NUM_CHANNELS, NUM_POINTS, TARGET_LENGTH)
SLOT_VALUE = np.asarray([-1.0, 1.0], dtype=np.float32)


def ensure_raw_shape(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim != 4:
        raise ValueError(f"expected raw fullcascade shape (T,2,27,C), got {raw.shape}")
    if raw.shape[1] != NUM_SLOTS or raw.shape[2] != POINTS_PER_SLOT or raw.shape[3] < 3:
        raise ValueError(f"expected raw fullcascade shape (T,2,27,C>=3), got {raw.shape}")
    return raw[..., :3]


def sample_indices(length: int, target_length: int = TARGET_LENGTH) -> np.ndarray:
    if length <= 0:
        return np.zeros((target_length,), dtype=np.int64)
    if length == target_length:
        return np.arange(length, dtype=np.int64)
    return np.linspace(0, length - 1, target_length).round().astype(np.int64)


def temporal_sample(raw: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
    raw = ensure_raw_shape(raw)
    length = raw.shape[0]
    if length == target_length:
        return raw
    if length > target_length:
        return raw[sample_indices(length, target_length)]
    out = np.zeros((target_length, NUM_SLOTS, POINTS_PER_SLOT, 3), dtype=np.float32)
    if length > 0:
        out[:length] = raw
    return out


def normalize_frame(frame: np.ndarray, min_scale: float = 0.15) -> np.ndarray:
    out = np.zeros((NUM_CHANNELS, NUM_POINTS), dtype=np.float32)
    valid = frame[..., 2] > 0.5
    if not np.any(valid):
        return out

    xy = frame[..., :2]
    valid_xy = xy[valid]
    center = valid_xy.mean(axis=0)
    span = valid_xy.max(axis=0) - valid_xy.min(axis=0)
    scale = max(float(span.max()), float(min_scale))

    for slot in range(NUM_SLOTS):
        slot_valid = valid[slot]
        base = slot * POINTS_PER_SLOT
        end = base + POINTS_PER_SLOT
        if not np.any(slot_valid):
            continue
        norm_xy = (xy[slot] - center) / scale
        norm_xy = np.clip(norm_xy, -2.0, 2.0)
        out[0, base:end] = norm_xy[:, 0] * slot_valid
        out[1, base:end] = norm_xy[:, 1] * slot_valid
        out[2, base:end] = slot_valid.astype(np.float32)
        out[3, base:end] = SLOT_VALUE[slot] * slot_valid
    return out


def fullcascade_to_model_input(raw: np.ndarray, target_length: int = TARGET_LENGTH) -> np.ndarray:
    seq = temporal_sample(raw, target_length)
    out = np.zeros((NUM_CHANNELS, NUM_POINTS, target_length), dtype=np.float32)
    for t in range(target_length):
        out[:, :, t] = normalize_frame(seq[t])
    return out


def add_batch(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.shape != INPUT_SHAPE:
        raise ValueError(f"expected {INPUT_SHAPE}, got {x.shape}")
    return x[None, ...]
