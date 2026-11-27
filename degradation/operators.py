from __future__ import annotations

from typing import Any, Mapping
import numpy as np
from scipy import ndimage


def _fit_spatial_shape(array: np.ndarray, target_hw) -> np.ndarray:
    h, w = target_hw
    return np.ascontiguousarray(array[:h, :w, ...], dtype=np.float32)


def apply_hsi_corruption(scene: np.ndarray, name: str, params: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(scene, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("HSI scene must have shape [H,W,B].")
    height, width, bands = x.shape
    if name == "gaussian":
        result = x + rng.normal(0.0, float(params["sigma"]), size=x.shape).astype(np.float32)
    elif name == "poisson":
        peak = float(params["peak"])
        result = rng.poisson(np.clip(x, 0, 1) * peak).astype(np.float32) / max(peak, 1e-6)
    elif name == "stripe":
        result = x.copy()
        count = max(1, int(round(width * float(params["ratio"]))))
        columns = rng.choice(width, size=count, replace=False)
        offsets = rng.uniform(-float(params["amplitude"]), float(params["amplitude"]), size=count).astype(np.float32)
        result[:, columns, :] += offsets[None, :, None]
    elif name == "dead_line":
        result = x.copy()
        count = max(1, int(round(width * float(params["ratio"]))))
        columns = rng.choice(width, size=count, replace=False)
        result[:, columns, :] = 0.0
    elif name == "blur":
        sigma = float(params["sigma"])
        kernel = int(params["kernel"])
        truncate = max((kernel - 1) / (2.0 * max(sigma, 1e-6)), 0.5)
        result = ndimage.gaussian_filter(x, sigma=(sigma, sigma, 0.0), mode="reflect", truncate=truncate).astype(np.float32)
    elif name == "downsample":
        scale = float(params["scale"])
        low = ndimage.zoom(x, zoom=(scale, scale, 1.0), order=1, mode="reflect", prefilter=False)
        back = ndimage.zoom(low, zoom=(height / max(low.shape[0],1), width / max(low.shape[1],1), 1.0), order=1, mode="reflect", prefilter=False)
        result = _fit_spatial_shape(back, (height, width))
    else:
        raise KeyError(f"Unknown HSI degradation: {name}")
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False)


def apply_lidar_corruption(scene: np.ndarray, name: str, params: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(scene, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("LiDAR/DSM scene must have shape [H,W,C].")
    height, width, _ = x.shape
    if name == "elevation_noise":
        result = x + rng.normal(0.0, float(params["sigma"]), size=x.shape).astype(np.float32)
    elif name == "holes":
        result = x.copy()
        ratio = float(params["ratio"])
        block = int(params["block_size"])
        target = max(1, int(round(height * width * ratio)))
        mask = np.zeros((height, width), dtype=bool)
        attempts = 0
        while int(mask.sum()) < target and attempts < max(200, target * 2):
            attempts += 1
            cy = int(rng.integers(0, height)); cx = int(rng.integers(0, width))
            half = block // 2
            y0, y1 = max(0, cy-half), min(height, cy+half+1)
            x0, x1 = max(0, cx-half), min(width, cx+half+1)
            mask[y0:y1, x0:x1] = True
        fill = float(params.get("fill_value", 0.0))
        result[mask, :] = fill
    elif name == "resolution":
        scale = float(params["scale"])
        low = ndimage.zoom(x, zoom=(scale, scale, 1.0), order=1, mode="reflect", prefilter=False)
        back = ndimage.zoom(low, zoom=(height / max(low.shape[0],1), width / max(low.shape[1],1), 1.0), order=1, mode="reflect", prefilter=False)
        result = _fit_spatial_shape(back, (height, width))
    else:
        raise KeyError(f"Unknown LiDAR degradation: {name}")
    return np.ascontiguousarray(result, dtype=np.float32)


def apply_cross_modal_corruption(scene: np.ndarray, name: str, params: Mapping[str, Any], rng: np.random.Generator) -> np.ndarray:
    """Apply a relation-level corruption to the LiDAR scene relative to HSI.

    Misregistration is intentionally not part of the LiDAR modality-local
    degradation knowledge bank: a pure LiDAR statistic cannot identify a
    relative HSI/LiDAR shift. It is reserved for cross-modal alignment tests.
    """
    x = np.asarray(scene, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("Cross-modal LiDAR scene must have shape [H,W,C].")
    if name == "misregistration":
        max_shift = int(params["max_shift_pixels"])
        if max_shift <= 0:
            return np.ascontiguousarray(x, dtype=np.float32)
        choices = [i for i in range(-max_shift, max_shift + 1) if i != 0]
        dy = int(rng.choice(choices))
        dx = int(rng.choice(choices))
        result = ndimage.shift(x, shift=(dy, dx, 0), order=1, mode="reflect", prefilter=False)
    else:
        raise KeyError(f"Unknown cross-modal degradation: {name}")
    return np.ascontiguousarray(result, dtype=np.float32)


__all__ = ["apply_hsi_corruption", "apply_lidar_corruption", "apply_cross_modal_corruption"]
