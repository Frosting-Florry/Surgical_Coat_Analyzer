from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .geometry import (
    HarmonicResult,
    contour_segments,
    polygon_mask,
    solve_harmonic_coordinate,
    upsample_coordinate,
)


@dataclass(frozen=True)
class AnalysisOutput:
    metrics: dict[str, object]
    contours: dict[str, list[list[list[float]]]]
    solver: dict[str, float | int]
    parameter_hash: str
    classification_maps: dict[str, np.ndarray]


def load_rgb(path: str | Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def analyze_image(
    image: Image.Image,
    polygon: list[list[float]],
    left_edges: list[int],
    right_edges: list[int],
    exclusions: list[list[list[float]]],
    gray_card: list[list[float]],
    split_left: float,
    split_right: float,
    gray_target_l: float = 50.0,
    white_depth_min: float = 0.0,
    white_depth_max: float = 25.0,
    black_depth_min: float = 60.0,
    black_depth_max: float = 100.0,
    max_side: int = 320,
) -> AnalysisOutput:
    if not 0 < split_left < split_right < 1:
        raise ValueError("分界值必须满足 0% < 左/中分界 < 中/右分界 < 100%")
    _validate_hair_ranges(
        white_depth_min, white_depth_max, black_depth_min, black_depth_max
    )
    harmonic = solve_harmonic_coordinate(
        polygon, left_edges, right_edges, image.size, max_side=max_side
    )
    coordinate = upsample_coordinate(harmonic, image.size)
    roi = polygon_mask(image.size, polygon)
    excluded = np.zeros_like(roi)
    for item in exclusions:
        excluded |= polygon_mask(image.size, item)
    valid = roi & ~excluded
    if int(valid.sum()) < 20:
        raise ValueError("排除无效区域后，没有足够像素用于统计")

    rgb_u8 = np.asarray(image, dtype=np.uint8)
    l_star = image_to_lstar(image)

    gray_info: dict[str, object] = {"available": False}
    corrected_l = l_star
    if len(gray_card) >= 3:
        gray_mask = polygon_mask(image.size, gray_card)
        if int(gray_mask.sum()) >= 20:
            gray_values = l_star[gray_mask]
            observed_l = float(np.median(gray_values))
            scale = gray_target_l / max(observed_l, 1e-6)
            corrected_l = np.clip(l_star * scale, 0.0, 100.0)
            gray_rgb = np.median(rgb_u8[gray_mask], axis=0)
            gray_info = {
                "available": True,
                "observed_l": observed_l,
                "target_l": float(gray_target_l),
                "l_scale": float(scale),
                "median_rgb": [float(v) for v in gray_rgb],
                "l_iqr": float(np.percentile(gray_values, 75) - np.percentile(gray_values, 25)),
                "pixel_count": int(gray_mask.sum()),
                "clipped_dark_fraction": float(np.mean(l_star[gray_mask] <= 1.0)),
                "clipped_light_fraction": float(np.mean(l_star[gray_mask] >= 99.0)),
            }
    region_masks = {
        "left": valid & (coordinate < split_left),
        "middle": valid & (coordinate >= split_left) & (coordinate < split_right),
        "right": valid & (coordinate >= split_right),
    }
    regions: dict[str, object] = {}
    for name, region_mask in region_masks.items():
        regions[name] = _region_metrics(l_star[region_mask], corrected_l[region_mask])

    raw_classes = _classification_map(
        100.0 - l_star, valid, white_depth_min, white_depth_max,
        black_depth_min, black_depth_max,
    )
    corrected_classes = _classification_map(
        100.0 - corrected_l, valid, white_depth_min, white_depth_max,
        black_depth_min, black_depth_max,
    )
    hair_classification = {
        "definition": {
            "white_depth_min": float(white_depth_min),
            "white_depth_max": float(white_depth_max),
            "black_depth_min": float(black_depth_min),
            "black_depth_max": float(black_depth_max),
        },
        "modes": {
            "raw": _classification_metrics(raw_classes, valid, region_masks),
            "gray_corrected": _classification_metrics(
                corrected_classes, valid, region_masks
            ),
        },
    }

    payload = {
        "polygon": polygon,
        "left_edges": left_edges,
        "right_edges": right_edges,
        "exclusions": exclusions,
        "gray_card": gray_card,
        "split_left": split_left,
        "split_right": split_right,
        "gray_target_l": gray_target_l,
        "white_depth_min": white_depth_min,
        "white_depth_max": white_depth_max,
        "black_depth_min": black_depth_min,
        "black_depth_max": black_depth_max,
    }
    parameter_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metrics = {
        "split_left": float(split_left),
        "split_right": float(split_right),
        "roi_pixel_count": int(roi.sum()),
        "excluded_pixel_count": int((excluded & roi).sum()),
        "valid_pixel_count": int(valid.sum()),
        "gray_card": gray_info,
        "regions": regions,
        "hair_classification": hair_classification,
    }
    return AnalysisOutput(
        metrics=metrics,
        contours={
            "left_middle": contour_segments(harmonic, split_left),
            "middle_right": contour_segments(harmonic, split_right),
        },
        solver={"iterations": harmonic.iterations, "residual": harmonic.residual},
        parameter_hash=parameter_hash,
        classification_maps={"raw": raw_classes, "gray_corrected": corrected_classes},
    )


def _validate_hair_ranges(
    white_min: float, white_max: float, black_min: float, black_max: float
) -> None:
    values = (white_min, white_max, black_min, black_max)
    if not all(np.isfinite(value) and 0.0 <= value <= 100.0 for value in values):
        raise ValueError("白发和黑发的颜色深度范围必须在 0–100 之间")
    if white_min >= white_max or black_min >= black_max:
        raise ValueError("白发和黑发范围的下限必须小于上限")
    if white_max > black_min:
        raise ValueError("白发与黑发范围不能重叠；两者之间可保留灰发范围")


def _classification_map(
    darkness: np.ndarray,
    valid: np.ndarray,
    white_min: float,
    white_max: float,
    black_min: float,
    black_max: float,
) -> np.ndarray:
    classes = np.zeros(valid.shape, dtype=np.uint8)
    classes[valid & (darkness >= white_min) & (darkness < white_max)] = 1
    classes[valid & (darkness >= black_min) & (darkness <= black_max)] = 2
    return classes


def _classification_metrics(
    classes: np.ndarray,
    valid: np.ndarray,
    region_masks: dict[str, np.ndarray],
) -> dict[str, object]:
    return {
        "overall": _area_metrics(classes, valid),
        "regions": {
            name: _area_metrics(classes, mask) for name, mask in region_masks.items()
        },
    }


def _area_metrics(classes: np.ndarray, denominator_mask: np.ndarray) -> dict[str, object]:
    total = int(denominator_mask.sum())
    white = int(np.count_nonzero((classes == 1) & denominator_mask))
    black = int(np.count_nonzero((classes == 2) & denominator_mask))
    return {
        "valid_pixel_count": total,
        "white_pixel_count": white,
        "white_area_fraction": white / total if total else 0.0,
        "black_pixel_count": black,
        "black_area_fraction": black / total if total else 0.0,
        "unclassified_pixel_count": total - white - black,
        "unclassified_area_fraction": (total - white - black) / total if total else 0.0,
    }


def rgb_to_lstar(rgb: np.ndarray) -> np.ndarray:
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    y = linear[..., 0] * 0.2126729 + linear[..., 1] * 0.7151522 + linear[..., 2] * 0.0721750
    delta = 6 / 29
    f = np.where(y > delta**3, np.cbrt(y), y / (3 * delta**2) + 4 / 29)
    return (116 * f - 16).astype(np.float32)


def image_to_lstar(image: Image.Image, chunk_rows: int = 512) -> np.ndarray:
    """Convert an RGB Pillow image in row chunks to limit peak memory on camera photos."""
    rgb = np.asarray(image, dtype=np.uint8)
    output = np.empty(rgb.shape[:2], dtype=np.float32)
    for start in range(0, rgb.shape[0], chunk_rows):
        stop = min(start + chunk_rows, rgb.shape[0])
        output[start:stop] = rgb_to_lstar(rgb[start:stop].astype(np.float32) / 255.0)
    return output


def _region_metrics(
    l_values: np.ndarray, corrected_l_values: np.ndarray
) -> dict[str, object]:
    if l_values.size == 0:
        return {"pixel_count": 0}
    percentiles = [10, 25, 50, 75, 90]
    raw_hist, edges = np.histogram(l_values, bins=100, range=(0.0, 100.0))
    corrected_hist, _ = np.histogram(corrected_l_values, bins=100, range=(0.0, 100.0))
    return {
        "pixel_count": int(l_values.size),
        "mean_l": float(np.mean(l_values)),
        "median_l": float(np.median(l_values)),
        "std_l": float(np.std(l_values)),
        "l_percentiles": {str(p): float(np.percentile(l_values, p)) for p in percentiles},
        "mean_darkness": float(100.0 - np.mean(l_values)),
        "median_darkness": float(100.0 - np.median(l_values)),
        "mean_corrected_darkness": float(100.0 - np.mean(corrected_l_values)),
        "median_corrected_darkness": float(100.0 - np.median(corrected_l_values)),
        "dark_fraction_l_below_25": float(np.mean(l_values < 25.0)),
        "histogram_l_raw": {"counts": raw_hist.astype(int).tolist(), "edges": edges.tolist()},
        "histogram_l_corrected": {"counts": corrected_hist.astype(int).tolist(), "edges": edges.tolist()},
    }
