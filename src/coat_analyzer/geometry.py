from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class HarmonicResult:
    coordinate: np.ndarray
    mask: np.ndarray
    scale_x: float
    scale_y: float
    iterations: int
    residual: float


def validate_partition_inputs(
    polygon: list[list[float]], left_edges: list[int], right_edges: list[int]
) -> None:
    if len(polygon) < 3:
        raise ValueError("请先画出手术区域")
    edge_count = len(polygon)
    left = {int(i) % edge_count for i in left_edges}
    right = {int(i) % edge_count for i in right_edges}
    if not left or not right:
        raise ValueError("请分别标记左边缘和右边缘")
    if left & right:
        raise ValueError("同一段边界不能同时属于左边缘和右边缘")
    if not _is_contiguous_cycle(left, edge_count) or not _is_contiguous_cycle(right, edge_count):
        raise ValueError("左边缘和右边缘都必须是连续的边界链")


def solve_harmonic_coordinate(
    polygon: list[list[float]],
    left_edges: list[int],
    right_edges: list[int],
    image_size: tuple[int, int],
    max_side: int = 320,
    max_iterations: int = 2500,
    tolerance: float = 1e-4,
) -> HarmonicResult:
    validate_partition_inputs(polygon, left_edges, right_edges)
    width, height = image_size
    if width < 2 or height < 2:
        raise ValueError("图片尺寸无效")
    shrink = min(1.0, max_side / max(width, height))
    grid_w = max(8, int(round(width * shrink)))
    grid_h = max(8, int(round(height * shrink)))
    scale_x = grid_w / width
    scale_y = grid_h / height
    points = [(float(p[0]) * scale_x, float(p[1]) * scale_y) for p in polygon]

    roi_image = Image.new("1", (grid_w, grid_h), 0)
    ImageDraw.Draw(roi_image).polygon(points, fill=1)
    mask = np.asarray(roi_image, dtype=bool)
    if int(mask.sum()) < 20:
        raise ValueError("手术区域过小")

    left_seed = _rasterize_edges(points, left_edges, (grid_w, grid_h), mask)
    right_seed = _rasterize_edges(points, right_edges, (grid_w, grid_h), mask)
    overlap = left_seed & right_seed
    if overlap.any():
        left_seed[overlap] = False
        right_seed[overlap] = False
    if not left_seed.any() or not right_seed.any():
        raise ValueError("左右边缘无法栅格化，请放大 ROI 后重新标记")

    yy, xx = np.mgrid[0:grid_h, 0:grid_w]
    left_center = np.array(np.nonzero(left_seed)).mean(axis=1)
    right_center = np.array(np.nonzero(right_seed)).mean(axis=1)
    dl = np.hypot(yy - left_center[0], xx - left_center[1])
    dr = np.hypot(yy - right_center[0], xx - right_center[1])
    u = dl / np.maximum(dl + dr, 1e-9)
    u[~mask] = 0.0
    u[left_seed] = 0.0
    u[right_seed] = 1.0
    fixed = left_seed | right_seed
    active = mask & ~fixed

    checkerboard = (xx + yy) & 1
    relaxation = 1.72
    residual = math.inf
    for iteration in range(1, max_iterations + 1):
        residual = 0.0
        for parity in (0, 1):
            total = np.zeros_like(u)
            count = np.zeros_like(u, dtype=np.uint8)
            total[1:, :] += u[:-1, :] * mask[:-1, :]
            count[1:, :] += mask[:-1, :]
            total[:-1, :] += u[1:, :] * mask[1:, :]
            count[:-1, :] += mask[1:, :]
            total[:, 1:] += u[:, :-1] * mask[:, :-1]
            count[:, 1:] += mask[:, :-1]
            total[:, :-1] += u[:, 1:] * mask[:, 1:]
            count[:, :-1] += mask[:, 1:]
            updated = total / np.maximum(count, 1)
            selected = active & (checkerboard == parity)
            if selected.any():
                change = relaxation * (updated[selected] - u[selected])
                residual = max(residual, float(np.max(np.abs(change))))
                u[selected] += change
            u[left_seed] = 0.0
            u[right_seed] = 1.0
        if iteration % 10 == 0 and residual < tolerance:
            break

    return HarmonicResult(u.astype(np.float32), mask, scale_x, scale_y, iteration, residual)


def contour_segments(result: HarmonicResult, level: float) -> list[list[list[float]]]:
    """Return marching-square line segments in original-image coordinates."""
    u = result.coordinate
    mask = result.mask
    segments: list[list[list[float]]] = []
    h, w = u.shape
    for y in range(h - 1):
        for x in range(w - 1):
            valid = [mask[y, x], mask[y, x + 1], mask[y + 1, x + 1], mask[y + 1, x]]
            if sum(valid) < 2:
                continue
            vals = [u[y, x], u[y, x + 1], u[y + 1, x + 1], u[y + 1, x]]
            crossings: list[tuple[float, float]] = []
            edges = [((x, y), (x + 1, y), 0, 1), ((x + 1, y), (x + 1, y + 1), 1, 2),
                     ((x + 1, y + 1), (x, y + 1), 2, 3), ((x, y + 1), (x, y), 3, 0)]
            for a, b, ia, ib in edges:
                if not (valid[ia] and valid[ib]):
                    continue
                va, vb = vals[ia], vals[ib]
                if (va < level <= vb) or (vb < level <= va):
                    t = (level - va) / (vb - va)
                    crossings.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
            if len(crossings) == 2:
                segments.append([
                    [float(crossings[0][0] / result.scale_x), float(crossings[0][1] / result.scale_y)],
                    [float(crossings[1][0] / result.scale_x), float(crossings[1][1] / result.scale_y)],
                ])
            elif len(crossings) == 4:
                segments.append([[float(crossings[0][0] / result.scale_x), float(crossings[0][1] / result.scale_y)],
                                 [float(crossings[1][0] / result.scale_x), float(crossings[1][1] / result.scale_y)]])
                segments.append([[float(crossings[2][0] / result.scale_x), float(crossings[2][1] / result.scale_y)],
                                 [float(crossings[3][0] / result.scale_x), float(crossings[3][1] / result.scale_y)]])
    return segments


def polygon_mask(size: tuple[int, int], polygon: list[list[float]]) -> np.ndarray:
    image = Image.new("1", size, 0)
    if len(polygon) >= 3:
        ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in polygon], fill=1)
    return np.asarray(image, dtype=bool)


def upsample_coordinate(result: HarmonicResult, size: tuple[int, int]) -> np.ndarray:
    scaled = np.clip(result.coordinate * 65535, 0, 65535).astype(np.uint16)
    image = Image.fromarray(scaled).resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32) / 65535.0


def _rasterize_edges(
    points: list[tuple[float, float]], edges: list[int], size: tuple[int, int], mask: np.ndarray
) -> np.ndarray:
    image = Image.new("1", size, 0)
    draw = ImageDraw.Draw(image)
    n = len(points)
    line_width = max(2, int(round(max(size) / 500)))
    for raw_index in edges:
        index = int(raw_index) % n
        draw.line([points[index], points[(index + 1) % n]], fill=1, width=line_width)
    seed = np.asarray(image, dtype=bool) & mask
    return seed


def _is_contiguous_cycle(indices: set[int], size: int) -> bool:
    if not indices:
        return False
    if len(indices) == size:
        return True
    starts = sum(1 for index in indices if (index - 1) % size not in indices)
    return starts == 1
