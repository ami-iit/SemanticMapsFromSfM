from typing import Dict, Tuple
import numpy as np
from collections import Counter

import pycolmap  # for type hints; not strictly required


def compute_map_bounds(
    points3D: Dict[int, pycolmap.Point3D],
    margin: float = 0.5,
) -> Tuple[float, float, float, float]:
    """
    Returns (min_x, max_x, min_y, max_y) with a margin.
    """
    xyz = np.array([p.xyz for p in points3D.values()], dtype=np.float64)
    xs, ys = xyz[:, 0], xyz[:, 1]
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()

    return min_x - margin, max_x + margin, min_y - margin, max_y + margin


def build_semantic_map(
    points3D: Dict[int, pycolmap.Point3D],
    point_labels: Dict[int, int],
    cell_size: float = 0.05,
    unknown_label: int = 255,
) -> Tuple[np.ndarray, dict]:
    """
    Build a 2D semantic map in the X–Y plane.

    Args:
        points3D: reconstruction.points3D (dict[int, Point3D])
        point_labels: dict[point3D_id -> label]
        cell_size: grid cell size in meters
        unknown_label: label used for cells with no points

    Returns:
      - semantic_map: H x W array of class ids
      - metadata dict with bounds & cell_size
    """
    labeled_points = {pid: p for pid, p in points3D.items() if pid in point_labels}
    if not labeled_points:
        raise ValueError("No 3D points have semantic labels. Check inputs.")

    min_x, max_x, min_y, max_y = compute_map_bounds(labeled_points)

    width = int(np.ceil((max_x - min_x) / cell_size))
    height = int(np.ceil((max_y - min_y) / cell_size))

    # For each cell, accumulate labels
    cell_labels: Dict[Tuple[int, int], list] = {}

    for pid, pt in labeled_points.items():
        x, y = pt.xyz[0], pt.xyz[1]
        j = int((x - min_x) / cell_size)  # col
        i = int((y - min_y) / cell_size)  # row
        if 0 <= i < height and 0 <= j < width:
            cell_labels.setdefault((i, j), []).append(point_labels[pid])

    semantic_map = np.full((height, width), unknown_label, dtype=np.int32)

    for (i, j), labels in cell_labels.items():
        label_counts = Counter(labels)
        semantic_map[i, j] = label_counts.most_common(1)[0][0]

    metadata = {
        "min_x": float(min_x),
        "max_x": float(max_x),
        "min_y": float(min_y),
        "max_y": float(max_y),
        "cell_size": float(cell_size),
        "unknown_label": int(unknown_label),
    }
    return semantic_map, metadata


def colorize_semantic_map(
    semantic_map: np.ndarray,
    palette: Dict[int, Tuple[int, int, int]],
    unknown_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """
    Convert semantic map (H x W, int) to RGB image (H x W x 3, uint8)
    using provided palette (label -> (R, G, B)).
    """
    h, w = semantic_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in palette.items():
        mask = semantic_map == label
        rgb[mask] = color

    mask_unknown = ~(np.isin(semantic_map, list(palette.keys())))
    rgb[mask_unknown] = unknown_color

    return rgb