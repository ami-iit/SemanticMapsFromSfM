from typing import Dict
import numpy as np
from collections import Counter
from PIL import Image
import os

import pycolmap  # official COLMAP Python bindings


def load_semantic_mask(mask_dir: str, image_name: str) -> np.ndarray:
    """
    Loads a semantic mask as a 2D array, dtype=int.
    Assumes filename in mask_dir matches image_name but with .png extension,
    or exactly image_name if masks already share the same name.
    """
    # try exact name
    path_exact = os.path.join(mask_dir, image_name)
    if os.path.exists(path_exact):
        mask_path = path_exact
    else:
        base, _ = os.path.splitext(image_name)
        mask_path = os.path.join(mask_dir, base + ".png")

    if not os.path.exists(mask_path):
        raise FileNotFoundError(
            f"Semantic mask not found for image: {image_name} at {mask_path}"
        )

    mask = np.array(Image.open(mask_path), dtype=np.int32)
    if mask.ndim == 3:
        # if saved as RGB, take one channel or decode as needed
        mask = mask[..., 0].astype(np.int32)
    return mask


def assign_semantics_to_points(
    reconstruction: pycolmap.Reconstruction,
    mask_dir: str,
    min_observations: int = 1,
) -> Dict[int, int]:
    """
    Returns dict: point3D_id -> semantic_label (int).

    Uses majority vote across all image observations where the 2D point
    lies inside the semantic mask bounds.

    reconstruction.images: dict[int, pycolmap.Image]
    reconstruction.points3D: dict[int, pycolmap.Point3D]
    Each Point3D has a .track of TrackElement(image_id, point2D_idx). :contentReference[oaicite:3]{index=3}
    Each Image has .points2D (ListPoint2D) and each Point2D has .xy pixel coords. :contentReference[oaicite:4]{index=4}
    """
    images = reconstruction.images
    points3D = reconstruction.points3D

    # Cache loaded masks by image name
    mask_cache: Dict[str, np.ndarray] = {}

    point_labels: Dict[int, int] = {}

    for pid, pt in points3D.items():
        labels = []

        # pt.track is an iterable of TrackElement(image_id, point2D_idx)
        for elem in pt.track.elements:
            img_id = elem.image_id
            point2D_idx = elem.point2D_idx

            if img_id not in images:
                continue
            img = images[img_id]

            # Lazy-load semantic mask for this image
            if img.name not in mask_cache:
                mask_cache[img.name] = load_semantic_mask(mask_dir, img.name)
            mask = mask_cache[img.name]

            # Get 2D keypoint
            p2d = img.points2D[point2D_idx]
            xy = p2d.xy  # np.array([x, y])
            x, y = int(round(xy[0])), int(round(xy[1]))

            if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
                label = int(mask[y, x])  # row=y, col=x
                labels.append(label)

        if len(labels) >= min_observations and len(labels) > 0:
            label_counts = Counter(labels)
            label = label_counts.most_common(1)[0][0]
            point_labels[pid] = label

    return point_labels