import argparse
import os
import json
import numpy as np
from PIL import Image

import pycolmap  # <-- new

from semantic_fusion import assign_semantics_to_points
from Old.map_builder import build_semantic_map, colorize_semantic_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate 2D semantic map from COLMAP (binary) model and per-image semantic masks."
    )
    parser.add_argument(
        "--model_dir",
        # required=True,
        help=(
            "Path to COLMAP sparse model directory containing cameras.bin, "
            "images.bin, points3D.bin (e.g. path/to/project/sparse/0)."
        ),
        default='/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Data/colmap_model'
    )
    # parser.add_argument("--mask_dir", required=True, help="Directory of per-image semantic masks.")
    # parser.add_argument("--out_dir", required=True, help="Output directory for semantic map.")
    parser.add_argument("--mask_dir", help="Directory of per-image semantic masks.", default='/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Data/semantic_masks')
    parser.add_argument("--out_dir", help="Output directory for semantic map.", default='/home/mtoso/Documents/Code/AMI_Collab/2DSemanticMap/Data/semantic_map')
    parser.add_argument("--cell_size", default=0.05, help="Grid cell size (meters).")
    parser.add_argument("--unknown_label", type=int, default=255, help="Label id for unknown cells.")
    parser.add_argument(
        "--palette_json",
        type=str,
        default=None,
        help="Optional JSON file mapping label (string) -> [R,G,B].",
    )
    parser.add_argument(
        "--min_observations",
        type=int,
        default=1,
        help="Minimum number of image observations for a 3D point to get a semantic label.",
    )
    return parser.parse_args()


def load_palette(palette_json_path):
    if palette_json_path is None:
        # default tiny palette example
        return {
            0: (0, 0, 0),       # background
            1: (0, 255, 0),     # floor
            2: (255, 0, 0),     # wall
            3: (0, 0, 255),     # furniture
        }

    with open(palette_json_path, "r") as f:
        data = json.load(f)
    palette = {}
    for k, v in data.items():
        label = int(k)
        rgb = tuple(int(x) for x in v)
        palette[label] = rgb
    return palette


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading COLMAP reconstruction (binary) with pycolmap...")
    # pycolmap.Reconstruction will look for cameras.{txt,bin}, images.{txt,bin}, points3D.{txt,bin} in model_dir. :contentReference[oaicite:6]{index=6}
    reconstruction = pycolmap.Reconstruction(args.model_dir)
    print(
        f"Loaded reconstruction with "
        f"{len(reconstruction.cameras)} cameras, "
        f"{len(reconstruction.images)} images, "
        f"{len(reconstruction.points3D)} points3D."
    )

    print("Assigning semantics to 3D points...")
    point_labels = assign_semantics_to_points(
        reconstruction,
        args.mask_dir,
        min_observations=args.min_observations,
    )
    print(f"Labeled {len(point_labels)} / {len(reconstruction.points3D)} 3D points.")

    print("Building 2D semantic map...")
    semantic_map, metadata = build_semantic_map(
        reconstruction.points3D,
        point_labels,
        cell_size=args.cell_size,
        unknown_label=args.unknown_label,
    )

    semantic_map_path = os.path.join(args.out_dir, "semantic_map.npy")
    np.save(semantic_map_path, semantic_map)
    print(f"Saved semantic map (npy) to {semantic_map_path}")

    meta_path = os.path.join(args.out_dir, "semantic_map_meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {meta_path}")

    print("Colorizing semantic map...")
    palette = load_palette(args.palette_json)
    rgb = colorize_semantic_map(semantic_map, palette)
    img_out_path = os.path.join(args.out_dir, "semantic_map.png")
    Image.fromarray(rgb).save(img_out_path)
    print(f"Saved colorized semantic map PNG to {img_out_path}")


if __name__ == "__main__":
    main()