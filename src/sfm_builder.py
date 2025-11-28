import os
from turtle import home
from PIL import Image

# Resize all images before the reconstruction, to speed up the process and later make 2DGS reconstruction feasible.

def resize_images(input_dir, output_dir, max_side=640):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Supported image file extensions
    exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(exts):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with Image.open(input_path) as img:
                # Determine scale factor
                w, h = img.size
                scale = max_side / max(w, h)

                # Compute new size
                new_size = (int(w * scale), int(h * scale))

                # Resize with high-quality resampling
                resized_img = img.resize(new_size, Image.LANCZOS)

                # Save image
                resized_img.save(output_path)

            print(f"Saved: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help='input directory with the raw, original images', default='/home/mtoso/Desktop/Workshop-Images-26-11-2025/Raw_Images')
    parser.add_argument("--output_dir", help='output directory where to store the scaled images', default='/home/mtoso/Desktop/Workshop-Images-26-11-2025/Scaled_Images')
    parser.add_argument("--size", help='desired maximum length of the scaled images', default=640)
    args = parser.parse_args()      
    resize_images(args.input_dir, args.output_dir, args.size)


