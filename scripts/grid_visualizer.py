import os
import cv2
import argparse
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
from typing import List

# Configuration & Constants
DPI = 300
GAP_MM = 2.5  # Gap between images in millimeters
LINE_WIDTH = 7  # Bounding box thickness
FONT_SIZE = 3  # Text font size
CONF_THRESHOLD = 0.25  # Confidence threshold

VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}


def calculate_gap_pixels(gap_mm: float, dpi: int) -> int:
    """Converts millimeters to pixels based on DPI."""
    gap_inch = gap_mm / 25.4
    return int(gap_inch * dpi)


def create_model_grid(model_path: Path, image_paths: List[Path], output_dir: Path, save_name: str):
    """
    Generates a 2x2 grid of predictions using a specific YOLO model.
    """
    if not model_path.exists():
        print(f"[Error] Model not found: {model_path}")
        return

    # Ensure we have exactly 4 images for a 2x2 grid
    if len(image_paths) < 4:
        print(f"[Error] Need at least 4 images for 2x2 grid. Found {len(image_paths)}.")
        return

    # Use only the first 4 images
    selected_images = image_paths[:4]
    print(f"[*] Loading model: {model_path.name}")
    model = YOLO(str(model_path))

    # --- 1. Determine Dimensions from First Image ---
    print("[*] Running inference to determine dimensions...")
    sample_res = model.predict(str(selected_images[0]), save=False, verbose=False)[0]
    sample_plot = sample_res.plot(line_width=LINE_WIDTH, font_size=FONT_SIZE)
    h, w, _ = sample_plot.shape

    # Calculate Gap
    gap_px = calculate_gap_pixels(GAP_MM, DPI)

    # Calculate Canvas Size
    out_h = 2 * h + gap_px
    out_w = 2 * w + gap_px

    # Create White Canvas
    canvas = 255 * np.ones((int(out_h), int(out_w), 3), dtype=np.uint8)

    # Define 2x2 Positions
    positions = [
        (0, 0),  # Top-Left
        (0, w + gap_px),  # Top-Right
        (h + gap_px, 0),  # Bottom-Left
        (h + gap_px, w + gap_px)  # Bottom-Right
    ]

    # --- 2. Process Images ---
    print(f"[*] Processing {len(selected_images)} images...")

    for img_path, (y, x) in zip(selected_images, positions):
        # Predict
        results = model.predict(str(img_path), conf=CONF_THRESHOLD, save=False, verbose=False)
        res_plot = results[0].plot(line_width=LINE_WIDTH, font_size=FONT_SIZE)

        # Resize if necessary (safety check)
        if res_plot.shape[:2] != (h, w):
            res_plot = cv2.resize(res_plot, (w, h))

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(res_plot, cv2.COLOR_BGR2RGB)

        # Paste onto canvas
        canvas[y:y + h, x:x + w] = img_rgb

    # --- 3. Save Results ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as TIFF (High Quality)
    save_path_tiff = output_dir / f"{save_name}.tiff"
    img_pil = Image.fromarray(canvas)
    img_pil.save(save_path_tiff, format="TIFF", compression="tiff_lzw", dpi=(DPI, DPI))

    print(f"[+] Grid saved to: {save_path_tiff}")


def main():
    # Setup Default Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DEFAULT_WEIGHTS = BASE_DIR / 'weights/best.pt'  # Default model name
    DEFAULT_IMAGES = BASE_DIR / 'data/images'  # Default image source
    DEFAULT_OUT = BASE_DIR / 'output/visual_results'

    parser = argparse.ArgumentParser(description="Tool 3: Prediction Grid Visualizer")

    parser.add_argument('--model', type=str, default=str(DEFAULT_WEIGHTS),
                        help='Path to the .pt model file')
    parser.add_argument('--source', type=str, default=str(DEFAULT_IMAGES),
                        help='Directory containing images (takes first 4)')
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUT),
                        help='Output directory')
    parser.add_argument('--name', type=str, default="grid_result",
                        help='Filename for the output')

    args = parser.parse_args()

    model_path = Path(args.model)
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    # Find images
    if not source_dir.exists():
        print(f"[Error] Source directory not found: {source_dir}")
        return

    images = sorted([
        p for p in source_dir.iterdir()
        if p.suffix.lower() in VALID_EXTENSIONS
    ])

    if not images:
        print(f"[Error] No images found in {source_dir}")
        return

    create_model_grid(model_path, images, output_dir, args.name)


if __name__ == "__main__":
    main()