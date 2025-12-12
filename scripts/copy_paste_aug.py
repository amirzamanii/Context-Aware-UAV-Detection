"""
YOLO Copy-Paste Data Augmentation Script
"""

import random
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import cv2 as cv
from albumentations import Compose, RandomBrightnessContrast, RandomGamma, PixelDropout

# Configuration & Constants

IOU_THRESHOLD = 0.1
MAX_PLACEMENT_ATTEMPTS = 100
SKY_REGION_RATIO = 0.4
MIN_SCALE = 0.4
MAX_SCALE = 1.1

# Augmentation pipeline
PATCH_AUGMENTATION = Compose([
    RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    PixelDropout(p=0.05)
])


# Helper Functions


def calculate_iou_overlap(new_box: Tuple[int, int, int, int], existing_boxes: List[Tuple[int, int, int, int]]) -> bool:
    """Checks if a new bounding box overlaps significantly with existing boxes."""
    x1, y1, w1, h1 = new_box
    for (x2, y2, w2, h2) in existing_boxes:
        inter_x = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        inter_y = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        inter_area = inter_x * inter_y
        if inter_area == 0:
            continue
        union_area = (w1 * h1) + (w2 * h2) - inter_area
        if (inter_area / union_area) > IOU_THRESHOLD:
            return True
    return False


def yolo_to_pixel(yolo_box: Tuple[int, float, float, float, float], img_w: int, img_h: int) -> Tuple[
    int, int, int, int]:
    """Converts YOLO normalized coordinates to pixel coordinates."""
    _, xc, yc, w, h = yolo_box
    pixel_w = int(w * img_w)
    pixel_h = int(h * img_h)
    x_min = int((xc - w / 2) * img_w)
    y_min = int((yc - h / 2) * img_h)
    return max(0, x_min), max(0, y_min), pixel_w, pixel_h


def find_valid_position(bg_shape: Tuple[int, int], patch_shape: Tuple[int, int],
                        existing_boxes: List[Tuple[int, int, int, int]]) -> Tuple[Optional[int], Optional[int]]:
    """Finds a random valid position for the patch within the allowed sky region."""
    bg_h, bg_w = bg_shape[:2]
    p_h, p_w = patch_shape[:2]
    max_y_allow = int(bg_h * SKY_REGION_RATIO) - p_h
    if max_y_allow < 0:
        max_y_allow = bg_h - p_h
    if p_w >= bg_w or p_h >= bg_h:
        return None, None

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        x = random.randint(0, bg_w - p_w)
        y = random.randint(0, max_y_allow)
        if not calculate_iou_overlap((x, y, p_w, p_h), existing_boxes):
            return x, y
    return None, None


def load_dataset_files(img_dir: Path, label_dir: Path) -> List[Dict]:
    """Scans directories and pairs images with label files."""
    data = []
    # Support multiple extensions
    img_paths = sorted(list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpeg')))
    for img_path in img_paths:
        base_name = img_path.stem
        label_path = label_dir / f"{base_name}.txt"
        if label_path.exists():
            data.append({'img_path': str(img_path), 'label_path': str(label_path), 'base_name': base_name})
    return data


def parse_yolo_labels(label_path: str) -> List[Tuple]:
    """Reads a YOLO txt file and returns a list of boxes."""
    boxes = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    boxes.append((cls_id, *coords))
    except Exception:
        pass
    return boxes


def save_yolo_labels(output_path: str, original_boxes: List, new_boxes: List):
    """Saves combined original and new boxes to a text file."""
    all_boxes = original_boxes + new_boxes
    lines = [f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cls, xc, yc, w, h in all_boxes]
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))


def generate_augmented_images(source_data, target_data, output_img_dir, output_lbl_dir, num_generated, copy_range):
    """Main loop to generate augmented images."""
    print(f"[*] Starting augmentation. Target: {num_generated} images.")
    count = 0

    for i in range(num_generated):
        target_item = random.choice(target_data)
        target_img = cv.imread(target_item['img_path'])
        if target_img is None: continue

        target_labels = parse_yolo_labels(target_item['label_path'])
        h_bg, w_bg = target_img.shape[:2]
        existing_pixel_boxes = [yolo_to_pixel(box, w_bg, h_bg) for box in target_labels]

        num_copies = random.randint(*copy_range)
        new_yolo_labels = []
        successful_pastes = 0

        for _ in range(num_copies):
            source_item = random.choice(source_data)
            src_img = cv.imread(source_item['img_path'])
            if src_img is None: continue

            src_labels = parse_yolo_labels(source_item['label_path'])
            if not src_labels: continue

            # Pick a random object from source
            src_box = random.choice(src_labels)
            cls_id = src_box[0]

            x, y, w, h = yolo_to_pixel(src_box, src_img.shape[1], src_img.shape[0])
            patch = src_img[y:y + h, x:x + w]
            if patch.size == 0: continue

            # Apply Augmentations
            patch = PATCH_AUGMENTATION(image=patch)['image']
            scale = random.uniform(MIN_SCALE, MAX_SCALE)
            patch = cv.resize(patch, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

            # Paste
            pos_x, pos_y = find_valid_position(target_img.shape, patch.shape, existing_pixel_boxes)

            if pos_x is not None:
                h_p, w_p = patch.shape[:2]
                target_img[pos_y:pos_y + h_p, pos_x:pos_x + w_p] = patch

                existing_pixel_boxes.append((pos_x, pos_y, w_p, h_p))

                xc_norm = (pos_x + w_p / 2) / w_bg
                yc_norm = (pos_y + h_p / 2) / h_bg
                w_norm = w_p / w_bg
                h_norm = h_p / h_bg
                new_yolo_labels.append((cls_id, xc_norm, yc_norm, w_norm, h_norm))
                successful_pastes += 1

        output_filename = f"aug_{i:04d}_{target_item['base_name']}_{successful_pastes}copies"
        cv.imwrite(str(output_img_dir / f"{output_filename}.jpg"), target_img)
        save_yolo_labels(str(output_lbl_dir / f"{output_filename}.txt"), target_labels, new_yolo_labels)

        count += 1
        if count % 10 == 0:
            print(f" -> Generated {count}/{num_generated} images.")


# Main Execution


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Defaults
    DEFAULT_DATA_DIR = BASE_DIR / 'data'
    DEFAULT_OUT_DIR = BASE_DIR / 'output/augmented_data'

    parser = argparse.ArgumentParser(description="Tool 2: Copy-Paste Augmentor")

    # Arguments
    parser.add_argument('--source-img', default=str(DEFAULT_DATA_DIR / 'images'), help="Directory of source images")
    parser.add_argument('--source-label', default=str(DEFAULT_DATA_DIR / 'labels'), help="Directory of source labels")
    parser.add_argument('--target-img', default=str(DEFAULT_DATA_DIR / 'images'), help="Directory of target images")
    parser.add_argument('--target-label', default=str(DEFAULT_DATA_DIR / 'labels'), help="Directory of target labels")

    parser.add_argument('--output-img', default=str(DEFAULT_OUT_DIR / 'images'),
                        help="Directory to save augmented images")
    parser.add_argument('--output-label', default=str(DEFAULT_OUT_DIR / 'labels'),
                        help="Directory to save augmented labels")

    parser.add_argument('--count', type=int, default=100, help="Number of augmented images to generate")

    # --- FIX: Added missing arguments ---
    parser.add_argument('--min-copies', type=int, default=2, help="Min objects to paste")
    parser.add_argument('--max-copies', type=int, default=5, help="Max objects to paste")

    args = parser.parse_args()

    # Paths
    source_img_dir = Path(args.source_img)
    source_lbl_dir = Path(args.source_label)
    target_img_dir = Path(args.target_img)
    target_lbl_dir = Path(args.target_label)
    out_img_dir = Path(args.output_img)
    out_lbl_dir = Path(args.output_label)

    # Validation
    if not source_img_dir.exists():
        print(f"[Error] Source image directory not found: {source_img_dir}")
        return
    if not target_img_dir.exists():
        print(f"[Error] Target image directory not found: {target_img_dir}")
        return

    # Create output directories
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading datasets...")
    source_data = load_dataset_files(source_img_dir, source_lbl_dir)
    target_data = load_dataset_files(target_img_dir, target_lbl_dir)

    if source_data and target_data:
        generate_augmented_images(
            source_data,
            target_data,
            out_img_dir,
            out_lbl_dir,
            args.count,
            (args.min_copies, args.max_copies)
        )
    else:
        print("[Error] No valid data found. Check your data directories.")


if __name__ == "__main__":
    main()
