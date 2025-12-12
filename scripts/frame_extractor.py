import cv2 as cv
import argparse
import sys
from pathlib import Path

VALID_EXTENSIONS = {'.avi', '.mp4', '.mov', '.mkv', '.mpeg', '.mpg'}


def extract_frames(video_path: Path, output_root: Path, interval: float):
    if not video_path.exists(): return

    save_dir = output_root / video_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened(): return

    fps = cap.get(cv.CAP_PROP_FPS)
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if fps <= 0: return

    current_time = 0.0
    while True:
        idx = int(current_time * fps)
        if idx >= total_frames: break
        cap.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: break

        cv.imwrite(str(save_dir / f"{video_path.stem}_{idx:06d}.jpg"), frame)
        current_time += interval
    cap.release()
    print(f"Processed: {video_path.name}")


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DEFAULT_IN = BASE_DIR / 'data/videos'
    DEFAULT_OUT = BASE_DIR / 'output/frames'

    parser = argparse.ArgumentParser(description="Tool 1: Video Frame Extractor")
    parser.add_argument('--source', default=str(DEFAULT_IN), help='Folder with videos')
    parser.add_argument('--output', default=str(DEFAULT_OUT), help='Output folder')
    parser.add_argument('--interval', type=float, default=0.5)
    args = parser.parse_args()

    src = Path(args.source)
    if not src.exists():
        print(f"Create '{src}' and put videos there.")
        return

    videos = [p for p in src.iterdir() if p.suffix.lower() in VALID_EXTENSIONS]
    for vid in videos:
        extract_frames(vid, Path(args.output), args.interval)


if __name__ == "__main__":
    main()