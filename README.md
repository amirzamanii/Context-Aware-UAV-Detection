````markdown
# UAV Object Detection & Data Augmentation Toolkit

This repository contains the implementation of data augmentation and evaluation tools designed for **Small Object Detection (UAVs)**. The toolkit focuses on enhancing YOLO-based models in challenging environments (e.g., adverse weather conditions) using an improved **Copy-Paste Augmentation** strategy.

## 📂 Project Structure

This project is organized as a collection of independent research tools located in the `scripts/` directory:


Context-Aware-UAV-Detection/
├── scripts/                # Core Python scripts
│   ├── frame_extractor.py     # Video preprocessing tool
│   ├── copy_paste_aug.py      # Data augmentation tool
│   ├── grid_visualizer.py     # Qualitative result visualizer
│   └── precision_plotter.py   # Benchmark plotting tool
├── data/                      # Input data directory (User populated)
├── output/                    # Generated results (Automatically created)
└── weights/                   # Model weights directory
````

## ⚙️ Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/amirzamanii/Context-Aware-UAV-Detection.git
    cd Context-Aware-UAV-Detection
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 🛠️ Tools & Usage

### 1\. Video Frame Extractor

Extracts frames from video datasets to prepare them for training or validation.

```bash
python scripts/frame_extractor.py \
  --source "data/videos" \
  --output "output/frames" \
  --interval 0.5
```

### 2. Copy-Paste Augmentation

Generates synthetic training data by extracting objects from source images and pasting them onto target backgrounds using intelligent placement logic.

```bash
python scripts/copy_paste_aug.py \
  --source-img "data/images/objects" \
  --source-label "data/labels/objects" \
  --target-img "data/images/backgrounds" \
  --target-label "data/labels/backgrounds" \
  --output-img "output/augmented_data/images" \
  --output-label "output/augmented_data/labels" \
  --count 100
```

### 3\. Qualitative Visualization (Grid)

Creates a comparative 2x2 grid of model predictions for qualitative analysis. This is useful for comparing the baseline model against the proposed method.

```bash
python scripts/grid_visualizer.py \
  --model "weights/best.pt" \
  --source "data/test_images" \
  --name "comparison_grid"
```

### 4\. Benchmark Plotter

Generates precision comparison charts based on evaluation metrics.

```bash
python scripts/precision_plotter.py --output "output/charts"
```

## 📊 Datasets

To evaluate the performance of the proposed method, this research utilizes standard publicly available benchmarks for UAV and small object detection, as well as datasets specifically collected for adverse weather analysis.

**Note on Data Availability:**
Due to copyright restrictions of the third-party benchmarks and the blind review policy, the dataset files are **not included** in this repository.

**Instructions for Users:**

1.  Please use your own standard UAV detection datasets or download public benchmarks from their official sources.
2.  Organize your images into the `data/` directory.
3.  Update the input paths in the command-line arguments accordingly.

*(Detailed dataset specifications are available in the related publication.)*

## 📜 Citation

If you find this code useful in your research, please cite our paper:

*(Citation details are withheld for the review process and will be updated upon acceptance.)*

```
```
