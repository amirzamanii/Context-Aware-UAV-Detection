import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict

# Configuration & Styling

# Default Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / 'output/charts'
OUTPUT_FILENAME = "precision_chart_example.tiff"

DPI = 300
FIGURE_SIZE = (16, 10)
BAR_WIDTH = 0.18

# Placeholder Reference Number
REF_NUM = "XX"

# Font configurations
FONT_CONFIG = {
    'ylabel': {'fontsize': 24, 'weight': 'bold'},
    'xtick': {'fontsize': 22, 'weight': 'bold'},
    'legend': {'fontsize': 22},
    'label_normal': {'fontsize': 16, 'weight': 'normal'},
    'label_bold': {'fontsize': 16, 'weight': 'bold'},
}

# Generic Color Palette (Gray, Red, Orange, Green)
COLOR_PALETTE = ['#7F7F7F', '#D62728', '#FF7F0E', '#2CA02C']


# Helper Functions

def annotate_bars(ax, rects: List, method_index: int, max_indices: np.ndarray):
    """
    Adds value labels on top of bars. Bolds the text if it's the maximum value.
    """
    for dataset_idx, rect in enumerate(rects):
        height = rect.get_height()
        is_max = (method_index == max_indices[dataset_idx])
        font_style = FONT_CONFIG['label_bold'] if is_max else FONT_CONFIG['label_normal']

        ax.annotate(
            f'{height:.1f}',
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 8),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=font_style['fontsize'],
            weight=font_style['weight']
        )


def plot_benchmark_chart(datasets: List[str], data: Dict[str, List[float]], output_dir: Path):
    """
    Generates and saves the precision comparison chart using generic logic.
    """
    # 1. Setup Data
    x = np.arange(len(datasets))
    method_names = list(data.keys())
    values_matrix = np.array(list(data.values()))
    max_indices = np.argmax(values_matrix, axis=0)

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    num_methods = len(method_names)
    offsets = np.linspace(
        -BAR_WIDTH * (num_methods - 1) / 2,
        BAR_WIDTH * (num_methods - 1) / 2,
        num_methods
    )

    # 3. Plotting Loop
    for idx, (method_key, method_scores) in enumerate(data.items()):
        # Select color cyclically from the palette
        color = COLOR_PALETTE[idx % len(COLOR_PALETTE)]

        rects = ax.bar(
            x + offsets[idx],
            method_scores,
            BAR_WIDTH,
            label=method_key,
            color=color
        )
        annotate_bars(ax, rects, idx, max_indices)

    # 4. Formatting
    ax.set_ylabel('Precision (%)', **FONT_CONFIG['ylabel'])
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, **FONT_CONFIG['xtick'])
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.35)

    # 5. Legend
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.12),
        ncol=num_methods,
        fontsize=FONT_CONFIG['legend']['fontsize'],
        frameon=False
    )

    # 6. Saving
    plt.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / OUTPUT_FILENAME

    print(f"[*] Saving chart to: {save_path}")
    plt.savefig(
        save_path,
        dpi=DPI,
        pil_kwargs={"compression": "tiff_lzw"}
    )


# Main Execution

def main():
    parser = argparse.ArgumentParser(description="Tool 4: Precision Benchmark Plotter")
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help='Directory to save the chart')
    args = parser.parse_args()

    # --- DUMMY DATA FOR DEMONSTRATION ---
    # Replace these lists with your actual experimental results.

    datasets_list = ['Dataset A', 'Dataset B', 'Dataset C', 'Dataset D']

    precision_data = {
        'Baseline Method': [70.5, 75.0, 72.0, 60.5],
        f'State-of-the-Art [{REF_NUM}]': [72.0, 78.5, 76.0, 68.0],
        'Alternative Strategy': [71.5, 76.0, 74.5, 65.0],
        'Proposed Method': [85.0, 90.0, 88.0, 82.0]
    }

    plot_benchmark_chart(datasets_list, precision_data, Path(args.output))


if __name__ == "__main__":
    main()
