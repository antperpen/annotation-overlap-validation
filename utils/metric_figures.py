import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.cohort_figures import FONT_SIZES, PAIR_FIGURE_SIZE, dataset_sort_key, find_combinations, save_figure
from utils.grade_figures import GRID_COLOR, INK_COLOR, MUTED_COLOR

# (key, label, color). The first three are read from fichero.json and DSC_KEY is calculated from
# ficheroImagenes.json. Colors of the categorical palette in a fixed order, close to those of
# grafica_solape_acierto.png (blue, red, green and yellow)
DSC_KEY = "valor medio DSC"
METRICS = (
    ("porcentaje acierto", "Accuracy", '#2a78d6'),
    ("porcentaje ROI cubiertos", "ROI\ncovered", '#eb6834'),
    ("porcentaje patches con grado gleason acertado", "Exact\ngrade", '#1baf7a'),
    (DSC_KEY, "DSC", '#eda100'),
)
# Gap to the right of the last overlap for the end-of-line labels (in X axis units).
# The long labels are split into two lines so that the gap does not narrow the axis and its numbers do not overlap
LABEL_MARGIN = 20
# Minimum separation between labels (larger than in grade_figures.py because some of them take two lines)
METRIC_LABEL_GAP = 0.12
# Minimum and maximum heights of the labels: a two-line label outside that range would overlap the
# X axis or go out at the top
LABEL_FLOOR = 0.08
LABEL_CEILING = 1.0


def place_labels(positions, min_gap, floor, ceiling):
    # If several labels end up at the same limit (floor or ceiling), the tie is broken by the real
    # value of their line so that the order of the labels matches the order of the lines
    ordered = sorted((min(max(y, floor), ceiling), y, key) for y, key in positions)
    heights = [clamped for clamped, _, _ in ordered]
    for i in range(1, len(heights)):
        heights[i] = max(heights[i], heights[i - 1] + min_gap)
    heights[-1] = min(heights[-1], ceiling)
    for i in range(len(heights) - 2, -1, -1):
        heights[i] = min(heights[i], heights[i + 1] - min_gap)
    return {key: y for y, (_, _, key) in zip(heights, ordered)}


def mean_dsc_per_overlap(results_dir, overlaps):
    with open(os.path.join(results_dir, 'ficheroImagenes.json'), 'r', encoding='utf-8') as json_file:
        images_json = json.load(json_file)
    values = {overlap: [] for overlap in overlaps}
    for entries in images_json.values():
        for entry in entries:
            if entry["porcentaje DSC"] != "N/A":
                values[entry["porcentaje solape"]].append(entry["porcentaje DSC"])
    return [sum(values[o]) / len(values[o]) if values[o] else float('nan') for o in overlaps]


def load_global_metrics(results_dir):
    with open(os.path.join(results_dir, 'fichero.json'), 'r', encoding='utf-8') as json_file:
        rows = sorted(json.load(json_file), key=lambda row: row["porcentaje solape"])
    overlaps = [row["porcentaje solape"] for row in rows]
    values = {key: [row[key] for row in rows] for key, _, _ in METRICS if key != DSC_KEY}
    values[DSC_KEY] = mean_dsc_per_overlap(results_dir, overlaps)
    return overlaps, values


def draw_metric_axes(ax, results_dir, title, ylabel):
    overlaps, values = load_global_metrics(results_dir)
    line_ends = []
    for key, label, color in METRICS:
        ax.plot(overlaps, values[key], color=color, linewidth=1.6, marker='o', markersize=3,
                markeredgecolor='white', markeredgewidth=0.6, zorder=4)
        valid = [(o, v) for o, v in zip(overlaps, values[key]) if not math.isnan(v)]
        if valid:
            line_ends.append((valid[-1], label, color))

    # Labels aligned in a column to the right of the last overlap (separated if they are close)
    label_x = overlaps[-1] + 5
    label_y = place_labels([(end_value, label) for (_, end_value), label, _ in line_ends],
                           METRIC_LABEL_GAP, LABEL_FLOOR, LABEL_CEILING)
    for end_point, label, color in line_ends:
        ax.annotate(label, end_point, xytext=(label_x, label_y[label]), textcoords='data', va='center',
                    fontsize=FONT_SIZES['legend'], fontweight='bold', color=color, linespacing=0.95,
                    annotation_clip=False)

    ax.set_title(title, fontsize=FONT_SIZES['title'], color=INK_COLOR, pad=5)
    ax.set_xticks(overlaps)
    ax.set_xlim(-2, overlaps[-1] + LABEL_MARGIN)
    ax.tick_params(labelsize=FONT_SIZES['tick'], length=0, pad=3, colors=MUTED_COLOR)
    ax.set_xlabel('Percentage of overlap', fontsize=FONT_SIZES['label'], labelpad=3, color=INK_COLOR)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'], labelpad=3, color=INK_COLOR)
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.6)
    ax.spines[['left', 'bottom']].set_color(MUTED_COLOR)


def plot_model_metrics(model, datasets, output_path):
    """One figure per model with the global metrics of its datasets, side by side"""
    ordered = sorted(datasets.items(), key=lambda item: dataset_sort_key(item[0]))
    fig, axes = plt.subplots(1, len(ordered), figsize=PAIR_FIGURE_SIZE, sharey=True, squeeze=False)
    for index, (ax, (dataset, results_dir)) in enumerate(zip(axes[0], ordered)):
        draw_metric_axes(ax, results_dir, dataset, 'Metric value' if index == 0 else None)
    # Title of the model centered over the pair of figures
    fig.suptitle(model, fontsize=FONT_SIZES['title'] + 0.5, color=INK_COLOR, y=1.02)
    fig.subplots_adjust(wspace=0.12)
    save_figure(fig, output_path)


def generate_metric_figures(combinations, output_dir):
    """Generates {model}_metricas.png in output_dir for each model of combinations with fichero.json"""
    os.makedirs(output_dir, exist_ok=True)
    for model, datasets in combinations.items():
        with_metrics = {dataset: path for dataset, path in datasets.items()
                        if os.path.isfile(os.path.join(path, 'fichero.json'))}
        if with_metrics:
            plot_model_metrics(model, with_metrics, os.path.join(output_dir, f"{model}_metricas.png"))


def main():
    parser = argparse.ArgumentParser(description="Global metrics versus the overlap per model and dataset")
    parser.add_argument('results_root', help="Folder with one results subfolder per combination")
    parser.add_argument('output_dir', help="Folder where the figures are saved")
    args = parser.parse_args()

    generate_metric_figures(find_combinations(args.results_root), args.output_dir)


if __name__ == '__main__':
    main()
