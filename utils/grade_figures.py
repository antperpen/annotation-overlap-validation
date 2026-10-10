"""
Evolution of the accuracy per Gleason grade versus the overlap calculated image by image, from the
ficheroImagenes.json files of run_PROMETEO.py. The figure grafica_solape_acierto_grados.png of
run_PROMETEO.py aggregates all the patches of all the images (mean weighted by the number of
patches); here the accuracy of each grade is first calculated in each image and then averaged
between images, so that each image has the same weight.

Accuracy (recall) of a grade in an image: patches with that true grade predicted with exactly that
grade, over all the patches of that true grade (row of the true-predicted matrix, same criterion
as grafica_solape_acierto_grados.png). An image only counts for a grade and an overlap if it has
patches of that true grade at that overlap.

With --pooled, the global recall of each grade is drawn instead, as in
grafica_solape_acierto_grados.png of run_PROMETEO.py: exact hits of the grade summed over all the
images over all the patches of that true grade summed over all the images (mean weighted by the
number of patches of each image).

Generates one figure per model with its datasets side by side ({model}_grados.png, or
{model}_grados_global.png with --pooled), with the size and fonts of the pairs of
utils/cohort_figures.py: centered titles and, instead of a legend, the name of each grade at the
end of its line.

Usage (from the root of the project):
    python3 utils/grade_figures.py <resultados_analisis_folder> <output_folder> [--pooled]
"""
import argparse
import json
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.cohort_figures import (FONT_SIZES, PAIR_FIGURE_SIZE, dataset_sort_key, find_combinations,
                                  save_figure)

GRADES = (0, 3, 4, 5)
# Categorical palette in a fixed order (validated for colour blindness). Yellow and green have low
# contrast with the white background: that is why each line has its direct label at the end
GRADE_COLORS = {0: '#eda100', 3: '#2a78d6', 4: '#eb6834', 5: '#1baf7a'}
INK_COLOR = '#0b0b0b'
MUTED_COLOR = '#52514e'
GRID_COLOR = '#e4e3df'
# Minimum separation (in Y axis units, 0-1) between end-of-line labels so that they do not overlap
MIN_LABEL_GAP = 0.07


def load_grade_accuracy(results_dir):
    """
    Returns (number of images, overlaps, {grade: {overlap: [(hits, patches of the true grade) of
    each image]}}) from the true-predicted matrix of each image in ficheroImagenes.json
    """
    with open(os.path.join(results_dir, 'ficheroImagenes.json'), 'r', encoding='utf-8') as json_file:
        images_json = json.load(json_file)
    accuracy = {grade: defaultdict(list) for grade in GRADES}
    overlaps = set()
    for entries in images_json.values():
        for entry in entries:
            overlap = entry["porcentaje solape"]
            overlaps.add(overlap)
            matrix = dict(entry["Grado Gleason Real - Predicho"])
            for grade in GRADES:
                true_patches = sum(matrix.get(f"{grade}-{predicted}", 0) for predicted in GRADES)
                if true_patches:
                    accuracy[grade][overlap].append((matrix.get(f"{grade}-{grade}", 0), true_patches))
    return len(images_json), sorted(overlaps), accuracy


def mean_per_image(counts_by_overlap, overlaps):
    """Mean recall between images at each overlap (NaN if no image has that grade)"""
    return [sum(hits / total for hits, total in counts_by_overlap[o]) / len(counts_by_overlap[o])
            if counts_by_overlap[o] else float('nan') for o in overlaps]


def pooled_recall(counts_by_overlap, overlaps):
    """
    Global recall at each overlap: hits of all the images over patches of the true grade of all
    the images (same calculation as pachesAcceptedGleason / totalPatchesTrueGG in run_PROMETEO.py)
    """
    curve = []
    for o in overlaps:
        total = sum(total for _, total in counts_by_overlap[o])
        curve.append(sum(hits for hits, _ in counts_by_overlap[o]) / total if total else float('nan'))
    return curve


def spread_labels(positions, min_gap):
    """
    Vertically relocates the end-of-line labels so that they are at least min_gap from each
    other, moving them as little as possible. positions: [(y, key)]; returns {key: label_y}
    """
    ordered = sorted(positions)
    placed = []
    for y, key in ordered:
        if placed and y - placed[-1][0] < min_gap:
            y = placed[-1][0] + min_gap
        placed.append((y, key))
    # If the last label goes out at the top, the whole block is moved down
    overflow = placed[-1][0] - 1.0 if placed else 0
    if overflow > 0:
        placed = [(y - overflow, key) for y, key in placed]
    return {key: y for y, key in placed}


def mark_not_applicable(ax, overlaps, curve):
    """
    Writes "N/A" (not applicable) at the overlaps where no image has patches of that true grade,
    at the height of the last previous point with data, so that the break of the line is not
    confused with a recall of 0
    """
    last_value = None
    for overlap, value in zip(overlaps, curve):
        if not math.isnan(value):
            last_value = value
        elif last_value is not None:
            ax.text(overlap, last_value, 'N/A', ha='center', va='center', fontsize=FONT_SIZES['tick'],
                    color=MUTED_COLOR, fontstyle='italic', zorder=5)


def draw_grade_axes(ax, results_dir, title, ylabel, aggregate):
    """
    Draws in ax the recall of each grade versus the overlap, with the label of the grade at the
    end of its line. aggregate combines the results of the images (mean_per_image or pooled_recall)
    """
    num_images, overlaps, accuracy = load_grade_accuracy(results_dir)
    missing = []
    line_ends = []
    for grade in GRADES:
        curve = aggregate(accuracy[grade], overlaps)
        counts = ", ".join(f"{o}%:{len(accuracy[grade][o])}" for o in overlaps)
        print(f"  {title} | Gleason {grade} | images per overlap: {counts}")
        valid = [(o, v) for o, v in zip(overlaps, curve) if not math.isnan(v)]
        if not valid:
            missing.append(f"G{grade}")
            continue
        ax.plot(overlaps, curve, color=GRADE_COLORS[grade], linewidth=1.6, marker='o', markersize=3,
                markeredgecolor='white', markeredgewidth=0.6, zorder=4)
        mark_not_applicable(ax, overlaps, curve)
        line_ends.append((valid[-1], grade))

    # Grade labels aligned in a column to the right of the last overlap, at the height of the
    # last point with data of each line (separated if they are too close)
    label_x = overlaps[-1] + 5
    label_y = spread_labels([(end_value, grade) for (_, end_value), grade in line_ends], MIN_LABEL_GAP)
    for (end_overlap, end_value), grade in line_ends:
        ax.annotate(f"G{grade}", (end_overlap, end_value), xytext=(label_x, label_y[grade]),
                    textcoords='data', va='center', fontsize=FONT_SIZES['legend'], fontweight='bold',
                    color=GRADE_COLORS[grade], annotation_clip=False)

    subtitle = f"{num_images} images" + (f" · {', '.join(missing)}: N/A" if missing else "")
    ax.set_title(f"{title} ({subtitle})", fontsize=FONT_SIZES['title'], color=INK_COLOR, pad=5)
    ax.set_xticks(overlaps)
    ax.set_xlim(-2, overlaps[-1] + 12)
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


def plot_model_grades(model, datasets, output_path, aggregate, ylabel):
    """One figure per model with the recall of each grade in its datasets, side by side"""
    ordered = sorted(datasets.items(), key=lambda item: dataset_sort_key(item[0]))
    fig, axes = plt.subplots(1, len(ordered), figsize=PAIR_FIGURE_SIZE, sharey=True, squeeze=False)
    for index, (ax, (dataset, results_dir)) in enumerate(zip(axes[0], ordered)):
        draw_grade_axes(ax, results_dir, dataset, ylabel if index == 0 else None, aggregate)
    # Title of the model centered over the pair of figures
    fig.suptitle(model, fontsize=FONT_SIZES['title'] + 0.5, color=INK_COLOR, y=1.02)
    fig.subplots_adjust(wspace=0.12)
    save_figure(fig, output_path)


def generate_grade_figures(combinations, output_dir, pooled):
    """
    Generates in output_dir, for each model of combinations (see find_combinations), {model}_grados.png
    (mean recall per image) or, with pooled, {model}_grados_global.png (weighted global recall)
    """
    if pooled:
        aggregate, ylabel, suffix = pooled_recall, 'Pooled recall', 'grados_global'
    else:
        aggregate, ylabel, suffix = mean_per_image, 'Mean recall per image', 'grados'
    os.makedirs(output_dir, exist_ok=True)
    for model, datasets in combinations.items():
        plot_model_grades(model, datasets, os.path.join(output_dir, f"{model}_{suffix}.png"), aggregate, ylabel)


def main():
    parser = argparse.ArgumentParser(description="Recall of each Gleason grade versus the overlap")
    parser.add_argument('results_root', help="Folder with one results subfolder per combination")
    parser.add_argument('output_dir', help="Folder where the figures are saved")
    parser.add_argument('--pooled', action='store_true',
                        help="Global recall (mean weighted by patches) instead of the mean per image")
    args = parser.parse_args()

    generate_grade_figures(find_combinations(args.results_root), args.output_dir, args.pooled)


if __name__ == '__main__':
    main()
