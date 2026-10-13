import argparse
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.fixed_cohort import (ALL_IMAGES_POOLED_COLOR, COHORT_MEDIAN_COLOR, COHORT_POOLED_COLOR,
                                get_fixed_cohort, pooled_accuracy)

# Half column of an article with two figures side by side: ~3.4 inches wide, at 300 dpi
FIGURE_SIZE = (3.4, 3.0)
# Pair of figures flattened so that several pairs fit per page: axes ~1.9 inches high, shared Y
# axis and a single legend for the pair
PAIR_FIGURE_SIZE = (6.8, 2.3)
DPI = 300
FONT_SIZES = {'title': 8.5, 'label': 8, 'tick': 7, 'legend': 7.5}
# Order of the datasets within each pair (those not listed go afterwards, in alphabetical order)
DATASET_ORDER = ['Valme', 'Panda']


def parse_combination(folder_name):
    """Returns (model, dataset) from the folder name {model}_{dataset}_{patch}_{stride}_{footprint}"""
    match = re.match(r'^(.*)_([^_]+)_\d+_\d+_\d+$', folder_name)
    if not match:
        raise ValueError("The folder does not follow the pattern {model}_{dataset}_{patch}_{stride}_{footprint}: " + folder_name)
    return match.group(1), match.group(2)


def load_images_value(results_dir):
    """Rebuilds imagesValue (overlap, patches, correct) from ficheroImagenes.json"""
    with open(os.path.join(results_dir, 'ficheroImagenes.json'), 'r', encoding='utf-8') as json_file:
        images_json = json.load(json_file)
    return {image: [(entry["porcentaje solape"], entry["número patches"], entry["patches acertados"])
                    for entry in entries]
            for image, entries in images_json.items()}


def find_combinations(results_root, models=None):
    """
    Returns {model: {dataset: folder}} with the results folders that contain ficheroImagenes.json.
    With models only those models are returned. The folders whose name does not follow the pattern
    {model}_{dataset}_{patch}_{stride}_{footprint} (e.g. backups) are ignored with a warning, and if
    a model has two folders of the same dataset the last one in alphabetical order is used, with a
    warning
    """
    combinations = defaultdict(dict)
    for folder in sorted(os.listdir(results_root)):
        if not os.path.isfile(os.path.join(results_root, folder, 'ficheroImagenes.json')):
            continue
        try:
            model, dataset = parse_combination(folder)
        except ValueError:
            print(f"WARNING: the folder {folder} is ignored: its name does not follow the pattern "
                  "{model}_{dataset}_{patch}_{stride}_{footprint}")
            continue
        if models is not None and model not in models:
            continue
        if dataset in combinations[model]:
            print(f"WARNING: {model} has several {dataset} folders; {folder} is used instead of "
                  f"{os.path.basename(combinations[model][dataset])}")
        combinations[model][dataset] = os.path.join(results_root, folder)
    return combinations


def dataset_sort_key(dataset):
    """Valme and Panda first, in that order; the rest afterwards in alphabetical order"""
    return (DATASET_ORDER.index(dataset), '') if dataset in DATASET_ORDER else (len(DATASET_ORDER), dataset)


def add_grouped_legend(ax, cohort_handles, all_images_handles):
    """
    Legend in two columns with a header per group ("Fixed cohort" and "All images"), so as not to
    repeat "(cohort)" and "(all images)" in each entry and to be able to use a larger font
    """
    def header():
        return plt.Line2D([], [], linestyle='none')

    rows = max(len(cohort_handles), len(all_images_handles))
    blanks = [(header(), '')] * (rows - len(all_images_handles))
    entries = ([(header(), 'Fixed cohort')] + cohort_handles
               + [(header(), 'All images')] + all_images_handles + blanks)
    legend = ax.legend([handle for handle, _ in entries], [label for _, label in entries],
                       loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False,
                       fontsize=FONT_SIZES['legend'], handlelength=2.2, columnspacing=1.2,
                       handletextpad=0.5, labelspacing=0.3)
    for text in legend.get_texts():
        if text.get_text() in ('Fixed cohort', 'All images'):
            text.set_fontweight('bold')


def add_pair_legend(fig, cohort_handles, all_images_handles, top):
    """
    A single legend for the whole pair, below the figures, in two rows: the first one with the
    header "Fixed cohort" and its entries and the second one with "All images" and its own
    """
    def header():
        return plt.Line2D([], [], linestyle='none')

    columns = max(len(cohort_handles), len(all_images_handles)) + 1
    first_row = [(header(), 'Fixed cohort')] + cohort_handles
    second_row = [(header(), 'All images')] + all_images_handles
    first_row += [(header(), '')] * (columns - len(first_row))
    second_row += [(header(), '')] * (columns - len(second_row))
    # matplotlib fills the legend by columns: the two rows are interleaved
    entries = [entry for pair in zip(first_row, second_row) for entry in pair]
    legend = fig.legend([handle for handle, _ in entries], [label for _, label in entries],
                        loc='upper center', bbox_to_anchor=(0.5, top), ncol=columns, frameon=False,
                        fontsize=FONT_SIZES['legend'], handlelength=2.2, columnspacing=1.2,
                        handletextpad=0.5, labelspacing=0.3)
    for text in legend.get_texts():
        if text.get_text() in ('Fixed cohort', 'All images'):
            text.set_fontweight('bold')


def draw_cohort_axes(ax, images_value, title, show_legend=True, ylabel='Percentage of successful patches'):
    """
    Draws in ax the fixed cohort figure of a combination. Returns the legend entries (cohort and
    all the images) to be able to build a common legend, or None if there is no cohort.
    With ylabel=None no label is placed on the Y axis (figures that share the axis with another one)
    """
    overlaps, per_image, cohort = get_fixed_cohort(images_value)
    if not cohort:
        ax.set_title(title, fontsize=FONT_SIZES['title'], pad=6)
        ax.text(0.5, 0.5, "No image has patches\nat every overlap", ha='center', va='center',
                fontsize=FONT_SIZES['label'], transform=ax.transAxes)
        ax.set_axis_off()
        return None
    accuracy = np.array([[values[o][1] / values[o][0] for o in overlaps] for values in cohort.values()])
    q1, q3 = np.percentile(accuracy, [25, 75], axis=0)

    iqr = ax.fill_between(overlaps, q1, q3, color=COHORT_MEDIAN_COLOR, alpha=0.15, linewidth=0, zorder=2)
    median, = ax.plot(overlaps, np.median(accuracy, axis=0), color=COHORT_MEDIAN_COLOR, linewidth=1.4,
                      marker='o', markersize=3.5, zorder=4)
    cohort_pooled, = ax.plot(overlaps, [pooled_accuracy(cohort, o) for o in overlaps], color=COHORT_POOLED_COLOR,
                             linewidth=1.2, marker='s', markersize=3.5, zorder=4)
    # Above the orange one: in small datasets it almost matches it and would be hidden
    all_pooled, = ax.plot(overlaps, [pooled_accuracy(per_image, o) for o in overlaps], color=ALL_IMAGES_POOLED_COLOR,
                          linewidth=1.4, linestyle=(0, (4, 2)), zorder=6)

    ax.set_xticks(overlaps)
    ax.tick_params(labelsize=FONT_SIZES['tick'], length=2.5, width=0.6, pad=2)
    ax.set_xlabel('Percentage of overlap', fontsize=FONT_SIZES['label'], labelpad=3)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_SIZES['label'], labelpad=3)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='lightgray', alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_linewidth(0.6)
    ax.set_title(f"{title}\nFixed cohort: {len(cohort)} of {len(per_image)} images",
                 fontsize=FONT_SIZES['title'], pad=6)
    cohort_entries = [(median, 'Median per image'), (iqr, 'Interquartile range'), (cohort_pooled, 'Pooled accuracy')]
    all_images_entries = [(all_pooled, 'Pooled accuracy')]
    if show_legend:
        add_grouped_legend(ax, cohort_entries, all_images_entries)
    return cohort_entries, all_images_entries


def save_figure(fig, output_path):
    """Saves the figure at print resolution and closes it"""
    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f"Figure saved in: {output_path}")


def plot_article_cohort(images_value, title, output_path):
    """A half-column fixed cohort figure for a model-dataset combination"""
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    entries = draw_cohort_axes(ax, images_value, title)
    save_figure(fig, output_path)
    return entries is not None


def plot_model_pair(model, datasets, output_path):
    """
    One figure per model with the figures of its datasets side by side, flattened so that several
    pairs fit per page: shared Y axis (label only on the first one) and a single legend
    """
    ordered = sorted(datasets.items(), key=lambda item: dataset_sort_key(item[0]))
    fig, axes = plt.subplots(1, len(ordered), figsize=PAIR_FIGURE_SIZE, sharey=True, squeeze=False)
    legend_entries = None
    for index, (ax, (dataset, results_dir)) in enumerate(zip(axes[0], ordered)):
        ylabel = 'Percentage of\nsuccessful patches' if index == 0 else None
        entries = draw_cohort_axes(ax, load_images_value(results_dir), f"{model} - {dataset}",
                                   show_legend=False, ylabel=ylabel)
        legend_entries = legend_entries or entries
    fig.subplots_adjust(wspace=0.12, bottom=0.22)
    if legend_entries:
        # The legend starts right below the X axis label, without a gap in between
        fig.canvas.draw()
        xlabel_box = axes[0][0].xaxis.label.get_window_extent().transformed(fig.transFigure.inverted())
        add_pair_legend(fig, *legend_entries, top=xlabel_box.y0 - 0.01)
    save_figure(fig, output_path)


def generate_cohort_pairs(combinations, output_dir):
    """Generates {model}_cohort.png in output_dir for each model of combinations (see find_combinations)"""
    os.makedirs(output_dir, exist_ok=True)
    for model, datasets in combinations.items():
        plot_model_pair(model, datasets, os.path.join(output_dir, f"{model}_cohort.png"))


def main():
    parser = argparse.ArgumentParser(description="Fixed cohort figures in article format")
    parser.add_argument('results_root', help="Folder with one results subfolder per combination")
    parser.add_argument('output_dir', help="Folder where the figures are saved")
    parser.add_argument('--only', help="Generates only the figure of this subfolder (to test the format)")
    parser.add_argument('--pairs', action='store_true',
                        help="One figure per model ({model}_cohort.png) with its datasets side by side")
    args = parser.parse_args()

    if args.pairs:
        generate_cohort_pairs(find_combinations(args.results_root), args.output_dir)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    folders = [args.only] if args.only else sorted(os.listdir(args.results_root))
    for folder in folders:
        results_dir = os.path.join(args.results_root, folder)
        if not os.path.isfile(os.path.join(results_dir, 'ficheroImagenes.json')):
            continue
        model, dataset = parse_combination(folder)
        plot_article_cohort(load_images_value(results_dir), f"{model} - {dataset}",
                            os.path.join(args.output_dir, f"cohort_{model}_{dataset}.png"))


if __name__ == '__main__':
    main()
