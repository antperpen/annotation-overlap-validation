import argparse
import json
import os
import re
import sys

import openslide

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import AVAILABLE_DATASETS
from utils.image_index import list_dataset_images

OVERLAP = 1


def parse_geometry(results_dir):
    match = re.search(r'_(\d+)_(\d+)_(\d+)$', os.path.basename(os.path.normpath(results_dir)))
    if not match:
        raise ValueError("The results folder does not end in _{patch}_{stride}_{footprint}: " + results_dir)
    return int(match.group(2)), int(match.group(3))


def count_image_patches(wsi_path, stride, footprint):
    width, height = openslide.OpenSlide(wsi_path).dimensions
    patches_x = (width - footprint) // stride + 1 if width >= footprint else 0
    patches_y = (height - footprint) // stride + 1 if height >= footprint else 0
    return patches_x * patches_y


def load_overlap_results(results_dir, overlap):
    with open(os.path.join(results_dir, 'ficheroImagenes.json'), 'r', encoding='utf-8') as json_file:
        images_json = json.load(json_file)
    results = {}
    for key, entries in images_json.items():
        for entry in entries:
            if entry["porcentaje solape"] == overlap:
                results[int(key.split()[-1])] = (entry["número patches"], entry["número de ROI"])
    return results


def build_rows(dataset, results_dir):
    stride, footprint = parse_geometry(results_dir)
    images = list_dataset_images(dataset)
    rows = []
    for number, (annotated_patches, num_roi) in sorted(load_overlap_results(results_dir, OVERLAP).items()):
        if number > len(images):
            raise ValueError(f"Imagen {number} does not exist in the current list of {dataset} ({len(images)} images): "
                             "config.py or the CSV do not match those of the run")
        wsi_path = images[number - 1]
        rows.append({'image': number,
                     'name': os.path.splitext(os.path.basename(wsi_path))[0],
                     'total': count_image_patches(os.path.join(_PROJECT_ROOT, wsi_path), stride, footprint),
                     'cancer': annotated_patches,
                     'roi': num_roi})
    return rows


def mean_row(rows):
    return {key: sum(row[key] for row in rows) / len(rows) for key in ('total', 'cancer', 'roi')}


def print_table(rows, mean=None, num_images=None):
    print(f"{'Image':>5} | {'Name':<34} | {'Number of patches':>17} | {'Patches with cancer cells':>25} | {'ROI':>5}")
    for row in rows:
        print(f"{row['image']:>5} | {row['name']:<34} | {row['total']:>17} | {row['cancer']:>25} | {row['roi']:>5}")
    if mean:
        label = f"Mean ({num_images} images)"
        print(f"{'':>5} | {label:<34} | {mean['total']:>17.1f} | {mean['cancer']:>25.1f} | {mean['roi']:>5.1f}")


def main():
    parser = argparse.ArgumentParser(description="Table of patches per image with a 1% overlap")
    parser.add_argument('dataset', choices=AVAILABLE_DATASETS)
    parser.add_argument('results_dir', help="Results folder of run_validation.py with ficheroImagenes.json")
    parser.add_argument('--top', type=int, help="Only the N images with the most patches in the whole image")
    parser.add_argument('--mean', action='store_true', help="Adds the mean of all the analysed images")
    args = parser.parse_args()

    rows = build_rows(args.dataset, args.results_dir)
    shown = sorted(rows, key=lambda row: row['total'], reverse=True)[:args.top] if args.top else rows
    print_table(shown, mean_row(rows) if args.mean else None, len(rows))


if __name__ == '__main__':
    main()
