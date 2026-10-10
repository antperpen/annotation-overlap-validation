"""
Read-only utility to find out which real file corresponds to each "Imagen N" shown in the
results of run_PROMETEO.py (ficheroImagenes.json, figures, confusion matrices, etc).

The numbering is set by run_PROMETEO.py by the position of each image in the list passed by
test.py: "Imagen N" is the N-th row of the dataset CSV (VALME.csv or PANDA.csv) whose WSI and XML
exist on disk, counting only those rows. This module reproduces the same filter as
collect_images() in test.py to be able to look up that correspondence without running the
validation again

Usage as a script (from the root of the project):
    python3 utils/image_index.py                 # all the images of all the datasets
    python3 utils/image_index.py valme            # only the given dataset
    python3 utils/image_index.py panda 7          # only Imagen 7 of that dataset
"""
import csv
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import AVAILABLE_DATASETS, DEV_IMAGE_LIST, DEV_NUM_IMAGES


def list_dataset_images(dataset):
    """
    Returns, in order, the WSI paths of a dataset ('valme' or 'panda') as numbered by
    run_PROMETEO.py ("Imagen 1", "Imagen 2", etc). Only the CSV rows whose WSI and XML exist on
    disk count, and if DEV_IMAGE_LIST is active in config.py, only the first DEV_NUM_IMAGES rows
    are considered (same filter as collect_images() in test.py)
    """
    dataset_config = AVAILABLE_DATASETS[dataset]
    with open(os.path.join(_PROJECT_ROOT, dataset_config['csv']), 'r', encoding='utf-8') as csv_file:
        rows = [row for row in csv.reader(csv_file) if row and row[0].strip()]
    if DEV_IMAGE_LIST:
        rows = rows[:DEV_NUM_IMAGES]

    images = []
    for row in rows:
        name = os.path.splitext(row[0].strip())[0]
        wsi_path = os.path.join(dataset_config['wsi_dir'], name + dataset_config['wsi_ext'])
        xml_path = os.path.join(dataset_config['xml_dir'], name + '.xml')
        if os.path.isfile(os.path.join(_PROJECT_ROOT, wsi_path)) and os.path.isfile(os.path.join(_PROJECT_ROOT, xml_path)):
            images.append(wsi_path)
    return images


def image_number(dataset, n):
    """WSI path of Imagen n (1-indexed) of a dataset, or None if n is out of range"""
    images = list_dataset_images(dataset)
    return images[n - 1] if 0 < n <= len(images) else None


if __name__ == '__main__':
    dataset_arg = sys.argv[1] if len(sys.argv) > 1 else None
    n_arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    for dataset in ([dataset_arg] if dataset_arg else AVAILABLE_DATASETS):
        images = list_dataset_images(dataset)
        print(f"\n== {dataset}: {len(images)} images")
        if n_arg:
            path = images[n_arg - 1] if 0 < n_arg <= len(images) else 'out of range'
            print(f"  Imagen {n_arg} -> {path}")
        else:
            for i, path in enumerate(images, start=1):
                print(f"  Imagen {i:>3} -> {path}")
