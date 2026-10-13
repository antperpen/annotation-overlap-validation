import csv
import json
import os
import subprocess
import sys
import tempfile

from config import AVAILABLE_DATASETS, DEV_IMAGE_LIST, DEV_NUM_IMAGES
from utils.cohort_figures import find_combinations, generate_cohort_pairs
from utils.grade_figures import generate_grade_figures
from utils.metric_figures import generate_metric_figures

MODELS = ['Densenet121_FS', 'Resnext101', 'Densenet121_TS_simclr']
DATASETS = ['valme','panda']
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Folder where run_validation.py stores one results subfolder per combination; the figures per
# model are stored in that same folder
RESULTS_ROOT = os.path.join(PROJECT_DIR, 'EXT', 'resultados_analisis')


def collect_images(dataset):
    """
    Returns the list of images of the dataset as dictionaries {'wsi': path, 'xml': path}. Only the
    first column of each CSV row is used (name of the image, with or without extension) and the
    paths come from the dataset configuration in config.py, without deducing the dataset of each
    row. With DEV_IMAGE_LIST only the first DEV_NUM_IMAGES rows are read. The paths are relative
    to the project folder, which is where run_validation.py is run from.
    """
    config = AVAILABLE_DATASETS[dataset]
    images = []
    with open(os.path.join(PROJECT_DIR, config['csv']), 'r', encoding='utf-8') as csv_file:
        rows = [row for row in csv.reader(csv_file) if row and row[0].strip()]
    if DEV_IMAGE_LIST:
        rows = rows[:DEV_NUM_IMAGES]
    for row in rows:
        name = os.path.splitext(row[0].strip())[0]
        wsi_path = os.path.join(config['wsi_dir'], name + config['wsi_ext'])
        xml_path = os.path.join(config['xml_dir'], name + '.xml')
        if os.path.isfile(os.path.join(PROJECT_DIR, wsi_path)) and os.path.isfile(os.path.join(PROJECT_DIR, xml_path)):
            images.append({'wsi': wsi_path, 'xml': xml_path})
        else:
            print(f"File not found: {wsi_path} or {xml_path}")
    return images


def run_combination(model, dataset, images):
    """
    Runs run_validation.py with a model, a dataset and its list of images and returns its exit
    code. The list is passed in a temporary JSON file that is deleted at the end, even if the run
    fails. Each run is an independent process, so the model, the GPU memory and the figures of
    one run do not affect the next one.
    """
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as images_file:
        json.dump(images, images_file)
    try:
        # Backend without windows: the final plt.show() of run_validation.py does not block the next run
        env = dict(os.environ, MPLBACKEND='Agg')
        command = [sys.executable, os.path.join(PROJECT_DIR, 'run_validation.py'), '--model', model,
                   '--dataset', dataset, '--images-file', images_file.name]
        return subprocess.run(command, cwd=PROJECT_DIR, env=env).returncode
    finally:
        os.remove(images_file.name)


def generate_summary_figures(failures):
    """
    Generates, only for the models in MODELS, the figures that compare their datasets side by side
    ({model}_cohort.png, {model}_grados.png, {model}_grados_global.png and {model}_metricas.png).
    They are generated even if some combination has failed, with the existing results, warning
    about the failed combinations and about the models that lack some dataset of DATASETS
    """
    for model, dataset, reason in failures:
        print(f"WARNING: the evaluation of {model} with {dataset} has failed ({reason}); if there are results "
              "from a previous run, the figures are generated with them")
    combinations = find_combinations(RESULTS_ROOT, MODELS)
    expected = {AVAILABLE_DATASETS[dataset]['name'] for dataset in DATASETS}
    for model in MODELS:
        missing = sorted(expected - set(combinations.get(model, {})))
        if model not in combinations:
            print(f"WARNING: there are no results for {model}; its figures are not generated")
        elif missing:
            print(f"WARNING: {model} lacks results for {', '.join(missing)}; its figures are incomplete")
    generate_cohort_pairs(combinations, RESULTS_ROOT)
    generate_grade_figures(combinations, RESULTS_ROOT, pooled=False)
    generate_grade_figures(combinations, RESULTS_ROOT, pooled=True)
    generate_metric_figures(combinations, RESULTS_ROOT)


def main():
    failures = []
    for model in MODELS:
        for dataset in DATASETS:
            print(f"===== Model: {model} | Dataset: {dataset} =====", flush=True)
            images = collect_images(dataset)
            if not images:
                print(f"===== FAILURE: model {model}, dataset {dataset} (no image was found) =====", flush=True)
                failures.append((model, dataset, 'no images'))
                continue
            return_code = run_combination(model, dataset, images)
            if return_code != 0:
                print(f"===== FAILURE: model {model}, dataset {dataset} (code {return_code}) =====", flush=True)
                failures.append((model, dataset, f'code {return_code}'))

    print("===== Generating the figures per model =====", flush=True)
    try:
        generate_summary_figures(failures)
    except Exception as error:
        # A failure when drawing must not hide the summary of the evaluations nor change their exit code
        print(f"===== FAILURE when generating the figures per model: {error!r} =====", flush=True)

    total = len(MODELS) * len(DATASETS)
    print(f"===== {total - len(failures)} of {total} evaluations completed =====")
    for model, dataset, reason in failures:
        print(f"  FAILURE: model {model}, dataset {dataset} ({reason})")
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
