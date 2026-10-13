# Overlap-Aware Validation of Gleason Grading Models

Validation framework for patch-based Gleason grading models on prostate whole-slide images (WSIs). Instead of evaluating a model with a single criterion, the predictions are compared with the regions annotated by the pathologists at **increasing minimum overlap thresholds** (1%, 10%, 20%, …, 100%) between each patch and the annotated tissue. This shows how much of the performance of a model depends on the patches located at the edges of the annotated regions, both globally and image by image.

> **This repository contains only the source code.** The trained models and the datasets (WSIs and annotations) are **not included**. To run the validation you must provide your own models and data, following the instructions of [Adding a model](#adding-a-model) and [Dataset configuration](#dataset-configuration).

## Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Dataset configuration](#dataset-configuration)
- [Adding a model](#adding-a-model)
- [Usage](#usage)
- [Outputs](#outputs)
- [Other options](#other-options)
- [Third-party code](#third-party-code)
<!-- - [Citation](#citation) -->
- [License](#license)

## Overview

For each model and dataset, the validation pipeline performs the following steps:

1. **Extraction geometry.** The read layer, patch size and scanning stride of each dataset are chosen from the training resolution of the model (`TARGET_MPP`), so that WSIs scanned at different magnifications feed the network with the same physical scale.
2. **Inference.** The WSI is scanned with a sliding window. Background patches are discarded with a white/black filter and a tissue score, and the remaining ones are classified by the model (Gleason grade 0, 3, 4 or 5). The predictions are stored as GeoJSON and cached on disk.
3. **Annotated grid.** The XML annotations are split into a grid of cells aligned with the scanning stride. A cell is kept when its overlap with the annotated region reaches the current threshold. The grid of each threshold is cached as a `.pkl` file.
4. **Matching and metrics.** The predictions are matched with the annotated cells at every overlap threshold and the following metrics are computed per image and globally:

| Metric | Definition |
|---|---|
| Accuracy | Correctly classified patches (benign/malignant coherence) over evaluated patches |
| ROI covered | Annotated regions with at least 30% of their cells correctly classified |
| Exact grade | Correctly classified patches whose Gleason grade matches exactly |
| Recall per grade | Patches of a true grade predicted with that exact grade (mean per image and pooled) |
| DSC | Dice coefficient between the correctly classified patches and the whole annotated tissue (reference: cells with at least 1% overlap) |

5. **Fixed cohort.** Since increasing the threshold leaves some images without patches, the results are also analysed on the fixed cohort of images that keep patches at every threshold, to separate the effect of removing edge patches from the effect of excluding images.

## Repository structure

```
.
├── test.py                         # Runs every model on every dataset and generates the summary figures
├── run_validation.py               # Inference and validation of one model on one dataset
├── config.py                       # Datasets, default model and execution options
├── requirements.txt                # Python dependencies
├── StudentModel_*.py               # Model definitions (examples of the interface a model must implement)
├── validation/
│   └── patchExtractor_strong_valme.py  # Annotated grid, matching and metrics
├── utils/                          # Figures and auxiliary scripts that read the generated results
└── tiles.py, filter.py, slide.py, util.py  # Tissue detection utilities (third-party, see below)
```

The `models/` (model weights) and `EXT/` (datasets and results) folders are not part of the repository and must be created by the user.

## Requirements

- Python 3.8 or later
- [OpenSlide](https://openslide.org/) C library (required by `openslide-python`)
- A CUDA-capable GPU is recommended (the code falls back to CPU)
- The Python libraries listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

> PyTorch must be older than 2.6: from that version on, `torch.load` cannot load models saved as complete objects.

## Dataset configuration

All the data are read from, and all the results are written to, an `EXT/` folder at the root of the repository (it can be a symbolic link to another disk). The datasets are described in `AVAILABLE_DATASETS` in `config.py`, where each dataset gives the location of its image list, WSIs and XML annotations:

```python
AVAILABLE_DATASETS = {
    'valme': {'name': 'Valme', 'csv': './EXT/VALME.csv', 'wsi_dir': './EXT/VALME', 'wsi_ext': '.tif',
              'xml_dir': './EXT/VALME/XML'},
    'panda': {'name': 'Panda', 'csv': './EXT/PANDA.csv', 'wsi_dir': './EXT/train_images', 'wsi_ext': '.tiff',
              'xml_dir': './EXT/output_xml/XML_mask'},
}
```

| Field | Description |
|---|---|
| key (`'valme'`) | Identifier used in `DATASETS` (`test.py`) and in `--dataset` (`run_validation.py`) |
| `name` | Name used in the results folders and in the figures |
| `csv` | Image list: only the first column of each row is used (image name, with or without extension) |
| `wsi_dir`, `wsi_ext` | Folder and extension of the WSIs: the WSI of an image is `wsi_dir/<name><wsi_ext>` |
| `xml_dir` | Folder with the annotations (any name): the annotation of an image is `xml_dir/<name>.xml` |

Only the images of the list whose WSI **and** XML exist are evaluated. The folders are freely named: the code only uses the paths given in `AVAILABLE_DATASETS`. A possible layout of the `EXT/` folder for the datasets above is:

```
EXT/
├── VALME.csv                       # Image list of VALME
├── PANDA.csv                       # Image list of PANDA
├── VALME/
│   ├── <name>.tif                  # WSIs of VALME
│   └── XML/<name>.xml              # Annotations of VALME
├── train_images/<name>.tiff        # WSIs of PANDA
├── output_xml/XML_mask/<name>.xml  # Annotations of PANDA
├── Predicciones_Guardadas/         # Prediction cache (generated)
└── resultados_analisis/            # Results (generated)
```

To use another dataset, create its folders and add a new entry to `AVAILABLE_DATASETS` with the same fields; then add its key to `DATASETS` in `test.py`.

**Annotation format.** XML files with `Annotation` nodes whose `Text` region gives the Gleason pattern (`0`, `3`, `4` or `5`) and whose `Vertices` give the polygon (`rtPolyline`, `rtNegativePen`, `rtRectangle` or `rtEllipse`). Datasets with other annotation formats (e.g. masks) must be converted to this XML format.

**Generated files.** The prediction cache is written to `EXT/Predicciones_Guardadas/` and the annotated grids (`wsi_patch_map_*.pkl`) to the parent folder of each `xml_dir` (e.g. `EXT/VALME/`). Later runs reuse them: delete them to force a new inference or a new partition.

**Stain normalization (optional).** By default the patches are classified without color normalization. To normalize them (Macenko method) with a reference image, install the optional libraries `staintools` and `spams` (commented in `requirements.txt`) and set `STAIN_REFERENCE_IMAGE` in `config.py`:

```python
STAIN_REFERENCE_IMAGE = './EXT/reference_patch.png'
```

## Adding a model

Each model is a Python module in the root of the repository (the `StudentModel_*.py` files are examples). Adding a model takes three steps.

**1. Create the model module** (e.g. `StudentModel_mynet.py`) and place the weights in the `models/` folder. The module must export:

| Name | Description |
|---|---|
| `build_model(device)` | Returns the model loaded, in evaluation mode and on `device` |
| `predict(model, patch)` | Returns the tuple `(logits, probabilities)` for a batch of patches |
| `INPUT_SIZE` | Input size of the network (pixels) |
| `PATCH_SIZE` | Patch size at the training resolution (pixels) |
| `STRIDE` | Scanning stride at the training resolution (pixels) |
| `TARGET_MPP` | Resolution of the training images (µm/pixel) |
| `RESAMPLE` | PIL filter used to resize the patch to `INPUT_SIZE` (e.g. `Image.BILINEAR`) |
| `NORMALIZE` | `(mean, std)` of the input normalization, or `None` for raw pixels in [0, 1] |

The output classes must follow the order **0: benign, 1: Gleason 3, 2: Gleason 4, 3: Gleason 5**.

```python
import torch
from PIL import Image

MODEL_WEIGHTS = 'models/my_weights.pth'
INPUT_SIZE = 224
PATCH_SIZE = 512
STRIDE = 512
TARGET_MPP = 0.5
RESAMPLE = Image.BILINEAR
NORMALIZE = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def build_model(device):
    model = MyNetwork()
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    return model.to(device).eval()


def predict(model, patch):
    logits = model(patch)
    return logits, torch.nn.functional.softmax(logits, dim=1)
```

`run_validation.py` checks this interface when the model is loaded and stops with an error if any attribute is missing. If the weights were saved as a complete object with `torch.save(model)` from a script where the class lived in `__main__`, register the class before loading them (see `build_model` in `StudentModel_densenet.py`).

**2. Register the model** in `AVAILABLE_MODELS` in `run_validation.py`:

```python
AVAILABLE_MODELS = {
    ...
    'MyNet': 'StudentModel_mynet',
}
```

**3. Include it in `test.py`** by adding its name to the `MODELS` list:

```python
MODELS = ['Densenet121_FS', 'Resnext101', 'Densenet121_TS_simclr', 'MyNet']
```

## Usage

All the scripts must be run from the root of the repository, with the models in `models/` and the datasets in `EXT/`.

```bash
python test.py
```

`test.py` runs `run_validation.py` once for every model in `MODELS` and every dataset in `DATASETS` (both lists are defined at the top of `test.py`), each run as an independent process. When all the runs have finished, it generates the summary figures of each model. Failed combinations are reported at the end and the exit code is `1` if any of them failed. To evaluate only some models or datasets, edit these two lists:

```python
MODELS = ['Densenet121_FS']
DATASETS = ['valme', 'panda']
```

A single combination can also be run directly with `python run_validation.py --model <model> --dataset <dataset> --images-file <images.json>`, where the images file is a JSON list of `{"wsi": <path>, "xml": <path>}` entries (`test.py` builds it automatically).

## Outputs

Each model/dataset combination writes its results to `EXT/resultados_analisis/{model}_{dataset}_{patch}_{stride}_{footprint}/`:

| File | Content |
|---|---|
| `fichero.json` | Global metrics at each overlap threshold |
| `ficheroImagenes.json` | Metrics and true-predicted matrix of each image at each overlap threshold |
| `grafica_solape_acierto.png` | Global metrics versus overlap |
| `grafica_solape_acierto_grados.png` | Pooled recall of each Gleason grade versus overlap |
| `grafica_cohorte_fija.png` | Accuracy of the fixed cohort versus overlap |
| `Matrices_Generadas/` | Confusion matrices of each image and global, at each overlap (SVG) |
| `img_analized/` | Debug images (only with `devPhase = True`) |

`test.py` also writes, in `EXT/resultados_analisis/`, the summary figures of each model comparing its datasets side by side: `{model}_cohort.png`, `{model}_grados.png` (mean recall per image), `{model}_grados_global.png` (pooled recall) and `{model}_metricas.png`.

> The keys of the JSON files and the names of the output files are in Spanish.

## Other options

`config.py` also contains the following options:

| Option | Description |
|---|---|
| `GLEASON_MODEL` | Default model of `run_validation.py` when `--model` is not given |
| `STAIN_REFERENCE_IMAGE` | Reference image for the optional stain normalization (`None` to disable it) |
| `devPhase` | `True` to save the debug images (`img_analized/`) and print timing information |
| `DEV_IMAGE_LIST`, `DEV_NUM_IMAGES` | Limit the evaluation to the first `DEV_NUM_IMAGES` images of each list |
| `DEV_MODE` | `True` to match the predictions in the main process instead of in parallel |
| `DEV_MODE_VER` | Verbose output of the matching step |
| `DEV_MODE_ALL_IMG_GG` | Debug images: `0` none, `1` patches predicted as GG5, `2` patches inside the ROIs |

The scripts in `utils/` read the results already generated and can be run independently (`cohort_figures.py`, `grade_figures.py`, `metric_figures.py`, `image_index.py`, `patch_table.py` and `roi_patch_overlay.py`; each one documents its usage at the top of the file).

## Third-party code

`tiles.py`, `filter.py`, `slide.py` and `util.py` come from the [deep-histopath](https://github.com/CODAIT/deep-histopath) project and are distributed under the Apache License 2.0, as stated in their headers. They are used for tissue detection when scanning the WSIs.

<!--
## Citation

If you use this code, please cite:

```bibtex
@article{<key>,
  title   = {<title>},
  author  = {<authors>},
  journal = {<journal>},
  year    = {<year>}
}
```
-->

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Authors

Lourdes Duran-Lopez, Juan P. Dominguez-Morales and Antonio M. Pérez Peña.
