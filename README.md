# Overlap-Aware Validation of Gleason Grading Models

Validation framework for patch-based Gleason grading models on prostate whole-slide images (WSIs). Instead of evaluating a model with a single criterion, the predictions are compared with the regions annotated by the pathologists at **increasing minimum overlap thresholds** (1%, 10%, 20%, …, 100%) between each patch and the annotated tissue. This shows how much of the performance of a model depends on the patches located at the edges of the annotated regions, both globally and image by image.

## Table of contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Outputs](#outputs)
- [Adding a new model](#adding-a-new-model)
- [Auxiliary scripts](#auxiliary-scripts)
- [Third-party code](#third-party-code)
- [Citation](#citation)
- [License](#license)

## Overview

For each model and dataset, the validation pipeline performs the following steps:

1. **Extraction geometry.** The read layer, patch size and scanning stride of each dataset are chosen from the training resolution of the model (`TARGET_MPP`), so that WSIs scanned at different magnifications (e.g. 40X and 16X) feed the network with the same physical scale.
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

Three models are currently supported:

| Name | Module | Architecture | Weights |
|---|---|---|---|
| `Densenet121_FS` | `StudentModel_densenet.py` | DenseNet121, fully supervised | `models/fully_supervised_model_strongly.pt` |
| `Densenet121_TS_simclr` | `StudentModel_Teacher_Student_simclr.py` | DenseNet121, SimCLR-pretrained backbone | `models/student_model_Teacher_Student_simclr_exp7.pt` |
| `Resnext101` | `StudentModel_REX.py` | ResNeXt-101 32x8d ([MicheleDamian/prostate-gleason-dataset](https://github.com/MicheleDamian/prostate-gleason-dataset)) | `models/resnext-101-32x8d.pth` |

> The model weights are not included in this repository and must be placed in the `models/` folder.

## Repository structure

```
.
├── test.py                         # Runs every model on every dataset and generates the summary figures
├── run_PROMETEO.py                 # Inference and validation of one model on one dataset
├── config.py                       # Datasets, default model and execution options
├── StudentModel_densenet.py        # Model module: DenseNet121 (fully supervised)
├── StudentModel_Teacher_Student_simclr.py  # Model module: DenseNet121 (SimCLR backbone)
├── StudentModel_REX.py             # Model module: ResNeXt-101 32x8d
├── validation/
│   ├── patchExtractor_strong_valme.py  # Annotated grid, matching and metrics (validationModel)
│   ├── VALME.csv                   # Image list of the VALME dataset
│   └── PANDA.csv                   # Image list of the PANDA dataset
├── utils/
│   ├── fixed_cohort.py             # Fixed cohort selection and figure
│   ├── cohort_figures.py           # Fixed cohort figures per model (article format)
│   ├── grade_figures.py            # Recall per Gleason grade versus overlap
│   ├── metric_figures.py           # Global metrics versus overlap
│   ├── image_index.py              # Maps "Imagen N" in the results to its WSI file
│   ├── patch_table.py              # Table of patches per image
│   └── roi_patch_overlay.py        # Draws the annotated ROIs and the patches that cover them
├── tiles.py, filter.py, slide.py, util.py  # Tissue detection utilities (third-party, see below)
└── models/                         # Model weights (not included)
```

## Requirements

- Python 3.8 or later
- [OpenSlide](https://openslide.org/) C library (required by `openslide-python`)
- A CUDA-capable GPU is recommended for inference (the code falls back to CPU)

Python libraries:

| Library | Purpose |
|---|---|
| `torch`, `torchvision` | Model definition and inference |
| `openslide-python` | Reading WSIs |
| `numpy`, `scipy`, `scikit-image` | Image processing and tissue detection |
| `Pillow`, `opencv-python` | Image I/O and drawing |
| `shapely` | Annotation polygons and overlap computation |
| `geojson` | Storage of the predictions |
| `staintools` | Stain normalizer (requires `spams`) |
| `matplotlib`, `seaborn` | Figures and confusion matrices |

## Installation

```bash
git clone <repository-url>
cd <repository-folder>

# OpenSlide C library (Ubuntu/Debian)
sudo apt-get install openslide-tools

# Python environment
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision openslide-python numpy scipy scikit-image Pillow opencv-python \
            shapely geojson staintools spams matplotlib seaborn
```

Install the PyTorch build that matches your CUDA version following the [official instructions](https://pytorch.org/get-started/locally/).

## Data preparation

All the data are read from, and all the results are written to, the `EXT/` folder at the root of the repository (it can be a symbolic link to another disk):

```
EXT/
├── VALME.csv                       # Image list of VALME
├── PANDA.csv                       # Image list of PANDA
├── VALME/
│   ├── <image>.tif                 # VALME WSIs
│   └── XML_perfeccionados_ampliado_nombresOriginales/<image>.xml
├── train_images/<image>.tiff       # PANDA WSIs
├── output_xml/XML_mask/<image>.xml # PANDA annotations converted to XML
├── SICAPV2/                        # Reference image of the stain normalizer
├── Predicciones_Guardadas/         # Prediction cache (generated)
└── resultados_analisis/            # Results (generated)
```

- **Image lists.** Only the first column of each CSV row is used: the name of the image, with or without extension.
- **Annotations.** XML files with `Annotation` nodes whose `Text` region gives the Gleason pattern (`0`, `3`, `4` or `5`) and whose `Vertices` give the polygon (`rtPolyline`, `rtNegativePen`, `rtRectangle` or `rtEllipse`).
- **Stain normalizer.** `run_PROMETEO.py` loads the reference image `EXT/SICAPV2/16B0006668_Block_Region_3_10_13_xini_67790_yini_77884.jpg_4.png`.

The paths of each dataset are defined in `AVAILABLE_DATASETS` in `config.py`, so other datasets can be added there.

## Configuration

`config.py` contains the following options:

| Option | Description |
|---|---|
| `GLEASON_MODEL` | Default model of `run_PROMETEO.py` when `--model` is not given |
| `AVAILABLE_DATASETS` | Name, image list, WSI folder, WSI extension and XML folder of each dataset |
| `devPhase` | `True` to save the debug images (`img_analized/`) and print timing information |
| `DEV_IMAGE_LIST`, `DEV_NUM_IMAGES` | Limit the evaluation to the first `DEV_NUM_IMAGES` images of each list |
| `DEV_MODE` | `True` to match the predictions in the main process instead of in parallel |
| `DEV_MODE_VER` | Verbose output of the matching step |
| `DEV_MODE_ALL_IMG_GG` | Debug images: `0` none, `1` patches predicted as GG5, `2` patches inside the ROIs |

## Usage

All the scripts must be run from the root of the repository.

### Full evaluation

```bash
python test.py
```

`test.py` runs `run_PROMETEO.py` once for every model in `MODELS` and every dataset in `DATASETS` (both lists are defined at the top of `test.py`). Each run is an independent process. When all the runs have finished, it generates the summary figures of each model in `EXT/resultados_analisis/`. Failed combinations are reported at the end and the exit code is `1` if any of them failed.

To evaluate only some models or datasets, edit the lists in `test.py`:

```python
MODELS = ['Densenet121_FS', 'Resnext101', 'Densenet121_TS_simclr']
DATASETS = ['valme', 'panda']
```

### Single model and dataset

```bash
python run_PROMETEO.py --model Resnext101 --dataset panda --images-file images.json
```

| Argument | Description |
|---|---|
| `--model` | Model name (`Densenet121_FS`, `Resnext101` or `Densenet121_TS_simclr`). Default: `GLEASON_MODEL` |
| `--dataset` | Dataset key of `AVAILABLE_DATASETS` (`valme` or `panda`). Default: `valme` |
| `--images-file` | JSON file with the images to validate (required) |

The images file is a list with the paths of each WSI and its annotation, relative to the root of the repository:

```json
[
  {"wsi": "./EXT/VALME/B16-004297-A3.tif", "xml": "./EXT/VALME/XML_perfeccionados_ampliado_nombresOriginales/B16-004297-A3.xml"}
]
```

`test.py` builds this file automatically from the image list of each dataset.

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

`test.py` also writes the summary figures of each model, comparing its datasets side by side: `{model}_cohort.png`, `{model}_grados.png` (mean recall per image), `{model}_grados_global.png` (pooled recall) and `{model}_metricas.png`.

The predictions of each WSI are cached in `EXT/Predicciones_Guardadas/{model}/` and the annotated grids next to the XML folder (`wsi_patch_map_{dataset}_{overlap}_patch{patch}_stride{stride}.pkl`), so later runs reuse them. Delete these files to force a new inference or a new partition.

> The keys of the JSON files and the names of the output files are kept in Spanish for compatibility with existing results.

## Adding a new model

Each model is a Python module in the root of the repository that exports a common interface. To add a new model:

**1. Create the model module** (e.g. `StudentModel_mynet.py`) exporting:

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

**2. Register the model** in `AVAILABLE_MODELS` in `run_PROMETEO.py`:

```python
AVAILABLE_MODELS = {
    ...
    'MyNet': 'StudentModel_mynet',
}
```

**3. Add it to the evaluation** in `MODELS` in `test.py` and copy its weights to `models/`.

`run_PROMETEO.py` checks the interface when the model is loaded and stops with an error if any attribute is missing. If the weights were saved as a complete object with `torch.save(model)` from a script where the class lived in `__main__`, register the class before loading them (see `build_model` in `StudentModel_densenet.py`).

## Auxiliary scripts

The scripts in `utils/` read the results already generated and can be run independently:

```bash
# Summary figures per model
python utils/cohort_figures.py EXT/resultados_analisis <output_folder> --pairs
python utils/grade_figures.py EXT/resultados_analisis <output_folder> [--pooled]
python utils/metric_figures.py EXT/resultados_analisis <output_folder>

# WSI file of each "Imagen N" of the results
python utils/image_index.py panda 7

# Table of patches per image at 1% overlap
python utils/patch_table.py panda <results_folder> --top 5 --mean

# Annotated ROIs (red) and the patches that cover them (blue)
python utils/roi_patch_overlay.py <wsi.tif> <wsi_patch_map_*.pkl> output.png --level 1
```

## Third-party code

`tiles.py`, `filter.py`, `slide.py` and `util.py` come from the [deep-histopath](https://github.com/CODAIT/deep-histopath) project and are distributed under the Apache License 2.0, as stated in their headers. They are used for tissue detection when scanning the WSIs.

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

## License

<!-- Add the license of the repository (e.g. MIT, Apache 2.0) and a LICENSE file. -->

## Authors

Lourdes Duran-Lopez, Juan P. Dominguez-Morales and Antonio M. Pérez Peña.

