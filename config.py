# Gleason grade classification model imported by run_validation.py
# Available values: 'Densenet121_FS' (StudentModel_densenet.py), 'Resnext101' (StudentModel_REX.py), 'Densenet121_TS_simclr'
GLEASON_MODEL = 'Densenet121_TS_simclr'

# Available datasets (used by test.py and run_validation.py). Only the first column of each CSV row is
# used (name of the image, with or without extension): the WSI is wsi_dir/name + wsi_ext and its
# annotation xml_dir/name.xml. The map of annotated patches (.pkl) is stored in the folder that
# contains xml_dir
AVAILABLE_DATASETS = {
    'valme': {'name': 'Valme', 'csv': './EXT/VALME.csv', 'wsi_dir': './EXT/VALME', 'wsi_ext': '.tif',
             'xml_dir': './EXT/VALME/XML'},
    'panda': {'name': 'Panda', 'csv': './EXT/PANDA.csv', 'wsi_dir': './EXT/train_images', 'wsi_ext': '.tiff',
              'xml_dir': './EXT/output_xml/XML_mask'},
}

# Reference image to normalize the color of the patches before the prediction (requires the optional
# libraries staintools and spams). None --> no stain normalization
STAIN_REFERENCE_IMAGE = None

# True if you want generate images reports
devPhase = False
# True to evaluate only the first DEV_NUM_IMAGES images of the list
DEV_IMAGE_LIST = True
DEV_NUM_IMAGES = 300

#PatchExtractor_strong_valme.py
# True --> Enable parallel beahavior with one process
DEV_MODE = False
DEV_MODE_VER = True
# 0 --> Dissable Generation of images, 1 --> Generate images with patches with GG5 marked, 2 --> Generate images with patches inside ROIs
DEV_MODE_ALL_IMG_GG = 1
