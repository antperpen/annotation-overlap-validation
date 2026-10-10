## Scripted by Lourdes Duran-Lopez and Juan P. Dominguez-Morales

import torch
import openslide
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import filter
import tiles
import time
import torchvision.transforms as transforms

import warnings
warnings.simplefilter("ignore")

import json

from geojson import Polygon, Feature, FeatureCollection, dump, load
from validation.patchExtractor_strong_valme import *
from utils.fixed_cohort import plot_fixed_cohort_accuracy

import sys
import importlib
import argparse
from math import log


import staintools
from config import *

# Settings
whiteFilter = 230
blackFilter = 100
scalarFactor = 8 # Factor used to convert the coordinates from one resolution layer to another
imDimLayer = 0 # Pixel coordinates refer to the highest resolution layer
resolutionLayer = 3 # The image is read at the 10X resolution layer
reductionLayer = 6 # Reduction factor from the lowest to the highest layer (0-->6)

#Maximum deviation allowed between the input scale and the training scale
MAX_SCALE_DIFF = 0.20

path_geoJSON_output = 'output/'

#Available models: model name --> model module
AVAILABLE_MODELS = {
    'Densenet121_FS': 'StudentModel_densenet',
    'Resnext101': 'StudentModel_REX',
    'Densenet121_TS_simclr': 'StudentModel_Teacher_Student_simclr',
}


parser = argparse.ArgumentParser(description="Validation of Gleason grade models")
parser.add_argument('--model', choices=AVAILABLE_MODELS, default=GLEASON_MODEL)
parser.add_argument('--dataset', choices=AVAILABLE_DATASETS, default='valme')
parser.add_argument('--images-file', help="JSON file with the list of images to validate")
args = parser.parse_args()
GLEASON_MODEL = args.model

def load_student_model(model_name):
    #Import the model module and check the common interface
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model '{model_name}' not available. Use one of: {list(AVAILABLE_MODELS.keys())}")
    module = importlib.import_module(AVAILABLE_MODELS[model_name])
    for attribute in ('build_model', 'predict', 'INPUT_SIZE', 'PATCH_SIZE', 'STRIDE',
                      'TARGET_MPP', 'RESAMPLE', 'NORMALIZE'):
        if not hasattr(module, attribute):
            raise AttributeError(f"Model '{model_name}' ({AVAILABLE_MODELS[model_name]}.py) does not define '{attribute}'")
    return module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Gleason model: {GLEASON_MODEL}")
student_model = load_student_model(GLEASON_MODEL)
model = student_model.build_model(device)


def _mpp_from_tiff_tags(properties):

    x_res = properties.get('tiff.XResolution')
    res_unit = properties.get('tiff.ResolutionUnit')
    if not x_res or not res_unit:
        return None
    x_res = float(x_res)
    if res_unit == 'centimeter':
        return 10000.0 / x_res
    if res_unit == 'inch':
        return 25400.0 / x_res
    return None


def resolve_extraction(image):
    mpp0 = image.properties.get('openslide.mpp-x') or _mpp_from_tiff_tags(image.properties)
    if not mpp0:
        raise ValueError("The image does not declare 'openslide.mpp-x' nor TIFF resolution tags "
                         "(tiff.XResolution/tiff.ResolutionUnit), the extraction layer for the "
                         "training scale of the model cannot be decided")
    mpp0 = float(mpp0)
    level = min(range(image.level_count),
                key=lambda L: abs(log(image.level_downsamples[L] * mpp0 / student_model.TARGET_MPP)))
    mpp_level = mpp0 * image.level_downsamples[level]
    tissue_um = student_model.PATCH_SIZE * student_model.TARGET_MPP
    read_size = max(1, round(tissue_um / mpp_level))
    #Patch footprint and scanning stride in layer 0 pixels
    footprint_size = max(1, round(read_size * image.level_downsamples[level]))
    stride_size = max(1, round(footprint_size * student_model.STRIDE / student_model.PATCH_SIZE))
    entry_mpp = mpp_level * read_size / student_model.INPUT_SIZE
    reference_mpp = student_model.TARGET_MPP * student_model.PATCH_SIZE / student_model.INPUT_SIZE
    if abs(entry_mpp - reference_mpp) / reference_mpp > MAX_SCALE_DIFF:
        raise ValueError(f"Incompatible scale: the network would receive {entry_mpp:.3f} um/px instead of "
                         f"the {reference_mpp:.3f} um/px of its training with this image "
                         f"(mpp={mpp0}). Check the resolution of the dataset or the geometry of the model")
    print(f"Extraction: layer {level} ({mpp_level:.3f} um/px), patch {read_size} px, "
          f"footprint {footprint_size} px and stride {stride_size} px in layer 0, "
          f"input scale {entry_mpp:.3f} um/px")
    return level, read_size, footprint_size, stride_size


def build_preprocess():
    steps = [
        transforms.ToTensor(),
        transforms.Resize((student_model.INPUT_SIZE, student_model.INPUT_SIZE),
                          interpolation=student_model.RESAMPLE),
    ]
    if student_model.NORMALIZE is not None:
        mean, std = student_model.NORMALIZE
        steps.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(steps)


def generateHeatmapValidation(WSI_path, val_object, probability_threshold = 0.7, listImages = None):
    WSI_name = os.path.splitext(os.path.basename(WSI_path))[0]
    WSI_name_ext = Path(WSI_path).name
    #If image has two layers dimension for each real layer on the image
    double_dim = 0
    #Each model saves its predictions in its own folder
    predictions_folder = os.path.join(os.getcwd(),"EXT", "Predicciones_Guardadas", GLEASON_MODEL)
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)
    #Prediction files are tagged with the model name
    model_tag = student_model.__name__
    imglayer_path = os.path.join(predictions_folder, f"imglayer_{WSI_name}_{model_tag}.npy")
    prediction_file_path = os.path.join(predictions_folder, f"prediccion_{WSI_name}_{model_tag}.json")

    if listImages != None and not listImages[WSI_path] and not os.path.isfile(prediction_file_path):
        print(f"Processing WSI: {WSI_name}")

        with torch.no_grad():
            features_gp0 = []
            features_gp3 = []
            features_gp4 = []
            features_gp5 = []
            imglayer = []
            #Reference image to normalize the color of the WSIs

            target = staintools.read_image(os.path.join(os.getcwd(),"EXT/SICAPV2", "16B0006668_Block_Region_3_10_13_xini_67790_yini_77884.jpg_4.png"))
            normalizer = staintools.StainNormalizer(method='macenko')
            normalizer.fit(target)
            if WSI_path.endswith(".mrxs") or WSI_path.endswith(".bif") or WSI_path.endswith(".tif") or WSI_path.endswith(".tiff"):
                image = openslide.OpenSlide(WSI_path)
                #Extraction geometry resolved in the validator
                extractLevel = val_object.extract_level
                patchSize = val_object.patch_size_read
                strideSize = val_object.stride
                if(reductionLayer >= image.level_count):
                    highestLayer = image.level_count -1
                else:
                    highestLayer = image.level_count - reductionLayer
                if(devPhase):
                    imglayer = image.read_region((0,0), highestLayer ,(image.level_dimensions[highestLayer][0],image.level_dimensions[highestLayer][1]))
                    imglayer.save("./Img_topLayer_oring.png")
                    imglayer = np.array((imglayer.convert("RGB")))
                for i in range(0, image.level_dimensions[0][1], strideSize):
                    if (i + strideSize < image.level_dimensions[0][1]):
                        for j in range(0, image.level_dimensions[0][0], strideSize):
                            if (j + strideSize < image.level_dimensions[0][0]):
                                # Read patch

                                tile = image.read_region((j, i), extractLevel, (patchSize, patchSize))
                                # White/Black filter
                                tile_mean = np.mean(tile)
                                if (tile_mean<=whiteFilter and tile_mean>=blackFilter):

                                    tile = np.array(tile)
                                    tile = tile[:,:,0:3]

                                    # Patch scoring
                                    color_factor = tiles.hsv_purple_pink_factor(tile)
                                    s_and_v_factor = tiles.hsv_saturation_and_value_factor(tile)

                                    t_p = filter.tissue_percent(tile)
                                    score = t_p/100 * color_factor * s_and_v_factor

                                    if score>= 0.7:

                                        # CNN prediction
                                        preprocess = build_preprocess()

                                        patch = preprocess(tile).float()
                                        patch = patch.unsqueeze(0)

                                        patch = patch.to(device)

                                        logits, classes = student_model.predict(model, patch)

                                        classes = classes.cpu()
                                        winning_class_label = int(np.argmax(classes))
                                        winning_class_confidence = float(np.max(np.array(classes), axis=1))

                                        print(classes, winning_class_label, winning_class_confidence)

                                        ## GEOJSON CREATION

                                        #Patch geometry in layer 0 (side = scanning stride)
                                        patch_x0 = j
                                        patch_y0 = i
                                        patch_x1 = j + strideSize
                                        patch_y1 = i + strideSize

                                        central_point = ((patch_x0 + patch_x1)/2, (patch_y0 + patch_y1)/2)
                                        try:
                                            polygon = Polygon([(patch_x0, patch_y0), (patch_x1, patch_y0),
                                                (patch_x1, patch_y1), (patch_x0, patch_y1)])
                                            if winning_class_label == 0:
                                                features_gp0.append(Feature(geometry=polygon, properties={"type": "Heatmap", "probability":
                                                round(winning_class_confidence, 2), "centralPoint": central_point}))
                                            elif winning_class_label == 1:
                                                features_gp3.append(Feature(geometry=polygon, properties={"type": "Heatmap", "probability":
                                                round(winning_class_confidence, 2), "centralPoint": central_point}))
                                            elif winning_class_label == 2:
                                                features_gp4.append(Feature(geometry=polygon, properties={"type": "Heatmap", "probability":
                                                round(winning_class_confidence, 2), "centralPoint": central_point}))
                                            elif winning_class_label == 3:
                                                features_gp5.append(Feature(geometry=polygon, properties={"type": "Heatmap", "probability":
                                                round(winning_class_confidence, 2), "centralPoint": central_point}))
                                        except Exception:
                                            print("############ Failed to create the polygon")

                feature_collection_gp0 = FeatureCollection(features_gp0)
                feature_collection_gp3 = FeatureCollection(features_gp3)
                feature_collection_gp4 = FeatureCollection(features_gp4)
                feature_collection_gp5 = FeatureCollection(features_gp5)
                dim_img = (image.level_dimensions[0][0], image.level_dimensions[0][1])
                feature_collections = ((feature_collection_gp0,0),(feature_collection_gp3,3), (feature_collection_gp4,4), (feature_collection_gp5,5))
                listImages[WSI_path] = (feature_collections, dim_img, highestLayer,imglayer)
                #Data structure to save the prediction
                data_to_save = {
                    "nombre_imagen": WSI_name_ext,
                    "dim_img": [image.level_dimensions[0][0], image.level_dimensions[0][1]],
                    "highestLayer": highestLayer,
                    "predicciones": {
                        "gp0": feature_collection_gp0,
                        "gp3": feature_collection_gp3,
                        "gp4": feature_collection_gp4,
                        "gp5": feature_collection_gp5
                    }
                }

                with open(prediction_file_path, "w") as prediction_file:
                    dump(data_to_save, prediction_file, indent=4)
                #Save the imglayer as a numpy array
                np.save(imglayer_path, imglayer)
                print(f"--- Prediction of image '{WSI_name}' successfully saved in: {prediction_file_path} ---")

    elif os.path.isfile(prediction_file_path) and os.path.isfile(imglayer_path) and not listImages[WSI_path]:
        start_time = time.time()
        with open(prediction_file_path, "r") as prediction_file:
            loaded_data = load(prediction_file)

        imglayer = np.load(imglayer_path)
        #imglayer is only needed to draw the debug images (devPhase)
        if len(loaded_data) == 0 or (devPhase and imglayer.size == 0):
            print(f"Error: The file '{prediction_file_path}' is empty or does not contain valid JSON data.")
            sys.exit()
        collection_gp0 = loaded_data["predicciones"]["gp0"]
        collection_gp3 = loaded_data["predicciones"]["gp3"]
        collection_gp4 = loaded_data["predicciones"]["gp4"]
        collection_gp5 = loaded_data["predicciones"]["gp5"]
        dim_img, highestLayer = loaded_data["dim_img"], loaded_data["highestLayer"]
        feature_collections = (((collection_gp0,0),(collection_gp3,3),(collection_gp4,4),(collection_gp5,5)))
        listImages[WSI_path] = (feature_collections, dim_img, highestLayer,imglayer)
        end_time = time.time()
        total_time = end_time - start_time
        if(devPhase):
            print(f"Time to load the prediction from the file: {total_time} seconds")
        print(f"--- Prediction of image '{WSI_name}' successfully loaded from: {prediction_file_path} ---")

    elif listImages == None:
        print("The list of images is empty")
        sys.exit()
    start_time = time.time()
    if (WSI_path.endswith(".tiff")):
        double_dim = 1
    status = val_object.validate_model(WSI_name_ext,listImages[WSI_path][0], probability_threshold, 0, listImages[WSI_path][1], imDimLayer,
    listImages[WSI_path][2], listImages[WSI_path][3], double_dim = double_dim)
    end_time = time.time()
    total_time = end_time - start_time
    if(devPhase):
        print(f"Model validation time: {total_time} seconds")
    return (listImages[WSI_path][0], status)

if __name__ == '__main__':
    result_list = []
    accuracy_percent = []
    overlap_values = []
    roi_percent = []
    dsc_percent = []
    gleason_grade0_percent = []
    gleason_grade3_percent = []
    gleason_grade4_percent = []
    gleason_grade5_percent = []
    #List of images generated by test.py (WSI and XML paths)
    if args.images_file is None:
        parser.error("missing --images-file with the list of images to validate (generated by test.py)")
    with open(args.images_file, 'r', encoding='utf-8') as f:
        images = json.load(f)
    if not images:
        parser.error(f"the list of images in {args.images_file} is empty")
    dataset = AVAILABLE_DATASETS[args.dataset]
    dataset_name = dataset['name']
    image_map = {image_entry['wsi']: () for image_entry in images}
    #List with the percentage of correctly classified patches in the area with correct Gleason grade classification
    gleason_patches_percent=[]
    NumImages = 0
    probability_threshold = 0.0
    #Extraction geometry resolved once from the first image of the dataset
    extractLevel, patchSize, footprintSize, strideSize = resolve_extraction(openslide.OpenSlide((next(iter(image_map)))))
    #Results folder for each model/dataset/geometry combination
    results_folder_name = f"{GLEASON_MODEL}_{dataset_name}_{patchSize}_{strideSize}_{footprintSize}"
    path_output = os.path.join(os.getcwd(), "EXT", "resultados_analisis", results_folder_name)
    os.makedirs(os.path.join(path_output, "img_analized"), exist_ok=True)
    imagesValue = {}
    #Map of annotated patches with 1% overlap (DSC reference)
    reference_patch_map = None
    imagesWithoutReference = set()
    for i in range (0,110,10):
        imod = 1
        if i != 0:
            imod = i

        validationObj = validationModel(patch_size=footprintSize, stride=strideSize,
        extract_level=extractLevel, dataset_name = dataset_name,annotation_overlap = imod, path_xml = dataset['xml_dir'], path_xml_ext= dataset['xml_dir'],
        path_csv= dataset['csv'], path_output= path_output, devPhase = devPhase, data_source_path=dataset['wsi_dir'], data_source_path_ext = dataset['wsi_dir'],
        image_list=images)
        validationObj.patch_size_read = patchSize
        validationObj.CSVTestListExtractor()
        if reference_patch_map is None:
            reference_patch_map = validationObj.WSI_patch_map
        validationObj.reference_patch_map = reference_patch_map
        NumImages = 0
        #Acumulative variables
        patchesAcepted = 0
        numPatches = 0
        DCSValue = 0
        falseGleason = 0
        numValROI = 0
        numROI = 0
        #Images included in the DSC mean
        numDSCImages = 0
        pachesAcceptedGleason = {0:0,3:0,4:0,5:0}
        totalPatchesTrueGG = {0:0,3:0,4:0,5:0}
        for i, image_path in enumerate(image_map):
            wsi_path = image_path
            NumImages = i+1
            result = generateHeatmapValidation(wsi_path, validationObj,probability_threshold,image_map)
            if(result[1] != 0):
                continue
            print(str(NumImages) + ". Number of patches found: " + str(validationObj.patch_num) + " Number of correctly classified patches: " + str(validationObj.A) +
            " Number of ROIs found: " + str(validationObj.num_ROI) + " Number of ROIs with at least one correctly classified patch: " + str(validationObj.num_val_ROI)+
            " Number of misclassified patches (Gleason): " +  str(validationObj.False_gleason)+" Number of accepted patches with grade 0: " +
            str(validationObj.gradeGleasonAccepted[0])+ " Number of accepted patches with grade 3: " +
            str(validationObj.gradeGleasonAccepted[3]) + " Number of accepted patches with grade 4: " + str(validationObj.gradeGleasonAccepted[4]) +
            " Number of accepted patches with grade 5: " + str(validationObj.gradeGleasonAccepted[5]))


            #N/A if there are no patches to evaluate at this overlap
            validPatchesPercent = validationObj.A/validationObj.patch_num if validationObj.patch_num != 0 else "N/A"
            #DSC: 0 if the image has no patches at this overlap, N/A if it has no annotated tissue
            if imod == 1 and validationObj.patch_num == 0:
                imagesWithoutReference.add(NumImages)
            hasReference = NumImages not in imagesWithoutReference
            imageDSC = validationObj.DSC if hasReference else "N/A"
            validGleasonPatchesPercent = (validationObj.A-validationObj.False_gleason)/validationObj.A if validationObj.A != 0 else 0
            #Evaluated patches of each true grade (row of the confusion matrix)
            trueEvaluatedGleason0 = sum(validationObj.gradeGleasonPredAccept[f"0-{p}"] for p in (0, 3, 4, 5))
            trueEvaluatedGleason3 = sum(validationObj.gradeGleasonPredAccept[f"3-{p}"] for p in (0, 3, 4, 5))
            trueEvaluatedGleason4 = sum(validationObj.gradeGleasonPredAccept[f"4-{p}"] for p in (0, 3, 4, 5))
            trueEvaluatedGleason5 = sum(validationObj.gradeGleasonPredAccept[f"5-{p}"] for p in (0, 3, 4, 5))
            #N/A if the image has no evaluated patches of the grade
            validGleasonGrade0Percent = validationObj.gradeGleasonAccepted[0]/trueEvaluatedGleason0 if trueEvaluatedGleason0 != 0 else "N/A"
            validGleasonGrade3Percent = validationObj.gradeGleasonAccepted[3]/trueEvaluatedGleason3 if trueEvaluatedGleason3 != 0 else "N/A"
            validGleasonGrade4Percent = validationObj.gradeGleasonAccepted[4]/trueEvaluatedGleason4 if trueEvaluatedGleason4 != 0 else "N/A"
            validGleasonGrade5Percent = validationObj.gradeGleasonAccepted[5]/trueEvaluatedGleason5 if trueEvaluatedGleason5 != 0 else "N/A"
            listGradeGleasonPredAccept = list(validationObj.gradeGleasonPredAccept.items())
            if NumImages not in imagesValue:
                imagesValue[NumImages] = [(imod,validationObj.patch_num, validationObj.A, validPatchesPercent,
                validationObj.num_ROI, validationObj.num_val_ROI, (validationObj.A-validationObj.False_gleason), imageDSC,
                validGleasonPatchesPercent, validGleasonGrade0Percent, validGleasonGrade3Percent, validGleasonGrade4Percent, validGleasonGrade5Percent,listGradeGleasonPredAccept)]
            else:
                imagesValue[NumImages].append((imod,validationObj.patch_num, validationObj.A, validPatchesPercent, validationObj.num_ROI,
                validationObj.num_val_ROI, (validationObj.A-validationObj.False_gleason), imageDSC,
                validGleasonPatchesPercent,validGleasonGrade0Percent, validGleasonGrade3Percent, validGleasonGrade4Percent, validGleasonGrade5Percent,listGradeGleasonPredAccept))

            #Acumulative values to calculate final metrics of all images with the same overlap
            patchesAcepted += validationObj.A
            numPatches += validationObj.patch_num
            if hasReference:
                DCSValue += validationObj.DSC
                numDSCImages += 1
            falseGleason += validationObj.False_gleason
            numValROI += validationObj.num_val_ROI
            numROI += validationObj.num_ROI
            #TODO: Add a metric for patches with GG0
            pachesAcceptedGleason[0] += validationObj.gradeGleasonAccepted[0]
            pachesAcceptedGleason[3] += validationObj.gradeGleasonAccepted[3]
            pachesAcceptedGleason[4] += validationObj.gradeGleasonAccepted[4]
            pachesAcceptedGleason[5] += validationObj.gradeGleasonAccepted[5]

            #Acumulate the evaluated patches of each true Gleason grade
            totalPatchesTrueGG[0] += trueEvaluatedGleason0
            totalPatchesTrueGG[3] += trueEvaluatedGleason3
            totalPatchesTrueGG[4] += trueEvaluatedGleason4
            totalPatchesTrueGG[5] += trueEvaluatedGleason5
        #TODO: check that numPatches and numROI are not 0
        accuracy_percent.append(patchesAcepted/numPatches) if numPatches != 0 else accuracy_percent.append(0)
        gleason_patches_percent.append((patchesAcepted - falseGleason)/patchesAcepted) if patchesAcepted != 0 else gleason_patches_percent.append(0)
        #NaN (N/A in the figure) if there are no evaluated patches of the grade
        gleason_grade0_percent.append(pachesAcceptedGleason[0]/totalPatchesTrueGG[0]) if totalPatchesTrueGG[0] !=0 else gleason_grade0_percent.append(float('nan'))
        gleason_grade3_percent.append(pachesAcceptedGleason[3]/totalPatchesTrueGG[3]) if totalPatchesTrueGG[3] !=0 else gleason_grade3_percent.append(float('nan'))
        gleason_grade4_percent.append(pachesAcceptedGleason[4]/totalPatchesTrueGG[4]) if totalPatchesTrueGG[4] !=0 else gleason_grade4_percent.append(float('nan'))
        gleason_grade5_percent.append(pachesAcceptedGleason[5]/totalPatchesTrueGG[5]) if totalPatchesTrueGG[5] !=0 else gleason_grade5_percent.append(float('nan'))


        #Calculation values of all imgages with the same overlap
        validPatchesPercent = patchesAcepted/numPatches if numPatches != 0 else 0
        validGleasonPatchesPercent = (patchesAcepted-falseGleason)/patchesAcepted if patchesAcepted != 0 else 0
        roi_percent.append(numValROI/numROI) if numROI != 0 else roi_percent.append(0)
        validROIpercent = numValROI/numROI if numROI != 0 else 0

        #Mean DSC of the images with annotated tissue (N/A if there are none)
        meanDSC = DCSValue / numDSCImages if numDSCImages != 0 else "N/A"
        dsc_percent.append(meanDSC if numDSCImages != 0 else float('nan'))
        overlap_values.append(imod)


        #List to store resultd in txt file
        result_list.append((imod,numPatches, patchesAcepted, numROI, numValROI,
        (patchesAcepted-falseGleason), meanDSC, validPatchesPercent, validROIpercent,
        validGleasonPatchesPercent))
    results_folder = path_output
    fileImgPath = os.path.join(results_folder, "ficheroImagenes.json")
    filePath = os.path.join(results_folder, "fichero.json")

    IMG_KEYS = [
        "porcentaje solape",
        "número patches",
        "patches acertados",
        "porcentaje acierto",
        "número de ROI",
        "número ROI acertados",
        "número de patches con grado gleason acertado",
        "porcentaje DSC",
        "porcentaje patches con grado gleason acertado",
        "porcentaje grado 0 acertado",
        "porcentaje grado 3 acertado",
        "porcentaje grado 4 acertado",
        "porcentaje grado 5 acertado",
        "Grado Gleason Real - Predicho"
    ]

    RESULT_KEYS = [
        "porcentaje solape",
        "número patches",
        "patches acertados",
        "número de ROI",
        "número ROI acertados",
        "número de patches con grado gleason acertado",
        "valor medio DSC",
        "porcentaje acierto",
        "porcentaje ROI cubiertos",
        "porcentaje patches con grado gleason acertado"
    ]

    #Results of the validated images
    images_json = {}
    for i in sorted(imagesValue):
        images_json[f"Imagen {i}"] = [
            dict(zip(IMG_KEYS, listInfImage))
            for listInfImage in imagesValue[i]
        ]

    with open(fileImgPath, 'w', encoding='utf-8') as f:
        json.dump(images_json, f, ensure_ascii=False, indent=4)

    result_json = [
        dict(zip(RESULT_KEYS, parameters))
        for parameters in result_list
    ]

    with open(filePath, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=4)


    # Configuration of the axes of the confusion matrices
    real_labels = ['0', '3', '4', '5']      # Rows (Y)
    pred_labels = ['0', '3', '4', '5'] # Columns (X)
    n_rows = len(real_labels)
    n_columns = len(pred_labels)

    dicMatrixGlobal = {1: np.zeros((n_rows, n_columns), dtype=float), 10: np.zeros((n_rows, n_columns),
                    dtype=float), 20: np.zeros((n_rows, n_columns), dtype=float), 30: np.zeros((n_rows, n_columns), dtype=float),
                    40: np.zeros((n_rows, n_columns), dtype=float),  50: np.zeros((n_rows, n_columns), dtype=float),
                    60: np.zeros((n_rows, n_columns), dtype=float), 70: np.zeros((n_rows, n_columns), dtype=float),
                    80: np.zeros((n_rows, n_columns), dtype=float), 90: np.zeros((n_rows, n_columns), dtype=float),
                    100: np.zeros((n_rows, n_columns), dtype=float)}
    dicMatrixPercentGlobal = {1: np.zeros((n_rows, n_columns), dtype=float), 10: np.zeros((n_rows, n_columns),
                    dtype=float), 20: np.zeros((n_rows, n_columns), dtype=float), 30: np.zeros((n_rows, n_columns), dtype=float),
                    40: np.zeros((n_rows, n_columns), dtype=float),  50: np.zeros((n_rows, n_columns), dtype=float),
                    60: np.zeros((n_rows, n_columns), dtype=float), 70: np.zeros((n_rows, n_columns), dtype=float),
                    80: np.zeros((n_rows, n_columns), dtype=float), 90: np.zeros((n_rows, n_columns), dtype=float),
                    100: np.zeros((n_rows, n_columns), dtype=float)}

    # Confusion matrix of each image
    for NumImages, data_list in imagesValue.items():
        for dataOverlap in data_list:
            imod = dataOverlap[0]
            confusion_data = dataOverlap[13]

            # Dynamic names for titles and files
            title = f"Image: {NumImages} - Overlap: {imod}%"
            file_name = f"Image_{NumImages}_Solape_{imod}.svg"
            output_folder = os.path.join(results_folder, "Matrices_Generadas")

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                print(f"Folder created: {output_folder}")

            mapping_real = {label: i for i, label in enumerate(real_labels)}
            mapping_pred = {label: i for i, label in enumerate(pred_labels)}

            # Initialize the matrices
            matrix = np.zeros((n_rows, n_columns), dtype=float)
            percent_matrix = np.zeros((n_rows, n_columns), dtype=float)

            # Process the true-predicted counts of the image
            for value in confusion_data:
                real, predicted = value[0].split('-')
                if real in mapping_real and predicted in mapping_pred:
                    dicMatrixGlobal[imod][mapping_real[real], mapping_pred[predicted]] += value[1]
                    matrix[mapping_real[real], mapping_pred[predicted]] = value[1]

            # Calculate the percentages per row
            for i in range(n_rows):
                row_total = matrix[i].sum()
                if row_total != 0:
                    percent_matrix[i] = (matrix[i] / row_total) * 100

            # Visual configuration (Seaborn)
            custom_cmap = LinearSegmentedColormap.from_list("custom_style", ["#f9f9f9", "#2f1b43"])

            plt.figure(figsize=(6, 4.5))
            ax = sns.heatmap(percent_matrix,
                        annot=True,
                        fmt=".2f",
                        cmap=custom_cmap,
                        xticklabels=pred_labels,
                        yticklabels=real_labels,
                        annot_kws={"size": 10, "weight": "bold"},
                        cbar=False)

            #Text of each cell (% and n), N/A if the true grade has no patches
            for t, (row, col) in zip(ax.texts, np.ndindex(matrix.shape)):
                if matrix[row].sum() == 0:
                    t.set_text("N/A")
                else:
                    abs_value = int(matrix[row, col])
                    t.set_text(f"{t.get_text()}%\n(n={abs_value})")

            plt.title(title)
            plt.xlabel('Predicted Grade')
            plt.ylabel('Real Grade')
            plt.tight_layout()

            # Save in SVG format
            full_path = os.path.join(output_folder, file_name)
            plt.savefig(full_path, format="svg", bbox_inches="tight")
            plt.close()

            print(f"Matrix of image {NumImages} successfully saved in: {full_path}")
    #Creation of the global matrix with the data accumulated from all the images

    for i in range(0,101,10):
        imod = i
        if i == 0:
            imod = 1

        global_title = f"Global - Overlap: {imod}%"
        global_file_name = f"Global_Solape_{imod}.svg"
        for i in range(n_rows):
            row_total = dicMatrixGlobal[imod][i].sum()
            if row_total != 0:
                dicMatrixPercentGlobal[imod][i] = (dicMatrixGlobal[imod][i] / row_total) * 100

        custom_cmap = LinearSegmentedColormap.from_list("custom_style", ["#f9f9f9", "#2f1b43"])

        plt.figure(figsize=(6, 4.5))
        ax = sns.heatmap(dicMatrixPercentGlobal[imod],
                    annot=True,
                    fmt=".2f",
                    cmap=custom_cmap,
                    xticklabels=pred_labels,
                    yticklabels=real_labels,
                    annot_kws={"size": 10, "weight": "bold"},
                    cbar=False)

        #Text of each cell (% and n), N/A if the true grade has no patches
        for t, (row, col) in zip(ax.texts, np.ndindex(dicMatrixGlobal[imod].shape)):
            if dicMatrixGlobal[imod][row].sum() == 0:
                t.set_text("N/A")
            else:
                abs_value = int(dicMatrixGlobal[imod][row, col])
                t.set_text(f"{t.get_text()}%\n(n={abs_value})")

        plt.title(global_title)
        plt.xlabel('Predicted Grade')
        plt.ylabel('Real Grade')
        plt.tight_layout()

        # Save in SVG format
        full_path = os.path.join(output_folder, global_file_name)
        plt.savefig(full_path, format="svg", bbox_inches="tight")
        plt.close()

        print(f"Global matrix successfully saved in: {full_path}")


    #Figure with the general values without taking the Gleason grades into account
    plt.figure(figsize=(8, 6))
    plt.plot(overlap_values, accuracy_percent, marker='o', linestyle='-', color='b', label='Accuracy')
    plt.plot(overlap_values, roi_percent, marker='o', linestyle='-', color='r', label='ROI covered')
    plt.plot(overlap_values, gleason_patches_percent, marker='o', linestyle='-', color='g', label='Patches with accurate Gleason score')
    plt.plot(overlap_values, dsc_percent, marker='o', linestyle='-', color='y', label='DSC')
    plt.xlabel('Percentage of overlap (%)', fontsize=12)
    plt.ylabel('Accuracy Rate (%)', fontsize=12)
    plt.title('Relationship between overlap and metrics', fontsize=14)

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(os.path.join(results_folder, 'grafica_solape_acierto.png'), bbox_inches='tight')

    #Second figure with the accuracy values of each Gleason grade with the global information
    plt.figure(figsize=(8, 6))
    grade_series = ((0, gleason_grade0_percent, 'goldenrod'), (3, gleason_grade3_percent, 'b'),
                     (4, gleason_grade4_percent, 'r'), (5, gleason_grade5_percent, 'g'))
    for grade, values, color in grade_series:
        #Overlaps without patches of the grade are marked as N/A in the legend
        no_data = np.isnan(values)
        label = f'Patches with accurate Gleason {grade}'
        if no_data.all():
            label += ' (N/A)'
        elif no_data.any():
            na_overlaps = ', '.join(f'{overlap}%' for overlap in np.array(overlap_values)[no_data])
            label += f' (N/A: {na_overlaps})'
        plt.plot(overlap_values, values, marker='o', linestyle='-', color=color, label=label)
    plt.xlabel('Percentage of overlap (%)', fontsize=12)
    plt.ylabel('Accuracy Rate (%)', fontsize=12)
    plt.title('Relationship between overlap and metrics across Gleason grades', fontsize=14)

    # Add grid and legend
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.savefig(os.path.join(results_folder, 'grafica_solape_acierto_grados.png'), bbox_inches='tight')

    #Fixed cohort figure (images with patches at every overlap)
    plot_fixed_cohort_accuracy(imagesValue, "{} - {}".format(GLEASON_MODEL, dataset_name),
                               os.path.join(results_folder, 'grafica_cohorte_fija.png'))
