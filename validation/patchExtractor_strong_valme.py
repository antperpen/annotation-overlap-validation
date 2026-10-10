## @package patchExtractor
#  Patch extractor module using pathologist's annotation. this module uses whole
#  slide images from different folders and extracts the patches selected by the pathologits
#
#  Scripted by Juan Pedro Domínguez Morales, Lourdes Duran Lopez and Antonio M. Pérez Peña.

import os
import numpy as np
import openslide
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon, box
from shapely.validation import explain_validity
import csv
import sys
import cv2

import pickle

import tiles
import filter
from multiprocessing import Pool
import time

from config import *




def findTruePatchesWorker(id_p, numProcesses, geo_json_list, patch_size, WSI_info, probability_threshold):
    #patch_size is only used in the result key, the matching is done by the origin coordinate
    dicPatchesAcceptedRAW = {():()}
    ROI_patches_WSI_optimized = []
    #We iterate over all the regions of interest marked by the pathologist
    for roi in WSI_info["covered_ROI"]:
        #roi[0] --> list of patches (x,y) that are in the ROI, roi[1] --> Gleason grade of the ROI, roi[2] --> id of the ROI
        roi_dict = {coord: perc for coord, perc in roi[0]}
        ROI_patches_WSI_optimized.append((roi_dict, roi[1], roi[2]))
    #For each list of patches detected by the network
    for geoJson_gle in geo_json_list:

        total_features = geoJson_gle[0]["features"]
        total_elements = len(total_features)

        chunk_size = total_elements // numProcesses
        init_element = id_p * chunk_size
        end_element = total_elements if id_p == (numProcesses - 1) else init_element + chunk_size
        if DEV_MODE_VER:
            print(f"Process {id_p} processing elements from {init_element} to {end_element} out of a total of {total_elements} + \n")
        # We iterate by index so as not to duplicate memory by slicing the list
        for idx in range(init_element, end_element):
            feature = total_features[idx]

            if(feature["properties"]["probability"] > probability_threshold):
                x0 = feature["geometry"]["coordinates"][0][0]
                #We change the order of the coordinates by the order of the coordinates in the image (x,y) --> (y,x)
                coord_x, coord_y = int(x0[0]), int(x0[1])


                for ROI_patches in ROI_patches_WSI_optimized:
                    variationGleasonValue = abs(int(ROI_patches[1]) - geoJson_gle[1])
                    #The key includes the ROI id so that a patch in several ROIs has one entry per ROI
                    check_tuple = (int(coord_x), int(coord_y), int(coord_x) + patch_size, int(coord_y) + patch_size, geoJson_gle[1], variationGleasonValue, ROI_patches[2])

                    if (coord_x, coord_y) in ROI_patches[0]:
                        dicPatchesAcceptedRAW[check_tuple] = (geoJson_gle[1], int(ROI_patches[1]), ROI_patches[2],ROI_patches[0][(coord_x, coord_y)])

    return dicPatchesAcceptedRAW
class validationModel:

    def __init__(self, patch_size = 800, dataset_name = "x", annotation_overlap = 0, path_xml = os.getcwd() + os.sep+"XML",path_xml_ext = os.getcwd() + os.sep+"XML_ext",  path_csv = os.getcwd() + os.sep+"CSV"
                ,path_geojson = os.getcwd() + os.sep+"geoJson", path_output= os.getcwd() + os.sep+"output", devPhase = False, data_source_path="./EXT", data_source_path_ext="./EXT", stride = 0, extract_level = 0, roi_coverage_threshold = 0.3, image_list = None):
        self.patch_size = patch_size
        #List of images to validate {'wsi': WSI path, 'xml': XML path}, used instead of path_csv
        self.image_list = image_list
        #Minimum fraction of coherent cells to consider a ROI covered
        if not 0 < roi_coverage_threshold <= 1:
            raise ValueError("The ROI coverage threshold (roi_coverage_threshold) must be in (0, 1]")
        self.roi_coverage_threshold = roi_coverage_threshold
        #Map of annotated patches with 1% overlap, used as DSC reference
        self.reference_patch_map = None
        #Scanning stride of the heatmap (grid of annotated patches), patch_size if it is not given
        if stride and (stride < 0 or stride > patch_size):
            raise ValueError("The scanning stride (stride) must be between 1 and the patch size")
        self.stride = stride if stride else patch_size
        # Layer of the pyramid from which the patches are read; it is only used when generating the heatmap
        self.extract_level = extract_level
        self.dataset_name = dataset_name
        self.annotation_overlap = annotation_overlap

        if not os.path.exists(path_csv):
            print("CSV path not found")
            sys.exit()
        self.path_csv = path_csv

        if not os.path.exists(path_xml):
            print("XML path not found")
            sys.exit()
        self.path_xml = path_xml

        if not os.path.exists(path_xml_ext):
            print("XML path not found")
            sys.exit()
        self.path_xml_ext = path_xml_ext

        if not os.path.exists(path_geojson):
            os.makedirs(path_geojson)
        self.path_geojson = path_geojson

        if not os.path.exists(path_output):
            os.makedirs(path_output)
        self.path_output = path_output

        if not os.path.exists(data_source_path):
            print("Data path not found")
            sys.exit()
        self.data_source_path = data_source_path
        if not os.path.exists(data_source_path_ext):
            print("Data path extended not found")
            sys.exit()
        self.data_source_path_ext = data_source_path_ext
        #{wsi_name: {geason_value: "", patches: [(x,y,value), (x,y,value), ...]}}
        self.WSI_patch_map = {}
        self.WSI_RAW_ROI = {}

        self.FN = 0
        self.A = 0
        self.num_ROI = 0
        self.num_val_ROI = 0
        self.False_gleason = 0
        self.patch_num = 0
        self.DSC = 0
        self.trueGradeGleason3 = 0
        self.trueGradeGleason4 = 0
        self.trueGradeGleason5 = 0
        self.devPhase = devPhase


    def areaSplitToPatchSize(self,roi_polygon,wsi_image):
            patch_list = []
            #Patches starting up to patch_size - stride before the ROI also contain annotated tissue
            x0 = max(0, roi_polygon.bounds[0] - (self.patch_size - self.stride))
            y0 = max(0, roi_polygon.bounds[1] - (self.patch_size - self.stride))
            x0_mod = x0
            y0_mod = y0
            patch_inside_bb = 0
            #Check if the polygon is valid, if not try to fix it
            if roi_polygon.is_valid == False:
                explanation = explain_validity(roi_polygon)
                print("Error ROI: " + explanation)
                geom_fixed = roi_polygon.buffer(0)
                if geom_fixed.is_valid == True:
                    print("The geometry has been fixed")
                    roi_polygon = geom_fixed
                elif geom_fixed.area != roi_polygon.area:
                    print("The geometry has been fixed but the area has changed, using make_valid")
                else:
                    raise ValueError("The geometry could not be corrected.")
            #Pending: adapt the extreme cases so that they do not modify i or j, but the loop limits
            #The grid advances with self.stride and each patch covers self.patch_size
            if(int(y0) % self.stride != 0):
                y0_mod = ((int)(y0/self.stride))*self.stride

            for j in range(int(y0_mod), int(roi_polygon.bounds[3]), self.stride):
                #We calulate the initial x point
                if(int(x0) % self.stride != 0):
                    x0_mod = ((int)(x0/self.stride))*self.stride

                #iteration through the image rows (y coordenate)
                for i in range(int(x0_mod), int(roi_polygon.bounds[2]), self.stride):
                    #If the patch area inside the roi is bigger than annotation overlap threshold

                    patchEvaluated = box(i, j, i + self.patch_size, j + self.patch_size)
                    interserctionArea = roi_polygon.intersection(patchEvaluated).area
                    PorcentageIntersection = (interserctionArea * 100) / (self.patch_size**2)
                    if PorcentageIntersection >= self.annotation_overlap and self.tissuePatchCheck(i,j, 0, wsi_image) and (i,j) not in patch_list:
                        patch_list.append(((i,j),PorcentageIntersection))
                        patch_inside_bb += 1
            return (patch_inside_bb, patch_list)

    #Extracts the bounding box that covers the whole area annotated by the pathologist and the number of patches needed to cover all the areas of the image
    def xmlPatchExtractor(self,wsi_path,xml_path, WSI_test_name):
        if os.path.isfile(xml_path):
            print("Annotations found for image", WSI_test_name)
            num_Patches = 0
            #List with the set of patches selected by the pathologist
            bb_ROI = []
            ROI_area = []
            ROI_covered_area = []
            idROI = 0
            dictPatches = {0:0,3:0,4:0,5:0}
            #Open image WSi to extract patches and check if this have tissue (we assume .tif)
            if os.path.isfile(wsi_path):
                imageWSI = openslide.OpenSlide(wsi_path)
            else:
                raise RuntimeError("WSI not found - Extract data")
            xml_root = ET.parse(xml_path).getroot()
            xml_nodes = xml_root.getchildren()

            for node in xml_nodes:
                if node.tag == 'Annotation':
                    roi_polygon = Polygon()
                    if(len(node[0]) == 0):
                        continue
                    for region in node[0][0]:
                        if region.tag == 'Text':
                            gleason_pattern = region.attrib['Value']
                            if gleason_pattern != '0' and gleason_pattern != '3' and gleason_pattern != '4' and gleason_pattern != '5':
                                continue
                        elif region.tag == 'Vertices':
                            if gleason_pattern != '0' and gleason_pattern != '3' and gleason_pattern != '4' and gleason_pattern != '5':
                                continue
                            if node[0][0].attrib['Type'] == 'rtPolyline' or node[0][0].attrib['Type'] == 'rtNegativePen':
                                roi_polygon = Polygon([[int(vertex.attrib['X']), int(vertex.attrib['Y'])] for vertex in region])
                            elif node[0][0].attrib['Type'] == 'rtRectangle' or node[0][0].attrib['Type'] == 'rtEllipse':
                                roi_polygon = Polygon([[int(region[0].attrib['X']), int(region[0].attrib['Y'])],
                                                        [int(region[1].attrib['X']), int(region[0].attrib['Y'])],
                                                        [int(region[1].attrib['X']), int(region[1].attrib['Y'])],
                                                        [int(region[0].attrib['X']), int(region[1].attrib['Y'])],
                                                        [int(region[0].attrib['X']), int(region[0].attrib['Y'])]])
                            if(roi_polygon.area != 0.0):
                                #ROI information added to make the image of area selected by the pathology
                                ROI_area.append((roi_polygon,gleason_pattern))
                                # V2 -> We use this function to count the patches within the bounding box and accumulate the
                                #value of patches in the entire wsi
                                resultSplitArea = self.areaSplitToPatchSize(roi_polygon, imageWSI)
                                num_Patches += resultSplitArea[0]
                                #Covered area by the patches accepted before the overlap threshold
                                ROI_covered_area.append((resultSplitArea[1],gleason_pattern, idROI))

                                #We update the number of patches for each gleason grade
                                dictPatches[int(gleason_pattern)] += resultSplitArea[0]
                                #Increment the ROI identifier
                                idROI += 1
                                #We add the bounding box of polygon with the gleason grade and the number of patches within the ROI to check if this region is represented
                                bb_ROI.append(list((int(roi_polygon.bounds[0]),int(roi_polygon.bounds[1]), int(roi_polygon.bounds[2]), int(roi_polygon.bounds[3]), gleason_pattern)))
            return (bb_ROI, num_Patches,ROI_area, ROI_covered_area,dictPatches)
        else:
            print("Annotations not found for the image: "+ WSI_test_name)
            return []

    def CSVTestListExtractor(self,write_csv = 0):
        status = 0
        #File to save de values of extracted patch from annotation in the XML files
        #The name includes the dataset and the extraction geometry
        patch_map_path = os.path.join(os.path.dirname(self.path_xml_ext), "wsi_patch_map_" + str(self.dataset_name)
            + "_" + str(self.annotation_overlap) + "_patch" + str(self.patch_size) + "_stride" + str(self.stride) + ".pkl")
        if self.image_list is None and not os.path.exists(self.path_csv):
            print("csv file not found")
            status = 1
        else:
            if self.image_list is not None:
                #Each row contains the image name and the paths of its WSI and XML
                WSIs_test = [[os.path.basename(img['wsi']), img['wsi'], img['xml']] for img in self.image_list]
            else:
                with open(self.path_csv, 'r') as csv_file:
                    WSIs_test = list(csv.reader(csv_file))
            dict_temp = dict()
            file_exists = False
            dict_dirty = False

            if os.path.exists(patch_map_path):
                with open(patch_map_path, "rb") as f:
                    dict_temp = pickle.load(f)
                file_exists = True
            img_count = 0
            for WSI_test in WSIs_test:
                if DEV_IMAGE_LIST and DEV_NUM_IMAGES == img_count:
                    break
                img_count += 1
                new_image = False
                WSI_test_name = WSI_test[0]

                #TODO: Handle the case of images that are not added to the dictionary because their XML files are not found ---> they are evaluated several times
                if file_exists:
                    # The keys of the map are stored without extension (see dict_temp[...] below)
                    new_image = os.path.splitext(WSI_test_name)[0] not in dict_temp
                    # A single new image is enough to save the map again
                    dict_dirty = dict_dirty or new_image
                if not file_exists or new_image:
                    if self.image_list is not None:
                        WSI_path, XML_path = WSI_test[1], WSI_test[2]
                    elif not WSI_test_name.lower().endswith(('.tif', '.tiff')):
                        WSI_path = os.path.join(self.data_source_path,WSI_test_name +".tif")
                        XML_path = os.path.join(self.path_xml,WSI_test_name+".xml")
                    else:
                        WSI_path = os.path.join(self.data_source_path_ext,WSI_test_name)
                        XML_path = os.path.join(self.path_xml_ext,os.path.splitext(WSI_test_name)[0]+".xml")
                    WSI_XML_out = os.path.join(os.getcwd(),WSI_test_name + ".xml")
                    #Extract the bb_list
                    wsi_patch_result = self.xmlPatchExtractor(WSI_path, XML_path, WSI_test_name)
                    if(not wsi_patch_result):
                        print("Patch list is empty")
                        status = 1
                    else:
                        if(len(wsi_patch_result[0]) == 0):
                            continue
                        bb_list = wsi_patch_result[0]
                        status = 0
                        gleason_value = bb_list[0][4]

                        #If we select the option to maka a CSV file with bounding box list of ROIs selected by the  pathologist
                        if(write_csv == 1):
                            value_XML = ET.Element("WSI_ROI")
                            value_XML.set("WSI_name", WSI_test_name)
                            ET.SubElement(value_XML, 'annotation_info', {"Value": gleason_value})
                            patches = ET.SubElement(value_XML, 'Annotation_patches')
                            for patch in bb_list:
                                ET.SubElement(patches, 'Patch', {"x": patch[0], "y":patch[1]})
                            ET.indent(value_XML)
                            et = ET.ElementTree(value_XML)
                            et.write(WSI_XML_out, xml_declaration=True)
                        else:
                            dict_temp[os.path.splitext(WSI_test_name)[0]] = {"bounding_box_list": bb_list, "patch_number" : wsi_patch_result[1], "raw_ROI": wsi_patch_result[2], "covered_ROI": wsi_patch_result[3], "num_patches_Gleason": wsi_patch_result[4]}
                            #We fill in the dictionary with the list of bounding box of pathologist selection for each image.
            if img_count == 0:
                print("CSV file is empty")
                status = 1
            elif not file_exists or dict_dirty:
                with open(patch_map_path, "wb") as f:
                    pickle.dump(dict_temp, f)
            self.WSI_patch_map = dict_temp
            return status


    #Function to extract the image selected mark the ROI selected by the pathology and the patch list generated with the network (function to extrapolate the points that define a region of interest from one layer to the next)
    def generateImgROI(self, listExtrapolatedPoints, SourceImage, imgFile):
        alpha = 0.4
        SourceImageCop = SourceImage.copy()
        for ROIs in listExtrapolatedPoints[0]:
            ROIpoints = ROIs[0].reshape((-1, 1, 2)).astype(np.int32)
            if ROIs[1] == "5" and DEV_MODE_ALL_IMG_GG != 0:
                cv2.polylines(SourceImageCop, [ROIpoints], isClosed=True, color=(0, 255, 255), thickness=1)
            else:
                cv2.polylines(SourceImageCop, [ROIpoints], isClosed=True, color=(0, 0, 255), thickness=1)
            #Draw bounding box with the ROI points
            x, y, w, h = cv2.boundingRect(ROIpoints)
            cv2.rectangle(SourceImageCop, (x, y), (x + w, y + h), (0, 255, 0), 1)

        overlay = SourceImage.copy()
        patchesDetected = SourceImage.copy()

        for patchPoints in listExtrapolatedPoints[1]:
            #Color for the patch depending on the diference of the Gleason value
            if(patchPoints[1]==0):
                color_patch = (0, 255, 0)
            elif(patchPoints[1]==1):
                color_patch = (0, 144, 255)
            else:
                color_patch = (0, 0, 255)

            cv2.rectangle(overlay, (patchPoints[0][0], patchPoints[0][1]), (patchPoints[0][2], patchPoints[0][3]),
            color=color_patch, thickness=-1)
            #Draw patches detected without gleason value error
            cv2.rectangle(patchesDetected, (patchPoints[0][0], patchPoints[0][1]), (patchPoints[0][2], patchPoints[0][3]),
            color=(255,0,0), thickness=1)

        cv2.addWeighted(overlay, alpha, SourceImageCop, 1 - alpha, 0, SourceImageCop)
        cv2.imwrite(self.path_output+ os.sep +"img_analized"+ os.sep +imgFile+"_analyzed.png",SourceImageCop)
        cv2.imwrite(self.path_output+ os.sep +"img_analized"+ os.sep + imgFile+"_analyzed_ng.png",patchesDetected)
        return 0


    def extrapolationCoords(self, listAcceptedPatches,listROI, sourceLayer, destLayer, dev_img = 0):
        scalingFactor = (float)(2**(abs(destLayer-sourceLayer)))
        ROIPointsList = []
        patchPointsList = []
        #If the source layer is a lower layer, the resulting coordinates are obtained by dividing the coordinate by the rescaling factor
        if(destLayer > sourceLayer):
            scalingFactor= 1/scalingFactor
        for polygonROI in listROI:
            poligonROIPoints = [(x*scalingFactor, y*scalingFactor) for x, y in polygonROI[0].exterior.coords]
            ROIPoints = np.array(poligonROIPoints, np.int32)
            ROIPointsList.append((ROIPoints,polygonROI[1]))
        if(dev_img== 1):
            for patch in listAcceptedPatches["features"]:
                x,y=patch["geometry"]["coordinates"][0][0]
                PatchPoints = np.array([int(coordPatchX)*scalingFactor for coordPatchX in (x,y,x+self.stride,y+self.stride)], np.int32)
                patchPointsList.append((PatchPoints,0))
        elif(dev_img==2):
            for ROIs in listAcceptedPatches:
                for patch in ROIs[0]:
                    x,y=patch[0]
                    PatchPoints = np.array([int(coordPatchX)*scalingFactor for coordPatchX in (x,y,x+self.stride,y+self.stride)], np.int32)
                    patchPointsList.append((PatchPoints,0))

        else:

            for patch in listAcceptedPatches:
                PatchPoints = np.array([int(coordPatchX)*scalingFactor for coordPatchX in patch[0]], np.int32)
                patchPointsList.append((PatchPoints,patch[2]))
        return (ROIPointsList, patchPointsList)

    def DSCValueOpt(self,dim_img, ROIpoints, patchesPoints):
        # Initialize the boolean mask
        ROIMask = np.zeros((dim_img[0], dim_img[1]), dtype=bool)

        # 2. OPTIMIZATION: Fill the ROI mask using NumPy slicing
        for roi in ROIpoints:
            for patch_comp in roi[0]:
                patch = patch_comp[0]
                i = int(patch[0])
                j = int(patch[1])
                # Replaces the two inner loops with a block assignment
                ROIMask[i:i + self.stride, j:j + self.stride] = True

        # The total number of unique ROI pixels is simply the count of 'True' values in the mask
        numPixelRoiLocal = np.sum(ROIMask)

        # 3. OPTIMIZATION: predicted patches are accumulated in a mask (union of pixels)
        patchesMask = np.zeros((dim_img[0], dim_img[1]), dtype=bool)
        for patchRAW in patchesPoints:
            patch = patchRAW[0]
            i_start, j_start, i_end, j_end = int(patch[0]), int(patch[1]), int(patch[2]), int(patch[3])
            patchesMask[i_start:i_end, j_start:j_end] = True

        numPixelPatchesLocal = np.sum(patchesMask)
        numOverlapLocal = np.sum(np.logical_and(ROIMask, patchesMask))

        # Final calculation of the Dice Similarity Coefficient (DSC)
        if (numPixelRoiLocal + numPixelPatchesLocal) == 0:
            return 0

        return (2. * numOverlapLocal) / (numPixelRoiLocal + numPixelPatchesLocal)

    #TODO: Implement the IoU function

    def tissuePatchCheck(self, cordi,cordj, Numlayer, wsi_image):
        whiteFilter = 230
        blackFilter = 100
        res = False
        # Read patch
        tile = wsi_image.read_region((cordi, cordj), Numlayer, (self.patch_size, self.patch_size))

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
                res = True
        return res



    #Modify so that the GeoJSON input data comes directly from the inference code
    def validate_model(self, WSI_name, geo_json_list, probability_threshold, write_csv, dim_img, sourceLayer=0, numTopLayer = 5, imgSource = 0, numProcesses = 8, double_dim = 0):
        listAcceptedPatches= set()
        status = 0
        variationGleasonValue = 0
        patch_overlap_mode = 2
        self.gradeGleasonAccepted = {0:0, 3:0, 4:0, 5:0}
        #Dictionary to count the number of patches accepted for each combination of true-predicted Gleason grades
        self.gradeGleasonPredAccept={"0-0":0,"0-3":0,"0-4":0,"0-5":0,"3-0":0,"3-3":0,"3-4":0,"3-5":0,"4-0":0,"4-3":0,"4-4":0,"4-5":0,"5-0":0,"5-3":0,"5-4":0,"5-5":0}
        #If the map is empty
        if(not self.WSI_patch_map):
            print("Validate model object is not initialized")
            status = self.CSVTestListExtractor(self, write_csv)

        #  x0 -----> y0
        #  |         |
        #  |         |
        #  |         |
        #  X1 <----- Y1
        if(status == 0):
            try:
                WSI_info = self.WSI_patch_map[os.path.splitext(WSI_name)[0]]
            except KeyError:
                print("Image name not found in the WSI dict")
                status = 2

            if(status == 0):
                start_time = time.time()
                if DEV_MODE:
                    print("Debug mode active: running in the main process...")
                    total_results = [
                        findTruePatchesWorker(0, 1, geo_json_list, self.stride, WSI_info, probability_threshold)
                    ]
                else:
                    with Pool(processes=numProcesses) as pool:
                        tasks = []

                        # We launch the asynchronous tasks passing them the parameters
                        for i in range(numProcesses):
                            task = pool.apply_async(
                                findTruePatchesWorker,
                                args=(i, numProcesses, geo_json_list, self.stride, WSI_info, probability_threshold)
                            )
                            tasks.append(task)

                        # .get() waits for them to finish and collects the returned values
                        total_results = [task.get() for task in tasks]

                    end_time = time.time()
                    total_time = end_time - start_time
                    if DEV_MODE_VER:
                        print("================Total parallel processing time===========================: ", total_time, " seconds")

                globalDict = {k: v for d in total_results for k, v in d.items()}
                reviewed_keys = set()
                duplicated_keys = set()
                duplicatedKeysCounter = 0
                deletedKeysCounter = 0
                #Delete the duplicated key from the list of accepted patches
                if patch_overlap_mode == 0:
                    for key,value in globalDict.items():
                        if key and value and key[:4] not in reviewed_keys:
                            reviewed_keys.add(key[:4])
                            #value[1] real GG value y value[0] predicted GG value
                            variationGleasonValue = key[5]
                            listAcceptedPatches.add((key[:4],value[0],variationGleasonValue, value[3], value[2], value[1])) # Adds the key and its value to the final list
                            self.gradeGleasonPredAccept[str(value[1]) + "-" + str(value[0])] += 1
                            if(	variationGleasonValue == 0):
                                #List to count the number of patches accepted for each Gleason grade
                                self.gradeGleasonAccepted[int(value[1])]+= 1

                        elif key and key[:4] in reviewed_keys:
                            duplicated_keys.add(key)
                            if DEV_MODE_VER:
                                print(f"Duplicated key found: {key}")
                            duplicatedKeysCounter += 1
                    #Delete the duplicated key from the list of accepted patches
                    for key in duplicated_keys:
                        searched_prefix = key[:4]
                        items_to_remove = [item for item in listAcceptedPatches if item[0] == searched_prefix]
                        for item in items_to_remove:
                            listAcceptedPatches.remove(item)
                            deletedKeysCounter += 1
                #We add all patches regardless of if the patch is duplicated
                elif patch_overlap_mode == 1:
                    for key,value in globalDict.items():
                        if key and value:
                            if key[:4] not in reviewed_keys:
                                reviewed_keys.add(key[:4])
                            else:
                                duplicated_keys.add(key)
                                if DEV_MODE_VER:
                                    print(f"Duplicated key found: {key}")

                            #value[1] real GG value y value[0] predicted GG value
                            variationGleasonValue = key[5]
                            listAcceptedPatches.add((key[:4],value[0],variationGleasonValue, value[3], value[2], value[1])) # Adds the key and its value to the final list
                            self.gradeGleasonPredAccept[str(value[1]) + "-" + str(value[0])] += 1
                            if(	variationGleasonValue == 0):
                                #List to count the number of patches accepted for each Gleason grade
                                self.gradeGleasonAccepted[int(value[1])]+= 1

                #We detect the duplicated patches and evaluate the overlaping percentage between the patch and ROI region
                else:
                    for key,value in globalDict.items():
                        #If the current patch has not been evaluated previously we accept the patch and we add this patch on a list
                        if key and value and key[:4] not in reviewed_keys:
                            reviewed_keys.add(key[:4])
                            #value[1] real GG value y value[0] predicted GG value
                            variationGleasonValue = key[5]
                            listAcceptedPatches.add((key[:4],value[0],variationGleasonValue, value[3], value[2], value[1])) # Adds the key and its value to the final list
                            self.gradeGleasonPredAccept[str(value[1]) + "-" + str(value[0])] += 1
                            if(	variationGleasonValue == 0):
                                #List to count the number of patches accepted for each Gleason grade
                                self.gradeGleasonAccepted[int(value[1])]+= 1
                        #We don´t add duplicated patches to list of patches (The first patch may have been added)
                        elif key and key[:4] in reviewed_keys:
                            duplicated_keys.add(key)
                            if DEV_MODE_VER:
                                print(f"Duplicated key found: {key}")
                            duplicatedKeysCounter += 1
                    #When an element is removed from listAcceptedPatches and we move on to the next
                    #duplicated key, it may have the same key and not be found because it was removed before.
                    #A dictionary is used where each key is the coordinates and the values are the duplicated patches at those coordinates
                    duplicated_keys_dict = dict()
                    #The patches with the same coordinates as the duplicated patches (added at the beginning of the evaluation) are searched, added to the dictionary of duplicated patches and removed from the list of accepted patches
                    for key in duplicated_keys:
                        searched_prefix = key[:4]
                        if searched_prefix in duplicated_keys_dict:
                            duplicated_keys_dict[searched_prefix].append((searched_prefix,key[4],key[5],key[6]))
                        else:
                            items_to_remove = [item for item in listAcceptedPatches if item[0] == searched_prefix]
                            # item = (coordinates, predicted_GG, variationGleasonValue, overlap, ROI id, real_GG)
                            duplicated_items = [(item[0], item[1], item[2], item[4]) for item in items_to_remove]
                            modified_key = (searched_prefix,key[4],key[5],key[6])
                            duplicated_items.append(modified_key)
                            duplicated_keys_dict[searched_prefix] = list(duplicated_items)
                            for item in items_to_remove:
                                listAcceptedPatches.remove(item)
                                deletedKeysCounter += 1

                                # The key is rebuilt to extract the real value from the original dictionary
                                # item = (coordinates, predicted_GG, variationGleasonValue, overlap, ROI id, real_GG)
                                original_key = (*item[0], item[1], item[2], item[4])
                                original_value = globalDict[original_key]

                                removed_true_gg = original_value[1]
                                removed_pred_gg = original_value[0]
                                removed_variation = item[2]

                                # 1. Subtract from the confusion matrix
                                self.gradeGleasonPredAccept[str(removed_true_gg) + "-" + str(removed_pred_gg)] -= 1

                                # 2. Subtract from the grade accuracy counter
                                if removed_variation == 0:
                                    self.gradeGleasonAccepted[int(removed_true_gg)] -= 1

                    #Evaluation of the overlap percentage with the annotated tissue. All the duplicated patches are traversed and their overlap percentage is analysed to find the best one
                    for key_set in duplicated_keys_dict.values():
                        patch_Max_Perc_Over = (0,0,0,0,0)

                        for key in key_set:
                            value = globalDict[(*key[0],key[1],key[2],key[3])]
                            if value[3] > patch_Max_Perc_Over[3]:
                                #We find the patch with highest ovelap value ((x,y,x+patch_size, y+patch_size) key[0], predicted_GG value[0], variationGleasonValue key[2], overlap percentage value[3], ROI of patch(value[2]), real_gg(value[1]))
                                patch_Max_Perc_Over = (key[0],value[0],key[2],value[3], value[2], value[1])
                        listAcceptedPatches.add(patch_Max_Perc_Over)
                        deletedKeysCounter -= 1

                        #We adjust the values based on the added patches
                        pred_gg_winner = patch_Max_Perc_Over[1]
                        true_gg_winner = patch_Max_Perc_Over[5]

                        # The confusion matrix is updated with the patch that actually won
                        self.gradeGleasonPredAccept[str(true_gg_winner) + "-" + str(pred_gg_winner)] += 1

                        if( patch_Max_Perc_Over[2] == 0):
                            #List to count the number of patches accepted for each Gleason grade
                            self.gradeGleasonAccepted[int(true_gg_winner)]+= 1

                #A ROI is covered if the fraction of its cells with coherent prediction reaches roi_coverage_threshold
                cells_per_ROI = {roi[2]: len(roi[0]) for roi in WSI_info["covered_ROI"]}
                coherent_patches_per_ROI = {}
                for key, value in globalDict.items():
                    #value[0] predicted GG, value[1] real GG of the ROI, value[2] id of the ROI
                    if key and value and (value[1] == 0) == (value[0] == 0):
                        coherent_patches_per_ROI[value[2]] = coherent_patches_per_ROI.get(value[2], 0) + 1
                listIdROI = [id_ROI for id_ROI, num_coherent in coherent_patches_per_ROI.items()
                             if num_coherent / cells_per_ROI[id_ROI] >= self.roi_coverage_threshold]

                if DEV_MODE_VER:
                    print(f"Total number of duplicated keys found: {duplicatedKeysCounter}")
                    print(f"Total number of deleted keys: {deletedKeysCounter}")

                if(self.devPhase):
                    listROI = WSI_info["raw_ROI"]
                    if(double_dim != 0):
                        numTopLayer = numTopLayer*2
                    listExtrapolatedPoints = self.extrapolationCoords(listAcceptedPatches,listROI,sourceLayer,numTopLayer, 0)
                    self.generateImgROI( listExtrapolatedPoints, imgSource,WSI_name+"_"+str(self.annotation_overlap))
                    #Paint the patches with GG5
                    if DEV_MODE_ALL_IMG_GG ==1:
                        #We use the patches with GG5 detected by the model (geo_json_list[3][0]) without taking the overlap percentage into account
                        listExtrapolatedPoints = self.extrapolationCoords(geo_json_list[3][0],listROI,sourceLayer,numTopLayer,DEV_MODE_ALL_IMG_GG)
                        self.generateImgROI( listExtrapolatedPoints, imgSource,WSI_name+"_"+str(self.annotation_overlap)+"GG_ALL")
                    if DEV_MODE_ALL_IMG_GG == 2:
                        listExtrapolatedPoints = self.extrapolationCoords(WSI_info["covered_ROI"],listROI,sourceLayer,numTopLayer,DEV_MODE_ALL_IMG_GG)
                        self.generateImgROI( listExtrapolatedPoints, imgSource,WSI_name+"_"+str(self.annotation_overlap)+"ROI_PATCH_ALL")

                #Accepted patches with coherent prediction (benign/malignant)
                # item = (coordinates, predicted_GG, variationGleasonValue, overlap, ROI id, real_GG)
                acceptedPatchesCoherent = [item for item in listAcceptedPatches if (int(item[5]) == 0) == (int(item[1]) == 0)]
                self.A = len(acceptedPatchesCoherent)
                # Accepted patches whose Gleason grade does not exactly match the true one
                self.False_gleason = sum(1 for item in acceptedPatchesCoherent if item[2] != 0)
                #False negatives are equal to paches marked as malignant and undetected.
                self.FN = WSI_info["patch_number"] - self.A
                #Number of annotated cells according to the duplicate handling mode
                cell_count = {}
                for roi in WSI_info["covered_ROI"]:
                    for coord, _ in roi[0]:
                        cell_count[coord] = cell_count.get(coord, 0) + 1
                if patch_overlap_mode == 0:
                    # The cells shared by several ROIs are discarded as ambiguous
                    self.patch_num = sum(1 for n in cell_count.values() if n == 1)
                elif patch_overlap_mode == 1:
                    # Each (cell, ROI) pair counts separately
                    self.patch_num = sum(cell_count.values())
                else:
                    # Each cell counts only once even if it belongs to several ROIs
                    self.patch_num = len(cell_count)
                self.num_val_ROI = len(listIdROI)
                self.num_ROI = len(WSI_info["bounding_box_list"])
                #DSC reference: annotated tissue with 1% overlap if it is given
                reference_info = (self.reference_patch_map.get(os.path.splitext(WSI_name)[0], WSI_info)
                                  if self.reference_patch_map else WSI_info)
                listPatchesOfROI = reference_info["covered_ROI"]
                #DSC with the accepted (coherent) patches
                self.DSC = self.DSCValueOpt(dim_img,listPatchesOfROI, acceptedPatchesCoherent)
                self.trueGradeGleason0=WSI_info["num_patches_Gleason"][0]
                self.trueGradeGleason3=WSI_info["num_patches_Gleason"][3]
                self.trueGradeGleason4=WSI_info["num_patches_Gleason"][4]
                self.trueGradeGleason5=WSI_info["num_patches_Gleason"][5]


            else:
                print("WSI not found in the patch list")
        return status
