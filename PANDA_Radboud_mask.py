import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import matplotlib.pyplot as plt
import openslide
import pandas as pd
import pdb
DEV_MODE = False
DEV_MODE_WSI = False


def mask_to_pathology_xml(mask_path, output_xml_path, wsi_name="imagen_original.tif", source_layer=5, target_layer=0):
    """
    Extracts the contours of a tissue mask by gray levels (2, 3, 4, 5) and
    saves them in an XML file assigning the corresponding label (0, 3, 4, 5).
    Saves an image of the mask with the detected areas highlighted.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"The mask could not be loaded from: {mask_path}")

    # Mapping dictionary: gray level in the mask -> label in the XML
    gray_level_mapping = {
        2: "0", # Benign
        3: "3", # Gleason 3
        4: "4", # Gleason 4
        5: "5"  # Gleason 5
    }

    # Initialize the XML tree
    root = ET.Element("Annotations", file=wsi_name)
    ET.SubElement(root, "SlideAnnotation", Text="", Voice="")

    grid_map = ET.SubElement(root, "GridMap")
    ET.SubElement(grid_map, "Mag", Factor="1")
    ET.SubElement(grid_map, "Mag", Factor="2")

    # The global Annotation and Regions containers are not created here

    scale_factor = (float)(2**(abs(target_layer-source_layer)))
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    total_regions = 0

    for gray_level, gleason_label in gray_level_mapping.items():
        # Create a binary mask exclusive to the current gray level
        thresh = np.where(mask == gray_level, 255, 0).astype(np.uint8)

        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue

        # Draw the contours found for this gray level (red)
        cv2.drawContours(mask_bgr, contours, -1, (0, 0, 255), 2)

        for contour in contours:
            if len(contour) < 3:
                continue

            # Create an Annotation and Regions block for EACH contour
            annotation_node = ET.SubElement(root, "Annotation", LineColor="65280")
            regions_node = ET.SubElement(annotation_node, "Regions")

            pixel_area = cv2.contourArea(contour) * (scale_factor ** 2)
            region = ET.SubElement(regions_node, "Region", Type="rtPolyline", regSelected="FALSE")

            # Assign the corresponding grade taken from the dictionary
            ET.SubElement(region, "Text", Value=gleason_label)
            ET.SubElement(region, "Voice", Path="")
            ET.SubElement(region, "Area", Value=str(int(pixel_area)))

            micron_area = pixel_area * 0.0625
            ET.SubElement(region, "Area", Microns=f"{micron_area:.2f}")

            vertices_node = ET.SubElement(region, "Vertices")

            for point in contour:
                x_m = int(point[0][0] * scale_factor)
                y_m = int(point[0][1] * scale_factor)
                ET.SubElement(vertices_node, "Vertex", X=str(x_m), Y=str(y_m))

            total_regions += 1

    # Save the view of the mask with the highlighted areas
    highlighted_mask_path = output_xml_path.replace(".xml", "_mask_areas.png")
    cv2.imwrite(highlighted_mask_path, mask_bgr)
    print(f"--> Mask view with highlighted areas saved in: {highlighted_mask_path}")

    xml_string = ET.tostring(root, encoding="utf-8")
    reparsed = minidom.parseString(xml_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"Success! {total_regions} regions in total have been processed in {output_xml_path}")


def draw_patches_from_xml(xml_path, wsi_path, display_layer, scale_factor, output_image_path, patch_size=224):
    """
    Reads a generated XML file, filters the regions by Gleason grade (0, 3, 4, 5)
    and generates 4 independent images drawing the grid of patches over the WSI
    for each grade (allowing patches with at least 1% of overlap).
    """
    try:
        slide = openslide.OpenSlide(wsi_path)
        dim = slide.level_dimensions[display_layer]
        imglayer = slide.read_region((0, 0), display_layer, dim).convert("RGB")
        base_img_np = np.array(imglayer)
    except Exception as e:
        print(f"Error opening the WSI for drawing: {e}")
        return

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error opening the XML file for drawing: {e}")
        return

    total_patch_area = patch_size * patch_size
    gleason_grades = ["0", "3", "4", "5"]

    # Generate an image for each Gleason grade
    for grade in gleason_grades:
        grade_img_np = base_img_np.copy()
        drawn_patches = 0

        for region in root.findall('.//Region'):
            # Get the grade assigned to this region in the <Text Value="..."> tag
            text_node = region.find('Text')
            region_grade = text_node.get('Value') if text_node is not None else None

            # Keep only the regions that match the current grade
            if region_grade != grade:
                continue

            vertices = []
            for vertex in region.findall('.//Vertex'):
                x = float(vertex.get('X'))
                y = float(vertex.get('Y'))
                vertices.append([x, y])

            if len(vertices) < 3:
                continue

            contour = np.array(vertices, dtype=np.int32)
            x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(contour)

            # Aligned to the grid (multiples of patch_size)
            start_x = (x_rect // patch_size) * patch_size
            start_y = (y_rect // patch_size) * patch_size
            end_x = x_rect + w_rect
            end_y = y_rect + h_rect

            # Iterate over the grid of the bounding box
            for px in range(start_x, end_x, patch_size):
                for py in range(start_y, end_y, patch_size):

                    local_contour = contour - [px, py]
                    patch_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
                    cv2.fillPoly(patch_mask, [local_contour], 255)

                    overlap_pixels = cv2.countNonZero(patch_mask)
                    overlap_percentage = (overlap_pixels / total_patch_area) * 100

                    # Overlap threshold >= 1%
                    if overlap_percentage >= 1.0:
                        pt1_x = int(px / scale_factor)
                        pt1_y = int(py / scale_factor)
                        pt2_x = int((px + patch_size) / scale_factor)
                        pt2_y = int((py + patch_size) / scale_factor)

                        # Draw the box of the patch
                        cv2.rectangle(grade_img_np, (pt1_x, pt1_y), (pt2_x, pt2_y), (255, 0, 0), 1)
                        drawn_patches += 1

        # Build the name of the output file with the corresponding grade
        base_path, ext = os.path.splitext(output_image_path)
        grade_output_path = f"{base_path}_grado_{grade}{ext}"

        # Convert RGB to BGR and save the image
        img_bgr = cv2.cvtColor(grade_img_np, cv2.COLOR_RGB2BGR)
        cv2.imwrite(grade_output_path, img_bgr)
        print(f"--> Preview with patches of Grade {grade} ({drawn_patches} patches) saved in: {grade_output_path}")


if __name__ == '__main__':
    csv_file  = os.path.join(os.getcwd(),"Images", "PANDA_test.csv")
    masks_folder_path = os.path.join(os.getcwd(),"masks/PANDA_strong/")
    output_path = os.path.join(os.getcwd(), "output_xml", "XML_mask")

    grade_paths = ["0", "1", "2", "3"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df = pd.read_csv(csv_file, header=None, names=['file_name', 'gleason_grade1','gleason_grade2'])

    # Load train.csv to filter the images by origin (Radboud)
    train_csv_path = os.path.join(os.getcwd(), "train.csv")
    if os.path.exists(train_csv_path):
        df_train = pd.read_csv(train_csv_path)
    else:
        df_train = None
        print("Warning: 'train.csv' was not found. The images will not be filtered by origin (Radboud).")

    # List that stores the records that pass the filter, used to generate PANDA.csv
    panda_records = []

    for index, row in df.iterrows():
        base_name = os.path.basename(str(row["file_name"]))

        if df_train is not None:
            # Remove the extension, if any, to search for the ID in train.csv
            image_id = base_name.split('.')[0]
            metadata_row = df_train.loc[df_train['image_id'] == image_id, 'data_provider']
            if metadata_row.empty or metadata_row.values[0] != 'radboud':
                continue
        # If the image has passed the Radboud filter, it is saved for the output CSV
            panda_records.append([base_name, row["gleason_grade1"]])

        mask_name = base_name + "_mask_use.png"
        full_mask_path = os.path.join(masks_folder_path, mask_name)

        if os.path.exists(full_mask_path):
            WSI_path = base_name
            full_WSI_path = os.path.join(os.getcwd(),os.path.join("train_images/" ,WSI_path))
            name_file = os.path.splitext(mask_name)[0].split(".")[0]

            if os.path.exists(full_WSI_path):
                image = openslide.OpenSlide(full_WSI_path)
                real_openslide_layer = image.level_count - 1
                highestLayer = real_openslide_layer * 2
            else:
                continue

            if DEV_MODE_WSI:
                imglayer = image.read_region((0,0), real_openslide_layer, (image.level_dimensions[real_openslide_layer][0], image.level_dimensions[real_openslide_layer][1]))
                imglayer.save("./Img_topLayer_oring.png")

            name_file = name_file + ".xml"
            name_WSI = mask_name.split("_mask_")[0]
            final_xml_path = os.path.join(output_path, name_file)

            # 1. Generate the XML file and save the highlighted mask
            mask_to_pathology_xml(
                mask_path=full_mask_path,
                output_xml_path=final_xml_path,
                wsi_name=name_WSI,
                source_layer=highestLayer,
                target_layer=0
            )

            # 2. Draw the patches with the 1% threshold
            wsi_scale_factor = (float)(2**(abs(0 - highestLayer)))
            patches_image_path = final_xml_path.replace(".xml", "_patches.png")

            draw_patches_from_xml(
                xml_path=final_xml_path,
                wsi_path=full_WSI_path,
                display_layer=real_openslide_layer,
                scale_factor=wsi_scale_factor,
                output_image_path=patches_image_path,
                patch_size=224
            )

    # Generate the PANDA.csv file
    panda_csv_path = os.path.join(os.getcwd(), "PANDA.csv")
    df_panda = pd.DataFrame(panda_records)
    # It is saved without indices or header names to keep the original format
    df_panda.to_csv(panda_csv_path, index=False, header=False)
    print(f"\nThe file {panda_csv_path} has been successfully generated with {len(panda_records)} records from Radboud.")
