"""
Description: This script generates templates from annotated images for model training.
Author: 
"""

import cv2
import os
import sys

FILE_FOLDER = os.path.dirname(__file__)
LABEL_FOLDER = os.path.join(FILE_FOLDER, "annotation", "obj_train_data")
IMAGE_FOLDER = os.path.join(FILE_FOLDER, "..", "camera_calibration", "calibrated_images")
TEMPLATE_FOLDER = os.path.join(FILE_FOLDER, "templates")

VISUALIZE = False

def get_file_names(folder):
    """Retrieve list of file names with .txt extension from a folder."""
    files = os.listdir(folder)
    return [file for file in files if file.endswith(".txt")]

if __name__ == '__main__':
    # Get list of annotation files
    annotation_files = get_file_names(LABEL_FOLDER)

    for annotation_file in annotation_files:
        # Read input image
        file_name = os.path.splitext(annotation_file)[0] + ".bmp"
        image = cv2.imread(os.path.join(IMAGE_FOLDER, file_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Open annotation file
        annotations_file_path = os.path.join(LABEL_FOLDER, annotation_file)
        labels_file = open(annotations_file_path)

        # Get image dimensions
        height, width = image.shape

        # Process each annotation in the annotation file
        for label_line in labels_file.readlines():
            if len(label_line) != 0:
                class_id, center_x, center_y, width_, height_ = label_line.split()

                # Calculate coordinates and dimensions
                center_x = int(float(center_x) * width)
                center_y = int(float(center_y) * height)
                half_width = int(float(width_) * width / 2)
                half_height = int(float(height_) * height / 2)

                x1 = int(center_x - half_width)
                y1 = int(center_y - half_height)
                x2 = int(center_x + half_width)
                y2 = int(center_y + half_height)

                # Extract template from the image
                template = image[y1:y2, x1:x2]

                # Visualize the template if enabled
                if VISUALIZE:
                    cv2.imshow("template", template)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        sys.exit(1)

                # Save the template image
                output_file_name = os.path.splitext(annotation_file)[0] + ".bmp"
                output_file_path = os.path.join(TEMPLATE_FOLDER, output_file_name)
                cv2.imwrite(output_file_path, template)




