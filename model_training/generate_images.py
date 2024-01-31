
"""
Image Generator Script for Training Data

This script generates synthetic training images for object detection tasks. 
It randomly selects objects from a set of templates, places them on a background image, 
and saves both the image and corresponding bounding box annotations. 
It can be configured to generate training or validation data by setting the TRAIN flag.

Author: 

"""

import os
import re
import cv2
import numpy as np
import multiprocessing as mp
# cfrom scipy import ndimages

# Define constants
INPUT_IMAGE_SIZE = (2448, 2048)
OUTPUT_IMAGE_SIZE = (640, 640)
MAX_NUMBER_OF_OBJECTS = 10 # Maximum number of templates in a single generated image
NUMBER_OF_IMAGES = 10000 # Total number of generated images
NUMBER_OF_PROCESSES = 10
TRAIN = True # True -> Training Images; False -> Validation images.
K = 3 # Number of reoccuring template in a single generated image

# List of object labels (customize according to your dataset)
LABELS = []

# Function to sort alphanumeric strings
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(data, key=alphanum_key)

# Read template files from folder and sort them alphanumerically
files = os.listdir('templates')
for file in files:
    if file.split('.')[1] == 'bmp':
        LABELS.append(file.split('.')[0])
LABELS = sorted_alphanumeric(LABELS)

# ######################################
# # To generate list for yaml file 
# for idx, lbl in enumerate(LABELS):
#     print(str(idx)+': '+str(lbl))
# ######################################

FILE_FOLDER = os.path.dirname(__file__)
TEMPLATE_FOLDER = os.path.join(FILE_FOLDER, "templates")   
TRAIN_IMAGES_FOLDER = os.path.join(FILE_FOLDER, "datasets", "images", "train")
TRAIN_LABELS_FOLDER = os.path.join(FILE_FOLDER, "datasets", "labels", "train")
VAL_IMAGES_FOLDER = os.path.join(FILE_FOLDER, "datasets", "images", "val")
VAL_LABELS_FOLDER = os.path.join(FILE_FOLDER, "datasets", "labels", "val")
TEMPLATE_IMAGE = os.path.join(FILE_FOLDER, "background_image.bmp")

# Function to get template filenames
def GetTemplates():
    files = os.listdir(TEMPLATE_FOLDER)
    fileList = []
    for file in files:
        if file.endswith(".bmp"):
            fileList.append(file)
    return fileList

# Function to load template image
def GetTemplateImage(name):
    image = cv2.imread(os.path.join(TEMPLATE_FOLDER, name))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

# Function to check overlapping gray areas in images
def CheckOverlappingGrayAreas(image, template, bb):
    # Threshold template to create mask
    _, mask = cv2.threshold(template, 225, 255, cv2.THRESH_BINARY)
    mask = mask / 255
    mask = mask.astype('uint8')
    _, mask2 = cv2.threshold(image[bb[1]:bb[3], bb[0]:bb[2]], 225, 255, cv2.THRESH_BINARY)
    mask2 = mask2 / 255
    mask2 = mask2.astype('uint8')
    # Check if gray areas overlap
    overlaps = np.sum(np.multiply(mask, mask2)) > 3
    return overlaps

# Function to check overlapping bounding boxes
def CheckOverlappingBoundingBoxes(bb1, bb2):
    return not ((bb1[0] >= bb2[2]) or (bb1[2] <= bb2[0]) or (bb1[3] <= bb2[1]) or (bb1[1] >= bb2[3]))

# Function to check overlap of bounding boxes
def BoundingBoxesOverlap(bb, boundingBoxes, image, template):
    if len(boundingBoxes) > 0:
        for b in boundingBoxes:
            if CheckOverlappingBoundingBoxes(bb, b):
                if CheckOverlappingGrayAreas(image, template, bb):
                    return True
    return False

# Function to get bounding box for a template
def GetBoundingBox(template, templateName):
    rng = np.random.default_rng()

    h, w = template.shape
    # Define the desired area within the background image
    area_x_min, area_x_max = 400, 2200
    area_y_min, area_y_max = 400, 1700

    x = rng.integers(area_x_min, area_x_max - w)
    y = rng.integers(area_y_min, area_y_max - h)

    return [x, y, x+w, y+h, templateName[:-4]]

def CropTemplate(template):
    top_row = 0
    bottom_row = -1
    left_column = 0
    right_column = -1
    # Remove completely white rows/columns
    while (True):
        if np.any(template[top_row] < 200):
            break
        else:
            top_row += 1
    while (True):
        if np.any(template[bottom_row] < 200):
            break
        else:
            bottom_row -= 1
    while (True):
        if np.any(template[:, left_column] < 200):
            break
        else:
            left_column += 1
    while (True):
        if np.any(template[:, right_column] < 200):
            break
        else:
            right_column -= 1
    if bottom_row == -1 and right_column == -1:
        out_image = template[top_row:, left_column:]
    elif bottom_row == -1:
        out_image = template[top_row:, left_column:right_column]
    elif right_column == -1:
        out_image = template[top_row:bottom_row, left_column:]
    else:
        bottom_row += 1
        right_column += 1
        out_image = template[top_row:bottom_row, left_column:right_column]
    return out_image

def GenerateOutput(name, image, boundingBoxes):
    if TRAIN:
        cv2.imwrite(os.path.join(TRAIN_IMAGES_FOLDER, name + ".bmp"), image)        
        file = open(os.path.join(TRAIN_LABELS_FOLDER, name + ".txt"), "w")
    else:
        cv2.imwrite(os.path.join(VAL_IMAGES_FOLDER, name + ".bmp"), image)
        file = open(os.path.join(VAL_LABELS_FOLDER, name + ".txt"), "w")
    def GenerateLabel(bb):
        x = ((bb[2] + bb[0]) / 2) / INPUT_IMAGE_SIZE[0]
        y = ((bb[3] + bb[1]) / 2) / INPUT_IMAGE_SIZE[1]
        w = (bb[2] - bb[0]) / INPUT_IMAGE_SIZE[0]
        h = (bb[3] - bb[1]) / INPUT_IMAGE_SIZE[1]
        return [str(LABELS.index(bb[4])), str(x), str(y), str(w), str(h)]
    for bb in boundingBoxes:
        label = " ".join(GenerateLabel(bb))
        file.write(label + "\n")
    file.close()

# Function to apply gray areas of template to background image
def apply_gray_areas(image, template, x1, x2, y1, y2):
    temp = np.copy(image[y1:y2, x1:x2])
    ret, mask = cv2.threshold(template, 225, 255, cv2.THRESH_BINARY)
    mask = mask / 255
    mask = mask.astype('uint8')
    ret, part = cv2.threshold(template, 225, 255, cv2.THRESH_TOZERO_INV)
    temp = np.multiply(temp, mask)
    temp = np.add(temp, part)
    image[y1:y2, x1:x2] = temp
    return image

def apply_color_change(image):
    color_change = np.random.randint(-10, 11)
    if color_change < 0:
        c = np.zeros(image.shape, dtype=np.uint8) + -1 * color_change
        np.putmask(image, c > image, c)
        image = image + color_change
    elif color_change > 0:
        c = np.ones(image.shape, dtype=np.uint8) * 255 - color_change
        np.putmask(image, c < image, c)
        image = image + color_change
    return image

def generate_images(number_of_images, offset):
    templates = GetTemplates()
    template_image = cv2.resize(cv2.cvtColor(cv2.imread(TEMPLATE_IMAGE), cv2.COLOR_BGR2GRAY), (INPUT_IMAGE_SIZE))
    label_counts = {}
    for label in templates:
        label_counts[label] = 0
    for i in range(number_of_images):
        rng = np.random.default_rng()
        image = np.copy(template_image)
        temp = sorted(set(label_counts.values()))[:K]        
        low_labels = [key for key in label_counts if label_counts[key] in temp]
        objects = rng.integers(2, MAX_NUMBER_OF_OBJECTS, endpoint=True)
        boundingBoxes = []
        trials = 0
        while len(boundingBoxes) < objects:
            if trials > 100:
                templateName = rng.choice(templates)
            else:
                templateName = rng.choice(low_labels)
            template = GetTemplateImage(templateName)
            rotation = rng.integers(0, 360)
            template = ndimage.rotate(template, rotation, cval=255)
            template = CropTemplate(template)
            bb = GetBoundingBox(template, templateName)
            if BoundingBoxesOverlap(bb, boundingBoxes, image, template):
                trials += 1
                continue
            boundingBoxes.append(bb)
            template = apply_color_change(template)
            image = apply_gray_areas(image, template, bb[0], bb[2], bb[1], bb[3]) 
            trials = 0
            label_counts[templateName] += 1
        image = cv2.resize(image, OUTPUT_IMAGE_SIZE)
        GenerateOutput(str(i+offset), image, boundingBoxes)

if __name__ == '__main__':
    templates = GetTemplates()
    template_image = cv2.cvtColor(cv2.imread(TEMPLATE_IMAGE), cv2.COLOR_BGR2GRAY)
    num_images_per_process = int(NUMBER_OF_IMAGES / NUMBER_OF_PROCESSES)
    processes = []
    for i in range(NUMBER_OF_PROCESSES):
        p = mp.Process(target=generate_images, args=(num_images_per_process, i * num_images_per_process))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
