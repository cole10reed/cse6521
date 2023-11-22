import cv2
from PIL import Image
import numpy as np
from pathlib import Path

def get_truth_image(filename, width, height):
    with Image.open(filename) as img:
        l = list(img.getdata())
        trutharray = [l[i:i+width] for i in range(0, width*height, width)]
        return np.array(trutharray)

def get_input_files(directory):
    pathlist = Path(directory).glob('**/Inputs/*_RGB.tif')
    return pathlist

def get_truth_files(directory):
    pathlist = Path(directory).glob('**/GT/*_GTI.tif')
    return pathlist

def get_truth_masks(truth_image):
    num_buildings = truth_image.max()
    truth_masks = np.zeros(num_buildings, 2048, 2048)
    for i in range(num_buildings + 1):
        if (i == 0):
            continue
        building_mask = np.where(truth_image==i, 1, 0)
        truth_masks[i - 1] = building_mask
    return truth_masks
