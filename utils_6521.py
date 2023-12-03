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
    truth_masks = np.zeros((num_buildings, truth_image.shape[0], truth_image.shape[1]))
    for i in range(num_buildings + 1):
        if (i == 0):
            continue
        building_mask = np.where(truth_image==i, 1, 0)
        truth_masks[i - 1] = building_mask
    return truth_masks

def resize_image(image, percent_resize):
    '''
    takes in an iage and resizes it based on percent_size
    ex. percent resize = 50 image goes from 2048*2048 to 1024*1024
    '''
    new_res = percent_resize/100
    new_dim = (int(image.shape[1] * new_res), int(image.shape[0] * new_res))
    image_resize = cv2.resize(image.astype('float32'), new_dim, cv2.INTER_AREA)
    image_resize = image_resize.astype('uint8')
    return image_resize
