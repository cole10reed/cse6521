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
