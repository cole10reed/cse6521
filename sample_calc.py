import cv2
from PIL import Image
import numpy as np
import torch
from torchmetrics import JaccardIndex

# truth_image = cv2.imread('Datasets/Urban_3D_Challenge/01-Provisional_Train/GT/JAX_Tile_004_GTL.tif')
# print(truth_image[:,:,0].max())

def get_truth_image(filename, width, height):
    with Image.open(filename) as img:
        l = list(img.getdata())
        trutharray = [l[i:i+width] for i in range(0, width*height, width)]
        return np.array(trutharray)

'''
image = get_truth_image('Datasets/Urban_3D_Challenge/01-Provisional_Train/GT/JAX_Tile_004_GTI.tif', 2048, 2048)
#print(image.max())
print(image.shape)


mask = np.where(image==686, 1, 0)
print(mask.shape)

j = JaccardIndex(task='binary')
t1 = torch.from_numpy(np.array([[True, False], 
      [True, False]]))
t2 = torch.from_numpy(np.array([[False, True],
      [True, True]]))
print(j(t1, t2))
'''
