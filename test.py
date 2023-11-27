import cv2
import torch
from ImageClassifier import ImageClassifier
from torchsummary import summary
import numpy as np

dataset_loc = 'Datasets/Urban_3D_Challenge/01-Provisional_Train/'

input_shape = (3, 128, 128)
classifier = ImageClassifier()
curr_anns = []
for i in range(10):
    ann = {
        'info1': i,
        'info2': i * 10,
        'info3': i * 4
    }
    curr_anns.append(ann)

print(curr_anns[4]['info1'])
print(curr_anns[8]['info3'])
curr_anns[2]['state'] = False
print(curr_anns[2]['state'])
for i in range(5):
    if ('state' in curr_anns[i].keys()):
        print('found', i)

# summary(classifier, input_shape)
input = torch.randn((3, 128, 128))
pred = classifier(input)
print(pred)
print(type(pred))

# input_image = cv2.imread(dataset_loc + r'Inputs/JAX_Tile_052_RGB.tif')
# bldg = cv2.resize(input_image[10:20, 40:50], (128,128))
# bldg = torch.from_numpy(bldg).float().transpose(1,2).transpose(0,1)
# print('BLDG shape:', bldg.shape)

