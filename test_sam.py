from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import torch
from torchmetrics import JaccardIndex
from sample_calc import get_truth_image
from pycocotools import mask as mask_utils

'''
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        print(ann)
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
'''

sam_checkpoint = 'Segment-Anything/checkpoints/sam_vit_h_4b8939.pth'
model_type = 'vit_h'

device = 'cuda'

image = cv2.imread('Datasets/Urban_3D_Challenge/01-Provisional_Train/Inputs/JAX_Tile_004_RGB.tif')

print('Shape:', image.shape)
print('Type:', image.dtype)
print('Max:', image.max())

#plt.figure(figsize=(20,20))
#plt.imshow(image)
#plt.axis('off')
#plt.show()



sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32)
masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())


truth_image = get_truth_image('Datasets/Urban_3D_Challenge/01-Provisional_Train/GT/JAX_Tile_004_GTI.tif', 2048, 2048)
print('Truth image shape:', truth_image.shape)

### BEGIN ACCURACY CALCULATION ###

jaccard = JaccardIndex(task='binary')

num_buildings = truth_image.max()
true_pos = list()
for i in range(num_buildings + 1):
    building_mask = np.where(truth_image==i, 1, 0)
    building_mask = torch.from_numpy(building_mask)
    for j in range(len(masks)):
        print('Shape of annotation: ', masks[j]['segmentation'].shape)
        print('Shape of building mask: ', building_mask.shape)
        segment = torch.from_numpy(masks[j]['segmentation'])
        res = jaccard(building_mask, segment)
        if (res >= 0.45):
            true_pos.append(i)
            del(masks[j])
            break

print('Number of true positives:', len(true_pos))
print('Number of false negatives:', num_buildings-len(true_pos))

'''
plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.savefig(fname='test_128pps')
plt.show()
'''
