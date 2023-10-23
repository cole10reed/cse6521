from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


sam_checkpoint = 'Segment-Anything/checkpoints/sam_vit_h_4b8939.pth'
model_type = 'vit_h'

image = cv2.imread('Datasets/Urban_3D_Challenge/01-Provisional_Train/Inputs/JAX_Tile_004_RGB.tif')

print('Shape:', image.shape)
print('Type:', image.dtype)
print('Max:', image.max())

#plt.figure(figsize=(20,20))
#plt.imshow(image)
#plt.axis('off')
#plt.show()



sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=64)
masks = mask_generator.generate(image)

print(len(masks))
print(masks[0].keys())

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()