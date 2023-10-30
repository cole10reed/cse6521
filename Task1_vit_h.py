from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import torch
from torchmetrics import JaccardIndex
import utils_6521 as utils

sam_checkpoint = 'Segment-Anything/checkpoints/sam_vit_h_4b8939.pth'
model_type = 'vit_h'

device = 'cuda'

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
# sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32)


input_images = utils.get_input_files('Datasets/Urban_3D_Challenge/02-Provisional_Test/')
truth_images = utils.get_truth_files('Datasets/Urban_3D_Challenge/02-Provisional_Test/')

input_list = list()
truth_list = list()
for i in input_images:
    input_list.append(str(i))
    print(str(i))
for i in truth_images:
    truth_list.append(str(i))

total_true_positives = 0
total_false_negatives = 0

for k in range(len(input_list)):
    print(input_list[k])
    print(truth_list[k])
    print('----------------------')

    input_image = cv2.imread(input_list[k])
    truth_image = utils.get_truth_image(truth_list[k], 2048, 2048)

    masks = mask_generator.generate(input_image)

    print('Number of masks:', len(masks))
    
    ### BEGIN ACCURACY CALCULATION ###

    jaccard = JaccardIndex(task='binary')  # This performs the IoU calculation.

    num_buildings = truth_image.max()
    true_pos = list()
    for i in range(num_buildings + 1):
        building_mask = np.where(truth_image==i, 1, 0)  # Create a mask specific to each building ID.
        building_mask = torch.from_numpy(building_mask)  # Turn the mask into a PyTorch tensor for the Jaccard calculations.
        for j in range(len(masks)):
            print('Shape of annotation: ', masks[j]['segmentation'].shape)
            print('Shape of building mask: ', building_mask.shape)
            segment = torch.from_numpy(masks[j]['segmentation'])
            result = jaccard(building_mask, segment)
            if (result >= 0.45):  # If the IoU result is greater than 45%, consider it a true positive.
                true_pos.append(i)
                del(masks[j])
                break

    print('Number of true positives:', len(true_pos))
    print('Number of false negatives:', num_buildings-len(true_pos))
    total_true_positives = total_true_positives + len(true_pos)
    total_false_negatives = total_false_negatives + num_buildings - len(true_pos)

print('XXXXXXXXXXXXXXXXX')
print('Final results:')
print('True Positives:', total_true_positives)
print('False Negatives:', total_false_negatives)
