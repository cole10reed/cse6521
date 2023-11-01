import utils_6521 as utils
import numpy as np
import torch
from torchmetrics import JaccardIndex
# (220, 140, 0) is the RGB value of a building pixel in GTC files.
# 6 represents a building pixel in GTL files.
# (224, 224, 224) is a background pixel in GTC files.
# 2 represents a background pixel in GTL files.
# (96, 96, 96) is an uncertain building (i.e. part of it is cut off at the border of the image) in GTC files.
# 65 represents an uncertain building in GTL files.

# See https://www.topcoder.com/challenges/db36b53a-c2f3-4899-9698-13e96148ffcd for more information.

truth_image = utils.get_truth_image('Datasets/Urban_3D_Challenge/01-Provisional_Train/GT/JAX_Tile_004_GTI.tif', 2048, 2048)
print('Truth image shape:', truth_image.shape)

num_buildings = truth_image.max()

a1 = [(10, 20, 30, 20), (20, 10, 15, 10), (10, 15, 40, 50), (20, 5, 30, 30)]

a2 = sorted(a1)

jaccard = JaccardIndex(task='binary')

true = np.ones(shape=(2048,2048))
true = torch.from_numpy(true)
false = np.zeros(shape=(2048,2048))
false = torch.from_numpy(false)

print(a2)
for i in range(num_buildings + 1):
    building_mask = np.where(truth_image==i, 1, 0)
    building_mask_tensor = torch.from_numpy(building_mask)
    arr = np.nonzero(building_mask)
    print('First el = ', arr[0][0], arr[1][0])
    print('Last el = ', arr[0][-1], arr[1][-1])

    bx1 = min(arr[0])
    bx2 = max(arr[0])
    by1 = min(arr[1])
    by2 = max(arr[1])
    print(bx1, bx2, by1, by2)
    print('true =', jaccard(building_mask_tensor, true))
    print('false =', jaccard(building_mask_tensor, false))
    print('self =', jaccard(building_mask_tensor, building_mask_tensor))
    # print(building_mask[(bx1-1):(bx2+2), (by1-1):(by2+2)])

