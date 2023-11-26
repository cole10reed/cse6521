import torch
from torchvision import transforms
import utils_6521 as utils
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


'''
What will be the training data?
 - Truth buildings (images with just the bbox of the building?)

What will be the testing data?

'''


def main(
        sam_checkpoint ='Segment-Anything/checkpoints/sam_vit_h_4b8939.pth',
         gpu_device = 'cuda',
          model_type = 'vit_h',
           dataset_loc = 'Datasets/Urban_3D_Challenge/01-Provisional_Train/'
        ):
    # image = cv2.imread(dataset_loc + r'Inputs/JAX_Tile_052_RGB.tif')
    input_image = Image.open(dataset_loc + r'Inputs/JAX_Tile_052_RGB.tif')
    truth_image = utils.get_truth_image(dataset_loc + r'GT/JAX_Tile_052_GTI.tif', 2048, 2048)
    
    num_buildings = truth_image.max()
    bldgs = list()
    x_offset = list()
    y_offset = list()
    bldg = None

    resize = transforms.Compose([transforms.Resize(size=(128, 128))])

    # This loop crops the image based on building bbox, then resizes the cropped image to 128x128.
    for i in range(num_buildings + 1):
        if i == 0:
            continue
        building_mask = np.where(truth_image==i, 1, 0)  # Create a mask specific to each building ID.
        building_mask = torch.from_numpy(building_mask)  # Turn the mask into a PyTorch tensor for the Jaccard calculations.
        arr = np.nonzero(building_mask)
        arr = np.transpose(arr)     # TODO: Figure out why arr needs to be transposed this time.  Didn't in Task1/2.
        print(arr)
        
        # Building bbox is as follows (least x value, greatest x val, least y value, greatest y val)
        # Forms a box with four corners (bx1,by1), (bx1, by2), (bx2, by1), (bx2, by2)
        by1 = min(arr[0]).item()
        by2 = max(arr[0]).item()
        bx1 = min(arr[1]).item()
        bx2 = max(arr[1]).item()
        
        print(f'bx1={bx1}, by1={by1}, bx2={bx2}, by2={by2}')
        # print(f'Type of bx1 is {type(bx1)}')
        # print(f'bx1 = {bx1}')

        bldg = resize(input_image.crop((bx1, by1, bx2, by2)))
        x_offset.append(bx1)
        y_offset.append(by1)
        print('Building size =', bldg.size)


    
    
    # image = image.transpose(2, 0, 1)
    print(input_image.size)
    print(resize(input_image).size)

    plt.figure(figsize=(20,20))
    # for i in range(num_buildings):
    #     plt.imshow(bldgs[i])
    plt.imshow(bldg)
    # show_anns(masks)
    plt.axis('off')
    # plt.savefig(fname='test_128pps')
    plt.show()


    return

if __name__ == '__main__':
    main()