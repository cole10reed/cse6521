from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchmetrics import JaccardIndex
import utils_6521 as utils
from pycocotools import mask as mask_utils
from multiprocessing import Pool
import time


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        # print(ann)
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def exe(sam_checkpoint ='Segment-Anything/checkpoints/sam_vit_h_4b8939.pth', device = 'cuda', model_type = 'vit_h', dataset_loc = r"C:\Users\Micha\Documents\CSE 6521\Datasets"):

    start = time.time()

    print('--------------Testing Sam Model ', model_type, ' on datasets located at ', dataset_loc)

    # input_images = utils.get_input_files('Datasets/Urban_3D_Challenge/02-Provisional_Test/')
    # truth_images = utils.get_truth_files('Datasets/Urban_3D_Challenge/02-Provisional_Test/')


    # for i in input_images:
    #     print(str(i))


    # print(type(input_images))

    image = cv2.imread(dataset_loc + r'\01-Provisional_Train\Inputs\JAX_Tile_052_RGB.tif')

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32)
    masks = mask_generator.generate(image)

    # print(len(masks))
    # print(masks[0].keys())

    truth_image = utils.get_truth_image(dataset_loc + r'\01-Provisional_Train\GT\JAX_Tile_052_GTI.tif', 2048, 2048)
    # print('Truth image shape:', truth_image.shape)

    ###show images so i can see what correct answers should roughly be
    # plt.figure(figsize=(20,20))
    # plt.imshow(truth_image)
    # plt.imshow(image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.savefig(fname='test_128pps')
    # plt.show()

    ### BEGIN ACCURACY CALCULATION ###

    jaccard = JaccardIndex(task='binary')

    num_buildings = truth_image.max()
    true_pos = list()

    masks = sorted(masks, key=(lambda x: x['bbox']))
    # input_masks = [torch.from_numpy(x['segmentation']) for x in masks]

    print("Starting outer for loop")

    nbreaks = 0
    ncontinue = 0

    timeforloopstart = time.time()
    for i in range(num_buildings + 1):
        building_mask = np.where(truth_image==i, 1, 0)
        building_mask_tensor = torch.from_numpy(building_mask)
        arr = np.nonzero(building_mask)
        
        # Building bbox is as follows (least x value, greatest x val, least y value, greatest y val)
        # Forms a box with four corners (bx1,by1), (bx1, by2), (bx2, by1), (bx2, by2)
        by1 = min(arr[0])
        by2 = max(arr[0])
        bx1 = min(arr[1])
        bx2 = max(arr[1])

        for j in range(len(masks)):
            x,y,h,w = masks[j]['bbox']
            # print('x,y', x,y)
            '''
            # Old calculation (works, but takes ~35 min)
            if ((x > arr[0][-1]) and (y > arr[1][-1])):
                nbreaks += 1 # First point of mask occurs after last point of building, no further masks will intersect.
                break
            if ((x + w < arr[0][0]) and (y + h < arr[1][0])):
                ncontinue += 1 # Last point of mask occurs before first point of building, keep looking but skip this mask.
                continue
            '''
             

            '''
            Current effort (broken, but fast!)
            '''
            
            if (x > bx2):   # Masks are past the building, no more possible intersections.
                # print('BREAKING')
                # print('mask = ',x,y,x+w,y+h)
                # print('bldg = ',bx1,by1,bx2,by2)
                nbreaks += 1
                break
            # if (((y+h) < by1) or (y > by2) or ((x+w) < bx1)): # Mask lies outside building bbox, keep looking but skip this iteration. 
                # print('CONTINUING')
                # print('mask = ',x,y,x+w,y+h)
                # print('bldg = ',bx1,by1,bx2,by2)
            #     ncontinue += 1
            #     continue
            
            if (y+h < by1):
                ncontinue += 1
                continue
            if (y > by2):
                ncontinue += 1
                continue
            if (x+w < bx1):
                ncontinue += 1
                continue
            # print(i, j)
            
            segment = torch.from_numpy(masks[j]['segmentation'])
            # segment = input_masks[j]
            res = jaccard(building_mask_tensor, segment)
            if (res >= 0.45):
                true_pos.append(i)
                # print('-------- FOUND MATCH --------')
                # print("bx1: ", bx1, " bx2: ", bx2, " by1: ", by1, " by2: ", by2)
                # print(" x1: ", x, "  x2: ", x+w, "  y1: ", y, "  y2: ", y+h)
                # print(segment[x:x+w+1,y:y+h+1])
                del(masks[j])
                # del(input_masks[j])
                break

    end = time.time()
    elapsed_time = end-start
    elapsed_for_loop_time = end - timeforloopstart
    print('elapsed for loop: ', elapsed_for_loop_time)
    print('Number of true positives:', len(true_pos))
    print('Number of false negatives:', num_buildings-len(true_pos))
    print('Execution time:', elapsed_time, 'seconds')
    print('Number of breaks', nbreaks)
    print('Number of continues', ncontinue)

    '''
    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(fname='test_128pps')
    plt.show()
    '''
    
'''
if __name__=='__main__':
    exe(dataset_loc='Datasets/Urban_3D_Challenge')
'''