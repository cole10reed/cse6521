import sys
sys.path.append('/users/PAS2622/mdsil11/cse6521/Segment-Anything/')
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


def exe(sam_checkpoint ='Segment-Anything/checkpoints/sam_vit_h_4b8939.pth', gpu_device = 'cuda', model_type = 'vit_h', dataset_loc = '/Datasets/02-Provisional_Test/'):

    start = time.time()

    print('--------------Testing Sam Model ', model_type, ' on datasets located at ', dataset_loc)

    print('Fetching Datasets from location ', dataset_loc)


    input_images = utils.get_input_files(dataset_loc)
    truth_images = utils.get_truth_files(dataset_loc)

    input_list = list()
    truth_list = list()
    for i in input_images:
        input_list.append(str(i))
        print(str(i))
    for i in truth_images:
        truth_list.append(str(i))

    # for i in input_images:
    #     print(str(i))
    # print(type(input_images))

    # image = cv2.imread(dataset_loc + r'\01-Provisional_Train\Inputs\JAX_Tile_004_RGB.tif')

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device = gpu_device)
    # sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32)

    print('Starting image loop')
    n = len(input_list) # number of images
    image_results_dic = dict() # store the image index as key and the values a list of results for each image
    total_stats = {'n_true_pos':0, 'n_false_neg':0, 'total_runtime':0 ,'avg_jac': 0} # array of total accuracy stats and total runtime for all images
    for k in range(len(input_list)):
    # masks = mask_generator.generate(image)

    # print(len(masks))
    # print(masks[0].keys())

    # truth_image = utils.get_truth_image(dataset_loc + r'\01-Provisional_Train\GT\JAX_Tile_004_GTI.tif', 2048, 2048)
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

        jaccard = JaccardIndex(task='binary', num_classes = 2).to(device = gpu_device)
        print(input_list[k])
        print(truth_list[k])
        print('----------------------')

        input_image = cv2.imread(input_list[k])
        truth_image = utils.get_truth_image(truth_list[k], 2048, 2048)
        masks = mask_generator.generate(input_image)
        print(' *********** Generated Masks *************** ' )

        num_buildings = truth_image.max()
        true_pos = list()

        masks = sorted(masks, key=(lambda x: x['bbox']))
    # input_masks = [torch.from_numpy(x['segmentation']) for x in masks]

        print("Starting calculations for image ", input_list[k], 'and truth image ', truth_list[k])
        print('Number_of_buildings: ', num_buildings)
        print('Number of masks: ', len(masks))

        nbreaks = 0
        ncontinue = 0

        image_results_dic[k] = {'n_true_pos': 0, 'n_false_neg': 0, 'ImgExecutionTime': 0, 'avg_jac': 0}

        timeforloopstart = time.time()
        print('Beggining loop over buildings for images')
        for i in range(num_buildings + 1):
            building_mask = np.where(truth_image==i, 1, 0)
            building_mask_tensor = torch.from_numpy(building_mask).cuda()
            arr = np.nonzero(building_mask)

    
            # Building bbox is as follows (least x value, greatest x val, least y value, greatest y val)
            # Forms a box with four corners (bx1,by1), (bx1, by2), (bx2, by1), (bx2, by2)
            if num_buildings == 672:
                print(i)
                print(arr)
                print(arr[0])
            
            ## weird scenario happening right now where some building truth masks are zero
            ## might be the yellow/different colored buildings?
            if(arr[0].size == 0 or arr[1].size == 0):
                print('************ Array is Empty ***********************')
                print("************skipping calculations ************")
                continue

                
            by1 = min(arr[0])
            by2 = max(arr[0])
            bx1 = min(arr[1])
            bx2 = max(arr[1])

            for j in range(len(masks)):
                mask = masks[j]
                x,y,h,w = mask['bbox']
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
                
                segment = torch.from_numpy(mask['segmentation']).cuda()
                # segment = input_masks[j]
                res = jaccard(building_mask_tensor, segment)
                avg_jac = image_results_dic[k]['avg_jac']
                avg_jac = (avg_jac + res)/len(masks)
                image_results_dic[k]['avg_jac'] = avg_jac
                if (res >= 0.45):
                    true_pos.append(i)
                    # print('-------- FOUND MATCH --------')
                    # print("bx1: ", bx1, " bx2: ", bx2, " by1: ", by1, " by2: ", by2)
                    # print(" x1: ", x, "  x2: ", x+w, "  y1: ", y, "  y2: ", y+h)
                    # print(segment[x:x+w+1,y:y+h+1])
                    del(masks[j])
                    # del(input_masks[j])
                    break

        end_img = time.time()
        elapsed_time = end_img-start
        elapsed_for_loop_time = end_img- timeforloopstart
        # build and store results for each image
        print('elapsed for loop: ', elapsed_for_loop_time)
        image_results_dic[k]['elapsed_for_loop_time'] = elapsed_for_loop_time
        print('Number of true positives:', len(true_pos))
        image_results_dic[k]['n_true_pos'] = len(true_pos)
        total_stats['n_true_pos'] = total_stats['n_true_pos'] + len(true_pos)
        print('Number of false negatives:', num_buildings-len(true_pos))
        image_results_dic[k]['n_false_neg'] = num_buildings-len(true_pos)
        total_stats['n_false_neg'] = total_stats['n_false_neg'] + (num_buildings-len(true_pos))
        print('Execution time:', elapsed_time, 'seconds')
        image_results_dic[k]['ImgExecutionTime'] = elapsed_time
        # print('Number of breaks', nbreaks)
        # print('Number of continues', ncontinue)

    print('------- *** Completed all images *** --------')

    end_total = time.time()

    total_stats['total_runtime'] = end_total - start
    #Calculate overal accuracy computation
    print('Writing results to file')

    print('Total results on ', len(input_list) ,' images with model type ', model_type, '\n')
    print('Total number of true positives: ', total_stats['n_true_pos'], '\n')
    print('Total number of false negatives: ', total_stats['n_false_neg'],  '\n')
    print('Total execution time in seconds: ', total_stats['total_runtime'])

    fname = 'results_' + model_type + '_' + dataset_loc + '.txt'
    with open(fname, mode = 'wt') as f:
        f.write('Total results on ' + str(len(input_list)) + ' images with model type ' + model_type + '\n')
        f.write('Total number of true positives: ' + str(total_stats['n_true_pos']) + '\n')
        f.write('Total number of false negatives: ' + str(total_stats['n_false_neg']) +  '\n')
        f.write('Total execution time in seconds: ' + str(total_stats['total_runtime']))



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