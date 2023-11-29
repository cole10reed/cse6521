import sys
sys.path.append('/users/PAS2622/mdsil11/cse6521/Segment-Anything/')
import segment_anything as sa
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.ops.boxes import batched_nms, box_area
from segment_anything import automatic_mask_generator 
from segment_anything.utils.amg import (
    MaskData,
    area_from_rle,
    batch_iterator,
    batched_mask_to_box,
    box_xyxy_to_xywh,
    build_all_layer_point_grids,
    calculate_stability_score,
    coco_encode_rle,
    generate_crop_boxes,
    is_box_near_crop_edge,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
    uncrop_boxes_xyxy,
    uncrop_masks,
    uncrop_points,
)
import torch
from torchmetrics import JaccardIndex
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import utils_6521 as utils
from segment_anything import SamPredictor
from segment_anything.modeling import Sam
from typing import Any, Dict, List, Optional, Tuple
from amgWithGrad import AutomaticMaskGenerator_WithGrad
from FineTune import fine_tune

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




def model_train(model: AutomaticMaskGenerator_WithGrad, image: np.ndarray):
    data = MaskData()
    orig_size = image.shape[:2]

    crop_boxes, layer_idxs = generate_crop_boxes(
    orig_size, n_layers = 0, overlap_ratio = 512 / 1500
    )

    for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
        crop_data = model._process_crop(image, crop_box, layer_idx, orig_size)
        data.cat(crop_data)
        
    # with torch.no_grad():
    #    im_h, im_w = orig_size
    #    crop_box = [0, 0, im_w, im_h]
        ### Replicating the model._process_crop() function below
    #    model.predictor.set_image(image)                            # generates the image embedding
      
    
    # points_scale = np.array(orig_size)[None, ::-1]
    # points_for_image = model.point_grids[0] * points_scale      # generates the points to be used for prompting

        # Remove duplicate masks between crops
    if len(crop_boxes) > 1:
        # Prefer masks from smaller crops
        scores = 1 / box_area(data["crop_boxes"])
        scores = scores.to(data["boxes"].device)
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            scores,
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=model.crop_nms_thresh,
        )
        data.filter(keep_by_nms)


    # batch = (0,0,0,0)
    # image, point_coords, point_labels, labels = batch
    # image_embeddings = model.image_encoder(image)
    # sparse_embeddings, dense_embeddings = model.prompt_encoder(
    # points=(point_coords, point_labels),
    # oxes=None,
    # masks=None,
    # )
    # Something goes on here for batch sizes greater than 1
    # low_res_masks, iou_predictions = model.mask_decoder(
    # image_embeddings=image_embeddings,
    # image_pe=model.prompt_encoder.get_dense_pe(),
    # sparse_prompt_embeddings=sparse_embeddings,
    # dense_prompt_embeddings=dense_embeddings,
    # multimask_output=False,
    # )

    ### Here is where the mask generation starts ###
    ### We will need to train these parameters ###
    # for (points,) in batch_iterator(model.points_per_batch, points_for_image):
    #     batch_data = model._process_batch(points, orig_size, crop_box, orig_size)
    #    data.cat(batch_data)
    #     del batch_data
    # model.predictor.reset_image()

    data.to_numpy()

    data["segmentations"] = [rle_to_mask(rle) for rle in data["rles"]]

    curr_anns = []
    for idx in range(len(data["segmentations"])):
        ann = {
            "segmentation": data["segmentations"][idx],
            "area": area_from_rle(data["rles"][idx]),
            "bbox": box_xyxy_to_xywh(data["boxes"][idx]).tolist(),
            "predicted_iou": data["iou_preds"][idx].item(),
            "point_coords": [data["points"][idx].tolist()],
            "stability_score": data["stability_score"][idx].item(),
            # "crop_box": box_xyxy_to_xywh(data["crop_boxes"][idx]).tolist(),
        }
        curr_anns.append(ann)

    # print(data.items())

    return curr_anns

    
def model_test(masks, truth_image, device = 'cpu'):
    start = time.time()

    num_buildings = truth_image.max()
    true_pos = list()

    masks = sorted(masks, key=(lambda x: x['bbox']))
    
    jaccard = JaccardIndex(task='binary').to(device = device)

    nbreaks = 0
    ncontinue = 0
    sum_iou = 0

    image_size = 2048 * 2048
    shape = (num_buildings, image_size)
    # print(shape)
    # all_truth_masks = np.zeros(shape)
    # print(all_truth_masks.shape)

    # all_pred_masks = np.zeros(shape)

    timeforloopstart = time.time()
    for i in range(num_buildings + 1):
        if (i == 0):
            continue
        building_mask = np.where(truth_image==i, 1, 0)
        # building_mask_2 = building_mask.reshape(building_mask.shape[0], -1).T #flattens mask into 1D
        building_mask_2 = building_mask.flatten()
        # all_truth_masks[i - 1] = building_mask_2 # append building mask to all truth mask
        # np.concatenate(all_truth_mask, building_mask)
        building_mask_tensor = torch.from_numpy(building_mask, device = device)

        arr = np.nonzero(building_mask)
        # print(f'Building {i}')
        
        # Building bbox is as follows (least x value, greatest x val, least y value, greatest y val)
        # Forms a box with four corners (bx1,by1), (bx1, by2), (bx2, by1), (bx2, by2)
        by1 = min(arr[0])
        by2 = max(arr[0])
        bx1 = min(arr[1])
        bx2 = max(arr[1])

        for j in range(len(masks)):
            x,y,h,w = masks[j]['bbox']
            
            if (x > bx2):   # Masks are past the building, no more possible intersections.
                nbreaks += 1
                break
            
            if (y+h < by1):
                ncontinue += 1
                continue
            if (y > by2):
                ncontinue += 1
                continue
            if (x+w < bx1):
                ncontinue += 1
                continue
            
            segment = torch.from_numpy(masks[j]['segmentation'])
            segment_num = masks[j]['segmentation'].flatten()
            res = jaccard(building_mask_tensor, segment)

            ### Here we've found a true positive, let's calculate the loss using loss_func and optimizer ###
            if (res >= 0.45):
                true_pos.append(i)
                sum_iou = sum_iou + res
                # all_pred_masks[i - 1] = segment_num

                # building_mask_tensor_float = building_mask_tensor.float()
                segment = segment.float()

                # loss = loss_func(building_mask_tensor_float, segment)
                # loss.requires_grad = True
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                
                del(masks[j])
                break

    end = time.time()
    elapsed_time = end-start
    elapsed_for_loop_time = end - timeforloopstart
    print('elapsed for loop: ', elapsed_for_loop_time)
    print('Number of true positives:', len(true_pos))
    print('Number of false negatives:', num_buildings-len(true_pos))
    print('Execution time:', elapsed_time, 'seconds')
    print('total number of masks created by model: ', len(masks))
    # print('Number of breaks', nbreaks)
    # print('Number of continues', ncontinue)
    return sum_iou, len(true_pos) #, all_truth_masks, all_pred_masks

def compare_tuned_to_base(sam: Sam, sam_tuned: Sam,
         device = 'cpu',
          model_type = 'vit_h',
           dataset_loc = 'Datasets/Urban_3D_Challenge/01-Provisional_Train/',
           tuned_checkpoint = 'tuned_models/model_4.pth'):
        # base model

    print('----- Starting model compare of base model ', model_type, 'with tuned model located at: ', tuned_checkpoint)

    # sam_tuned = sam_model_registry[model_type](checkpoint=tuned_checkpoint)#.to(device = gpu_device)
    points_per_side = 32

    #mask generator for base
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side)

    #mask generator for tuned model
    mask_generator_tuned = SamAutomaticMaskGenerator(sam_tuned, points_per_side)

    #start with 1 image for now
    # image = cv2.imread(dataset_loc + r'Inputs/JAX_Tile_052_RGB.tif')

    # print(image)
    # truth_image = utils.get_truth_image(dataset_loc + r'GT/JAX_Tile_052_GTI.tif', 2048, 2048)
    # print('Image shape', truth_image.shape)

    # next steps:
    # 1. run image test code on model 1 
    # run image test code on tuned model
    # compare the results
    #move to osc

    print('Fetching Datasets from location ', dataset_loc)

    input_list = list()
    truth_list = list()

    input_images = utils.get_input_files(dataset_loc)
    truth_images = utils.get_truth_files(dataset_loc)

    for i in input_images:
        input_list.append(str(i))
        print(str(i))
    for i in truth_images:
        truth_list.append(str(i))

    
    ## start looping over all images and test each model
    image_results_dic = dict()
    for k in range(len(input_list)):

        jaccard = JaccardIndex(task='binary', num_classes = 2).to(device = device)
        print('----------------------')

        print('**** Reading in image ', k, 'out of image ', len(input_list) )
        input_image = cv2.imread(input_list[k])
        truth_image = utils.get_truth_image(truth_list[k], 2048, 2048)
        masks_tuned = mask_generator_tuned.generate(input_image)
        masks = mask_generator.generate(input_image)
        print(' *********** Generated Masks *************** ' )

        num_buildings = truth_image.max()
        true_pos = list()

        masks = sorted(masks, key=(lambda x: x['bbox']))
        masks_tuned = sorted(masks_tuned, key=(lambda x: x['bbox']))
    # input_masks = [torch.from_numpy(x['segmentation']) for x in masks]

        print("Starting calculations for image ", input_list[k], 'and truth image ', truth_list[k])
        print('Number_of_buildings: ', num_buildings)
        print('Number of masks for original model: ', len(masks))
        print('Number of masks for tuned model: ', len(masks_tuned))

        # nbreaks = 0
        # ncontinue = 0

        # image_results_dic[k] = {'n_true_pos': 0, 'n_false_neg': 0, 'ImgExecutionTime': 0, 'avg_jac': 0}

        sum_iou, nTruePos = model_test(masks, truth_image, device)
        sum_iou_tuned, nTruePos_tuned = model_test(masks_tuned, truth_image, device)

        print('Completing comparison of base model with tuned model at path: ' , tuned_checkpoint)
        print('**** IoU Comparison ****')
        print('Avg IoU for base model: ' , sum_iou/nTruePos)
        print('Avg IoU for tuned model: ', sum_iou_tuned/nTruePos_tuned)

        print('***** End comparison *****')
        #### compare masks on image k



def main(
        device = 'cpu',
        tune_model = False,
        sam_checkpoint ='Segment-Anything/checkpoints/sam_vit_h_4b8939.pth',
        model_type = 'vit_h',
        dataset_loc = 'Datasets/Urban_3D_Challenge/01-Provisional_Train/',
        num_epochs = 5
           ):

    print('Started Task2 Main with input args: \n', 
          'device: ', device, '\n',
          'tune_model: ', tune_model, '\n',
          'sam_checkpoint: ', sam_checkpoint, '\n',
          'model_type: ', model_type, '\n',
          'dataset_loc: ', dataset_loc, '\n',
          'num_epochs: ', num_epochs, '\n')
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device = device)

    ## task 2: first tune models. Then compare tuned to base

    # get list of all input and truth images in the specified directory
    input_images = utils.get_input_files(dataset_loc)
    truth_images = utils.get_truth_files(dataset_loc)

    input_list = list()
    truth_list = list()
    for i in input_images:
        input_list.append(str(i))
    for i in truth_images:
        truth_list.append(str(i))

    points_per_side = 32

    ## creates SAM Aut mask generator except this class has overriden the functions that use torch_nograd
    model_in_training = AutomaticMaskGenerator_WithGrad(sam, points_per_side=points_per_side)

    optimizer = torch.optim.Adam(sam.mask_decoder.parameters())
    loss_func = torch.nn.MSELoss().to(device = device)

    if tune_model:
        for j in range(num_epochs):
            for k in range(len(input_list)):
                input_image = cv2.imread(input_list[k])
                # print(image)
                truth_image = utils.get_truth_image(truth_list[k], 2048, 2048)
                
                # *** FINE TUNING DONE HERE ***
                print('*** Beginning fine tuning for model ', model_type, 'at checkpoint ', sam_checkpoint)
                print('Image:', input_list[k])
                print('Iteration:', j)

                model_dic = fine_tune(sam, model_in_training, input_image, truth_image, optimizer, loss_func, 1)
                print(f'Image{k}: {input_list[k]} is complete.')
            print(f'***COMPLETED ITERATION {j}***')

        print('**SUCCESSFULLY TUNED MODEL**')

    #step 2: compare tuned to base. for now, compare last model
    if(tune_model):
        last_tuned_model_path = model_dic.pop['fpath']
    else:
        last_tuned_model_path = 'tuned_models/model_4.pth' # default to some path
    sam_tuned = sam_model_registry[model_type](checkpoint=last_tuned_model_path).to(device = device)

    compare_tuned_to_base(sam, sam_tuned, device = device)

    #store params
    paramdic_base = {}
    paramdic_tuned = {}

    #sam has an iterator to iterate over params. Save them in corresponding dic
    for (param_name, param) in sam.mask_decoder.named_parameters():
        paramdic_base[param_name] = param
    
    for(param_name, param) in sam_tuned.mask_decoder.named_parameters():
        paramdic_tuned[param_name] = param

    #### look for changes in parameters
    weights_updated = False
    weight_diff = {}
    for (param_name, param) in paramdic_base.items():
        if not paramdic_tuned[param_name].equal(param): # tensor.equal returns true if tensor is exactly the same (values and all other elements of tensor)
            weights_updated = True
            weight_diff[param_name] = paramdic_tuned[param_name].eq(param) # this goes element by element and returns a tensor of true and falses corresponding to which exact elements have changed (i.e are different)
    
    if not weights_updated: # not one change in parameter detected
        print('Model weights are exactly the same')

    ''''TODO: I havent actually run the new models and got full prediction calculations like we do in Task1 - 
    after we see changes in weights we should start test running the new tuned checkpoints'''

   # plt.figure(figsize=(20,20))
   # plt.imshow(image)
   # show_anns(masks)
   # plt.axis('off')
   # plt.savefig(fname='test_task2')
   # plt.show()
    

if __name__ == '__main__':
    #set values of paramters if they exist in command line args else default. sys.argv[0] is name of program
    device = sys.argv[1] if len(sys.argv) > 1 else 'cuda'
    tune_model = sys.argv[2] if len(sys.argv) > 2 else False
    sam_checkpoint = sys.argv[3] if len(sys.argv) > 3 else 'Segment-Anything/checkpoints/sam_vit_h_4b8939.pth'
    model_type = sys.argv[4] if len(sys.argv) > 4 else 'vit_h'
    dataset_loc = sys.argv[5] if len(sys.argv) > 5 else 'Datasets/Urban_3D_Challenge/01-Provisional_Train/'
    num_epochs = sys.argv[6] if len(sys.argv) > 6 else 5
    main(device, tune_model, sam_checkpoint, model_type, dataset_loc, num_epochs)