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

    
def grad_descent(masks, truth_image, loss_func, optimizer):
    start = time.time()

    num_buildings = truth_image.max()
    true_pos = list()

    masks = sorted(masks, key=(lambda x: x['bbox']))
    
    jaccard = JaccardIndex(task='binary')

    nbreaks = 0
    ncontinue = 0
    sum_iou = 0

    image_size = 2048 * 2048
    shape = (num_buildings, image_size)
    print(shape)
    all_truth_masks = np.zeros(shape)
    print(all_truth_masks.shape)

    all_pred_masks = np.zeros(shape)

    timeforloopstart = time.time()
    for i in range(num_buildings + 1):
        if (i == 0):
            continue
        building_mask = np.where(truth_image==i, 1, 0)
        # building_mask_2 = building_mask.reshape(building_mask.shape[0], -1).T #flattens mask into 1D
        building_mask_2 = building_mask.flatten()
        all_truth_masks[i - 1] = building_mask_2 # append building mask to all truth mask
        # np.concatenate(all_truth_mask, building_mask)
        building_mask_tensor = torch.from_numpy(building_mask)

        arr = np.nonzero(building_mask)
        print(f'Building {i}')
        
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
                all_pred_masks[i - 1] = segment_num

                building_mask_tensor_float = building_mask_tensor.float()
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
    return sum_iou, len(true_pos), all_truth_masks, all_pred_masks



def main(
        sam_checkpoint ='Segment-Anything/checkpoints/sam_vit_h_4b8939.pth',
         gpu_device = 'cuda',
          model_type = 'vit_h',
           dataset_loc = 'Datasets/Urban_3D_Challenge/01-Provisional_Train/'
           ):
    # with torch.no_grad():

    # model to fine tune
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)#.to(device = gpu_device)

    points_per_side = 32

    image = cv2.imread(dataset_loc + r'Inputs/JAX_Tile_052_RGB.tif')
    # print(image)
    truth_image = utils.get_truth_image(dataset_loc + r'GT/JAX_Tile_052_GTI.tif', 2048, 2048)
    print('Image shape', truth_image.shape)

    all_truth_masks = utils.get_truth_masks(truth_image)

    ## creates SAM Aut mask generator except this class has overriden the functions that use torch_nograd
    model_in_training = AutomaticMaskGenerator_WithGrad(sam, all_truth_masks , points_per_side=points_per_side) 

    # *** FINE TUNING DONE HERE ***
    print('*** Begging fine tuning for model ', model_type, 'at checkpoint ', sam_checkpoint)

    optimizer = torch.optim.Adam(sam.mask_decoder.parameters())
    loss_func = torch.nn.MSELoss()

    model_dic = fine_tune(sam, model_in_training, image, truth_image, optimizer, loss_func, 1)

    print('**SUCCESSFULLY TUNED MODEL**')


    # After fine tuning, test and see if new models have any changes to parameters
    sam_tuned = sam_model_registry[model_type](checkpoint='tuned_models/model_4.pth')#.to(device = gpu_device)

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
    main(dataset_loc='Datasets/Urban_3D_Challenge/01-Provisional_Train/')