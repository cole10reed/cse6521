from amgWithGrad import AutomaticMaskGenerator_WithGrad
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

def fine_tune(sam: Sam, model: AutomaticMaskGenerator_WithGrad, image: np.ndarray, truth_image: np.ndarray, optimizer, loss_func, num_epochs = 1):
    fpaths = []
    model_tuning_dict = []
    for i in range(num_epochs):
        # set model tunig dict. Each element of array is an epoch. Each dictionary will contain all the batch runs for each epoch
 
        print('Beggining epoch loop for epoch: ', i)

        ## below is cropping code (only produces 1 crop i.e the original image anyway - keeping uncommented to keep code as similiar as possible)
        data = MaskData()
        orig_size = image.shape[:2]

        crop_boxes, layer_idxs = generate_crop_boxes(
        orig_size, n_layers = 0, overlap_ratio = 512 / 1500
        )

    # should only run once  - below is process_crop function
        # crop_data = model._process_crop(image, crop_boxes[0], layer_idxs[0], orig_size)
        # data = crop_data

    # **** Instead of calling model.process crop the code for this function is below ****

    # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_boxes[0]
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        model.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = model.point_grids[layer_idxs[0]] * points_scale

        # Generate masks for this crop in batches

        '''
        From what I understand, process_batch runs on the whole image and generates points across the image 
        in batches to use as reference points. So basically, what this originally did 
        was run 16 times, each batch of points being slightly different (but still across the whole 2048x2048 domain of the image) then catted and deduped the results. 
        However, since the cat function was giving us issues,
        i'm calling process_batch but instead of returning the batch data I am returning some stats about the batch
        and doing gradient descent on the predictions. 
        Because I am doing grad_descent in this function I have added the necessary params (truth image, loss func, etc)
        '''

        # data = MaskData()
        j = 0 # keep track of number of batches
        print('** Beginning batch iterator **')

        # this iterator, which is the same as the original, has 16 iterations. Feel free to cut it after a certain number of iterations if it takes to long
        for (points,) in batch_iterator(model.points_per_batch, points_for_image):
            print("Beginning batch run for batch: ", j)
            # print(points)
            IoU_avg, loss, nMasks = model._process_batch_and_do_grad_desc(truth_image, loss_func, optimizer, points, cropped_im_size, crop_boxes[0], orig_size)
            print('*** IoU: ', IoU_avg)
            print('*** Loss ', loss)
            print('**** nMasks for this batch: ', nMasks)
            
            if (j % 20 == 0):
                fpath = f"tuned_models/model_{j}.pth"
                print("Saving model to path: ", fpath)
                torch.save(sam.state_dict(), f"tuned_models/model_{j}.pth")
                model_dic = {'IoU_avg':IoU_avg, 'loss': loss, 'nMasks': nMasks, 'fpath': fpath}
                model_tuning_dict.append(model_dic)
            
            print('*** Completed Batch **** : ', j)
            j += 1
        model.predictor.reset_image()

        print('Completed Epoch')


        # Remove duplicates within this crop.
        # keep_by_nms = batched_nms(
        #    data["boxes"].float(),
        #    data["iou_preds"],
        #     torch.zeros_like(data["boxes"][:, 0]),  # categories
        #     iou_threshold=model.box_nms_thresh,
        #     )
        #cdata.filter(keep_by_nms)

        # Return to the original image frame
        # data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_boxes[0])
        # data["points"] = uncrop_points(data["points"], crop_boxes[0])
        # data["crop_boxes"] = torch.tensor([crop_boxes[0] for _ in range(len(data["rles"]))])
    print(' **** SUCESSFULLY MADE IT THROUGH FINE TUNING *****')
    return model_tuning_dict

def model_train(model: AutomaticMaskGenerator_WithGrad, image: np.ndarray):
    data = MaskData()
    orig_size = image.shape[:2]

    crop_boxes, layer_idxs = generate_crop_boxes(
    orig_size, n_layers = 0, overlap_ratio = 512 / 1500
    )

# should only run once  - below is process_crop function
    # crop_data = model._process_crop(image, crop_boxes[0], layer_idxs[0], orig_size)
    # data = crop_data

# Crop the image and calculate embeddings
    x0, y0, x1, y1 = crop_boxes[0]
    cropped_im = image[y0:y1, x0:x1, :]
    cropped_im_size = cropped_im.shape[:2]
    model.predictor.set_image(cropped_im)

    # Get points for this crop
    points_scale = np.array(cropped_im_size)[None, ::-1]
    points_for_image = model.point_grids[layer_idxs[0]] * points_scale

    # Generate masks for this crop in batches
    data = MaskData()

    for (points,) in batch_iterator(model.points_per_batch, points_for_image):
        batch_data = model._process_batch(points, cropped_im_size, crop_boxes[0], orig_size)
        data.cat(batch_data)
        del batch_data
    model.predictor.reset_image()

    # Remove duplicates within this crop.
    keep_by_nms = batched_nms(
        data["boxes"].float(),
        data["iou_preds"],
        torch.zeros_like(data["boxes"][:, 0]),  # categories
        iou_threshold=model.box_nms_thresh,
        )
    data.filter(keep_by_nms)

    # Return to the original image frame
    data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_boxes[0])
    data["points"] = uncrop_points(data["points"], crop_boxes[0])
    data["crop_boxes"] = torch.tensor([crop_boxes[0] for _ in range(len(data["rles"]))])

    return data


    return masks, iou_preds, low_res_masks



def save_checkpoint(model_in_training):
    return 0


def retrieve_latest_tuned_checkpoint(original_model = 'vit_b'):
    '''
    retrieves latest tuned checkpoint of vit_b/l/h
    '''

def retrieve_tuned_checkpoint(checkpoint_path):
    '''
    retrieves checkpoint at given file path
    '''