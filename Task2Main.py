import segment_anything as sa
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
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
from typing import Optional, Tuple, List, Dict, Any
from copy import deepcopy
from torchvision.ops.boxes import batched_nms, box_area  # type: ignore
import utils_6521 as utils


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


def predict_torch_grad(predictor,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if not predictor.is_image_set:
        raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

    if point_coords is not None:
        points = (point_coords, point_labels)
    else:
        points = None

    # Embed prompts
    sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
        points=points,
        boxes=boxes,
        masks=mask_input,
    )

    # Predict masks
    low_res_masks, iou_predictions = predictor.model.mask_decoder(
        image_embeddings=predictor.features,
        image_pe=predictor.model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
    )

    # Upscale the masks to the original image resolution
    masks = predictor.model.postprocess_masks(low_res_masks, predictor.input_size, predictor.original_size)

    if not return_logits:
        masks = masks > predictor.model.mask_threshold

    return masks, iou_predictions, low_res_masks


def process_batch_grad(
        model,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
    orig_h, orig_w = orig_size

    transformed_points = model.predictor.transform.apply_coords(points, orig_size)
    in_points = torch.as_tensor(transformed_points, device=model.predictor.device)
    in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
    masks, iou_preds, _ = predict_torch_grad(
        model.predictor,
        in_points[:, None, :],
        in_labels[:, None],
        multimask_output=True,
        return_logits=True,
    )
        
    # Serialize predictions and store in MaskData
    data = MaskData(
        masks=masks.flatten(0, 1),
        iou_preds=iou_preds.flatten(0, 1),
        points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
    )
    del masks

    # Filter by predicted IoU
    if model.pred_iou_thresh > 0.0:
        keep_mask = data["iou_preds"] > model.pred_iou_thresh
        data.filter(keep_mask)

    # Calculate stability score
    data["stability_score"] = calculate_stability_score(
        data["masks"], model.predictor.model.mask_threshold, model.stability_score_offset
    )
    if model.stability_score_thresh > 0.0:
        keep_mask = data["stability_score"] >= model.stability_score_thresh
        data.filter(keep_mask)

    # Threshold masks and calculate boxes
    data["masks"] = data["masks"] > model.predictor.model.mask_threshold
    data["boxes"] = batched_mask_to_box(data["masks"])

    # Filter boxes that touch crop boundaries
    keep_mask = ~is_box_near_crop_edge(data["boxes"], crop_box, [0, 0, orig_w, orig_h])
    if not torch.all(keep_mask):
        data.filter(keep_mask)

    # Compress to RLE
    data["masks"] = uncrop_masks(data["masks"], crop_box, orig_h, orig_w)
    data["rles"] = mask_to_rle_pytorch(data["masks"])
    del data["masks"]

    return data


def process_crop_grad(
        model,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
    ) -> MaskData:
    # Crop the image and calculate embeddings
    x0, y0, x1, y1 = crop_box
    cropped_im = image[y0:y1, x0:x1, :]
    cropped_im_size = cropped_im.shape[:2]
    model.predictor.set_image(cropped_im)

    # Get points for this crop
    points_scale = np.array(cropped_im_size)[None, ::-1]
    points_for_image = model.point_grids[crop_layer_idx] * points_scale

    # Generate masks for this crop in batches
    data = MaskData()
    for (points,) in batch_iterator(model.points_per_batch, points_for_image):
        batch_data = process_batch_grad(model, points, cropped_im_size, crop_box, orig_size)
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
    data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
    data["points"] = uncrop_points(data["points"], crop_box)
    data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

    return data


def generate_masks_grad(model, image: np.ndarray) -> MaskData:
    orig_size = image.shape[:2]
    crop_boxes, layer_idxs = generate_crop_boxes(
        orig_size, model.crop_n_layers, model.crop_overlap_ratio
    )
        
    # Iterate over image crops
    data = MaskData()
    for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
        crop_data = model._process_crop(image, crop_box, layer_idx, orig_size)
        data.cat(crop_data)

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

    data.to_numpy()
    return data



def generate_grad(model, image: np.ndarray) -> List[Dict[str, Any]]:
    
    # Generate masks
    mask_data = generate_masks_grad(model, image)

    # Filter small disconnected regions and holes in masks
    if model.min_mask_region_area > 0:
        mask_data = model.postprocess_small_regions(
            mask_data,
            model.min_mask_region_area,
            max(model.box_nms_thresh, model.crop_nms_thresh),
        )

    # Encode masks
    if model.output_mode == "coco_rle":
        mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
    elif model.output_mode == "binary_mask":
        mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
    else:
        mask_data["segmentations"] = mask_data["rles"]

    # Write mask records
    curr_anns = []
    for idx in range(len(mask_data["segmentations"])):
        ann = {
            "segmentation": mask_data["segmentations"][idx],
            "area": area_from_rle(mask_data["rles"][idx]),
            "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
            "predicted_iou": mask_data["iou_preds"][idx].item(),
            "point_coords": [mask_data["points"][idx].tolist()],
            "stability_score": mask_data["stability_score"][idx].item(),
            "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
        }
        curr_anns.append(ann)

    return curr_anns


'''
def model_train(model: SamAutomaticMaskGenerator, image: np.ndarray):
    data = MaskData()
    orig_size = image.shape[:2]
    
    with torch.no_grad():
        im_h, im_w = orig_size
        crop_box = [0, 0, im_w, im_h]
        ### Replicating the model._process_crop() function below
        model.predictor.set_image(image)                            # generates the image embedding
      
    
    points_scale = np.array(orig_size)[None, ::-1]
    points_for_image = model.point_grids[0] * points_scale      # generates the points to be used for prompting
    

    ### Here is where the mask generation starts ###
    ### We will need to train these parameters ###
    for (points,) in batch_iterator(model.points_per_batch, points_for_image):
        # batch_data = model._process_batch(points, orig_size, crop_box, orig_size) # TODO: split this up. Calls predict_torch which has @torch.no_grad()
        
        batch_data = process_batch_grad(model, points, orig_size, crop_box, orig_size)
        data.cat(batch_data)
        del batch_data
    model.predictor.reset_image()

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
'''
    
def grad_descent(masks, truth_image, loss_func, optimizer):
    start = time.time()

    num_buildings = truth_image.max()
    true_pos = list()

    masks = sorted(masks, key=(lambda x: x['bbox']))
    
    jaccard = JaccardIndex(task='binary')

    nbreaks = 0
    ncontinue = 0
    sum_iou = 0

    timeforloopstart = time.time()
    for i in range(num_buildings + 1):
        if (i == 0):
            continue
        building_mask = np.where(truth_image==i, 1, 0)
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
            res = jaccard(building_mask_tensor, segment)

            ### Here we've found a true positive, let's calculate the loss using loss_func and optimizer ###
            if (res >= 0.45):
                true_pos.append(i)
                sum_iou = sum_iou + res

                building_mask_tensor_float = building_mask_tensor.float()
                segment = segment.float()

                loss = loss_func(building_mask_tensor_float, segment)
                loss.requires_grad = True
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                del(masks[j])
                break

    end = time.time()
    elapsed_time = end-start
    elapsed_for_loop_time = end - timeforloopstart
    print('elapsed for loop: ', elapsed_for_loop_time)
    print('Number of true positives:', len(true_pos))
    print('Number of false negatives:', num_buildings-len(true_pos))
    print('Execution time:', elapsed_time, 'seconds')
    # print('Number of breaks', nbreaks)
    # print('Number of continues', ncontinue)
    return sum_iou, len(true_pos)



def main(
        sam_checkpoint ='Segment-Anything/checkpoints/sam_vit_h_4b8939.pth',
         gpu_device = 'cuda',
          model_type = 'vit_h',
           dataset_loc = 'Datasets/Urban_3D_Challenge/01-Provisional_Train/'
           ):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)#.to(device = gpu_device)

    points_per_side = 32

    model_in_training = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side) 
    optimizer = torch.optim.Adam(sam.mask_decoder.parameters())
    loss_func = torch.nn.MSELoss()

    image = cv2.imread(dataset_loc + r'Inputs/JAX_Tile_052_RGB.tif')
    # print(image)
    truth_image = utils.get_truth_image(dataset_loc + r'GT/JAX_Tile_052_GTI.tif', 2048, 2048)
    
    for i in range(5):
        # masks = model_train(model_in_training, image)
        masks = generate_grad(model_in_training, image)
        ### Here we calculate loss building-by-building and call optimizer ###
        sum_iou, num_true_positive = grad_descent(masks=masks, truth_image=truth_image, loss_func=loss_func, optimizer=optimizer)
        print(f'Iteration  {i}: {(sum_iou / num_true_positive)}')


    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(fname='test_task2')
    plt.show()

    

if __name__ == '__main__':
    main(dataset_loc='Datasets/Urban_3D_Challenge/01-Provisional_Train/')