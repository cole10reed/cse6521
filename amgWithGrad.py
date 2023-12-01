from torchmetrics import JaccardIndex
import segment_anything as sa
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from torchvision.ops.boxes import batched_nms, box_area 
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
import numpy as np
from copy import deepcopy
from SamPredictorWithGrad import SamPredictor_WithGrad
from segment_anything.modeling import Sam
from typing import Any, Dict, List, Optional, Tuple

class AutomaticMaskGenerator_WithGrad(SamAutomaticMaskGenerator):
    '''
    This class is an identical child class of SamAutomaticMaskGenerator EXCEPT we override two things:

    1. wrote _process_batch_and_do_grad_descent which is similiar to process batch except after predictions we dedup right away and run grad descent.

    2. In the init function, self.predictor is now the SamPredictor_WithGrad instead of SamPredictor, which overrides .predict so we can call it with gradients.
    It is identical to its parent class function except torch_no_grad is removed.
    '''
    def __init__(
        self,
        model: Sam,
        # labels: torch.Tensor,
        points_per_side: Optional[int] = 32,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        stability_score_offset: float = 1.0,
        box_nms_thresh: float = 0.7,
        crop_n_layers: int = 0,
        crop_nms_thresh: float = 0.7,
        crop_overlap_ratio: float = 512 / 1500,
        crop_n_points_downscale_factor: int = 1,
        point_grids: Optional[List[np.ndarray]] = None,
        min_mask_region_area: int = 0,
        output_mode: str = "binary_mask",
    ) -> None:
        """
        Using a SAM model, generates masks for the entire image.
        Generates a grid of point prompts over the image, then filters
        low quality and duplicate masks. The default settings are chosen
        for SAM with a ViT-H backbone.

        Arguments:
          model (Sam): The SAM model to use for mask prediction.
          points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
          points_per_batch (int): Sets the number of points run simultaneously
            by the model. Higher numbers may be faster but use more GPU memory.
          pred_iou_thresh (float): A filtering threshold in [0,1], using the
            model's predicted mask quality.
          stability_score_thresh (float): A filtering threshold in [0,1], using
            the stability of the mask under changes to the cutoff used to binarize
            the model's mask predictions.
          stability_score_offset (float): The amount to shift the cutoff 
            calculated the stability score.
          box_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks.
          crop_n_layers (int): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run, where each
            layer has 2**i_layer number of image crops.
          crop_nms_thresh (float): The box IoU cutoff used by non-maximal
            suppression to filter duplicate masks between different crops.
          crop_overlap_ratio (float): Sets the degree to which crops overlap.
            In the first crop layer, crops will overlap by this fraction of
            the image length. Later layers with more crops scale down this overlap.
          crop_n_points_downscale_factor (int): The number of points-per-side
            sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
          point_grids (list(np.ndarray) or None): A list over explicit grids
            of points used for sampling, normalized to [0,1]. The nth grid in the
            list is used in the nth crop layer. Exclusive with points_per_side.
          min_mask_region_area (int): If >0, postprocessing will be applied
            to remove disconnected regions and holes in masks with area smaller
            than min_mask_region_area. Requires opencv.
          output_mode (str): The form masks are returned in. Can be 'binary_mask',
            'uncompressed_rle', or 'coco_rle'. 'coco_rle' requires pycocotools.
            For large resolutions, 'binary_mask' may consume large amounts of
            memory.
        """

        assert (points_per_side is None) != (
            point_grids is None
        ), "Exactly one of points_per_side or point_grid must be provided."
        if points_per_side is not None:
            self.point_grids = build_all_layer_point_grids(
                points_per_side,
                crop_n_layers,
                crop_n_points_downscale_factor,
            )
        elif point_grids is not None:
            self.point_grids = point_grids
        else:
            raise ValueError("Can't have both points_per_side and point_grid be None.")

        assert output_mode in [
            "binary_mask",
            "uncompressed_rle",
            "coco_rle",
        ], f"Unknown output_mode {output_mode}."
        if output_mode == "coco_rle":
            from pycocotools import mask as mask_utils  # type: ignore # noqa: F401

        if min_mask_region_area > 0:
            import cv2  # type: ignore # noqa: F401

        self.predictor = SamPredictor_WithGrad(model)
        self.points_per_batch = points_per_batch
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.stability_score_offset = stability_score_offset
        self.box_nms_thresh = box_nms_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_nms_thresh = crop_nms_thresh
        self.crop_overlap_ratio = crop_overlap_ratio
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor
        self.min_mask_region_area = min_mask_region_area
        self.output_mode = output_mode
        # self.labels = labels

    def _process_batch_and_do_grad_desc(
        self,
        truth_image: np.ndarray,
        loss_func,
        optimizer,
        points: np.ndarray,
        im_size: Tuple[int, ...],
        crop_box: List[int],
        orig_size: Tuple[int, ...],
    ) -> MaskData:
        orig_h, orig_w = orig_size

        print(' ** Making predicted masks **')
        print("*** Self.predictor.device: ", self.predictor.device)


        # Run model on this batch
        transformed_points = self.predictor.transform.apply_coords(points, im_size)
        in_points = torch.as_tensor(transformed_points, device=self.predictor.device)
        in_labels = torch.ones(in_points.shape[0], dtype=torch.int, device=in_points.device)
        masks, iou_preds, _ =  self.predictor.predict_torch(
            in_points[:, None, :],
            in_labels[:, None],
            multimask_output=False,
            # multimask_output=True,
            return_logits=True,
         )
        print('Orig masks grad_fn:', masks.grad_fn)
        masks = masks.flatten(0, 1)
        bin_masks = masks > 0 # Turn logit mask into binary mask of building (True = building_prediction, False = background_prediction)
        unique = bin_masks[0][0].unique(return_counts=True)
        
        print('-------------')
        print('Generated masks info:')
        print('Shape: ', bin_masks.shape)
        print('First mask: ', masks[0][0])
        print('Unique vals:', unique)
        print('Num unique vals:', len(unique[0]))
        print('Max val:', torch.max(unique[0]))
        print('Grad:', masks.grad_fn)
        
        """
                # Serialize predictions and store in MaskData
        data = MaskData(
            masks=masks.flatten(0, 1),
            iou_preds=iou_preds.flatten(0, 1),
            points=torch.as_tensor(points.repeat(masks.shape[1], axis=0)),
        )
        del masks
        """
        """
        ********************************
         *** Below is all the filtering and deduping done in the original generate_masks, process_crop and process_batch ***

         *** TODO ***
         I THINK, what the issue is, is we need to run the loss function and optimizer on the original tensor -
         masks , because that tensor actually has some of the gradient needed for backpropogation and I THINK
         this is getting lost in all these transforms. Good news is that we probably can then remove all this filtering and processing gibberish,
         bad news is we will need some kind of logic to convert the original tensor to match the shape of the truth masks

         Note - there is also a postprocessing step in SamPredictor.predict. We may need to check that as well in SamPRedictor.WithGrad.predict
         ***** Second note - I shared some pictures of debugging on teams that I think proves my above theory correct. grad_func is missing when we pass the tensor to the loss function
        """
        
        print(' *** Grad Descent Begginging ***')
        num_buildings = truth_image.max()

        true_pos = list()
        jaccard = JaccardIndex(task='binary').to(device = self.predictor.device)

        image_size = 2048 * 2048
        shape = (num_buildings, image_size)
        loss_sum = 0

        # all_pred_masks = np.zeros(shape)
        nbreaks = 0
        ncontinue = 0
        sum_iou = 0

        '''
        Below is same logic as we did in task one, that loops through all the masks, find matches, and calculates IoU and runs grad descent.
        I starting fiddling with trying to do the calculation and gradient descent
        on one whole tensor with all the masks instead of one mask at a time, i think that would be more efficient,
        but I haven't finished that yet.
        '''
        h, w = truth_image.shape
        num_matches = 0
        reordered_truth_masks = torch.from_numpy(np.zeros(shape=(len(masks), h, w))).to(device = self.predictor.device)
        # Reordered masks will be used to put building/prediction matches at the same row, in order to calculate loss all at once

        for i in range(num_buildings + 1):
          building_mask = np.where(truth_image==i, 1, 0)
          building_mask_tensor = torch.from_numpy(building_mask).to(device = self.predictor.device)
          arr = np.nonzero(building_mask)

          if(arr[0].size == 0 or arr[1].size == 0):
              print('*** SKIPPING MASK FOR BUILDING ', i, ' AS TRUTH MASKS ARE EMPTY AND WILL CAUSE FAILURE OF MIN/MAX FUNCTION***')
              continue

          # Building bbox is as follows (least x value, greatest x val, least y value, greatest y val)
          # Forms a box with four corners (bx1,by1), (bx1, by2), (bx2, by1), (bx2, by2)
          by1 = min(arr[0])
          by2 = max(arr[0])
          bx1 = min(arr[1])
          bx2 = max(arr[1])

          for j in range(len(masks)):
            arr = np.nonzero(bin_masks[j])
            my1 = min(arr[0])
            my2 = max(arr[0])
            mx1 = min(arr[1])
            mx2 = max(arr[1])

            
            if (mx1 > bx2):   # Masks are past the building, no more possible intersections.
                nbreaks += 1
                break
            
            if (my2 < by1):
                ncontinue += 1
                continue
            if (my1 > by2):
                ncontinue += 1
                continue
            if (mx2 < bx1):
                ncontinue += 1
                continue
            
            segment = bin_masks[j]
            res = jaccard(building_mask_tensor, segment)

            ### Here we've found a true positive, let's calculate the loss using loss_func and optimizer ###
            if (res >= 0.45):
                true_pos.append(i)
                
                if (not reordered_truth_masks[j].any()):
                    reordered_truth_masks[j] = building_mask_tensor.float()
                
                num_matches = num_matches + 1
                sum_iou = sum_iou + res
                
                break

          ##now, if bounding box is within batch coordinates, we add the mask


        print('------GRADIENT COMPUTATION------')
        masks = masks.float()
        reordered_truth_masks = reordered_truth_masks.float()
        loss = loss_func(reordered_truth_masks, masks)
        # loss.requires_grad = True
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(len(true_pos) > 0):
            IoU_avg = sum_iou/len(true_pos)
        else:
            IoU_avg = 0

        return IoU_avg, loss.item(), len(masks)
