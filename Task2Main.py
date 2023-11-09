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
import numpy as np
import cv2
'''
def train(
        sam_model: sam_model_registry,
         optimizer: torch.optim,
          loss_func = torch.nn.MSELoss(),
           dataset_loc = '/Datasets/Urban_3D_Challenge/01-Provisional_Train'
           ):
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
        batch_data = model._process_batch(points, orig_size, crop_box, orig_size)
        data.cat(batch_data)
        del batch_data
    model.predictor.reset_image()

    data.to_numpy()
    return data

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

    image = cv2.imread(dataset_loc + r'Inputs/JAX_Tile_004_RGB.tif')
    print(image)

    model_train(model_in_training, image)

    

if __name__ == '__main__':
    main(dataset_loc='Datasets/Urban_3D_Challenge/01-Provisional_Train/')