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
import matplotlib.pyplot as plt


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

    masks = model_train(model_in_training, image)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig(fname='test_task2')
    plt.show()

    

if __name__ == '__main__':
    main(dataset_loc='Datasets/Urban_3D_Challenge/01-Provisional_Train/')