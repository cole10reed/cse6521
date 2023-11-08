import segment_anything as sa
from segment_anything import sam_model_registry
import torch
import cv2
'''
def train(
        sam_model: sam_model_registry,
         optimizer: torch.optim,
          loss_func = torch.nn.MSELoss(),
           dataset_loc = '/Datasets/Urban_3D_Challenge/01-Provisional_Train'
           ):
    '''


def main(
        sam_checkpoint ='Segment-Anything/checkpoints/sam_vit_h_4b8939.pth',
         gpu_device = 'cuda',
          model_type = 'vit_h',
           dataset_loc = '/Datasets/Urban_3D_Challenge/01-Provisional_Train/'
           ):
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to(device = gpu_device)

    optimizer = torch.optim.Adam(sam.mask_decoder.parameters())
    loss_func = torch.nn.MSELoss()

    image = cv2.imread(dataset_loc + r'Inputs/JAX_Tile_004_RGB.tif')

    with torch.no_grad():
        image_embedding = sam.image_encoder(image)
        # sparse, dense = sam.prompt_encoder(points=, boxes=, masks=) # TODO: understand the parameters here

if __name__ == '__main__':
    main(dataset_loc='/Datasets/Urban_3D_Challenge/01_Provisional_Train/')