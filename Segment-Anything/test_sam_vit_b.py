from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
# import utils_6521 as utils
import glob


path = glob.glob(r"C:\Users\Micha\Documents\CSE 6521\Datasets\Test_Small_Sample\*.tif")
cv_img = []
for img in path:
    n = cv2.imread(img)
    cv_img.append(n)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

model_type = 'vit_b'

sam = sam_model_registry[model_type](checkpoint="Segment-Anything\checkpoints\sam_vit_b_01ec64.pth")
predictor = SamPredictor(sam)

# image = cv2.imread(r"C:\Users\Micha\Documents\CSE 6521\Datasets\02-Provisional_Test\Inputs\JAX_Tile_000_RGB.tif")
print("this is a test of the program")

mask_generator = SamAutomaticMaskGenerator(sam)

for img in cv_img:
    masks = mask_generator.generate(img)
    print(len(masks))
    print(masks[0].keys())
    plt.figure(figsize=(20,20))
    plt.imshow(img)
    show_anns(masks)
    plt.axis('off')
    plt.show()
    # cv2_imshow(image)

print("reached end of program")
### works for one image, now generate masks for a set of images and store results 



# predictor.set_image(image)
# masks, _, _ = predictor.predict(<input_prompts>)


# run predictions on test dataset in Urban 3D


## Gather predictions and compare to ground truth


## Save info for report
