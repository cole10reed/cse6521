from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
from torchvision import transforms
from torchmetrics import JaccardIndex
import utils_6521 as utils
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from ImageClassifier import ImageClassifier


'''
What will be the training data?
 - Truth buildings (images with just the bbox of the building?)

What will be the testing data?

'''

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
        # if ('Found' in ann.keys()):
        if ann['Found']:
            color_mask = np.concatenate([(0, 0, 1), [0.35]])
        else:
            color_mask = np.concatenate([(1, 0, 0), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



def train_step(classifier, image, label, optimizer, loss_func):
    # print(f'train_step Image type: {type(image)}')
    
    predicted = classifier(image)
    difference = predicted - label
    # print(f'Predicted: {predicted}')
    # print(f'Label: {label}')
    # print(f'Difference: {difference}')
    loss = loss_func(predicted, torch.Tensor([label]))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return predicted, difference.item()


def main(
        sam_checkpoint ='Segment-Anything/checkpoints/sam_vit_h_4b8939.pth',
         gpu_device = 'cuda',
          model_type = 'vit_h',
           dataset_loc = 'Datasets/Urban_3D_Challenge/01-Provisional_Train/'
        ):
    # input_image = cv2.imread(dataset_loc + r'Inputs/JAX_Tile_052_RGB.tif')
    # print('Image type =', type(input_image))
    # train_image = Image.open(dataset_loc + r'Inputs/JAX_Tile_052_RGB.tif')
    # truth_image = utils.get_truth_image(dataset_loc + r'GT/JAX_Tile_052_GTI.tif', 2048, 2048)
    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)  # TODO: change checkpoint to the fine-tuned model for validation.
    # sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=32)

    classifier = ImageClassifier()

    jaccard = JaccardIndex(task='binary')  # This performs the IoU calculation.
    optimizer = torch.optim.Adam(classifier.parameters())
    loss_func = torch.nn.MSELoss()
    
    # resize = transforms.Compose([transforms.Resize(size=(128, 128))])
    resize_dim = (128, 128)

    input_images = utils.get_input_files(dataset_loc)
    truth_images = utils.get_truth_files(dataset_loc)

    input_list = list()
    truth_list = list()
    for i in input_images:
        input_list.append(str(i))
        print(str(i))
    for i in truth_images:
        truth_list.append(str(i))

    sum_iou = 0
    sum_error = 0
    correct_preds = 0
    incorrect_preds = 0
    
    true_pos = list()
        
    for k in range(len(input_list)):
        input_image = cv2.imread(input_list[k])
        truth_image = utils.get_truth_image(truth_list[k], 2048, 2048)

        masks = mask_generator.generate(input_image)
        masks = sorted(masks, key=(lambda x: x['bbox']))
        num_masks = len(masks)
        print('Num masks: ', num_masks)

        num_buildings = truth_image.max()
        bldg = None


        # This loop crops the image based on building bbox, then resizes the cropped image to 128x128.
        for i in range(num_buildings + 1):
            if i == 0:
                continue
            building_mask = np.where(truth_image==i, 1, 0)  # Create a mask specific to each building ID.
            building_mask_tensor = torch.from_numpy(building_mask)  # Turn the mask into a PyTorch tensor for the Jaccard calculations.
            arr = np.nonzero(building_mask)
            # print(arr)
            print(f'Building {i}')
            # Building bbox is as follows (least x value, greatest x val, least y value, greatest y val)
            # Forms a box with four corners (bx1,by1), (bx1, by2), (bx2, by1), (bx2, by2)
            if len(arr[0]) == 0:
                break
            if len(arr[1]) == 0:
                break
            by1 = min(arr[0]).item()
            by2 = max(arr[0]).item()
            bx1 = min(arr[1]).item()
            bx2 = max(arr[1]).item()
            
            # print(f'bx1={bx1}, by1={by1}, bx2={bx2}, by2={by2}')
            # print(f'Type of bx1 is {type(bx1)}')
            # print(f'bx1 = {bx1}')

            # bldg = resize(input_image.crop((bx1, by1, bx2, by2)))
            

            for j in range(len(masks)):
                x,y,h,w = masks[j]['bbox']
                # print(f'Mask {j}')
                if (x > bx2):   # Masks are past the building, no more possible intersections.
                    break
                
                if (y+h < by1):
                    continue
                if (y > by2):
                    continue
                if (x+w < bx1):
                    continue
                
                segment = torch.from_numpy(masks[j]['segmentation'])
                res = jaccard(building_mask_tensor, segment)

                ### Here we've found a true positive, let's calculate the loss using loss_func and optimizer ###
                if (res >= 0.45):
                    true_pos.append(i)
                    sum_iou = sum_iou + res
                    bldg = cv2.resize(input_image[y:y+h, x:x+w], resize_dim)
                    bldg = torch.from_numpy(bldg).float().transpose(1,2).transpose(0,1)
                    
                    # Train the image classifier with a true building mask (label=1).
                    
                    pred, err = train_step(classifier=classifier, image=bldg, label=10, optimizer=optimizer, loss_func=loss_func)
                    sum_error += err

                    
                    print(f'Predicted: {pred}')
                    print(f'Label: 1')
                    print(f'Difference: {err}')
                    
                    if pred.item() >= 5:
                        masks[j]['Found'] = True
                        correct_preds += 1
                    else:
                        masks[j]['Found'] = False
                        incorrect_preds += 1
                    # del(masks[j])
                    break

            # Now all masks that were matched have been deleted.
        for i in range(len(masks)):
            # Train on remaining masks that are not buildings (label=0).
            if ('Found' in masks[i].keys()):
                continue
            x,y,h,w = masks[i]['bbox']
            bldg = cv2.resize(input_image[y:y+h, x:x+w], resize_dim)
            bldg = torch.from_numpy(bldg).float().transpose(1,2).transpose(0,1)
                    
            pred, err = train_step(classifier=classifier, image=bldg, label=0, optimizer=optimizer, loss_func=loss_func)
            sum_error += err
            
            print(f'Predicted: {pred}')
            print(f'Label: 0')
            print(f'Difference: {err}')

            if pred.item() < 5:
                masks[i]['Found'] = True
                correct_preds += 1
            else:
                masks[i]['Found'] = False
                incorrect_preds += 1
            # loss = loss_func(predictions, labels)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
    print('Total Error:', sum_error)
    print('Num masks:', num_masks)
    print('True Positives:', len(true_pos))
    print('Correct Preds:', correct_preds)
    print('Incorrect Preds:', incorrect_preds)
    torch.save(classifier.state_dict(), f"image_classifier/tuned_1.pth")
        

    
    
    # image = image.transpose(2, 0, 1)
    # print(input_image.shape)
    # print(resize(input_image).size)

    # plt.figure(figsize=(20,20))
    # for i in range(num_buildings):
    #     plt.imshow(bldgs[i])
    # plt.imshow(bldg)
    # plt.imshow(input_image)
    # show_anns(masks)
    # plt.axis('off')
    # plt.savefig(fname='test_128pps')
    # plt.show()


    return

if __name__ == '__main__':
    main()