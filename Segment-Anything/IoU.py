import numpy as np
import cv2

## takes in a singular ground truth / predicted file pair and calculats IoU (Intersection over union
# intersection over union is in plain english the area of overlap between the 
# predictions and the GT divided by the area of union (total mask coverage)


def IoU(GT, Predictions):
    # Convert pictures to BW
    GT_BW = cv2.threshold(GT, 180, 255, cv2.THRESH_BINARY)[1]
    Preds_BW = cv2.threshold(Predictions, 180, 255, cv2.THRESH_BINARY)[1]
    # Use cv2 bitwise opertors to calculate intersection and union

    intersection = cv2.bitwise_and(GT_BW, Preds_BW)
    cv2.imshow(intersection, "AND")
    union = cv2.bitwise_or(GT_BW, Preds_BW)
    cv2.imshow(union, "OR")

    intersection_non_mask_area = cv2.countNonZero(intersection)
    intersection_area = intersection.size() - intersection_non_mask_area # total minus non mask area

    union_non_mask_area = union.countNonero(union)
    union_area = union.size() - union_non_mask_area

    return intersection_area/union_area
