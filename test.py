import numpy
import numpy as np
import copy
from cityscape_dataset import *
from ext_transforms import ExtUnNormalize, ExtNormalize

target_classes = [2, 0]

temporary_class_id = 255

acc_iou = 0
for elem in target_classes:
    temp_mask = np.array([[0, 0, 2, 2, ],
                          [0, 0, 2, 2, ],
                          [0, 0, 2, 2, ],
                          [1, 1, 1, 1, ]])
    temp_mask[temp_mask != elem] = temporary_class_id
    temp_mask[temp_mask == elem] = 1
    temp_mask[temp_mask == temporary_class_id] = 0

    temp_pred_mask = np.array([[3, 0, 1, 1, ],
                               [2, 0, 1, 1, ],
                               [2, 0, 2, 2, ],
                               [1, 1, 1, 1, ]])
    temp_pred_mask = temp_pred_mask.astype('uint8')
    # print(np.unique(temp_pred_mask))
    temp_pred_mask[temp_pred_mask == elem] = temporary_class_id
    temp_pred_mask[temp_pred_mask != temporary_class_id] = 1
    temp_pred_mask[temp_pred_mask == temporary_class_id] = 0
    attacked = np.sum(temp_pred_mask * temp_mask)
    original_total = np.sum(temp_mask)
    acc_iou += attacked / original_total
iou = acc_iou / len(target_classes)
print(temp_mask)
print(temp_pred_mask)
print(iou)
