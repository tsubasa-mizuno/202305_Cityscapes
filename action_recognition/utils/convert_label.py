import torch
import numpy as np
import sys
sys.path.append("..")
from category_sampling import COCO_category


def convert_label(label):
    '''
    0(unlabeled class)以外のカテゴリを詰めずにしたもので返す．
    '''
    category = COCO_category()
    id = torch.unique(label).cpu().numpy()
    if 0 in id:
        id = np.delete(id, 0)

    for i in id:
        new_id = category[i - 1]['id']
        label[label == i] = new_id

    return label
