import torch
import numpy as np
from copy import copy
import sys
sys.path.append("..")
from category_sampling import COCO_category


def plot_label(image):
    format = COCO_category()
    list = []
    for i in range(3):
        x = copy(image[0]) + 1
        for category in format:
            np.place(x, x == category["id"], category["color"][i])
        list.append(torch.from_numpy(x / 255))

    source = torch.stack(list, dim=0)

    return source
