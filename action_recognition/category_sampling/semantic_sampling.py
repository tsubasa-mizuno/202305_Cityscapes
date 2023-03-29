import torch
import random
import numpy as np


def semantic_sampling(label_map, opt, id, category_distance):
    id = id[id >= opt.shuffle_over_category]

    for i in id:
        distance_dict = category_distance[i].copy()
        id_list = list(distance_dict.keys())
        distance_list = list(distance_dict.values())

        distance_np = np.array(distance_list)
        np.exp(distance_np)
        sum_distance = np.sum(distance_np)

        distance_ratio = (distance_np / sum_distance).tolist()
        convert_id = np.random.choice(id_list, 1, distance_ratio)

        label_map[label_map == i] = convert_id[0]

        # max_value = max(distance_list)
        # max_index = distance_list.index(max_value)
        # max_key = id_list[max_index]
        # label_map[label_map == i] = max_key

    return label_map
