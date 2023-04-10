import torch
# import random
import numpy as np
from numpy.random import default_rng
from person_paste import person_paste
# from detectron2.data import MetadataCatalog
from category_sampling import (
    semantic_sampling,
    random_sampling
)

rng = default_rng()


def label2image(generator, label_map, instance_map, target, opt, category_distance):
    id = torch.unique(label_map).cpu().numpy()

    metadata = MetadataCatalog.get('coco_2017_train_panoptic_separated')
    stuff_data = metadata.stuff_dataset_id_to_contiguous_id
    panoptic_id = np.array(list(stuff_data.keys()))
    stuff_id = panoptic_id[panoptic_id <= 182]

    if opt.category_sampling == 'Semantic':
        label_map = semantic_sampling(label_map, opt, id, category_distance)
    elif opt.category_sampling == 'Random':
        label_map = random_sampling(label_map, id, panoptic_id, stuff_id, rng)
    else:
        merged_id = id[id > 182]
        for i in merged_id:
            new_0 = rng.choice(stuff_id[stuff_id != 0])
            label_map[label_map == i] = new_0

    new_0 = rng.choice(stuff_id[stuff_id != 0])
    label_map[label_map == 0] = new_0

    label = label_map - 1

    input_dict = {
        # BCHW, float32, necessary but not used for inference (None or scalar 0 may be ok)
        'image': target,

        # [B,1,224,224], (converetd to edge (edge white, bg black ))
        'instance': instance_map.float(),

        # [B,1,224,224], (converted to long in the model), from 0 to 182
        'label': label.float(),
    }
    with torch.no_grad():
        gen = generator(input_dict, mode='inference')

    not_paste = gen.clone()

    if opt.paste:
        gen = person_paste(label.detach(), target.detach(), gen.detach())

    return gen, label, not_paste
