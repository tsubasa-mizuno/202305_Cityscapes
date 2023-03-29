import random


def random_sampling(label_map, id, panoptic_id, stuff_id, rng):

    other_id = panoptic_id[panoptic_id > 182]

    num_other = len(other_id)
    other_id_replaced = rng.choice(stuff_id[stuff_id != 0], size=num_other)

    id = id[id >= 91]
    id = id[id <= 182]

    for old, new in zip(other_id, other_id_replaced):
        label_map[label_map == old] = new

    for i in id:
        rand = random.randint(0, len(stuff_id) - 1)
        label_map[label_map == i] = stuff_id[rand]

    return label_map
