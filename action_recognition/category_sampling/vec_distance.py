from scipy.spatial import distance


def vec_distance(opt, category_vec):
    id_distance = {}

    keys = list(category_vec.keys())
    for key in keys:
        distance_dict = {}
        for other_key in keys:
            if key != other_key and other_key <= 182:
                dist = distance.cosine(category_vec[key].detach().numpy(), category_vec[other_key].detach().numpy())
                distance_dict[other_key] = dist

        id_distance[key] = distance_dict

    # id_min_distance = {id:dict{id:distance}, id:dict{id:distance}, ...}
    return id_distance
