from transformers import AutoTokenizer, AutoModel
from .coco_category import COCO_category
import torch


def word2vec(opt):
    device = torch.device(
        'cuda:' + str(opt.gpu) if torch.cuda.is_available() else 'cpu'
    )
    category = COCO_category()

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased').to(device)

    vec_dict = {}
    for i in range(len(category)):
        id = category[i]["id"]
        if id >= opt.shuffle_over_category:
            text = category[i]["name"]
            encoded_inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(device)
            output = model(**encoded_inputs)
            vec_dict[id] = output[0][0][1].cpu()

    return vec_dict
