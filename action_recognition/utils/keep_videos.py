import os
import numpy as np
from torchvision.utils import save_image
from . import plot_label
import torch


def denorm(x):
    """De-normalization"""
    out = (x + 1) / 2
    return out.clamp(0, 1).type(torch.FloatTensor)


def keep_videos(data, dir, opt, index, norm=True):
    if opt.paste:
        if opt.shift:
            if opt.category_sampling == 'Semantic':
                output_dir = os.path.join(
                    opt.keep_path,
                    'paste_shift_sampling',
                    dir
                )
            elif opt.category_sampling == 'Random':
                output_dir = os.path.join(
                    opt.keep_path,
                    'paste_shift',
                    dir
                )
        else:
            if opt.category_sampling == 'Semantic':
                output_dir = os.path.join(
                    opt.keep_path,
                    'paste_sampling',
                    dir
                )
            elif opt.category_sampling == 'Random':
                output_dir = os.path.join(
                    opt.keep_path,
                    'paste',
                    dir
                )
    else:
        output_dir = os.path.join(opt.keep_path, 'not_paste', dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dir == 'source' or dir == 'source_not_shuffle':
        label_list = []
        for i in range(data.size()[0]):
            image = plot_label(np.uint8(data[i].numpy()))
            label_list.append(image)
        data = torch.stack(label_list, dim=0).to(torch.float32)

    if norm:
        data = denorm(data)

    save_image(
        data,
        os.path.join(
            output_dir,
            'Epoch_%03d.png' % (index)
        ),
        nrow=opt.num_frames,
    )
