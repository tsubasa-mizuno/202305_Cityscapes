import os.path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import glob
import torch
import math
from pathlib import Path
from skimage.io import imread


# バッチ単位で乱数をふる


class AlignedDataset(Dataset):
    def __init__(self, config, mimetics) -> None:
        # データセットクラスの初期化
        self.config = config
        self.num_frames = config.num_frames
        self.num_intervals = config.num_intervals
        self.class_idx_dict = self.classToIdx(config, mimetics)
        self.target_list = []
        if mimetics:
            target_video_path = os.path.join(config.mimetics_path, '*', '*')
            self.target_list = sorted(glob.glob(target_video_path))
        else:
            if config.dataset == 'HMDB':
                target_video_path = os.path.join(config.target, 'test', '*', '*')
                self.target_list = sorted(glob.glob(target_video_path))
            elif config.dataset == 'UCF':
                f = open(config.test_split, 'r')
                data = f.readlines()
                # del data[100:]

                for i in range(len(data)):
                    video_name = data[i].split()[0].replace('.avi', '').rstrip("\n")
                    target_video_path = os.path.join(config.target, video_name)

                    self.target_list.append(target_video_path)
            elif config.dataset == 'Kinetics':
                target_video_path = os.path.join(config.target, 'val', '*', '*')
                self.target_list = sorted(glob.glob(target_video_path))

        self.target_file_list = []
        for i in range(len(self.target_list)):
            for _, _, fnames in sorted(os.walk(self.target_list[i])):
                target_file_path_list = []

                for fname in fnames:
                    target_file_path = os.path.join(
                        self.target_list[i], fname
                    )
                    target_file_path_list.append(target_file_path)

            if len(target_file_path_list) > self.num_frames * self.num_intervals:
                target_file_path_list.sort()

                self.target_file_list.append(target_file_path_list)

    def classToIdx(self, config, mimetics) -> dict:
        path = config.target
        if mimetics:
            path = config.mimetics_path
        else:
            if config.dataset == 'HMDB':
                path = os.path.join(path, 'test')
            elif config.dataset == 'Kinetics':
                path = os.path.join(path, 'val')
        class_list = sorted(
            entry.name for entry in os.scandir(path)
            if entry.is_dir())

        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_list)}

        return class_to_idx

    def short_side(self, w, h, size):
        # https://github.com/facebookresearch/pytorchvideo/blob/a77729992bcf1e43bf5fa507c8dc4517b3d7bc4c/pytorchvideo/transforms/functional.py#L118
        if w < h:
            new_h = int(math.floor((float(h) / w) * size))
            new_w = size
        else:
            new_h = size
            new_w = int(math.floor((float(w) / h) * size))
        return new_w, new_h

    def make_dataset(self, index, target_list) -> dict:
        # ランダムなindexの画像を取得
        target_file_path_list = target_list[index]

        num_frames = len(target_file_path_list) - (self.num_frames * self.num_intervals)
        num_video_start = self.config.num_multis
        stride = math.floor(num_frames / num_video_start)
        start_list = []
        for i in range(num_video_start):
            start_list.append(i * stride)

        dict_key = ''

        multi_target = []

        for start_index in start_list:
            sampling_target = []
            for i in range(start_index, start_index + (self.num_frames * self.num_intervals), self.num_intervals):
                # ----UCFframe-----
                target_path = target_file_path_list[i]
                target_numpy = imread(target_path)
                target = torch.from_numpy(target_numpy).permute(2, 0, 1)

                sampling_target.append(target)

            # target_tensor.size():16*3*H*W
            target_tensor = torch.stack(sampling_target, dim=0)

            for i in range(self.config.num_crops):
                target_tensor = self.random_crop(target_tensor)
                multi_target.append(target_tensor)

        # total_multi*16*3*224*224
        target_tensor = torch.stack(multi_target, dim=0)

        dict_key = str(Path(target_path).parent.parent.name)
        label = self.class_idx_dict[dict_key]

        return {
            'target': target_tensor,
            'labels': label
        }

    def random_crop(self, tensor):
        h, w = self.short_side(tensor.size()[2], tensor.size()[3], 256)
        transform_list = [
            transforms.Resize([h, w], Image.NEAREST),
            transforms.RandomCrop(self.config.crop_size)
        ]
        transform = transforms.Compose(transform_list)
        tensor = transform(tensor.float())

        return tensor

    def __getitem__(self, index) -> dict:

        data = self.make_dataset(
            index,
            self.target_file_list
        )

        return data

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.target_file_list)
