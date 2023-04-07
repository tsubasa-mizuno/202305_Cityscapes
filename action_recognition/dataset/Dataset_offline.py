import os.path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import random
import math
import glob
from pathlib import Path
from skimage.io import imread

# バッチ単位で乱数をふる


class AlignedDataset(Dataset):
    def __init__(self, config, purpose) -> None:
        # データセットクラスの初期化
        self.config = config
        self.num_frames = config.num_frames  # 16フレーム
        self.num_intervals = config.num_intervals   # 間隔5

        # labelsファイルのリスト
        self.labels_list = []
        # instanceファイルのリスト
        self.instance_list = []
        # imageファイルのリスト
        self.image_list = []

        # trainがTrueの時，trainのパスを指定
        # 大元のpathをargsで指定
        # 大元以下は結合
        if purpose == 'train':
            self.labels_list = glob.glob(config.train_labels_folder)
            self.instance_list = glob.glob(config.train_instance_forder)
            self.image_list = glob.glob(config.train_image_forder)
        # trainがFalseの時，testのパスを指定
        else:
            self.labels_list = glob.glob(config.test_labels_folder)
            self.instance_list = glob.glob(config.test_instance_forder)
            self.image_list = glob.glob(config.test_image_forder)

        print(self.labels_list)

        # ソートする
        self.image_files.sort()
        self.labels_files.sort()

        # ここを作らないといけない
        # フレームまでのパスが欲しい
        # ラベルマップのついているフレームだけを取り出したい

        self.image_file_list = []
        self.labels_file_list = []

    # 短い方をsizeの値に合わせるように，アスペクト比を保ったままリサイズする
    def short_side(self, w, h, size):
        # https://github.com/facebookresearch/pytorchvideo/blob/a77729992bcf1e43bf5fa507c8dc4517b3d7bc4c/pytorchvideo/transforms/functional.py#L118
        if w < h:
            new_h = int(math.floor((float(h) / w) * size))
            new_w = size
        else:
            new_h = size
            new_w = int(math.floor((float(w) / h) * size))
        return new_w, new_h

    def make_dataset(self, index, image_list, labels_list) -> dict:
        # ランダムなindexの画像を取得
        image_file_path = image_list[index]
        labels_file_path = labels_list[index]
        # target_file_path = target_list[index]

        # 開始場所をランダムで指定する
        rand_index = random.randint(
            0, len(image_file_path)
        )

        dict_key = ''
        # sampling_target = []
        sampling_image = []
        sampling_labels = []
        path_list = []

        seed = random.randint(0, 2**32)
        # for in range(開始値，最大値，間隔)
        for i in range(
            rand_index,
            rand_index + len(image_file_path) - rand_index,
            1
        ):

            # shape:H*W*3が欲しい
            # ----UCFframe-----
            # target_numpy = imread(target_file_path_list[i])
            # target = torch.from_numpy(target_numpy).permute(2, 0, 1)

            # shape:H*W欲しい
            # ----image----
            image_path = image_file_path[i]
            label_numpy = imread(image_path)
            image = torch.from_numpy(label_numpy).unsqueeze(0)

            # shape:H*Wが欲しい
            # ----インスタンスマップ----
            # まずnumpyにする
            labels_numpy = imread(labels_file_path[i])
            # numpyをテンソルに変換する
            labels = torch.from_numpy(labels_numpy).unsqueeze(0)

            # sampling_target.append(target)
            sampling_image.append(image)
            sampling_labels.append(labels)

        path_list.append(str(Path(image_path).parent.parent.name))

        # リサイズされた画像の縦と横の長さをリスト化する
        # target.size()[1]：縦の長さ，target.size()[2]：横の長さ
        h, w = self.short_side(target.size()[1], target.size()[2], 256)
        transform_list = [
            transforms.Resize([h, w], Image.NEAREST),
            transforms.RandomCrop(self.config.crop_size)
        ]

        # transform.Compose：複数のTransformを連続して行うTransform
        transform = transforms.Compose(transform_list)

        # ここでは16*□*224*224になっているが，3*224*224に変換する
        # target_tensor.size():16*3*224*224
        # torch.stack：新しいdimを作成し，そそのdimに沿ってテンソルを連結
        # target_tensor = torch.stack(sampling_target, dim=0)
        # torch.manual_seed(seed)
        # target_tensor = transform(target_tensor.float())

        # # source_tensor.size():16*1*224*224
        # source_tensor = torch.stack(sampling_sourse, dim=0)
        # torch.manual_seed(seed)
        # source_tensor = transform(source_tensor)

        # image_tensor.size():3*224*224にしたい

        # torch.stack：sampling_imageをdim=0の方向に連結
        image_tensor = torch.stack(sampling_image, dim=0)
        torch.manual_seed(seed)
        image_tensor = transform(image_tensor)

        # # instance_tensor.size():16*1*224*224
        # instance_tensor = torch.stack(sampling_instance, dim=0)
        # torch.manual_seed(seed)
        # instance_tensor = transform(instance_tensor)

        # labels_tensor.size():3*224*224にしたい
        labels_tensor = torch.stack(sampling_labels, dim=0)
        torch.manual_seed(seed)
        labels_tensor = transform(labels_tensor)

        # dict_key = str(Path(image_path).parent.parent.parent.name)
        # label = self.class_idx_dict[dict_key]

        return {
            # 'source': source_tensor,
            # 'instance': instance_tensor,
            # 'target': target_tensor,
            # 'labels': label,
            # 'path': path_list
            'image': image_tensor,
            'labels': labels_tensor
        }

    # クラス名とデータセット内の対応するインデックスを対応づけるメソッド
    # def classToIdx(self, args):
    #     class_list = sorted(
    #         entry.name for entry in os.scandir(args.image)
    #         if entry.is_dir())

    #     class_to_idx = {cls_name: i for i, cls_name in enumerate(class_list)}

    #     return class_to_idx

    # __getitem__が呼び出された時，index（ランダム引数），source_file_list，instance_file_list，target_file_listをdataにして返す
    def __getitem__(self, index) -> dict:

        data = self.make_dataset(
            index,
            self.image_file_list,
            self.labels_file_list,
            # self.target_file_list
        )

        return data

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.image_file_list)
