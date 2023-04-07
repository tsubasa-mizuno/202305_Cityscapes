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

        # labelsファイルのパスのリスト
        self.labels_list = []
        # instanceファイルのパスのリスト
        self.instance_list = []
        # imageファイルのパスのリスト
        self.image_list = []

        # purpose == trainの時，trainのパスを指定
        if purpose == 'train':
            self.labels_list = glob.glob(os.path.join(config.gtFine_folder, "train/*/*_gtFine_labelIds.png"))
            self.instance_list = glob.glob(config.gtFine_folder + 'train/*/*_gtFine_instanceIds.png')
            self.image_list = glob.glob(os.path.join(config.image_folder, "train/*/*_leftImg8bit.png"))
        # purpose == valの時，valのパスを指定
        if purpose == 'val':
            self.labels_list = glob.glob(os.path.join(config.gtFine_folder, "val/*/*_gtFine_labelIds.png"))
            self.instance_list = glob.glob(config.gtFine_folder + 'val/*/*_gtFine_instanceIds.png')
            self.image_list = glob.glob(os.path.join(config.image_folder, "val/*/*_leftImg8bit.png"))
        # purpose == testの時，testのパスを指定
        else:
            self.labels_list = glob.glob(os.path.join(config.gtFine_folder, 'test/*/*_gtFine_labelIds.png'))
            self.instance_list = glob.glob(os.path.join(config.gtFine_folder, 'test/*/*_gtFine_instanceIds.png'))
            self.image_list = glob.glob(os.path.join(config.image_folder, 'test/*/*_leftImg8bit.png'))

        # ソートする
        self.labels_list.sort()
        self.instance_list.sort()
        self.image_list.sort()
        # これでlabels，instance，imageのそれぞれの画像までのパス一覧が順番に並んだリストができた

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

    def make_dataset(self, index, labels_list, instance_list, image_list) -> dict:
        # ランダムなindexの画像を取得
        print(index)
        labels_file_path = lanels_list[index]
        instance_file_path = instance_list[index]
        image_file_path = image_list[index]

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
            self.labels_list,
            self.instance_file_list,
            self.image_file_list
        )

        return data

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.labels_list)
