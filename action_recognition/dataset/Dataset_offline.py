import os.path
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import random
import math
from pathlib import Path
from skimage.io import imread

# バッチ単位で乱数をふる


class AlignedDataset(Dataset):
    def __init__(self, config, purpose) -> None:
        # データセットクラスの初期化
        self.config = config
        self.num_frames = config.num_frames
        self.num_intervals = config.num_intervals
        self.class_idx_dict = self.classToIdx(config)

        self.source_list = []
        self.instance_list = []
        self.target_list = []

        if purpose == 'train':
            f = open(config.train_split, 'r')
        else:
            f = open(config.test_split, 'r')

        data = f.readlines()
        del data[100:]

        for i in range(len(data)):
            #
            video_name = data[i].split()[0].replace('.avi', '').rstrip("\n")
            # os.path.join(a, b)->a/b
            source_video_path = os.path.join(config.source, video_name, 'label_map')
            instance_map_path = os.path.join(config.source, video_name, 'instance_map')
            target_video_path = os.path.join(config.target, video_name)

            # ディレクトリまでのパスのリストを生成
            self.source_list.append(source_video_path)
            self.instance_list.append(instance_map_path)
            self.target_list.append(target_video_path)

        # ここを作らないといけない
        # フレームまでのパスが欲しい
        # ラベルマップのついているフレームだけを取り出したい
        self.source_file_list = []
        self.instance_file_list = []
        self.target_file_list = []
        for i in range(len(self.source_list)):

            # sourcefile
            for _, _, fnames in sorted(os.walk(self.source_list[i])):
                # os.walk(a) -> 現在のディレクトリ，内包するディレクトリ一覧，内包するファイル一覧
                # _のところは無視
                source_file_path_list = []

                for fname in fnames:
                    source_file_path = os.path.join(
                        self.source_list[i], fname
                    )
                    source_file_path_list.append(source_file_path)

            # instance
            for _, _, fnames in sorted(os.walk(self.instance_list[i])):
                instance_file_path_list = []

                for fname in fnames:
                    instance_file_path = os.path.join(
                        self.instance_list[i], fname
                    )
                    instance_file_path_list.append(instance_file_path)

            # targetfile
            for _, _, fnames in sorted(os.walk(self.target_list[i])):
                target_file_path_list = []

                for fname in fnames:
                    target_file_path = os.path.join(
                        self.target_list[i], fname
                    )
                    target_file_path_list.append(target_file_path)

            # 16*5=80フレーム以上のファイルのみ，ソートする
            if len(source_file_path_list) > self.num_frames * self.num_intervals:
                source_file_path_list.sort()
                instance_file_path_list.sort()
                target_file_path_list.sort()

                self.source_file_list.append(source_file_path_list)
                self.instance_file_list.append(instance_file_path_list)
                self.target_file_list.append(target_file_path_list)

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

    def make_dataset(self, index, source_list, instance_list, target_list) -> dict:
        # ランダムなindexの画像を取得
        source_file_path_list = source_list[index]
        instance_file_path_list = instance_list[index]
        target_file_path_list = target_list[index]

        # 開始場所をランダムで指定する
        rand_index = random.randint(
            0, len(source_file_path_list) - (self.num_frames * self.num_intervals)
        )

        dict_key = ''
        sampling_target = []
        sampling_source = []
        sampling_instance = []
        path_list = []

        seed = random.randint(0, 2**32)
        # for in range(開始値，最大値，間隔)
        for i in range(
            rand_index,
            rand_index + (self.num_frames * self.num_intervals),
            self.num_intervals
        ):

            # shape:H*W*3が欲しい
            # ----UCFframe-----
            target_numpy = imread(target_file_path_list[i])
            target = torch.from_numpy(target_numpy).permute(2, 0, 1)

            # shape:H*W欲しい
            # ----ラベル画像----
            source_path = source_file_path_list[i]
            label_numpy = imread(source_path)
            source = torch.from_numpy(label_numpy).unsqueeze(0)

            # shape:H*Wが欲しい
            # ----インスタンスマップ----
            # まずnumpyにする
            instance_numpy = imread(instance_file_path_list[i])
            # numpyをテンソルに変換する
            instance = torch.from_numpy(instance_numpy).unsqueeze(0)

            sampling_target.append(target)
            sampling_source.append(source)
            sampling_instance.append(instance)

        path_list.append(str(Path(source_path).parent.parent.name))

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
        target_tensor = torch.stack(sampling_target, dim=0)
        torch.manual_seed(seed)
        target_tensor = transform(target_tensor.float())

        # source_tensor.size():16*1*224*224
        source_tensor = torch.stack(sampling_source, dim=0)
        torch.manual_seed(seed)
        source_tensor = transform(source_tensor)

        # instance_tensor.size():16*1*224*224
        instance_tensor = torch.stack(sampling_instance, dim=0)
        torch.manual_seed(seed)
        instance_tensor = transform(instance_tensor)

        dict_key = str(Path(source_path).parent.parent.parent.name)
        label = self.class_idx_dict[dict_key]

        return {
            'source': source_tensor,
            'instance': instance_tensor,
            'target': target_tensor,
            'labels': label,
            'path': path_list
        }

    def classToIdx(self, args):
        class_list = sorted(
            entry.name for entry in os.scandir(args.source)
            if entry.is_dir())

        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_list)}

        return class_to_idx

    # __getitem__が呼び出された時，index（ランダム引数），source_file_list，instance_file_list，target_file_listをdataにして返す
    def __getitem__(self, index) -> dict:

        data = self.make_dataset(
            index,
            self.source_file_list,
            self.instance_file_list,
            self.target_file_list
        )

        return data

    def __len__(self):
        # 全画像ファイル数を返す
        return len(self.source_file_list)
