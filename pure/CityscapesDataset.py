import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CityscapesDataset(Dataset):
    def __init__(self, data_path, split='train', img_size=512):
        self.data_path = data_path
        self.split = split
        self.img_size = img_size

        self.img_dir = os.path.join(data_path, 'leftImg8bit', split)
        self.mask_dir = os.path.join(data_path, 'gtFine', split)

        self.img_files = sorted(os.listdir(self.img_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))

        assert len(self.img_files) == len(self.mask_files)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        mask_file = self.mask_files[idx]

        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        # リサイズ
        img = img.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), resample=Image.NEAREST)

        # numpy 配列に変換
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.int64)

        # クラスごとの one-hot マスクに変換
        mask = self.one_hot_encode(mask)

        # 画像の正規化
        img = transforms.ToTensor()(img)
        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        # numpy 配列から Tensor に変換
        mask = torch.from_numpy(mask).long()

        return img, mask

    def one_hot_encode(self, mask):
        num_classes = 19
        shape = mask.shape[:2] + (num_classes,)
        mask_one_hot = np.zeros(shape, dtype=np.float32)
        for i in range(num_classes):
            mask_one_hot[..., i][mask == i] = 1.0
        return mask_one_hot
