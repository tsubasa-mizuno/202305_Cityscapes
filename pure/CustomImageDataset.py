import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, write_jpeg import glob
from torch.utils.data import DataLoader
from torchvision import transforms as transforms import numpy as np


class CustomImageDataset(Dataset):
    def __init__(self, train=True):
        self.train = train
        if train:
            self.image_path = glob.glob("./dataset/train/images/*.jpg")
            self.label_path = glob.glob("./dataset/train/labels/*.png")
        else:
            self.image_path = glob.glob("./dataset/val/images/*.jpg") self.label_path = glob.glob("./dataset/val/labels/*.png")
            self.image_path.sort()
            self.label_path.sort()
            self.gray = transforms.Grayscale()
            self.resize = transforms.Resize((320, 640))

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, idx):
        img_path = self.image_path[idx]
        label_path = self.label_path[idx]
        image = read_image(img_path)
        label = self.gray(read_image(label_path))

        # 車以外のラベルを 0 にする
        label[label != 25] = 0
        label[label == 25] = 1

        image = self.resize(image)
        label = self.resize(label)

        # ランダムで左右にフリップする
        if torch.randn(1) > 0:
            image = transforms.functional.hflip(image)
            label = transforms.functional.hflip(label)
        # ランダムで 256x256 をクロッピングする
        i, j, h, w = transforms.RandomCrop.get_params(image, (256, 256))
        image = transforms.functional.crop(image, i, j, h, w)
        label = transforms.functional.crop(label, i, j, h, w)

        label = torch.nn.functional.one_hot(label[0].long(), num_classes=2)
        label = torch.permute(label, (2, 0, 1))
        sample = {"image": image.float(), "label": label.float()}
        return sample

    if __name__ == "__main__":
        bsize = 8
        train_data = CustomImageDataset()
        train_dataloader = DataLoader(train_data, batch_size=bsize, shuffle=True)

        data = next(iter(train_dataloader))
        images = data["image"]
        labels = data["label"]
        for i in range(bsize):
            image = images[i]
            label = labels[i]
            label = torch.stack([label[1], label[1], label[1]])
            print(label.size())
            write_jpeg(image.to(torch.uint8), "image_{0:01d}.jpg".format(i))
            write_jpeg(label.to(torch.uint8) * 255, "label_{0:01d}.jpg".format(i))
