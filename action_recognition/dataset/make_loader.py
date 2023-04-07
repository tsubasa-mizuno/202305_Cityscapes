from . import (
    Dataset_offline,
    Dataset_online,
    Kinetics_Dataset,
    HMDB_Dataset,
    multiview_valDataset
)
from torch.utils.data import DataLoader
import pytorch_lightning as pl



class MyDataModule(pl.LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def train_dataloader(self):
        if self.opt.dataset == 'HMDB':
            train_dataset = HMDB_Dataset.AlignedDataset(self.opt, 'train')
        elif self.opt.dataset == 'UCF':
            if self.opt.online_mode:
                train_dataset = Dataset_online.AlignedDataset(self.opt, 'train')
            else:
                train_dataset = Dataset_offline.AlignedDataset(self.opt, 'train')
        elif self.opt.dataset == 'Kinetics':
            train_dataset = Kinetics_Dataset.AlignedDataset(self.opt, 'train')

        train_loader = DataLoader(
            train_dataset,
            self.opt.train_num_batchs,
            shuffle=True,
            drop_last=True,
            num_workers=self.opt.num_workers
        )

        return train_loader

    def val_dataloader(self):
        if self.opt.dataset == 'HMDB':
            if self.opt.multiview:
                val_dataset = multiview_valDataset.AlignedDataset(self.opt, False)
            else:
                val_dataset = HMDB_Dataset.AlignedDataset(self.opt, 'test')
        elif self.opt.dataset == 'UCF':
            if self.opt.multiview:
                val_dataset = multiview_valDataset.AlignedDataset(self.opt, False)
            else:
                val_dataset = Dataset_online.AlignedDataset(self.opt, 'val')
        elif self.opt.dataset == 'Kinetics':
            val_dataset = multiview_valDataset.AlignedDataset(self.opt, False)

        val_loader = DataLoader(
            val_dataset,
            self.opt.val_num_batchs,
            shuffle=True,
            drop_last=True,
            num_workers=self.opt.num_workers
        )

        return val_loader

    def mimetmics_dataloader(self):
        mimetics_dataset = multiview_valDataset.AlignedDataset(self.opt, True)
        mimetics_loader = DataLoader(
            mimetics_dataset,
            self.opt.val_num_batchs,
            shuffle=True,
            drop_last=True,
            num_workers=self.opt.num_workers
        )

        return mimetics_loader


def make_loader(opt):

    if opt.dataset == 'HMDB':
        train_dataset = HMDB_Dataset.AlignedDataset(opt, 'train')
        if opt.multiview:
            val_dataset = multiview_valDataset.AlignedDataset(opt, False)
        else:
            val_dataset = HMDB_Dataset.AlignedDataset(opt, 'test')

    elif opt.dataset == 'Cityscapes':
        train_dataset = Dataset_offline.AlignedDataset(opt, 'train')
        val_dataset = Dataset_offline.AlignedDataset(opt, 'val')

    elif opt.dataset == 'UCF':
        if opt.online_mode:
            train_dataset = Dataset_online.AlignedDataset(opt, 'train')
        else:
            train_dataset = Dataset_offline.AlignedDataset(opt, 'train')
        if opt.multiview:
            val_dataset = multiview_valDataset.AlignedDataset(opt, False)
        else:
            val_dataset = Dataset_online.AlignedDataset(opt, 'val')

    elif opt.dataset == 'Kinetics':
        train_dataset = Kinetics_Dataset.AlignedDataset(opt, 'train')
        val_dataset = multiview_valDataset.AlignedDataset(opt, False)

        if opt.mimetics:
            mimetics_dataset = multiview_valDataset.AlignedDataset(opt, True)

    train_dataloader = DataLoader(
        train_dataset,
        opt.train_num_batchs,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        opt.val_num_batchs,
        shuffle=True,
        drop_last=True,
        num_workers=opt.num_workers
    )

    if opt.dataset == 'Kinetics' and opt.mimetics:
        mimetics_loader = DataLoader(
            mimetics_dataset,
            opt.val_num_batchs,
            shuffle=True,
            drop_last=True,
            num_workers=opt.num_workers
        )

        return train_dataloader, val_loader, mimetics_loader
    else:
        return train_dataloader, val_loader
