import torch
import pytorch_lightning as pl
import torch.nn as nn
from x3d_m import X3D_M
import torchvision.transforms as transforms
import random
from utils import (
    accuracy,
    label2image,
    convert_label,
)
from category_sampling import (
    word2vec,
    vec_distance
)
import sys
sys.path.append("../SegGen/seggen")
from segmentation import segmentation
from model_factory import get_segmentor, get_generator


# def noise(opt):
#     return 'real' if random.random() >= opt.probability else 'fake'

def use_real(opt):
    return True if random.random() >= opt.probability else False


class MyLightningModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.model = X3D_M(101, pretrain=True)

        if self.opt.online_mode:
            self.seg_model = get_segmentor("Mask2Former", self.opt.gpu)

        self.spade = get_generator("SPADE", self.opt.gpu, self.opt)

        coco_category_vec = word2vec(self.opt)
        self.category_distance = vec_distance(self.opt, coco_category_vec)
        self.transform = transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.opt.lr,
            weight_decay=self.opt.wd
        )
        return optimizer

    def training_step(self, batch, batch_idx):

        real_video = batch['target']
        real_video = self.transform(real_video / 255)
        action_labels = batch['labels']

        bs = len(real_video)

        videos_data = []

        if use_real(self.opt):
            videos = real_video.permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW
        else:
            with torch.no_grad():
                if self.opt.online_mode:
                    label_data = []
                    instance_data = []
                    for b in range(bs):
                        label, instance = segmentation(
                            self.opt,
                            real_video[b],
                            self.seg_model
                        )
                        label_data.append(label)
                        instance_data.append(instance)
                    seg_label_img = torch.stack(label_data, dim=0)
                    instances = torch.stack(instance_data, dim=0)
                else:
                    seg_label_img = batch['source']
                    instances = batch['instance']

                for b in range(bs):
                    video, source, not_paste = label2image(
                        self.spade.model,
                        convert_label(seg_label_img[b].clone()),
                        instances[b],
                        real_video[b],
                        self.opt,
                        self.category_distance
                    )
                    videos_data.append(video)  # TCHW

                videos = torch.stack(
                    videos_data, dim=0).permute(0, 2, 1, 3, 4)  # BTCHW -> BCTHW

        outputs = self.model(videos)
        loss = self.criterion(outputs, action_labels)
        acc = accuracy(outputs, action_labels)

        self.log_dict(
            {
                'train_loss': loss,
                'train_acc': acc,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            rank_zero_only=True,
            sync_dist=False,
            batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):

        real_video = batch['target']
        real_video = self.transform(real_video / 255)
        action_labels = batch['labels']

        bs = len(real_video)

        if self.opt.multiview:
            videos = real_video.permute(0, 1, 3, 2, 4, 5)
        else:
            videos = real_video.permute(0, 2, 1, 3, 4)

        if self.opt.multiview:
            mean_stack = []
            for b in range(bs):
                val_outputs = self.model(videos[b])
                mean = torch.mean(input=val_outputs, dim=0)
                mean_stack.append(mean)
            val_outputs = torch.stack(mean_stack, dim=0)
        else:
            val_outputs = self.model(videos)
        loss = self.criterion(val_outputs, action_labels)
        acc1, acc5 = accuracy(val_outputs, action_labels, topk=(1, 5))

        self.log_dict(
            {
                'val_loss': loss,
                'val_top1': acc1,
                'val_top5': acc5,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=bs)

        return
