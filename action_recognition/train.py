
# 警告
import warnings
import torch
import torchvision.transforms as transforms
import random
from tqdm import tqdm
from dataset import make_loader, MyDataModule
# from model_lightning import MyLightningModel
from pytorch_lightning.callbacks import ModelSummary
import pytorch_lightning as pl


from logger import get_loggers

from utils import (
    AverageMeter,   # average_meter.pyのAverageMeterクラス
    cometml,        # cometml.pyのcometmlクラス
    get_args,        # config.pyの中のget_args
    label2image,    # label2im.pyの中のlabel2image
    convert_label,  # convert_label.pyの中のconvert_label
    keep_videos,    # keep_videos.pyの中のkeep_videos
)

# __init__.pyから，word2vecとvec_distanceをimport
from category_sampling import (
    word2vec,
    vec_distance
)
import ar
import sys
sys.path.append("../SegGen/seggen")
# from segmentation import segmentation
# from model_factory import get_segmentor, get_generator

warnings.filterwarnings('ignore')


# parametersを定義
def to_dict(args):
    parameters = {
        'epochs': args.num_epochs,
        'train_batch_size': args.train_num_batchs,
        'val_batch_size': args.val_num_batchs,
        'crop_size': args.crop_size,
        'num_workers': args.num_workers,
        'probability': args.probability,
        'person_paste': args.paste,
        'multiview': args.multiview,
        'num_multi': args.num_multis,
        'category_sampling': args.category_sampling,
        'shift': args.shift,
        'dataset': args.dataset
    }
    return parameters


def noise(args):
    return 'real' if random.random() >= args.probability else 'fake'


def main():
    # argsの値はget_argsで定義
    # get_argsでは，parserによって必要な引数を変える
    args = get_args()

    # dataloaderがKineticsでかつmimetics(見せかけ)の時，mimetics_loaderをmake_loaderに
    if args.dataset == 'Kinetics' and args.mimetics:
        dataloader, val_loader, mimetics_loader = make_loader(args)
    # それ以外の時は，val_loaderをmake_loaderに
    else:
        dataloader, val_loader = make_loader(args)

    # data_moduleは，make_loader.pyのMyDataModuleクラスで定義
    data_module = MyDataModule(args)
    model_lightning = MyLightningModel(args)

    # loggersとcheckpoint_callbackは，logger.pyの中で定義
    # loggers = (comet_logger, tb_logger)
    loggers, checkpoint_callback = get_loggers(args)

    # pytorch-lightning で学習を回す
    trainer = pl.Trainer(
        devices=args.gpus,
        accelerator='gpu',
        strategy='ddp_find_unused_parameters_false' if args.gpus > 1 else None,
        max_epochs=args.num_epochs,
        logger=loggers,
        log_every_n_steps=10,  # default:50
        # accumulate_grad_batches=args.grad_upd,
        # precision=16,
        num_sanity_val_steps=0,
        # #
        # # only for debug
        # #
        # fast_dev_run=True,
        # fast_dev_run=5,
        limit_train_batches=5,
        limit_val_batches=5,
        callbacks=[
            ModelSummary(max_depth=3),
            checkpoint_callback,
        ]
    )

    trainer.fit(
        model=model_lightning,
        datamodule=data_module,
        ckpt_path=args.ckpt_path
    )


# 今回のmain関数はこっち
def main_old():
    args = get_args()
    device = torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu'
    )

    if args.mimetics:
        dataloader, val_loader, mimetics_loader = make_loader(args)
    else:
        dataloader, val_loader = make_loader(args)

    X3D = ar.ActionRecognition(args)

    if args.online_mode:
        seg_model = get_segmentor("Cityscapes", args.gpu)  # input: RGB
        # seg_model = get_segmentor("MaskDINO", gpu_id, "resnet")  # input: RGB
        # seg_model = get_segmentor("Mask-RCNN", gpu_id)  # input: BGR

    # spade = get_generator("SPADE", args.gpu, args)

    # coco_category_vec = word2vec(args)
    # category_distance = vec_distance(args, coco_category_vec)

    experiment = cometml.comet()
    train_log_loss = AverageMeter()
    train_log_top1 = AverageMeter()
    train_log_top5 = AverageMeter()

    val_log_loss = AverageMeter()
    val_log_top1 = AverageMeter()
    val_log_top5 = AverageMeter()

    hyper_params = to_dict(args)
    experiment.log_parameters(hyper_params)

    with tqdm(range(args.num_epochs)) as pbar_epoch:

        for epoch in pbar_epoch:
            print("===> Train:Epoch[{}]".format(epoch + 1))
            with tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                leave=True,
                smoothing=0,
            ) as pbar_batch:

                train_log_loss.reset()
                train_log_top1.reset()
                train_log_top5.reset()

                val_log_loss.reset()
                val_log_top1.reset()
                val_log_top5.reset()

                for batch_num, data in pbar_batch:
                    batches_done = epoch * len(dataloader) + batch_num + 1

                    choice_data = noise(args)

                    image = data['image']
                    transform = transforms.Normalize(
                        (0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5)
                    )
                    image = transform(image / 255)

                    videos_data = []
                    if choice_data == 'real':
                        videos = image.permute(0, 2, 1, 3, 4).to(device)
                    else:
                        with torch.no_grad():
                            if args.online_mode:
                                label_data = []
                                instance_data = []
                                for i in range(len(data['image'])):
                                    label, instance = segmentation(
                                        args,
                                        data['image'][i],
                                        seg_model
                                    )
                                    label_data.append(label)
                                    instance_data.append(instance)
                                data['labels'] = torch.stack(label_data, dim=0)
                                data['instance'] = torch.stack(instance_data, dim=0)

                            for i in range(len(data['labels'])):
                                video, labels, not_paste = label2image(
                                    # spade.model,
                                    # ここでtorch.Size([16, 16, 3, 224, 224])の5次元テンソルが生成
                                    convert_label(data['labels'][i].clone()),
                                    data['instance'][i],
                                    image[i],
                                    args,
                                    # category_distance
                                )

                                keep_videos(video.cpu(), 'gen', args, i,)
                                keep_videos(image[i].cpu(), 'image', args, i)
                                keep_videos(
                                    (convert_label(data['labels'][i].clone()) - 1).cpu(),
                                    'labels_not_shuffle', args, i, norm=False)
                                keep_videos(labels.cpu(), 'labels', args, i, norm=False)
                                keep_videos((labels / 255).cpu(), 'grey', args, i, norm=False)
                                keep_videos(not_paste.cpu(), 'not_paste', args, i)

                                videos_data.append(video.to(device))

                            videos = torch.stack(
                                videos_data, dim=0).permute(0, 2, 1, 3, 4)

                    X3D.action_rec(
                        videos,
                        data['labels'].to(device),
                        batches_done,
                        pbar_epoch,
                        experiment,
                        train_log_loss,
                        train_log_top1,
                        train_log_top5,
                        args.train_num_batchs
                    )

            experiment.log_metric(
                "train_epoch_loss", train_log_loss.avg, step=epoch + 1
            )
            experiment.log_metric(
                "train_epoch_top1", train_log_top1.avg, step=epoch + 1
            )
            experiment.log_metric(
                "train_epoch_top5", train_log_top5.avg, step=epoch + 1
            )

            print("===> Validation:Epoch[{}]".format(epoch + 1))
            with tqdm(
                enumerate(val_loader),
                total=len(val_loader),
                leave=True,
                smoothing=0,
            ) as pbar_batch:

                # offline_mode and multiview -> data['image'].size() = B*total_multis*T*C*H*W
                for batch_num, data in pbar_batch:
                    batches_done = epoch * len(val_loader) + batch_num + 1

                    # B*total_multis*16*3*224*224
                    image = data['image']
                    transform = transforms.Normalize(
                        (0.5, 0.5, 0.5),
                        (0.5, 0.5, 0.5)
                    )
                    image = transform(image / 255)

                    if args.multiview:
                        videos = image.permute(0, 1, 3, 2, 4, 5).to(device)
                    else:
                        videos = image.permute(0, 2, 1, 3, 4).to(device)

                    X3D.val_action_rec(
                        videos,
                        data['labels'].to(device),
                        pbar_epoch,
                        val_log_loss,
                        val_log_top1,
                        val_log_top5,
                        args
                    )

            experiment.log_metric(
                "val_epoch_loss", val_log_loss.avg, step=epoch + 1
            )
            experiment.log_metric(
                "val_epoch_top1", val_log_top1.avg, step=epoch + 1
            )
            experiment.log_metric(
                "val_epoch_top5", val_log_top5.avg, step=epoch + 1
            )

            if args.mimetics:
                print("===> Mimetics Val:Epoch[{}]".format(epoch + 1))
                with tqdm(
                    enumerate(mimetics_loader),
                    total=len(mimetics_loader),
                    leave=True,
                    smoothing=0,
                ) as pbar_batch:

                    # offline_mode and multiview -> data['image'].size() = B*total_multis*T*C*H*W
                    for batch_num, data in pbar_batch:
                        batches_done = epoch * len(mimetics_loader) + batch_num + 1

                        # B*total_multis*16*3*224*224
                        image = data['image']
                        transform = transforms.Normalize(
                            (0.5, 0.5, 0.5),
                            (0.5, 0.5, 0.5)
                        )
                        image = transform(image / 255)

                        if args.multiview:
                            videos = image.permute(0, 1, 3, 2, 4, 5).to(device)
                        else:
                            videos = image.permute(0, 2, 1, 3, 4).to(device)

                        X3D.val_action_rec(
                            videos,
                            data['labels'].to(device),
                            pbar_epoch,
                            val_log_loss,
                            val_log_top1,
                            val_log_top5,
                            args
                        )

                experiment.log_metric(
                    "mimetics_epoch_loss", val_log_loss.avg, step=epoch + 1
                )
                experiment.log_metric(
                    "mimetics_epoch_top1", val_log_top1.avg, step=epoch + 1
                )
                experiment.log_metric(
                    "mimetics_epoch_top5", val_log_top5.avg, step=epoch + 1
                )


if __name__ == '__main__':
    main_old()
