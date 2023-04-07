import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        default='UCF',
        choices=['UCF', 'HMDB', 'Kinetics']
    )
    # parser.add_argument(
    #     '-l',
    #     '--labels',
    #     type=str,
    #     help='input image to generator'
    # )
    parser.add_argument(
        '-l',
        '--labels',
        type=str,
        required=True,
        help='ground truth'
    )
    parser.add_argument(
        '-im',
        '--image',
        type=str,
        required=True,
        help='ground truth'
    )
    parser.add_argument(
        '-p',
        '--probability',
        type=float,
        default=1,
    )
    parser.add_argument(
        '-e',
        '--num_epochs',
        default=10,
        type=int
    )
    parser.add_argument(
        '--train_num_batchs',
        type=int,
        default=16
    )
    parser.add_argument(
        '--crop_size',
        type=int,
        default=224
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='number of dataloader workders. default 12'
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU No. to be used for model. default 0'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='how many GPUs to be used for model. default 1'
    )

    # dataset_path
    parser.add_argument(
        '--train_gtFine_folder',
        default='/mnt/mizuno/dataset/cityscapes/gtFine_trainvaltest/gtFine/train/aachen',
        type=str
    )
    parser.add_argument(
        '--train_image_folder',
        default='/mnt/mizuno/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train',
        type=str
    )
    parser.add_argument(
        '--test_gtFine_folder',
        default='/mnt/mizuno/dataset/cityscapes/gtFine_trainvaltest/gtFine/test',
        type=str
    )
    parser.add_argument(
        '--test_image_folder',
        default='/mnt/mizuno/dataset/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/test',
        type=str
    )


    # parser.add_argument(
    #     '--num_frames',
    #     type=int,
    #     default=16
    # )
    # parser.add_argument(
    #     '--num_intervals',
    #     type=int,
    #     default=5
    # )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='learning rate. default to 0.0001'
    )
    parser.add_argument(
        '--wd',
        type=float,
        default=5e-5,
        help='weight decay. default to 0.00005'
    )
    parser.add_argument(
        '--keep_path',
        type=str,
        default='image'
    )
    parser.add_argument(
        '--paste',
        action='store_true'
    )
    parser.add_argument(
        '--online_mode',
        action='store_true'
    )
    parser.add_argument(
        '--category_sampling',
        default='Not',
        choices=['Not', 'Random', 'Semantic']
    )
    parser.add_argument(
        '--shift',
        action='store_true'
    )
    parser.add_argument(
        '--shuffle_over_category',
        type=int,
        default=91
    )
    parser.add_argument(
        '--not_keep',
        action='store_true'
    )
    parser.add_argument(
        '--mimetics_path',
        default='/mnt/SSD4TB/sugiura/MimeticsFrame'
    )
    parser.add_argument(
        '--mimetics',
        action='store_true'
    )

    # spade config
    parser.add_argument(
        '--aspect_ratio',
        default=1.0,
        type=int
    )
    parser.add_argument(
        '--checkpoints_dir',
        default='../SegGen/SPADE/checkpoints'
    )
    parser.add_argument(
        '--name',
        default='coco_pretrained'
    )
    parser.add_argument(
        '--which_epoch',
        default='latest'
    )
    parser.add_argument(
        '--contain_dontcare_label',
        action='store_false'
    )
    parser.add_argument(
        '--init_type',
        default='xavier'
    )
    parser.add_argument(
        '--init_variance',
        default=0.02,
        type=int
    )
    parser.add_argument(
        '--isTrain',
        action='store_true'
    )
    parser.add_argument(
        '--label_nc',
        default=182,
        type=int
    )
    parser.add_argument(
        '--semantic_nc',
        default=184,
        type=int
    )
    parser.add_argument(
        '--netG',
        default='spade'
    )
    parser.add_argument(
        '--ngf',
        default=64,
        type=int
    )
    parser.add_argument(
        '--no_instance',
        action='store_true'
    )
    parser.add_argument(
        '--norm_G',
        default='spectralspadesyncbatch3x3'
    )
    parser.add_argument(
        '--num_upsampling_layers',
        default='normal'
    )
    parser.add_argument(
        '--use_vae',
        action='store_true'
    )

    # ----multiview config----
    parser.add_argument(
        '--multiview',
        action='store_true'
    )
    parser.add_argument(
        '--val_num_batchs',
        type=int,
        default=4
    )
    parser.add_argument(
        '--num_multis',
        type=int,
        default=10
    )
    parser.add_argument(
        '--num_crops',
        type=int,
        default=3
    )
    # ------------------------

    parser.add_argument(
        '--ckpt_path', type=str,
        help='path or URL of the checkpoint, or "last" or "hpc".',
        default=None)

    parser.add_argument('--comet_logs', type=str, default='./comet_logs/')
    parser.add_argument('--tf_logs', type=str, default='./tf_logs/')

    args = parser.parse_args()

    return args
