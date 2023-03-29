
from datetime import datetime
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
from os import path
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)


def get_loggers(args):
    exp_name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S:%f')

    comet_logger = CometLogger(
        # api_key, workspace: ~/.comet.config
        # workspace, project_name: ./.comet.config (gitignored)
        save_dir=args.comet_logs,
        experiment_name=exp_name,
        parse_args=True,
    )

    tb_logger = TensorBoardLogger(
        save_dir=args.tf_logs,
        name=exp_name
    )

    ckpt_dir = path.join(args.comet_logs, exp_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor='val_acc',
        mode='max',  # larger acc is better
        save_top_k=2,
        filename='epoch={epoch}-step={step}-val_acc={val_acc:.2f}',
        auto_insert_metric_name=False,
    )

    return (comet_logger, tb_logger), checkpoint_callback
