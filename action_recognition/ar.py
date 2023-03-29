from utils import accuracy
import torch
import torch.nn as nn
from x3d_m import X3D_M


class ActionRecognition():
    def __init__(self, opt):
        device = torch.device(
            'cuda:' + str(opt.gpu) if torch.cuda.is_available() else 'cpu'
        )
        self.model_X3D = X3D_M(101, pretrain=True).to(device)
        self.optimizer = torch.optim.Adam(
            self.model_X3D.parameters(),
            lr=opt.lr,
            weight_decay=opt.wd
        )
        self.criterion = nn.CrossEntropyLoss()

    def action_rec(
        self,
        videos,
        labels,
        batches_done,
        pbar_epoch,
        experiment,
        log_loss,
        log_top1,
        log_top5,
        batch_size
    ) -> None:
        self.model_X3D.train()
        self.optimizer.zero_grad()
        outputs = self.model_X3D(videos)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step()

        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
        log_loss.update(loss.item(), batch_size)
        log_top1.update(acc1.item(), batch_size)
        log_top5.update(acc5.item(), batch_size)
        pbar_epoch.set_postfix_str(
            ' | loss={:6.04f}, top1={:6.04f}, top5={:6.04f}'
            .format(
                log_loss.avg,
                log_top1.avg,
                log_top5.avg,
            ))

        experiment.log_metric(
            "train_batch_loss", log_loss.avg, step=batches_done
        )
        experiment.log_metric(
            "train_batch_top1", log_top1.avg, step=batches_done
        )
        experiment.log_metric(
            "train_batch_top5", log_top5.avg, step=batches_done
        )

    def val_action_rec(
        self,
        videos,
        labels,
        pbar_epoch,
        log_loss,
        log_top1,
        log_top5,
        opt
    ) -> None:
        self.model_X3D.eval()

        with torch.no_grad():
            if opt.multiview:
                mean_stack = []
                for i in range(videos.size()[0]):
                    val_outputs = self.model_X3D(videos[i])
                    mean = torch.mean(input=val_outputs, dim=0)
                    mean_stack.append(mean)
                val_outputs = torch.stack(mean_stack, dim=0)
            else:
                val_outputs = self.model_X3D(videos)
            loss = self.criterion(val_outputs, labels)

        acc1, acc5 = accuracy(val_outputs, labels, topk=(1, 5))
        log_loss.update(loss, opt.val_num_batchs)
        log_top1.update(acc1, opt.val_num_batchs)
        log_top5.update(acc5, opt.val_num_batchs)
        pbar_epoch.set_postfix_str(
            ' | loss={:6.04f}, top1={:6.04f}, top5={:6.04f}'
            .format(
                log_loss.avg,
                log_top1.avg,
                log_top5.avg,
            ))
