import time
from copy import deepcopy

import numpy as np
import torch
import utils

from .own.impl import iterative_unlearn
from .own.utils import train_phase


class OwnSVD:
    @staticmethod
    def train_iter(data_loaders, model, criterion, optimizer, epoch, args):
        forget_loader = data_loaders["forget"]
        retain_loader = data_loaders["retain"]

        losses = utils.AverageMeter()
        top1 = utils.AverageMeter()

        # switch to train mode
        # model.train()
        train_phase(model, changed_layers_class=["customlinear", "customconv2d"])

        # --------------------------------------------------
        # random set targets for forget loader
        forget_dataset = deepcopy(forget_loader.dataset)

        if hasattr(forget_dataset, 'targets'):
            train_dataset = forget_dataset
        else:
            train_dataset = forget_dataset.dataset

        targets_unlearn = np.random.randint(0, args.num_classes, len(train_dataset.targets))
        targets_unlearn = np.where(
            targets_unlearn == train_dataset.targets,
            np.remainder(targets_unlearn + 1, args.num_classes),
            targets_unlearn
        )
        train_dataset.targets = targets_unlearn
        # --------------------------------------------------

        if args.use_retaining_data:
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, retain_loader.dataset])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        start = time.time()
        for i, (image, target) in enumerate(train_loader):
            if epoch < args.warmup:
                utils.warmup_lr(epoch, i + 1, optimizer, one_epoch_step=len(train_loader), args=args)

            image = image.cuda()
            target = target.cuda()

            output_clean = model(image)
            loss = criterion(output_clean, target)

            assert (
                torch.isfinite(loss).all().item()
            ), f"Loss is NaN or Infinite, get: {loss}"

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output = output_clean.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = utils.accuracy(output.data, target)[0]

            losses.update(loss.item(), image.size(0))
            top1.update(prec1.item(), image.size(0))

            if (i + 1) % args.print_freq == 0:
                end = time.time()
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                    "Time {3:.2f}".format(
                        epoch, i, len(train_loader), end - start, loss=losses, top1=top1
                    )
                )
                start = time.time()

        print("train_accuracy {top1.avg:.3f}".format(top1=top1))

        return top1.avg

    @staticmethod
    def validation_iter(model, val_loader, epoch, args):
        top1 = utils.AverageMeter()

        model.eval()

        start = time.time()
        with torch.no_grad():
            for i, (image, target) in enumerate(val_loader):
                image, target = image.cuda(), target.cuda()
                output = model(image)

                prec1 = utils.accuracy(output.data, target)[0]
                top1.update(prec1.item(), image.size(0))

                if (i + 1) % args.print_freq == 0:
                    end = time.time()
                    print(
                        "Valid epoch: [{0}][{1}/{2}]\t"
                        "Accuracy {top1.val:.3f} ({top1.avg:.3f})\t"
                        "Time {3:.2f}".format(
                            epoch, i, len(val_loader), end - start, top1=top1
                        )
                    )
                    start = time.time()
        print("valid_accuracy {top1.avg:.3f}".format(top1=top1))

        return top1.avg


@iterative_unlearn
def own_svd():
    return OwnSVD
