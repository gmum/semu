import time

import numpy as np
import torch
from .transform_model import transform_model, set_requires_grad, transform_text_layer


class EarlyExit:
    def __init__(self, patience=5, min_delta=0.01):
        """
        Initializes the early exit procedure.
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :param min_delta: Minimum change in the monitored value to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        """
        Checks the early stopping condition.
        :param current_score: The current value of the monitored metric (e.g., validation accuracy or loss).
        """
        if self.best_score is None:
            self.best_score = current_score
        elif current_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0



def _iterative_unlearn_impl(unlearn_obj):
    def _wrapped(data_loaders, model, criterion, args):
        decreasing_lr = list(map(int, args.decreasing_lr.split(",")))

        start = time.time()
        transform_model(
            model,
            data_loaders["forget"],
            criterion,
            ["linear", "conv2d"],
            getattr(args, "explained_variance_ratio", None),
            use_projection_grad=args.use_projection_grad,
        )
        print("Transform model duration: {:.4f}".format(time.time() - start))
        set_requires_grad(model, changed_layers_class=["customlinear", "customconv2d"])

        # Display layer for which we calculate gradients
        print("\nLayers for which calculate gradients:", end="\n" + "=" * 50 + "\n")
        for name, param in model.named_parameters():
            if param.requires_grad:
                key = name[: name.rfind(".")]
                layer = eval(f"model.{transform_text_layer(key)}")
                print(name, layer.weight.shape, layer.__class__.__name__, sep=",  ")
        print("=" * 75, end="\n\n")

        # Filter parameters with requires_grad
        parameters_to_optimize = [param for param in model.parameters() if param.requires_grad]
        total_params = sum(p.numel() for p in parameters_to_optimize)
        print("Total trainable parameters:", total_params)

        optimizer = torch.optim.SGD(
            parameters_to_optimize,
            args.unlearn_lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        if args.imagenet_arch and args.unlearn == "retrain":
            lambda0 = (
                lambda cur_iter: (cur_iter + 1) / args.warmup
                if cur_iter < args.warmup
                else (
                    0.5
                    * (
                        1.0
                        + np.cos(
                            np.pi
                            * (
                                (cur_iter - args.warmup)
                                / (args.unlearn_epochs - args.warmup)
                            )
                        )
                    )
                )
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=decreasing_lr, gamma=0.1
            )  # 0.1 is fixed
        if args.rewind_epoch != 0:
            # learning rate rewinding
            for _ in range(args.rewind_epoch):
                scheduler.step()

        early_exit = None
        if args.early_exit:
            early_exit = EarlyExit(patience=args.early_exit_patience, min_delta=args.early_exit_min_delta)

        for epoch in range(0, args.unlearn_epochs):
            start_time = time.time()

            print(
                "Epoch #{}, Learning rate: {}".format(
                    epoch, optimizer.state_dict()["param_groups"][0]["lr"]
                )
            )

            unlearn_obj.train_iter(data_loaders, model, criterion, optimizer, epoch, args)
            scheduler.step()

            if early_exit is not None:
                val_forget = unlearn_obj.validation_iter(model, data_loaders["forget"], epoch, args)
                val_retain = unlearn_obj.validation_iter(model, data_loaders["retain"], epoch, args)

                # Check for early stopping condition
                early_exit((100 - val_forget + val_retain) / 2)
                if early_exit.early_stop:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

            print("one epoch duration:{:.4f}".format(time.time() - start_time))

    return _wrapped


def iterative_unlearn(func):
    """usage:

    @iterative_unlearn

    def func(data_loaders, model, criterion, optimizer, epoch, args)"""
    return _iterative_unlearn_impl(func())
