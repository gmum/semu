import copy
import logging
import os
import pickle
import random
import time

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as tvu
import tqdm
import torchvision
import torch.nn as nn
from datasets import (
    all_but_one_class_path_dataset,
    data_transform,
    get_dataset,
    get_forget_dataset,
    inverse_data_transform,
)
from functions import create_class_labels, cycle, get_optimizer
from functions.denoising import generalized_steps_conditional
from functions.losses import loss_registry_conditional
from models.diffusion import Conditional_Model
from models.ema import EMAHelper
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.datasets import ImageFolder

from torch import svd_lowrank as truncated_svd
import math

import svd_utils


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
                np.linspace(
                    beta_start ** 0.5,
                    beta_end ** 0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def save_fim(self):
        args, config = self.args, self.config
        bs = (
            torch.cuda.device_count()
        )  # process 1 sample per GPU, so bs == number of gpus
        fim_dataset = ImageFolder(
            os.path.join(args.ckpt_folder, "class_samples"),
            transform=transforms.ToTensor(),
        )
        fim_loader = DataLoader(
            fim_dataset,
            batch_size=bs,
            num_workers=config.data.num_workers,
            shuffle=True,
        )

        print("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(self.config)
        states = torch.load(
            os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        model.eval()

        # calculate FIM
        fisher_dict = {}
        fisher_dict_temp_list = [{} for _ in range(bs)]

        for name, param in model.named_parameters():
            fisher_dict[name] = param.data.clone().zero_()

            for i in range(bs):
                fisher_dict_temp_list[i][name] = param.data.clone().zero_()

        # calculate Fisher information diagonals
        for step, data in enumerate(
                tqdm.tqdm(fim_loader, desc="Calculating Fisher information matrix")
        ):
            x, c = data
            x, c = x.to(self.device), c.to(self.device)

            b = self.betas
            ts = torch.chunk(torch.arange(0, self.num_timesteps), args.n_chunks)

            for _t in ts:
                for i in range(len(_t)):
                    e = torch.randn_like(x)
                    t = torch.tensor([_t[i]]).expand(bs).to(self.device)

                    # keepdim=True will return loss of shape [bs], so gradients across batch are NOT averaged yet
                    if i == 0:
                        loss = loss_registry_conditional[config.model.type](
                            model, x, t, c, e, b, keepdim=True
                        )
                    else:
                        loss += loss_registry_conditional[config.model.type](
                            model, x, t, c, e, b, keepdim=True
                        )

                # store first-order gradients for each sample separately in temp dictionary
                # for each timestep chunk
                for i in range(bs):
                    model.zero_grad()
                    if i != len(loss) - 1:
                        loss[i].backward(retain_graph=True)
                    else:
                        loss[i].backward()
                    for name, param in model.named_parameters():
                        fisher_dict_temp_list[i][name] += param.grad.data
                del loss

            # after looping through all 1000 time steps, we can now aggregrate each individual sample's gradient and square and average
            for name, param in model.named_parameters():
                for i in range(bs):
                    fisher_dict[name].data += (
                                                      fisher_dict_temp_list[i][name].data ** 2
                                              ) / len(fim_loader.dataset)
                    fisher_dict_temp_list[i][name] = (
                        fisher_dict_temp_list[i][name].clone().zero_()
                    )

            if (step + 1) % config.training.save_freq == 0:
                with open(os.path.join(args.ckpt_folder, "fisher_dict.pkl"), "wb") as f:
                    pickle.dump(fisher_dict, f)

        # save at the end
        with open(os.path.join(args.ckpt_folder, "fisher_dict.pkl"), "wb") as f:
            pickle.dump(fisher_dict, f)

    def train(self):
        args, config = self.args, self.config
        D_train_loader = get_dataset(args, config)
        D_train_iter = cycle(D_train_loader)

        model = Conditional_Model(config)

        optimizer = get_optimizer(self.config, model.parameters())
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        model.train()

        start = time.time()
        for step in range(0, self.config.training.n_iters):

            model.train()
            x, c = next(D_train_iter)
            n = x.size(0)
            x = x.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(
                low=0, high=self.num_timesteps, size=(n // 2 + 1,)
            ).to(self.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            loss = loss_registry_conditional[config.model.type](model, x, t, c, e, b)

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                logging.info(
                    f"step: {step}, loss: {loss.item()}, time: {end - start}"
                )
                start = time.time()

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass
            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, "ckpt.pth"),
                )
                # torch.save(states, os.path.join(self.config.ckpt_dir, "ckpt_latest.pth"))

                test_model = ema_helper.ema_copy(model) if self.config.model.ema else copy.deepcopy(model)
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model

    def train_forget(self):
        args, config = self.args, self.config
        logging.info(
            f"Training diffusion forget with contrastive and EWC. Gamma: {config.training.gamma}, lambda: {config.training.lmbda}"
        )
        D_train_loader = all_but_one_class_path_dataset(
            config,
            os.path.join(args.ckpt_folder, "class_samples"),
            args.label_to_forget,
        )
        D_train_iter = cycle(D_train_loader)

        print("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        optimizer = get_optimizer(config, model.parameters())

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            # model = ema_helper.ema_copy(model_no_ema)
        else:
            ema_helper = None

        with open(os.path.join(args.ckpt_folder, "fisher_dict.pkl"), "rb") as f:
            fisher_dict = pickle.load(f)

        params_mle_dict = {}
        for name, param in model.named_parameters():
            params_mle_dict[name] = param.data.clone()

        label_choices = list(range(config.data.n_classes))
        label_choices.remove(args.label_to_forget)

        for step in range(0, config.training.n_iters):
            model.train()
            x_remember, c_remember = next(D_train_iter)
            x_remember, c_remember = x_remember.to(self.device), c_remember.to(
                self.device
            )
            x_remember = data_transform(config, x_remember)

            n = x_remember.size(0)
            channels = config.data.channels
            img_size = config.data.image_size
            c_forget = (torch.ones(n, dtype=int) * args.label_to_forget).to(self.device)
            x_forget = (
                               torch.rand((n, channels, img_size, img_size), device=self.device) - 0.5
                       ) * 2.0
            e_remember = torch.randn_like(x_remember)
            e_forget = torch.randn_like(x_forget)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            loss = loss_registry_conditional[config.model.type](
                model, x_forget, t, c_forget, e_forget, b, cond_drop_prob=0.0
            ) + config.training.gamma * loss_registry_conditional[config.model.type](
                model, x_remember, t, c_remember, e_remember, b, cond_drop_prob=0.0
            )
            forgetting_loss = loss.item()

            ewc_loss = 0.0
            for name, param in model.named_parameters():
                _loss = (
                        fisher_dict[name].to(self.device)
                        * (param - params_mle_dict[name].to(self.device)) ** 2
                )
                loss += config.training.lmbda * _loss.sum()
                ewc_loss += config.training.lmbda * _loss.sum()

            if (step + 1) % config.training.log_freq == 0:
                logging.info(
                    f"step: {step}, loss: {loss.item()}, forgetting loss: {forgetting_loss}, ewc loss: {ewc_loss}"
                )

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    # epoch,
                    step,
                ]
                if config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(config.ckpt_dir, "ckpt.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model

    def retrain(self):
        args, config = self.args, self.config

        D_remain_loader, _ = get_forget_dataset(
            args, config, args.label_to_forget
        )
        D_remain_iter = cycle(D_remain_loader)

        model = Conditional_Model(config)

        optimizer = get_optimizer(self.config, model.parameters())
        model.to(self.device)
        model = torch.nn.DataParallel(model)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None

        model.train()

        start = time.time()
        for step in range(0, self.config.training.n_iters):
            model.train()
            x, c = next(D_remain_iter)
            # x, c = next(D_train_iter)

            n = x.size(0)
            x = x.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            loss = loss_registry_conditional[config.model.type](model, x, t, c, e, b)

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                logging.info(f"step: {step}, loss: {loss.item()}, time: {end - start}")
                start = time.time()

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass
            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, "ckpt.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                del test_model

    def saliency_unlearn(self):
        args, config = self.args, self.config

        D_remain_loader, D_forget_loader = get_forget_dataset(
            args, config, args.label_to_forget
        )
        D_remain_iter = cycle(D_remain_loader)
        D_forget_iter = cycle(D_forget_loader)

        if args.mask_path:
            mask = torch.load(args.mask_path)
        else:
            mask = None

        print("Loading checkpoints {}".format(args.ckpt_folder))

        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        optimizer = get_optimizer(config, model.parameters())
        criteria = torch.nn.MSELoss()

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            # model = ema_helper.ema_copy(model_no_ema)
        else:
            ema_helper = None

        model.train()
        start = time.time()
        for step in range(0, self.config.training.n_iters):
            model.train()

            # remain stage
            remain_x, remain_c = next(D_remain_iter)
            n = remain_x.size(0)
            remain_x = remain_x.to(self.device)
            remain_x = data_transform(self.config, remain_x)
            e = torch.randn_like(remain_x)
            b = self.betas

            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            remain_loss = loss_registry_conditional[config.model.type](
                model, remain_x, t, remain_c, e, b
            )

            # forget stage
            forget_x, forget_c = next(D_forget_iter)

            n = forget_x.size(0)
            forget_x = forget_x.to(self.device)
            forget_x = data_transform(self.config, forget_x)
            e = torch.randn_like(forget_x)
            b = self.betas

            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            if args.method == "ga":
                forget_loss = -loss_registry_conditional[config.model.type](
                    model, forget_x, t, forget_c, e, b
                )

            else:
                a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                forget_x = forget_x * a.sqrt() + e * (1.0 - a).sqrt()

                output = model(forget_x, t.float(), forget_c, mode="train")

                if args.method == "rl":
                    pseudo_c = torch.full(
                        forget_c.shape,
                        (args.label_to_forget + 1) % 10,
                        device=forget_c.device,
                    )
                    pseudo = model(forget_x, t.float(), pseudo_c, mode="train").detach()
                    forget_loss = criteria(pseudo, output)

            loss = forget_loss + args.alpha * remain_loss

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                logging.info(f"step: {step}, loss: {loss.item()}, time: {end - start}")
                start = time.time()

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            if mask:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad *= mask[name].to(param.grad.device)
            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    os.path.join(self.config.ckpt_dir, "ckpt.pth"),
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                if (step + 1) % args.eval_freq == 0 and args.classifier_evaluation:
                    self.classifier_evaluation(test_model, args.cond_scale)
                    self.classifier_evaluation_ta(test_model, args.cond_scale)
                del test_model

            if (step + 1) % config.training.n_iters == 0:
                if args.sample_fid:
                    test_model = (
                        ema_helper.ema_copy(model)
                        if self.config.model.ema
                        else copy.deepcopy(model)
                    )
                    test_model.eval()
                    self.sample_fid(test_model, args.cond_scale, add_name=args.add_name)
                    del test_model

    def load_ema_model(self):
        model = Conditional_Model(self.config)
        states = torch.load(
            os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            test_model = ema_helper.ema_copy(model)
        else:
            ema_helper = None

        model.eval()
        return model

    def sample(self):
        model = Conditional_Model(self.config)
        states = torch.load(
            os.path.join(self.args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )

        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            test_model = ema_helper.ema_copy(model)
        else:
            ema_helper = None

        model.eval()
        test_model = locals().get("test_model", model)

        if self.args.mode == "sample_fid":
            self.sample_fid(test_model, self.args.cond_scale)
        elif self.args.mode == "sample_classes":
            self.sample_classes(test_model, self.args.cond_scale)
        elif self.args.mode == "visualization":
            self.sample_visualization(
                model, str(self.args.cond_scale), self.args.cond_scale
            )

    def my_sample_fid(self):

        model = torch.load(os.path.join(self.args.ckpt_folder, "model.pth"), map_location=self.device)
        print(model)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        model.eval()

        self.sample_fid(model, self.args.cond_scale)

    def sample_classes(self, model, cond_scale):
        """
        Samples each class from the model. Can be used to calculate FIM, for generative replay
        or for classifier evaluation. Stores samples in "./class_samples/<class_label>".
        """
        config = self.config
        args = self.args
        sample_dir = os.path.join(args.ckpt_folder, "class_samples")
        os.makedirs(sample_dir, exist_ok=True)
        img_id = 0
        classes, _ = create_class_labels(
            args.classes_to_generate, n_classes=config.data.n_classes
        )
        n_samples_per_class = args.n_samples_per_class

        for i in classes:
            os.makedirs(os.path.join(sample_dir, str(i)), exist_ok=True)
            if n_samples_per_class % config.sampling.batch_size == 0:
                n_rounds = n_samples_per_class // config.sampling.batch_size
            else:
                n_rounds = n_samples_per_class // config.sampling.batch_size + 1
            n_left = n_samples_per_class  # tracker on how many samples left to generate

            with torch.no_grad():
                for j in tqdm.tqdm(
                        range(n_rounds),
                        desc=f"Generating image samples for class {i} to use as dataset",
                ):
                    if n_left >= config.sampling.batch_size:
                        n = config.sampling.batch_size
                    else:
                        n = n_left

                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                    c = torch.ones(x.size(0), device=self.device, dtype=int) * int(i)
                    x = self.sample_image(x, model, c, cond_scale)
                    x = inverse_data_transform(config, x)

                    for k in range(n):
                        tvu.save_image(
                            x[k],
                            os.path.join(sample_dir, str(c[k].item()), f"{img_id}.png"),
                            normalize=True,
                        )
                        img_id += 1

                    n_left -= n

    def sample_one_class(self, model, cond_scale, class_label):
        """
        Samples one class only for classifier evaluation.
        """
        config = self.config
        args = self.args
        sample_dir = os.path.join(args.ckpt_folder, "class_" + str(class_label))
        os.makedirs(sample_dir, exist_ok=True)
        img_id = 0
        total_n_samples = 500

        if total_n_samples % config.sampling.batch_size == 0:
            n_rounds = total_n_samples // config.sampling.batch_size
        else:
            n_rounds = total_n_samples // config.sampling.batch_size + 1
        n_left = total_n_samples  # tracker on how many samples left to generate

        with torch.no_grad():
            for j in tqdm.tqdm(
                    range(n_rounds),
                    desc=f"Generating image samples for class {class_label}",
            ):
                if n_left >= config.sampling.batch_size:
                    n = config.sampling.batch_size
                else:
                    n = n_left

                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                c = torch.ones(x.size(0), device=self.device, dtype=int) * class_label
                x = self.sample_image(x, model, c, cond_scale)
                x = inverse_data_transform(config, x)

                for k in range(n):
                    tvu.save_image(
                        x[k], os.path.join(sample_dir, f"{img_id}.png"), normalize=True
                    )
                    img_id += 1

                n_left -= n

    def sample_fid(self, model, cond_scale, add_name=None):
        config = self.config
        args = self.args
        img_id = 0

        classes, excluded_classes = create_class_labels(
            args.classes_to_generate, n_classes=config.data.n_classes
        )
        n_samples_per_class = args.n_samples_per_class

        sample_dir = f"fid_samples_guidance_{args.cond_scale}"
        if excluded_classes:
            excluded_classes_str = "_".join(str(i) for i in excluded_classes)
            sample_dir = f"{sample_dir}_excluded_class_{excluded_classes_str}"

        if add_name is not None:
            sample_dir = f"{sample_dir}/{add_name}"
        sample_dir = os.path.join(args.ckpt_folder, sample_dir)
        os.makedirs(sample_dir, exist_ok=True)

        for i in classes:
            if n_samples_per_class % config.sampling.batch_size == 0:
                n_rounds = n_samples_per_class // config.sampling.batch_size
            else:
                n_rounds = n_samples_per_class // config.sampling.batch_size + 1
            n_left = n_samples_per_class  # tracker on how many samples left to generate

            with torch.no_grad():
                for j in tqdm.tqdm(
                        range(n_rounds),
                        desc=f"Generating image samples for class {i} for FID",
                ):
                    if n_left >= config.sampling.batch_size:
                        n = config.sampling.batch_size
                    else:
                        n = n_left

                    x = torch.randn(
                        n,
                        config.data.channels,
                        config.data.image_size,
                        config.data.image_size,
                        device=self.device,
                    )
                    c = torch.ones(x.size(0), device=self.device, dtype=int) * int(i)
                    x = self.sample_image(x, model, c, cond_scale)
                    x = inverse_data_transform(config, x)

                    for k in range(n):
                        tvu.save_image(
                            x[k],
                            os.path.join(sample_dir, f"{img_id}.png"),
                            normalize=True,
                        )
                        img_id += 1

                    n_left -= n

    def sample_image(self, x, model, c, cond_scale, last=True):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import generalized_steps_conditional

            xs = generalized_steps_conditional(
                x, c, seq, model, self.betas, cond_scale, eta=self.args.eta
            )
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                        np.linspace(
                            0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                        )
                        ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from functions.denoising import ddpm_steps_conditional

            x = ddpm_steps_conditional(x, c, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def sample_visualization(self, model, name, cond_scale):
        config = self.config
        total_n_samples = config.training.visualization_samples
        assert total_n_samples % config.data.n_classes == 0
        n_rounds = (
            total_n_samples // config.sampling.batch_size
            if config.sampling.batch_size < total_n_samples
            else 1
        )

        # esd
        # c = torch.repeat_interleave(torch.arange(config.data.n_classes), total_n_samples//config.data.n_classes)
        c = torch.repeat_interleave(
            torch.arange(config.data.n_classes),
            total_n_samples // config.data.n_classes,
        ).to(self.device)

        c_chunks = torch.chunk(c, n_rounds, dim=0)

        with torch.no_grad():
            all_imgs = []
            for i in tqdm.tqdm(
                    range(n_rounds), desc="Generating image samples for visualization."
            ):
                c = c_chunks[i]
                n = c.size(0)
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model, c, cond_scale)
                x = inverse_data_transform(config, x)

                all_imgs.append(x)

            all_imgs = torch.cat(all_imgs)
            grid = tvu.make_grid(
                all_imgs,
                nrow=total_n_samples // config.data.n_classes,
                normalize=True,
                padding=0,
            )

            try:
                tvu.save_image(
                    grid, os.path.join(self.config.log_dir, f"sample-{name}.png")
                )  # if called during training of base model
            except AttributeError:
                tvu.save_image(
                    grid, os.path.join(self.args.ckpt_folder, f"sample-{name}.png")
                )  # if called from sample.py

    def generate_mask(self):
        args, config = self.args, self.config
        logging.info(
            f"Generating mask of diffusion to achieve gradient sparsity. Gamma: {config.training.gamma}, lambda: {config.training.lmbda}"
        )

        _, D_forget_loader = get_forget_dataset(args, config, args.label_to_forget)

        print("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        optimizer = get_optimizer(config, model.parameters())

        gradients = {}
        for name, param in model.named_parameters():
            gradients[name] = 0

        model.eval()

        for x, forget_c in D_forget_loader:
            n = x.size(0)
            x = x.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            # loss 1
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            x = x * a.sqrt() + e * (1.0 - a).sqrt()
            output = model(
                x, t.float(), forget_c, cond_scale=args.cond_scale, mode="test"
            )

            # https://github.com/clear-nus/selective-amnesia/blob/a7a27ab573ba3be77af9e7aae4a3095da9b136ac/ddpm/models/diffusion.py#L338
            loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradient = param.grad.data.cpu()
                        gradients[name] += gradient

        with torch.no_grad():

            for name in gradients:
                gradients[name] = torch.abs_(gradients[name])

            mask_path = os.path.join('results/cifar10/mask', str(args.label_to_forget))
            os.makedirs(mask_path, exist_ok=True)

            threshold_list = [0.5]
            for i in threshold_list:
                print(i)
                sorted_dict_positions = {}
                hard_dict = {}

                # Concatenate all tensors into a single tensor
                all_elements = - torch.cat(
                    [tensor.flatten() for tensor in gradients.values()]
                )

                # Calculate the threshold index for the top 10% elements
                threshold_index = int(len(all_elements) * i)

                # Calculate positions of all elements
                positions = torch.argsort(all_elements)
                ranks = torch.argsort(positions)

                start_index = 0
                for key, tensor in gradients.items():
                    num_elements = tensor.numel()
                    tensor_ranks = ranks[start_index: start_index + num_elements]

                    sorted_positions = tensor_ranks.reshape(tensor.shape)
                    sorted_dict_positions[key] = sorted_positions

                    # Set the corresponding elements to 1
                    threshold_tensor = torch.zeros_like(tensor_ranks)
                    threshold_tensor[tensor_ranks < threshold_index] = 1
                    threshold_tensor = threshold_tensor.reshape(tensor.shape)
                    hard_dict[key] = threshold_tensor
                    start_index += num_elements

                torch.save(hard_dict, os.path.join(mask_path, f'with_{str(i)}.pt'))

    def classifier_evaluation(self, model, cond_scale):
        """
        Samples one class only for classifier evaluation.
        """
        config = self.config
        args = self.args

        class_label = args.label_to_forget

        total_n_samples = args.evaluation_samples

        if total_n_samples % config.sampling.batch_size == 0:
            n_rounds = total_n_samples // config.sampling.batch_size
        else:
            n_rounds = total_n_samples // config.sampling.batch_size + 1
        n_left = total_n_samples  # tracker on how many samples left to generate

        dataset = []

        with torch.no_grad():
            for _ in tqdm.tqdm(
                    range(n_rounds),
                    desc=f"Generating image samples for class {class_label}",
            ):
                if n_left >= config.sampling.batch_size:
                    n = config.sampling.batch_size
                else:
                    n = n_left

                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )
                c = torch.ones(x.size(0), device=self.device, dtype=int) * class_label
                x = self.sample_image(x, model, c, cond_scale)
                # x = inverse_data_transform(config, x)
                dataset.append(x)

                n_left -= n

        dataset = torch.cat(dataset)
        # dataset = TensorDataset(dataset)
        # loader = DataLoader(dataset, batch_size=config.sampling.batch_size)

        classifier_model_path = f"results/classifier_models/{config.data.dataset.lower()}_resnet34.pth"

        classifier_model = torchvision.models.resnet34(pretrained=False)
        num_ftrs = classifier_model.fc.in_features
        classifier_model.fc = nn.Linear(num_ftrs, 10)

        classifier_model.load_state_dict(
            torch.load(classifier_model_path, map_location=self.device)
        )

        classifier_model = classifier_model.to(self.device)

        entropy_cum_sum = 0
        forgotten_prob_cum_sum = 0
        accuracy_cum_sum = 0
        classifier_model.eval()

        logits = classifier_model(dataset.to(self.device))

        pred = torch.argmax(logits, dim=-1)
        accuracy = (pred == class_label).sum()
        accuracy_cum_sum += accuracy / total_n_samples

        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        entropy = -torch.multiply(probs, log_probs).sum(1)
        avg_entropy = torch.sum(entropy) / total_n_samples
        entropy_cum_sum += avg_entropy.item()
        forgotten_prob_cum_sum += (
            (probs[:, class_label] / total_n_samples).sum().item()
        )

        print(f"Average entropy: {entropy_cum_sum}")
        print(f"Average prob of forgotten class: {forgotten_prob_cum_sum}")
        print(f"Average accuracy of forgotten class: {accuracy_cum_sum}")

    def classifier_evaluation_ta(self, model, cond_scale):
        """
        Samples one class only for classifier evaluation.
        """
        config = self.config
        args = self.args

        class_label = args.label_to_forget

        total_n_samples_per_class = args.evaluation_samples

        if total_n_samples_per_class % config.sampling.batch_size == 0:
            n_rounds = total_n_samples_per_class // config.sampling.batch_size
        else:
            n_rounds = total_n_samples_per_class // config.sampling.batch_size + 1
        n_left = total_n_samples_per_class  # tracker on how many samples left to generate

        classifier_model_path = f"results/classifier_models/{config.data.dataset.lower()}_resnet34.pth"

        classifier_model = torchvision.models.resnet34(pretrained=False)
        num_ftrs = classifier_model.fc.in_features
        classifier_model.fc = nn.Linear(num_ftrs, 10)

        classifier_model.load_state_dict(
            torch.load(classifier_model_path, map_location=self.device)
        )

        classifier_model = classifier_model.to(self.device)

        entropy_cum_sum = 0
        accuracy_cum_sum = 0
        classifier_model.eval()

        dataset = []
        labels = []

        with torch.no_grad():
            for lab in range(10):
                if lab != class_label:
                    total_n_samples_per_class = args.evaluation_samples
                    if total_n_samples_per_class % config.sampling.batch_size == 0:
                        n_rounds = total_n_samples_per_class // config.sampling.batch_size
                    else:
                        n_rounds = total_n_samples_per_class // config.sampling.batch_size + 1
                    n_left = total_n_samples_per_class  # tracker on how many samples left to generate
                    for _ in tqdm.tqdm(
                            range(n_rounds),
                            desc=f"Generating image samples for class {lab}",
                    ):
                        if n_left >= config.sampling.batch_size:
                            n = config.sampling.batch_size
                        else:
                            n = n_left

                        x = torch.randn(
                            n,
                            config.data.channels,
                            config.data.image_size,
                            config.data.image_size,
                            device=self.device,
                        )
                        c = torch.ones(x.size(0), device=self.device, dtype=int) * lab
                        x = self.sample_image(x, model, c, cond_scale)
                        # x = inverse_data_transform(config, x)
                        dataset.append(x)
                        labels.append(torch.ones(x.size(0), device=self.device, dtype=int) * lab)

                        n_left -= n

        dataset = torch.cat(dataset)
        labels = torch.cat(labels)
        # dataset = TensorDataset(dataset)
        # loader = DataLoader(dataset, batch_size=config.sampling.batch_size)

        logits = classifier_model(dataset.to(self.device))

        pred = torch.argmax(logits, dim=-1)
        accuracy = (pred == labels).sum()
        accuracy_cum_sum += accuracy / (total_n_samples_per_class * 9)

        probs = torch.nn.functional.softmax(logits, dim=-1)
        log_probs = torch.log(probs)
        entropy = -torch.multiply(probs, log_probs).sum(1)
        avg_entropy = torch.sum(entropy) / (total_n_samples_per_class * 9)
        entropy_cum_sum += avg_entropy.item()
        # forgotten_prob_cum_sum += (
        #     (probs[:, class_label] / (total_n_samples_per_class*9)).sum().item()
        # )

        print(f"TA - Average entropy: {entropy_cum_sum}")
        # print(f"TA - Average prob of remaining classes: {forgotten_prob_cum_sum}")
        print(f"TA - Average accuracy of remaining classes: {accuracy_cum_sum}")

    def generate_svd2(self):
        args, config = self.args, self.config
        logging.info(
            f"Generating mask of diffusion to achieve gradient sparsity. Gamma: {config.training.gamma}, lambda: {config.training.lmbda}"
        )

        removing_list = ['bias', 'null', 'norm', 'conv_out']

        _, D_forget_loader = get_forget_dataset(args, config, args.label_to_forget)

        print("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        # added
        print("Model:")
        print(model)

        optimizer = get_optimizer(config, model.parameters())

        gradients = {}
        weights = {}
        updated_gradients = {}
        self.args.perpendicular = True

        for name, param in model.named_parameters():
            if all(x not in name for x in removing_list):
                gradients[name] = 0
                if self.args.perpendicular:
                    weights[name] = param.data.clone().detach()
                    updated_gradients[name] = 0

        model.eval()

        for x, forget_c in D_forget_loader:
            n = x.size(0)
            x = x.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            # loss 1
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            x = x * a.sqrt() + e * (1.0 - a).sqrt()
            output = model(
                x, t.float(), forget_c, cond_scale=args.cond_scale, mode="test"
            )

            # https://github.com/clear-nus/selective-amnesia/blob/a7a27ab573ba3be77af9e7aae4a3095da9b136ac/ddpm/models/diffusion.py#L338
            loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if all(x not in name for x in removing_list):
                        if param.grad is not None:
                            # gradient = param.grad.data.cpu()
                            gradient = param.grad.data
                            gradients[name] += gradient

        with torch.no_grad():

            svds_path = os.path.join('results/cifar10/svds', str(args.label_to_forget))
            os.makedirs(svds_path, exist_ok=True)

            if self.args.perpendicular:
                for name in updated_gradients:
                    gradients_flattened = torch.flatten(gradients[name], 1)
                    weights_flattened = torch.flatten(weights[name], 1)
                    scale = torch.matmul(gradients_flattened, weights_flattened.T) / torch.norm(weights_flattened,
                                                                                                dim=1)
                    updated_gradients[name] = gradients[name] - (scale @ weights_flattened).reshape(weights[name].shape)

            for name in gradients:
                fraction = 0.05
                rank = 10
                self.args.niter = 100

                gradient = updated_gradients[name] if self.args.perpendicular else gradients[name]

                gradient_shape = gradient.shape
                if gradient.dim() > 2:
                    gradient = gradient.flatten(1)

                u, s, v = truncated_svd(gradient, q=rank, niter=self.args.niter)

                params = torch.zeros((s.shape[0], s.shape[0]), device=s.device)
                svd_projection = u @ params @ v.T
                svd_projection_reshaped = svd_projection.reshape(gradient_shape)

                # second approach: using the fraction of the parameters

                # num_elements = gradient.numel()
                # approx_rank = int(math.sqrt(num_elements * fraction))
                # print(f"approx_rank = {approx_rank}")
                # u, s, v = truncated_svd(gradient, q=approx_rank, niter=self.args.niter)
                # print(f"u.shape={u.shape}")
                # print(f"s.shape={s.shape}")
                # print(f"v.shape={v.shape}")

                # svd_inverse = u @ torch.diag(s) @ v.T
                # print(f"svd - shape = {svd_inverse.shape}")
                # params = torch.zeros((s.shape[0], s.shape[0]), device=s.device)

                # my_svd_inverse = u @ params @ v.T
                # print(f"my_svd - shape = {my_svd_inverse.shape}")

                # my_svd_reshaped = my_svd_inverse.reshape(gradient_shape)
                # print(f"my_svd_reshaped - shape = {my_svd_reshaped.shape}")

            #     torch.save(hard_dict, os.path.join(svds_path, f'with_{str(i)}.pt'))

    def generate_svd(self):
        args, config = self.args, self.config
        # logging.info(
        # f"Generating mask of diffusion to achieve gradient sparsity. Gamma: {config.training.gamma}, lambda: {config.training.lmbda}"
        # )

        criterion = None

        _, D_forget_loader = get_forget_dataset(args, config, args.label_to_forget)

        print("Loading checkpoints {}".format(args.ckpt_folder))
        model = Conditional_Model(config)
        states = torch.load(
            os.path.join(args.ckpt_folder, "ckpts/ckpt.pth"),
            map_location=self.device,
        )
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        # added
        print("Model:")
        print(model)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")

        optimizer = get_optimizer(config, model.parameters())

        model.eval()

        def compute_gradients_function(x, forget_c):
            optimizer.zero_grad()
            n = x.size(0)
            x = x.to(self.device)
            x = data_transform(self.config, x)
            e = torch.randn_like(x)
            b = self.betas

            # antithetic sampling
            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            # loss 1
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            x = x * a.sqrt() + e * (1.0 - a).sqrt()
            output = model(
                x, t.float(), forget_c, cond_scale=args.cond_scale, mode="test"
            )

            # https://github.com/clear-nus/selective-amnesia/blob/a7a27ab573ba3be77af9e7aae4a3095da9b136ac/ddpm/models/diffusion.py#L338
            loss = (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            gradients_dict = {
                name[: name.rfind(".")]: param.grad.clone()
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            return gradients_dict

        svd_utils.transform_model(model=model, data_loader_unlearn=D_forget_loader, criterion=criterion,
                                  compute_gradients_function=compute_gradients_function,
                                  changed_layers_class=args.changed_layers_class,
                                  explained_variance_ratio=args.explained_variance_ratio,
                                  use_projection_grad=args.use_projection_grad,
                                  attention_only=args.attention_only, others_only=args.others_only,
                                  explained_variance_ratio_attention=args.explained_variance_ratio_attention)

        print("Updated model:")
        print(model)

        changed_layers = '_'.join(args.changed_layers_class)
        model_properties = f"attention_{args.attention}_others_{args.others}_use_projection_grad_{args.use_projection_grad}_explained_variance_ratio_{args.explained_variance_ratio}_changed_layers_class_{changed_layers}"
        svd_path = os.path.join(args.ckpt_folder, "svd_models", str(args.label_to_forget), model_properties)
        os.makedirs(svd_path, exist_ok=True)

        model_path = os.path.join(svd_path, "model.pth")
        print(f"Saving model to {model_path}")
        torch.save(model, model_path)

    def semu_unlearn(self):
        args, config = self.args, self.config

        if args.lr is not None:
            config.optim.lr = args.lr

        D_remain_loader, D_forget_loader = get_forget_dataset(
            args, config, args.label_to_forget
        )
        D_remain_iter = cycle(D_remain_loader)
        D_forget_iter = cycle(D_forget_loader)

        changed_layers = '_'.join(args.changed_layers_class)

        model_properties = f"attention_{args.attention}_others_{args.others}_use_projection_grad_{args.use_projection_grad}_explained_variance_ratio_{args.explained_variance_ratio}_changed_layers_class_{changed_layers}"
        svd_path = os.path.join(args.ckpt_folder, "svd_models", str(args.label_to_forget), model_properties)
        os.makedirs(svd_path, exist_ok=True)
        model_path = os.path.join(svd_path, "model.pth")

        unlearn_model_properties = model_properties + f"_method_{args.method}_lr_{config.optim.lr}"
        unlearn_path = os.path.join(args.ckpt_folder, "unlearn_models", str(args.label_to_forget),
                                    unlearn_model_properties)
        os.makedirs(unlearn_path, exist_ok=True)
        unlearn_model_path = os.path.join(unlearn_path, "model.pth")

        if os.path.exists(model_path) is False:
            print("The model file does not exist. Computing SVDs projections.")
            self.generate_svd()

        print("Loading checkpoints {}".format(model_path))

        model = torch.load(model_path, map_location=self.device)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        print("Loaded model:")
        print(model)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")

        for param in model.parameters():
            param.requires_grad = False

        if args.changed_layers_class is not None:
            new_layers_classes = ['custom' + str(lclass) for lclass in args.changed_layers_class]
        else:
            new_layers_classes = None
        svd_utils.set_requires_grad(model, changed_layers_class=new_layers_classes)

        # Filter parameters with requires_grad
        parameters_to_optimize = [param for param in model.parameters() if param.requires_grad]
        total_params = sum(p.numel() for p in parameters_to_optimize)

        num_comps = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                key = name[: name.rfind(".")]
                num = eval(f"model.{svd_utils.transform_text_layer(key)}.n_imp_comps")
                num_comps.append(torch.flatten(num))

        total_s_params = sum(num ** 2 for num in torch.flatten(torch.cat(num_comps)))
        print("Total trainable parameters - S matrices:", total_s_params.item())

        print("All layers with requires_grad=True:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

        optimizer = get_optimizer(config, parameters_to_optimize)

        criteria = torch.nn.MSELoss()

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            # model = ema_helper.ema_copy(model_no_ema)
        else:
            ema_helper = None

        model.train()
        if args.changed_layers_class is not None or args.changed_layers_name is not None:
            svd_utils.train_phase(
                model,
                changed_layers_class=new_layers_classes,
                changed_layers_name=args.changed_layers_name,
            )
        start = time.time()
        for step in range(0, self.config.training.n_iters):
            model.train()
            if args.changed_layers_class is not None or args.changed_layers_name is not None:
                svd_utils.train_phase(
                    model,
                    changed_layers_class=new_layers_classes,
                    changed_layers_name=args.changed_layers_name,
                )

            # forget stage
            forget_x, forget_c = next(D_forget_iter)

            n = forget_x.size(0)
            forget_x = forget_x.to(self.device)
            forget_x = data_transform(self.config, forget_x)
            e = torch.randn_like(forget_x)
            b = self.betas

            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            if args.method == "ga":
                forget_loss = -loss_registry_conditional[config.model.type](
                    model, forget_x, t, forget_c, e, b
                )

            else:
                a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                forget_x = forget_x * a.sqrt() + e * (1.0 - a).sqrt()

                output = model(forget_x, t.float(), forget_c, mode="train")

                if args.method == "rl":
                    pseudo_c = torch.full(
                        forget_c.shape,
                        (args.label_to_forget + 1) % 10,
                        device=forget_c.device,
                    )
                    pseudo = model(forget_x, t.float(), pseudo_c, mode="train").detach()
                    forget_loss = criteria(pseudo, output)
                elif args.method == "rl++":
                    random_label = torch.randint(0, args.num_classes, torch.Size([]))

                    if random_label == args.label_to_forget:
                        random_label = (random_label + 1) % args.num_classes

                    pseudo_c = torch.full(
                        forget_c.shape,
                        random_label,
                        device=forget_c.device,
                    )
                    pseudo = model(forget_x, t.float(), pseudo_c, mode="train").detach()
                    forget_loss = criteria(pseudo, output)

            loss = forget_loss

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                logging.info(f"step: {step}, loss: {loss.item()}, time: {end - start}")
                start = time.time()

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            svd_utils.clip_grad(model=model)

            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    unlearn_model_path,
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)
                if (step + 1) % args.eval_freq == 0 and args.classifier_evaluation:
                    self.classifier_evaluation(test_model, args.cond_scale)
                    self.classifier_evaluation_ta(test_model, args.cond_scale)
                del test_model

            if (step + 1) % config.training.n_iters == 0:
                if args.sample_fid:
                    test_model = (
                        ema_helper.ema_copy(model)
                        if self.config.model.ema
                        else copy.deepcopy(model)
                    )
                    test_model.eval()
                    self.sample_fid(test_model, args.cond_scale, add_name=args.add_name)

                    del test_model

    def semu_unlearn_retrain(self):
        args, config = self.args, self.config

        if args.lr is not None:
            config.optim.lr = args.lr

        if args.subset_size is not None:
            print(f"Using equal sizes of forgeting and remaining subsets: {args.subset_size}")
            D_remain_loader, D_forget_loader = get_forget_dataset(
                args, config, args.label_to_forget, args.subset_size)
        else:
            D_remain_loader, D_forget_loader = get_forget_dataset(
                args, config, args.label_to_forget
            )
        D_remain_iter = cycle(D_remain_loader)
        D_forget_iter = cycle(D_forget_loader)

        changed_layers = '_'.join(args.changed_layers_class)

        model_properties = f"attention_{args.attention}_others_{args.others}_use_projection_grad_{args.use_projection_grad}_explained_variance_ratio_{args.explained_variance_ratio}_changed_layers_class_{changed_layers}"
        svd_path = os.path.join(args.ckpt_folder, "svd_models", str(args.label_to_forget), model_properties)
        os.makedirs(svd_path, exist_ok=True)
        model_path = os.path.join(svd_path, "model.pth")

        unlearn_model_properties = model_properties + f"_method_{args.method}_alpha_{args.alpha}_lr_{config.optim.lr}"
        if args.subset_size is not None:
            unlearn_path = os.path.join(args.ckpt_folder, "unlearn_subset_retrain_models", str(args.label_to_forget),
                                        unlearn_model_properties)
            unlearn_model_properties += f"_subset_size_{args.subset_size}"
        else:
            unlearn_path = os.path.join(args.ckpt_folder, "unlearn_retrain_models", str(args.label_to_forget),
                                        unlearn_model_properties)
        os.makedirs(unlearn_path, exist_ok=True)
        unlearn_model_path = os.path.join(unlearn_path, "model.pth")

        if os.path.exists(model_path) is False:
            print("The model file does not exist. Computing SVDs projections.")
            self.generate_svd()

        print("Loading checkpoints {}".format(model_path))

        model = torch.load(model_path, map_location=self.device)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)

        print("Loaded model:")
        print(model)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")

        for param in model.parameters():
            param.requires_grad = False

        if args.changed_layers_class is not None:
            new_layers_classes = ['custom' + str(lclass) for lclass in args.changed_layers_class]
        else:
            new_layers_classes = None
        svd_utils.set_requires_grad(model, changed_layers_class=new_layers_classes)

        # Filter parameters with requires_grad
        parameters_to_optimize = [param for param in model.parameters() if param.requires_grad]
        total_params = sum(p.numel() for p in parameters_to_optimize)

        num_comps = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                key = name[: name.rfind(".")]
                num = eval(f"model.{svd_utils.transform_text_layer(key)}.n_imp_comps")
                num_comps.append(torch.flatten(num))

        total_s_params = sum(num ** 2 for num in torch.flatten(torch.cat(num_comps)))
        print("Total trainable parameters - S matrices:", total_s_params.item())

        print("All layers with requires_grad=True:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

        optimizer = get_optimizer(config, parameters_to_optimize)

        criteria = torch.nn.MSELoss()

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            # model = ema_helper.ema_copy(model_no_ema)
        else:
            ema_helper = None

        model.train()
        if args.changed_layers_class is not None or args.changed_layers_name is not None:
            svd_utils.train_phase(
                model,
                changed_layers_class=new_layers_classes,
                changed_layers_name=args.changed_layers_name,
            )
        start = time.time()
        for step in range(0, self.config.training.n_iters):
            model.train()
            if args.changed_layers_class is not None or args.changed_layers_name is not None:
                svd_utils.train_phase(
                    model,
                    changed_layers_class=new_layers_classes,
                    changed_layers_name=args.changed_layers_name,
                )

            # remain stage
            remain_x, remain_c = next(D_remain_iter)
            n = remain_x.size(0)
            remain_x = remain_x.to(self.device)
            remain_x = data_transform(self.config, remain_x)
            e = torch.randn_like(remain_x)
            b = self.betas

            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
            remain_loss = loss_registry_conditional[config.model.type](
                model, remain_x, t, remain_c, e, b
            )

            # forget stage
            forget_x, forget_c = next(D_forget_iter)

            n = forget_x.size(0)
            forget_x = forget_x.to(self.device)
            forget_x = data_transform(self.config, forget_x)
            e = torch.randn_like(forget_x)
            b = self.betas

            t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(
                self.device
            )
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]

            if args.method == "ga":
                forget_loss = -loss_registry_conditional[config.model.type](
                    model, forget_x, t, forget_c, e, b
                )

            else:
                a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
                forget_x = forget_x * a.sqrt() + e * (1.0 - a).sqrt()

                output = model(forget_x, t.float(), forget_c, mode="train")

                if args.method == "rl":
                    pseudo_c = torch.full(
                        forget_c.shape,
                        (args.label_to_forget + 1) % 10,
                        device=forget_c.device,
                    )
                    pseudo = model(forget_x, t.float(), pseudo_c, mode="train").detach()
                    forget_loss = criteria(pseudo, output)
                elif args.method == "rl++":
                    random_label = torch.randint(0, args.num_classes, torch.Size([]))

                    if random_label == args.label_to_forget:
                        random_label = (random_label + 1) % args.num_classes

                    pseudo_c = torch.full(
                        forget_c.shape,
                        random_label,
                        device=forget_c.device,
                    )
                    pseudo = model(forget_x, t.float(), pseudo_c, mode="train").detach()
                    forget_loss = criteria(pseudo, output)

            loss = forget_loss + args.alpha * remain_loss

            if (step + 1) % self.config.training.log_freq == 0:
                end = time.time()
                logging.info(f"step: {step}, loss: {loss.item()}, time: {end - start}")
                start = time.time()

            optimizer.zero_grad()
            loss.backward()

            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass

            svd_utils.clip_grad(model=model)

            optimizer.step()

            if self.config.model.ema:
                ema_helper.update(model)

            if (step + 1) % self.config.training.snapshot_freq == 0:
                states = [
                    model.state_dict(),
                    optimizer.state_dict(),
                    step,
                ]
                if self.config.model.ema:
                    states.append(ema_helper.state_dict())

                torch.save(
                    states,
                    unlearn_model_path,
                )

                test_model = (
                    ema_helper.ema_copy(model)
                    if self.config.model.ema
                    else copy.deepcopy(model)
                )
                test_model.eval()
                self.sample_visualization(test_model, step, args.cond_scale)

                if (step + 1) % args.eval_freq == 0 and args.classifier_evaluation:
                    self.classifier_evaluation(test_model, args.cond_scale)
                    self.classifier_evaluation_ta(test_model, args.cond_scale)
                del test_model

            if (step + 1) % config.training.n_iters == 0:
                if args.sample_fid:
                    test_model = (
                        ema_helper.ema_copy(model)
                        if self.config.model.ema
                        else copy.deepcopy(model)
                    )
                    test_model.eval()
                    self.sample_fid(test_model, args.cond_scale, add_name=args.add_name)

                    del test_model