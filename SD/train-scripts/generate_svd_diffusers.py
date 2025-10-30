import argparse
import os

import copy
from time import sleep

import torch
from dataset import setup_forget_data, setup_forget_nsfw_data, setup_model, setup_forget_data_mock, setup_forget_nsfw_data_mock
from tqdm import tqdm
from convertModels import savemodelDiffusers
import svd_utils_diffusers

import pandas as pd
import torch
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UNet2DConditionModel,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer
from esd_utils import StableDiffuser


def save_svd_model(
    model,
    name,
    classes,
    device="cpu",
    save_compvis=True,
):
    # SAVE MODEL
    folder_path = f"models/svd_models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if classes is not None:
        path = f"{folder_path}/{name}_unlearn_classes_{classes}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    if save_compvis:
        torch.save(model.state_dict(), path)
    return path


def generate_svd(
    classes,
    c_guidance,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    device,
    image_size=512,
    num_timesteps=1000,
    return_model=False,
    args=None
):

    train_dl, descriptions = setup_forget_data(classes, batch_size, image_size)

    nsteps = 50

    # diffuser = StableDiffuser(scheduler='DDIM').to(device)
    diffuser = StableDiffuser().to(device)
    diffuser.eval()

    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(diffuser.unet.parameters(), lr=lr)

    unet_params = [name for name, param in diffuser.unet.named_parameters()]

    print("Model params:")
    print("Model diffusion params:")
    print(unet_params)
    print(diffuser.unet)

    gradients_dict = {}
    for name, param in diffuser.unet.named_parameters():
        gradients_dict[name] = 0

    
    # with torch.no_grad():
    #     scaling_factor = 0.18215
    #     latents = vae.encode(image).latent_dist.sample().detach().cpu() * scaling_factor

    noise_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True, steps_offset=1)

    # COMPUTING GRADIENTS DICT
    with tqdm(total=len(train_dl)) as time:
        for i, (images, labels) in enumerate(train_dl):
            optimizer.zero_grad()

            prompts = [descriptions[label] for label in labels]
            forget_batch = images.to(device)

            forget_input = diffuser.encode(forget_batch)

            #TODO - check if this is correct!
            text_embeddings = diffuser.get_text_embeddings(prompts, n_imgs=1)

            t = torch.randint(
                0, diffuser.scheduler.config.num_train_timesteps, (forget_input.shape[0],), device=device
            ).long()
            noise = torch.randn_like(forget_input, device=device)

            forget_noisy = diffuser.add_batch_noises(forget_input, noise, t)

            # TODO: guidance scale???

            preds = diffuser.predict_noise_mine(t, forget_noisy, text_embeddings, guidance_scale=c_guidance)

            loss = -criteria(noise, preds)

            loss.backward()

            with torch.no_grad():
                for name, param in diffuser.unet.named_parameters():
                    if param.grad is not None:
                        gradients_dict[name] += param.grad.clone()
            
            time.set_description("Batch %i" % i)
            time.set_postfix(loss=loss.item() / batch_size)
            sleep(0.1)
            time.update(1)

    all_num_comps = svd_utils_diffusers.transform_model(model=diffuser.unet, gradients_dict=gradients_dict, 
                                changed_layers_class=args.changed_layers_class, 
                                explained_variance_ratio=args.explained_variance_ratio, 
                                use_projection_grad=args.use_projection_grad, 
                                attention_only=args.attention_only, 
                                others_only=args.others_only, 
                                explained_variance_ratio_attention=args.explained_variance_ratio_attention)


    print(f"Updated model with SVD layers:")
    print(diffuser.unet)

    optimizer.zero_grad()
    diffuser.zero_grad()

    # trainable_params = sum([value ** 2 for key, value in all_num_comps.items()])

    trainable_params = all_num_comps

    del optimizer, loss, train_dl, descriptions
    if return_model:
        return diffuser, trainable_params


def generate_nsfw_svd(
    c_guidance,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    diffusers_config_path,
    device,
    image_size=512,
    num_timesteps=1000,
    return_model=False,
    args=None
):

    # train_dl, descriptions = setup_forget_data(classes, batch_size, image_size)

    # MODEL TRAINING SETUP
    train_dl, _ = setup_forget_nsfw_data(batch_size, image_size)

    word_nude = "a photo of a nude person"

    diffuser = StableDiffuser().to(device)
    diffuser.eval()

    criteria = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(diffuser.unet.parameters(), lr=lr)

    unet_params = [name for name, param in diffuser.unet.named_parameters()]

    print("Model params:")
    print("Model diffusion params:")
    print(unet_params)
    print(diffuser.unet)

    gradients_dict = {}
    for name, param in diffuser.unet.named_parameters():
        gradients_dict[name] = 0

    noise_scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000, skip_prk_steps=True, steps_offset=1)

    # COMPUTING GRADIENTS DICT
    with tqdm(total=len(train_dl)) as time:
        for i, images in enumerate(train_dl):
            optimizer.zero_grad()

            prompts = [word_nude] * batch_size
            forget_batch = images.to(device)

            forget_input = diffuser.encode(forget_batch)

            text_embeddings = diffuser.get_text_embeddings(prompts, n_imgs=1)

            t = torch.randint(
                0, diffuser.scheduler.config.num_train_timesteps, (forget_input.shape[0],), device=device
            ).long()
            noise = torch.randn_like(forget_input, device=device)

            forget_noisy = diffuser.add_batch_noises(forget_input, noise, t)

            preds = diffuser.predict_noise_mine(t, forget_noisy, text_embeddings, guidance_scale=c_guidance)

            loss = -criteria(noise, preds)

            loss.backward()

            with torch.no_grad():
                for name, param in diffuser.unet.named_parameters():
                    if param.grad is not None:
                        gradients_dict[name] += param.grad.clone()
            
            time.set_description("Batch %i" % i)
            time.set_postfix(loss=loss.item() / batch_size)
            sleep(0.1)
            time.update(1)

    all_num_comps = svd_utils_diffusers.transform_model(model=diffuser.unet, gradients_dict=gradients_dict, 
                                changed_layers_class=args.changed_layers_class, 
                                explained_variance_ratio=args.explained_variance_ratio, 
                                use_projection_grad=args.use_projection_grad, 
                                attention_only=args.attention_only, 
                                others_only=args.others_only, 
                                explained_variance_ratio_attention=args.explained_variance_ratio_attention)


    print(f"Updated model with SVD layers:")
    print(diffuser.unet)

    optimizer.zero_grad()
    diffuser.zero_grad()

    # trainable_params = sum([value ** 2 for key, value in all_num_comps.items()])

    trainable_params = all_num_comps

    del optimizer, loss, train_dl
    if return_model:
        return diffuser, trainable_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )

    parser.add_argument(
        "--classes",
        help="class corresponding to concept to erase",
        type=str,
        required=False,
        default="6",
    )
    parser.add_argument(
        "--c_guidance",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=7.5,
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=1
    )
    parser.add_argument(
        "--lr",
        help="learning rate used to train",
        type=float,
        required=False,
        default=1e-5,
    )
    parser.add_argument(
        "--ckpt_path",
        help="ckpt path for stable diffusion v1-4",
        type=str,
        required=False,
        default="models/ldm/stable-diffusion-v1/sd-v1-4-full-ema.ckpt",
    )
    parser.add_argument(
        "--config_path",
        help="config path for stable diffusion v1-4 inference",
        type=str,
        required=False,
        default="configs/stable-diffusion/v1-inference.yaml",
    )
    parser.add_argument(
        "--diffusers_config_path",
        help="diffusers unet config json path",
        type=str,
        required=False,
        default="diffusers_unet_config.json",
    )
    parser.add_argument(
        "--device",
        help="cuda devices to train on",
        type=str,
        required=False,
        default="4",
    )
    parser.add_argument(
        "--image_size",
        help="image size used to train",
        type=int,
        required=False,
        default=512,
    )
    parser.add_argument(
        "--num_timesteps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--nsfw", help="class or nsfw", type=bool, required=False, default=False
    )
    parser.add_argument(
        "--explained_variance_ratio", type=float, default=0.95, help="explained variance ratio"
    )
    parser.add_argument(
        "--explained_variance_ratio_attention", type=float, default=1.0, help="explained variance ratio"
    )
    parser.add_argument(
        "--attention_only", default=False, action='store_true', help="whether unlearn attention layers only"
    )
    parser.add_argument(
        "--others_only", default=False, action='store_true', help="whether unlearn other layers only"
    )
    parser.add_argument(
        "--use_projection_grad", default=False, action='store_true', help="whether use projection gradient"
    )
    parser.add_argument(
        "--changed_layers_class", type=str, default=['linear', 'conv2d'], nargs='*', help="changed layers class"
    )
    parser.add_argument(
        "--changed_layers_name", type=str, default=None, nargs='*', help="changed layers name"
    )
    parser.add_argument(
        "--attention", type=bool, default=True, help="if unlearn in attention layers"
    )
    parser.add_argument(
        "--others", type=bool, default=True, help="if unlearn in attention layers"
    )
    args = parser.parse_args()

    if args.attention and not args.others:
        args.attention_only = True

    if args.others and not args.attention:
        args.others_only = True

    # classes = [int(d) for d in args.classes.split(',')]
    classes = int(args.classes)
    c_guidance = args.c_guidance
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    config_path = args.config_path
    ckpt_path = args.ckpt_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    num_timesteps = args.num_timesteps
    args.explained_variance_ratio_attention = args.explained_variance_ratio

    if args.nsfw:
        generate_nsfw_svd(
            c_guidance,
            batch_size,
            epochs,
            lr,
            config_path,
            ckpt_path,
            diffusers_config_path,
            device,
            image_size,
            num_timesteps,
            False,
            args
        )
    else:
        generate_svd(
            classes,
            c_guidance,
            batch_size,
            epochs,
            lr,
            config_path,
            ckpt_path,
            diffusers_config_path,
            device,
            image_size,
            num_timesteps,
            False,
            args
        )