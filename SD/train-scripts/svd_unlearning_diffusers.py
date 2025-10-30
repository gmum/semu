import argparse
import os
from time import sleep
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from dataset import setup_forget_data, setup_model, setup_remain_data, setup_forget_data_mock
from diffusers import LMSDiscreteScheduler
from tqdm import tqdm
from generate_svd_diffusers import generate_svd
from svd_utils_diffusers import clip_grad, transform_text_layer, CustomLinear, CustomConv2d, set_requires_grad, train_phase
from generate_images_svd import generate_images, generate_images_grid
from esd_diffusers import StableDiffuser

CLASS_LABELS = ["tench", "english springer", "cassette player", "chain saw", "church", "french horn", "garbage truck", "gas pump", "golf ball", "parachute"]

def backward_hook(module, grad_input, grad_output):
    if grad_input is None or grad_output is None:
        print(f"GRADIENT PROBLEM!!!")
        print(f"Layer: {module.__class__.__name__}")
        print(f"Grad Input: {grad_input}")
        print(f"Grad Output: {grad_output}")
    

def grad_hook(grad):
    print("Gradient:", grad)
    return grad

def check_grad_flow(named_parameters):
    for name, param in named_parameters:
        if param.requires_grad and param.grad is None:
            print(f"No gradient for {name}.")
        elif param.requires_grad:
            print(f"Gradient for {name}: {param.grad.abs().mean():.4f}")

def svd_unlearning(
    class_to_forget,
    train_method,
    alpha,
    batch_size,
    epochs,
    lr,
    config_path,
    ckpt_path,
    mask_path,
    diffusers_config_path,
    device,
    image_size=512,
    ddim_steps=50,
    args=None
):
    # MODEL TRAINING SETUP

    diffuser, trainable_params = generate_svd(classes=class_to_forget, c_guidance=args.c_guidance, batch_size=batch_size, 
                                              epochs=None, lr=lr, config_path=config_path, ckpt_path=ckpt_path, diffusers_config_path=None, 
                                              device=device, image_size=image_size, num_timesteps=1000, return_model=True, args=args)

    # print(f"Trainable parameters: {trainable_params}")

    # diffuser = StableDiffuser().to(device)

    diffuser.to(device)

    prompts_path = "prompts/imagenette.csv"
    model_name = f"compvis-cl-class_{str(class_to_forget)}-method_{train_method}-alpha_{alpha}-epoch_{epochs}-lr_{lr}-explained_variance_ratio_{args.explained_variance_ratio}-use_projection_grad_{args.use_projection_grad}-retrain_{args.retrain}-retrain_method_{args.retrain_method}"

    for name, param in diffuser.unet.named_parameters():
        param = param.requires_grad_(False)

    new_layers_classes = ['custom'+str(lclass) for lclass in args.changed_layers_class]

    set_requires_grad(
        diffuser.unet,
        changed_layers_class=new_layers_classes,
        changed_layers_name=[],
    )

    criteria = torch.nn.MSELoss()

    if args.retrain:
        remain_dl, descriptions = setup_remain_data(class_to_forget, batch_size, image_size)
    else:
        descriptions = [f"an image of a {label}" for label in CLASS_LABELS]
    forget_dl, _ = setup_forget_data(class_to_forget, batch_size, image_size)
    
    

    # set model to train
    diffuser.eval()
    diffuser.unet.train()
    losses = []

    # choose parameters to train based on train_method

    parameters = []
    names = []
    for name, param in diffuser.unet.named_parameters():
        # train only x attention layers
        if train_method == "xattn":
            if "attn2" in name:
                if name.endswith("weight"):
                    _module = f"diffuser.unet.{transform_text_layer(name[:-7])}"
                    _module = eval(_module)
                    _module_type = type(_module)
                    if _module_type is CustomLinear or _module_type is CustomConv2d:
                        parameters.append(param)
                        names.append(name)
        # train all layers
        if train_method == "full":
            if name.endswith("weight"):
                _module = f"diffuser.unet.{transform_text_layer(name[:-7])}"
                _module = eval(_module)
                _module_type = type(_module)
                if _module_type is CustomLinear or _module_type is CustomConv2d:
                    parameters.append(param)
                    names.append(name)

    parameters_to_optimize = [param for name, param in diffuser.unet.named_parameters() if param.requires_grad]
    param_names = [name for name, param in diffuser.unet.named_parameters() if param.requires_grad]
    
    optimizer = torch.optim.Adam(parameters, lr=lr)

    # TRAINING CODE
    #TODO - TRAIN PHASE!!!
    for epoch in range(epochs):
        diffuser.eval()
        diffuser.unet.train()
        diffuser.set_scheduler_timesteps(args.num_inference_steps)
        with tqdm(total=len(forget_dl)) as time:

            for i, (images, labels) in enumerate(forget_dl):
                optimizer.zero_grad()
                diffuser.zero_grad()

                train_phase(
                    diffuser.unet,
                    changed_layers_class=new_layers_classes,
                    changed_layers_name=args.changed_layers_name,
                )
                set_requires_grad(
                diffuser.unet,
                changed_layers_class=new_layers_classes,
                changed_layers_name=[],
               )

                forget_images, forget_labels = next(iter(forget_dl))

                forget_prompts = [descriptions[label] for label in forget_labels]

                if args.retrain_method == "rl":
                    ps = [(int(class_to_forget) + 1) % 10 for label in forget_labels]
                elif args.retrain_method == "rl++":
                    ps = [random.randint(0, 9) for label in forget_labels]
                    ps = [(int(class_to_forget) + 1) % 10 if p == class_to_forget else p for p in ps]
                else:
                    None
                
                pseudo_prompts = [descriptions[p] for p in ps]
                   
                if args.retrain:
                    remain_images, remain_labels = next(iter(remain_dl))

                    remain_prompts = [descriptions[label] for label in remain_labels]
                    remain_images = remain_images.to(device)

                    remain_text_embeddings = diffuser.get_single_text_embeddings(remain_prompts, n_imgs=1)

                    remain_input = diffuser.encode(remain_images)

                    t = torch.randint(0, 
                    diffuser.scheduler.config.num_train_timesteps, 
                    (remain_input.shape[0],), device=device).long()

                    noise = torch.randn_like(remain_input, device=device)

                    remain_noisy = diffuser.add_batch_noises(remain_input, noise, t)

                    remain_out = diffuser.predict_out_noise(t, remain_noisy, remain_text_embeddings)

                    remain_loss = criteria(remain_out, noise)


                # forget stage
                forget_images = forget_images.to(device)

                forget_input = diffuser.encode(forget_images)

                forget_text_embeddings = diffuser.get_text_embeddings(forget_prompts, neg_prompts=pseudo_prompts, n_imgs=1)

                t = torch.randint(0, 
                diffuser.scheduler.config.num_train_timesteps, 
                (forget_input.shape[0],), device=device).long()

                noise = torch.randn_like(forget_input, device=device)

                forget_noisy = diffuser.add_batch_noises(forget_input, noise, t)

                forget_out, pseudo_out = diffuser.predict_outs(t, forget_noisy, forget_text_embeddings)

                forget_loss = criteria(forget_out, pseudo_out)

                # total loss
                loss = forget_loss 
                if args.retrain:
                    loss += alpha * remain_loss

                loss.backward()
                losses.append(loss.item() / batch_size)

                clip_grad(diffuser.unet)

                optimizer.step()
                time.set_description("Epoch %i" % epoch)
                time.set_postfix(loss=loss.item() / batch_size)
                sleep(0.1)
                time.update(1)

        if (epoch +1) % args.sample_every == 0:
            diffuser.eval()
            cases = [int(class_to_forget), (int(class_to_forget) + 1) % 10, (int(class_to_forget) + 2) % 10]
            if args.add_name is not None:
                generate_images_grid(diffuser=diffuser, model_name=model_name, prompts_path=prompts_path, save_path=f"{args.add_name}/intermediate_images/{model_name}/epoch_{epoch+1}", device=device, num_samples=1, cases=cases)
            else:
                generate_images_grid(diffuser=diffuser, model_name=model_name, prompts_path=prompts_path, save_path=f"NEW_results_grid/intermediate_images/{model_name}/epoch_{epoch+1}", device=device, num_samples=1, cases=cases)
            # generate_images(diffuser=diffuser, model_name=model_name, prompts_path=prompts_path, save_path=f"results_new/intermediate_images/{model_name}/epoch_{epoch+1}", device=device, num_samples=1)

    diffuser.eval()
    if args.add_name is not None:
        generate_images(diffuser=diffuser, model_name=model_name, prompts_path=prompts_path, save_path=f"{args.add_name}/final_images/{model_name}", device=device, num_samples=10)
    else:
        generate_images(diffuser=diffuser, model_name=model_name, prompts_path=prompts_path, save_path=f"NEW_results_grid/final_images/{model_name}", device=device, num_samples=10)
    save_model(
        diffuser,
        name,
        None
    )

    save_history(losses, name, classes)


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_loss(losses, path, word, n=100):
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in trainings", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)


def save_model(
    model,
    name,
    num,
):
    # SAVE MODEL
    folder_path = f"svd_unlearn_models_new/{name}"
    os.makedirs(folder_path, exist_ok=True)
    if num is not None:
        path = f"{folder_path}/{name}_epoch_{num}.pt"
    else:
        path = f"{folder_path}/{name}.pt"
    torch.save(model.state_dict(), path)


def save_history(losses, name, word_print):
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([str(i) for i in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train", description="train a stable diffusion model from scratch"
    )
    parser.add_argument(
        "--class_to_forget",
        help="class corresponding to concept to erase",
        type=str,
        required=True,
        default="0",
    )
    parser.add_argument(
        "--train_method", help="method of training", type=str, required=True
    )
    parser.add_argument(
        "--alpha",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=0.1,
    )
    parser.add_argument(
        "--batch_size",
        help="batch_size used to train",
        type=int,
        required=False,
        default=8,
    )
    parser.add_argument(
        "--sample_every",
        help="batch_size used to train",
        type=int,
        required=False,
        default=5,
    )
    parser.add_argument(
        "--num_inference_steps",
        help="batch_size used to train",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--epochs", help="epochs used to train", type=int, required=False, default=5
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
        "--mask_path",
        help="mask path for stable diffusion v1-4",
        type=str,
        required=False,
        default=None,
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
        "--ddim_steps",
        help="ddim steps of inference used to train",
        type=int,
        required=False,
        default=50,
    )
    parser.add_argument(
        "--explained_variance_ratio", type=float, default=0.95, help="explained variance ratio"
    )
    parser.add_argument(
        "--explained_variance_ratio_attention", type=float, default=None, help="explained variance ratio"
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
    parser.add_argument(
        "--c_guidance",
        help="guidance of start image used to train",
        type=float,
        required=False,
        default=7.5,
    )
    parser.add_argument(
        "--add_name", type=str, default=None, help="additional name"
    )
    parser.add_argument(
        "--retrain", default=False, action='store_true', help="whether use the retraining dataset"
    )
    parser.add_argument(
        "--retrain_method", type=str, default="rl", help="random labeling types (rl and rl++)"
    )
    args = parser.parse_args()

    if args.attention and not args.others:
        args.attention_only = True

    if args.others and not args.attention:
        args.others_only = True

    if args.explained_variance_ratio_attention is None:
        args.explained_variance_ratio_attention = args.explained_variance_ratio

    # classes = [int(d) for d in args.classes.split(',')]
    classes = int(args.class_to_forget)
    train_method = args.train_method
    alpha = args.alpha
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    ckpt_path = args.ckpt_path
    mask_path = args.mask_path
    config_path = args.config_path
    diffusers_config_path = args.diffusers_config_path
    device = f"cuda:{int(args.device)}"
    image_size = args.image_size
    ddim_steps = args.ddim_steps

    svd_unlearning(
        classes,
        train_method,
        alpha,
        batch_size,
        epochs,
        lr,
        config_path,
        ckpt_path,
        mask_path,
        diffusers_config_path,
        device,
        image_size,
        ddim_steps,
        args=args
    )