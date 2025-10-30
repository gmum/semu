import re

import torch
import torch.nn as nn
from torch import linalg as LA
from torch.utils.data import DataLoader
from typing import List, Callable
from tqdm import tqdm
import numpy as np

from .utils import set_requires_grad, replace_layers_with_custom, get_requires_grad_list, CustomLinear, CustomConv2d


def transform_text_layer(text):
    return re.sub(r"(^|\.)([0-9]+)(?=\.|$)", lambda match: f"[{match.group(2)}]", text)


def transform_model(
    model: nn.Module,
    gradients_dict: dict = {},
    changed_layers_class: List[str] = None,
    explained_variance_ratio: float = None,
    use_projection_grad: bool = False,
    attention_only: bool = False,
    others_only: bool = False,
    explained_variance_ratio_attention: float = 1.0,
) -> None:
    device = next(model.parameters()).device

    if changed_layers_class is None:
        changed_layers_class = ["linear", "conv2d"]


    if use_projection_grad:
        # Projecting the gradient onto the space perpendicular to the weights
        for layer_name, grad in gradients_dict.items():
            _weight = f"model.{transform_text_layer(layer_name)}"
            _weight = eval(_weight)
            proj_grad = grad - (torch.sum(grad * _weight) / LA.norm(_weight)) * _weight
            gradients_dict[layer_name] = proj_grad

    u_matrices = {}
    vh_matrices = {}
    num_components = {}
    for key, G in gradients_dict.items():
        if G.dim() == 2:
            u, s, vh = torch.linalg.svd(G, full_matrices=False)

            total_variance = torch.square(s).sum()
            explained_variance = torch.square(s) / total_variance
            cumulative_explained_variance = torch.cumsum(explained_variance, dim=0)
            if "attn" in key:
                num_comp = torch.searchsorted(
                cumulative_explained_variance, explained_variance_ratio_attention, side="right"
            )
            else:
                num_comp = torch.searchsorted(
                    cumulative_explained_variance, explained_variance_ratio, side="right"
            )

            # print("Check u*s*vh = G")
            # print(torch.dist(G, u @ torch.diag(s) @ vh))
            # print(torch.dist(torch.einsum("ab,b,bc->ac", u, s, vh), G))
        elif G.dim() == 4:
            _weight = torch.permute(G, (1, 0, 2, 3))
            _weight = torch.flatten(_weight, start_dim=2, end_dim=-1)

            u, s, vh = torch.linalg.svd(_weight, full_matrices=False)

            total_variance = torch.square(s).sum(dim=1, keepdim=True)
            explained_variance = torch.square(s) / total_variance
            cumulative_explained_variance = torch.cumsum(explained_variance, dim=1)
            
            if "attn" in key:
                num_comp = torch.searchsorted(
                cumulative_explained_variance,
                torch.full((s.shape[0], 1), explained_variance_ratio_attention, device=s.device),
                side="right",
            )
            else:
                num_comp = torch.searchsorted(
                    cumulative_explained_variance,
                    torch.full((s.shape[0], 1), explained_variance_ratio, device=s.device),
                    side="right",
                )
        else:
            print(f"Layer skipped: {key}")
            # raise NotImplemented(
            #     "Operations are only suitable for layers: Conv2d and Linear"
            # )
        # print("-" * 75)
        u_matrices[key] = u.clone().detach()
        vh_matrices[key] = vh.clone().detach()
        num_components[key] = num_comp.clone().detach()

    requires_grad_list = [k  for  k in  u_matrices.keys()]

    replace_layers_with_custom(model, u_matrices, vh_matrices, num_components, attention_only=attention_only, others_only=others_only, changed_layers_class=changed_layers_class, requires_grad_list=requires_grad_list)
    return num_components


def clip_grad(model: nn.Module):
    for name, param in model.named_parameters():
        if param.grad is not None:
            key = name[: name.rfind(".")]
            # grad = param.grad 
            if name.endswith("weight"):
                _module = eval(f"model.{transform_text_layer(key)}")
                _module_type = type(_module)

                if _module_type is CustomLinear or _module_type is CustomConv2d:
                    num_comps = eval(f"model.{transform_text_layer(key)}.n_imp_comps")
                    if param.grad.dim() == 2:
                        mask = torch.arange(param.grad.shape[-1], device=param.grad.device).unsqueeze(
                            -1
                        ) < num_comps.unsqueeze(-1)
                        mask = mask & mask.transpose(0, 1)
                    elif param.grad.dim() == 3:
                        mask = torch.arange(param.grad.shape[-1], device=param.grad.device).unsqueeze(
                            0
                        ).unsqueeze(-1) < num_comps.unsqueeze(-1)
                        mask = mask & mask.transpose(1, 2)
                    else:
                        raise NotImplemented(
                            "Operations are only suitable for layers: Conv2d and Linear"
                        )
                else:
                    mask = torch.zeros_like(param.grad)
            else:
                mask = torch.zeros_like(param.grad)
                
            param.grad = param.grad.mul_(mask)