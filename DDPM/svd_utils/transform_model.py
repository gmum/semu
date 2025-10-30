import re

import torch
import torch.nn as nn
from torch import linalg as LA
from torch.utils.data import DataLoader
from typing import List, Callable

from .utils import set_requires_grad, replace_layers_with_custom


def transform_text_layer(text):
    return re.sub(r"(^|\.)([0-9]+)(?=\.|$)", lambda match: f"[{match.group(2)}]", text)


def transform_model(
        model: nn.Module,
        data_loader_unlearn: DataLoader,
        criterion: nn.Module,
        compute_gradients_function: Callable = None,
        changed_layers_class: List[str] = None,
        explained_variance_ratio: float = None,
        use_projection_grad: bool = False,
        attention_only: bool = False,
        others_only: bool = False,
        explained_variance_ratio_attention: float = 1.0,
) -> None:
    """
    Transform the model by replacing the last layer with a new linear layer.

    Args:
        model (nn.Module): The input model.
        data_loader_unlearn (DataLoader): DataLoader for the unlearning dataset.
        criterion (nn.Module): Loss function.
        changed_layers_class (list[str], optional): List of layer class names for which parameters should remain trainable.
        explained_variance_ratio (float, optional): Explained variance ratio for the new linear layer.
        use_projection_grad: Make projection gradients onto the space perpendicular to the weights

    Returns:
        nn.Module: The transformed model.
    """
    device = next(model.parameters()).device

    if changed_layers_class is None:
        changed_layers_class = ["linear", "conv2d"]

    def compute_gradients(data, target):
        """
        Compute gradients for the unlearning dataset
        """
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        gradients_dict = {
            name[: name.rfind(".")]: param.grad.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        return gradients_dict

    if compute_gradients_function is None:
        compute_gradients_function = compute_gradients

    set_requires_grad(model, changed_layers_class=changed_layers_class)

    sum_gradients = None
    for images, targets in data_loader_unlearn:
        images, targets = images.to(device), targets.to(device)
        gradients = compute_gradients_function(images, targets)
        if sum_gradients is None:
            sum_gradients = gradients
        else:
            for key, val in gradients.items():
                sum_gradients[key] += val

    if use_projection_grad:
        # Projecting the gradient onto the space perpendicular to the weights
        for layer_name, grad in sum_gradients.items():
            _weight = f"model.{transform_text_layer(layer_name)}.weight"
            _weight = eval(_weight)
            proj_grad = grad - (torch.sum(grad * _weight) / LA.norm(_weight)) * _weight
            sum_gradients[layer_name] = proj_grad

    u_matrices = {}
    vh_matrices = {}
    num_components = {}
    for key, G in sum_gradients.items():
        # TODO - check the attention case!
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
            raise NotImplemented(
                "Operations are only suitable for layers: Conv2d and Linear"
            )
        # print("-" * 75)
        u_matrices[key] = u
        vh_matrices[key] = vh
        num_components[key] = num_comp

    replace_layers_with_custom(model, u_matrices, vh_matrices, num_components, attention_only=attention_only,
                               others_only=others_only, changed_layers_class=changed_layers_class)


def clip_grad(model: nn.Module):
    for name, param in model.named_parameters():
        if param.grad is not None and name.endswith("weight"):
            key = name[: name.rfind(".")]
            grad = param.grad

            num_comps = eval(f"model.{transform_text_layer(key)}.n_imp_comps")
            if grad.dim() == 2:
                mask = torch.arange(grad.shape[-1], device=grad.device).unsqueeze(
                    -1
                ) < num_comps.unsqueeze(-1)
                mask = mask & mask.transpose(0, 1)
            elif grad.dim() == 3:
                mask = torch.arange(grad.shape[-1], device=grad.device).unsqueeze(
                    0
                ).unsqueeze(-1) < num_comps.unsqueeze(-1)
                mask = mask & mask.transpose(1, 2)
            else:
                raise NotImplemented(
                    "Operations are only suitable for layers: Conv2d and Linear"
                )

            grad.mul_(mask)