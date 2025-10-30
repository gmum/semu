import torch
from torch import nn, Tensor


class CustomLinear(nn.Linear):
    def __init__(self, m: nn.Linear, u: Tensor, vh: Tensor):
        super().__init__(
            in_features=m.in_features,
            out_features=m.out_features,
            bias=m.bias is not None,
            device=m.weight.device,
            dtype=m.weight.dtype,
        )
        factory_kwargs = {"device": m.weight.device, "dtype": m.weight.dtype}

        self.weight = nn.Parameter(
            torch.zeros((u.shape[1], vh.shape[0]), **factory_kwargs)
        )

        if m.bias is not None:
            self.bias.data = m.bias.data.clone()

        self.register_buffer("a", m.weight)
        self.register_buffer("u", u)
        self.register_buffer("vh", vh)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the CustomLinear layer using reconstructed weights from SVD components.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the linear transformation.
        """
        _weight = self.a + torch.einsum("ij,jk,kl->il", self.u, self.weight, self.vh)
        return nn.functional.linear(x, _weight, self.bias)


class CustomConv2d(nn.Conv2d):
    def __init__(self, m: nn.Conv2d, u: Tensor, vh: Tensor):
        super().__init__(
            in_channels=m.in_channels,
            out_channels=m.out_channels,
            kernel_size=m.kernel_size,
            stride=m.stride,
            padding=m.padding,
            dilation=m.dilation,
            groups=m.groups,
            bias=m.bias is not None,
            padding_mode=m.padding_mode,
            device=m.weight.device,
            dtype=m.weight.dtype,
        )
        factory_kwargs = {"device": m.weight.device, "dtype": m.weight.dtype}

        self.weight = nn.Parameter(
            torch.zeros((u.shape[0], u.shape[-1], vh.shape[1]), **factory_kwargs)
        )
        if m.bias is not None:
            self.bias.data = m.bias.data.clone()

        self.register_buffer("a", m.weight)
        self.register_buffer("u", u)
        self.register_buffer("vh", vh)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the CustomConv2d layer using reconstructed weights from SVD components.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying the convolution.
        """
        _weight = torch.einsum("nij,njk,nkl->nil", self.u, self.weight, self.vh)
        _weight = torch.unflatten(_weight, 2, self.kernel_size)
        _weight = torch.permute(_weight, (1, 0, 2, 3))
        _weight = self.a + _weight
        return self._conv_forward(x, _weight, self.bias)


def replace_layers_with_custom(
    model: nn.Module,
    u: dict[str, Tensor],
    vh: dict[str, Tensor],
    parent_name: str = "",
) -> nn.Module:
    """
    Replace all nn.Conv2d layers with CustomConv2d and nn.Linear layers with CustomLinear in the given model.

    Args:
        model (nn.Module): The input model.
        u (dict[str, Tensor]): Dictionary of left singular vectors (u) for each layer.
        vh (dict[str, Tensor]): Dictionary of right singular vectors (vh) for each layer.
        parent_name (str, optional): The name of the parent module. Defaults to "".

    Returns:
        nn.Module: The model with replaced layers.
    """
    for name, module in model.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if type(module) is nn.Conv2d:
            # Replace Conv2d with CustomConv2d
            setattr(
                model,
                name,
                CustomConv2d(module, u[full_name], vh[full_name]),
            )
        elif type(module) is nn.Linear:
            # Replace Linear with CustomLinear
            setattr(
                model,
                name,
                CustomLinear(module, u[full_name], vh[full_name]),
            )
        else:
            # Recursively replace layers in child modules
            replace_layers_with_custom(module, u, vh, full_name)


def set_requires_grad(
    module,
    changed_layers_class: list[str] = None,
    changed_layers_name: list[str] = None,
    include_bias: bool = False,
):
    """
    Set requires_grad=False for all parameters in the module except for layers that match the specified class names or name prefixes.

    Args:
    - module (torch.nn.Module): The module whose parameters are to be modified.
    - changed_layers_class (list[str], optional): List of layer class names for which parameters should remain trainable.
    - changed_layers_name (list[str], optional): List of layer name prefixes for which parameters should remain trainable.
    - include_bias (bool): If True set requires_grad=True for bias parameters, else requires_grad = False
    """
    # Default empty lists to avoid NoneType checks
    changed_layers_class = changed_layers_class or []
    changed_layers_name = changed_layers_name or []

    # Recursive function to set requires_grad
    def _set_requires_grad(module, prefix=""):
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name

            # Check if this module matches criteria for trainable layers
            requires_grad = False

            if any(
                type(child).__name__.lower() == cls_name.lower()
                for cls_name in changed_layers_class
            ):
                requires_grad = True

            if any(
                child_name.lower().startswith(prefix.lower())
                for prefix in changed_layers_name
            ):
                requires_grad = True

            # Set requires_grad for all parameters of the current module
            for name_param, param in child.named_parameters():
                if not include_bias and name_param.lower().endswith("bias"):
                    param.requires_grad = False
                else:
                    param.requires_grad = requires_grad

            # Recurse into child modules
            _set_requires_grad(child, child_name)

    # Disables the requires_grad flag in all parameters￥
    for param in module.parameters():
        param.requires_grad = False

    # Start the recursive setting of requires_grad
    _set_requires_grad(module)


def train_phase(
    module,
    changed_layers_class: list[str] = None,
    changed_layers_name: list[str] = None,
):
    """
    Set the training mode for the given module and its children layers based on specified classes or layer names.

    Args:
    - module (torch.nn.Module): The module to set the training mode for.
    - changed_layers_class (list[str], optional): List of layer class names to be set to train mode.
    - changed_layers_name (list[str], optional): List of layer name prefixes to be set to train mode.
    """
    # Ensure the entire model starts in eval mode
    module.train(False)

    # Stack-based traversal for hierarchical modules
    stack = [(module, "")]
    while stack:
        current_module, prefix = stack.pop()

        for name, child in current_module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name

            # Check if the child matches the criteria for train mode
            train_mode = False

            if changed_layers_class:
                train_mode = any(
                    type(child).__name__.lower() == layer.lower()
                    for layer in changed_layers_class
                )

            if changed_layers_name:
                train_mode = train_mode or any(
                    child_name.lower().startswith(layer.lower())
                    for layer in changed_layers_name
                )

            # Set the mode and continue traversal
            child.train(mode=train_mode)
            stack.append((child, child_name))


def test_training_children_layers(module):
    """
    Function to iterate over the children layers of a module and yield their names and module objects.

    Args:
    - module: torch.nn.Module: The module whose children layers need to be iterated over.

    Yields:
    Tuple[str, torch.nn.Module]: A tuple containing the name and the module object of each child layer.
    """
    stack = [(module, "")]
    while stack:
        module, prefix = stack.pop()
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            yield child_name, child
            stack.append((child, child_name))


# # train_phase(model, model.changed_layers)
# model.train()
# for name, module in test_training_children_layers(model):
#     if module.training:
#         print(f"Name: {name} | {module.training}")
#     # print(f"Name: {name} | {module.training}")
