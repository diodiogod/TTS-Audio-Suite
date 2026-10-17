import torch


def move_module(module, device):
    """
    Module.to() plus the plain tensor attributes it skips (s2mel causal_mask/freqs_cis,
    old-style weight_norm weights).
    """
    module.to(device)
    for sub in module.modules():
        for key, value in list(vars(sub).items()):
            if torch.is_tensor(value) and not isinstance(value, torch.nn.Parameter):
                setattr(sub, key, value.to(device))
    return module
