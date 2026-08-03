"""Model state helpers."""
import torch
import torch.nn as nn


def freeze_model(model: nn.Module) -> nn.Module:
    model.eval().requires_grad_(False)
    return model


def set_trainable(model: nn.Module) -> nn.Module:
    model.train().requires_grad_(True)
    return model


@torch.no_grad()
def update_ema(ema_model: nn.Module, student_model: nn.Module, decay: float) -> None:
    for ema_param, student_param in zip(
        ema_model.parameters(),
        student_model.parameters(),
        strict=True,
    ):
        ema_param.mul_(decay).add_(student_param, alpha=1.0 - decay)
