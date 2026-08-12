"""Shared model state, input, and tensor helpers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from distillation.schema import VALossWeights, VAMasks, VAPair, VATimesteps
from wan_va.utils.scheduler import FlowMatchScheduler


def video_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute frame-balanced video MSE for ``[B, C, F, V, H, W]``.

    Elementwise errors are averaged within each frame before supervised frames
    are averaged per sample. Consequently, adding views or changing latent
    resolution does not change the relative scale of the video loss. The target
    is detached so EMA or score targets never create a backward path.
    """
    # pred/target: [B,C,F,V,H,W], where B=batch, C=latent channel,
    # F=frame, V=view, and H/W are the latent spatial dimensions.
    # mask: [B,F], where only True frames participate in supervision.

    # Stage 1/3: Compute elementwise squared errors in FP32. Detaching the
    # target ensures that gradients propagate only through pred.
    # loss: [B,C,F,V,H,W]
    loss = F.mse_loss(
        pred.float(),
        target.float().detach(),
        reduction="none",
    )

    batch_size, _, frames = pred.shape[:3]
    mask = mask.reshape(batch_size, frames)  # [B,F]

    # Stage 2/3: Move frame F to the second dimension:
    # [B,C,F,V,H,W] -> [B,F,C,V,H,W] -> [B,F,C*V*H*W]
    #
    # Average over every channel, view, and spatial position per frame:
    # [B,F,C*V*H*W] -> [B,F]
    #
    # Example: errors [1,1,1,1,9,9,9,9] produce a frame loss of 5,
    # rather than the element-count-dependent sum of 40.
    loss = loss.permute(0, 2, 1, 3, 4, 5).reshape(
        batch_size,
        frames,
        -1,
    ).mean(-1)

    # Stage 3/3: Zero unsupervised frame errors so they do not enter the
    # numerator.
    loss = torch.where(mask, loss, torch.zeros_like(loss))

    # Normalize each sample by its valid frame count, then average the batch.
    # clamp_min(1) makes an all-invalid sample contribute zero instead of NaN.
    return (
        loss.sum(1) / mask.sum(1).clamp_min(1)
    ).mean()

def action_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Compute token-mask-aware action MSE for ``[B, C, F, N, 1]``.

    Errors are first averaged over valid action elements within each frame, then
    averaged over frames containing at least one valid element. Padding therefore
    contributes neither error nor denominator mass. The target is detached from
    autograd in the same way as the video target.
    """
    # pred/target: [B,C,F,N,1], where C=action channel, F=frame, and N is
    # the number of action tokens per frame. The mask broadcasts to this
    # layout and can select only part of a frame's channels or tokens.

    # Stage 1/3: Compute elementwise squared errors without backpropagating
    # through the target.
    # loss: [B,C,F,N,1]
    loss = F.mse_loss(
        pred.float(),
        target.float().detach(),
        reduction="none",
    )

    # Align the mask with every action loss element.
    mask = mask.expand_as(loss)

    batch_size, _, frames = loss.shape[:3]

    # Stage 2/3: Move the frame axis to the second dimension and flatten
    # all action elements belonging to each frame.
    #
    # [B,C,F,N,1] -> [B,F,C,N,1] -> [B,F,C*N]
    # loss and mask are now [B,F,E], where E=C*N.
    loss = loss.permute(0, 2, 1, 3, 4).reshape(
        batch_size,
        frames,
        -1,
    )
    mask = mask.permute(0, 2, 1, 3, 4).reshape(
        batch_size,
        frames,
        -1,
    )

    # Number of valid elements per frame: [B,F].
    valid = mask.sum(-1)

    # Stage 3/3: Average only valid elements within each frame.
    #
    # Example: if only the first two of eight positions are valid and their
    # errors are [4,16], frame_loss=(4+16)/2=10 rather than dividing by 8.
    #
    # Frames without any valid tokens contribute zero.
    frame_loss = torch.where(
        valid > 0,
        torch.where(mask, loss, 0).sum(-1)
        / valid.clamp_min(1),
        0,
    )

    # Average over all frames in the batch that contain at least one valid
    # token. This weights valid action frames equally rather than first
    # averaging within each sample.
    return frame_loss.sum() / (valid > 0).sum().clamp_min(1)


def va_loss(
    pred: VAPair,
    target: VAPair,
    masks: VAMasks,
    weights: VALossWeights,
    name: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Combine masked video/action MSE values and construct standard metrics.

    ``weights`` supplies independent ``video`` and ``action`` multipliers. The
    returned scalar retains gradients, while every metric is detached and named
    under ``distill/{name}_...`` for consistent trainer logging.
    """
    video_loss = video_mse(pred.video, target.video, masks.video)
    action_loss = action_mse(pred.action, target.action, masks.action)
    total = weights.video * video_loss + weights.action * action_loss
    return total, {
        f"distill/{name}_video_loss": video_loss.detach(),
        f"distill/{name}_action_loss": action_loss.detach(),
        f"distill/{name}_total_loss": total.detach(),
    }


def sigmas_for_timesteps(
    scheduler: FlowMatchScheduler,
    timesteps: torch.Tensor,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Look up the scheduler sigma nearest to each requested timestep.

    ``scheduler.timesteps`` and ``scheduler.sigmas`` are parallel one-dimensional
    tables. ``timesteps`` may have any shape, although distillation normally uses
    ``[B, F]``. The returned tensor preserves that shape and is placed on the
    timestep device. ``dtype`` optionally casts the selected sigma values for the
    arithmetic performed by the caller.
    """
    # scheduler.timesteps: [S]
    # scheduler.sigmas:    [S]
    # Entries correspond by index: timesteps[i] <-> sigmas[i].
    #
    # timesteps is normally [B,F], but can also be scalar, [F], or [B,F,...].
    # The result has the same shape as timesteps.

    # Stage 1/3: Move the one-dimensional scheduler table to the timestep
    # device and reshape it to [S,1,1,...] with timesteps.ndim trailing axes.
    #
    # For timesteps [B,F], scheduler_timesteps changes [S] -> [S,1,1].
    #
    # For S=4 and table [1000,750,500,250], broadcasting compares every
    # requested [B,F] value against all four entries.
    scheduler_timesteps = scheduler.timesteps.to(
        timesteps.device
    ).reshape(
        -1,
        *([1] * timesteps.ndim),
    )

    # Stage 2/3: Find the closest scheduler entry for every requested timestep.
    #
    # timesteps.unsqueeze(0):
    # [B,F] -> [1,B,F]
    #
    # Subtraction broadcasts as follows:
    # [S,1,1] - [1,B,F] -> [S,B,F]
    #
    # Each [b,f] position has S candidate distances. argmin(dim=0) selects
    # the closest entry along the scheduler axis:
    # [S,B,F] -> [B,F].
    #
    # Example:
    # scheduler.timesteps = [1000, 750, 500, 250]
    # timesteps[0,0] = 620
    # distances = [380,130,120,370]
    # indices[0,0] = 2, corresponding to 500.
    indices = (
        scheduler_timesteps - timesteps.unsqueeze(0)
    ).abs().argmin(dim=0)

    # Stage 3/3: Index the parallel sigma table using the nearest timestep IDs.
    #
    # scheduler.sigmas: [S]
    # scheduler.sigmas[indices]: [B,F]
    #
    # to(dtype=dtype) preserves the scheduler dtype when dtype=None, or casts
    # to the caller's requested arithmetic dtype.
    return scheduler.sigmas.to(
        timesteps.device
    )[indices].to(dtype=dtype)


def broadcast_frame_values(
    values: torch.Tensor,
    sample: torch.Tensor,
) -> torch.Tensor:
    """Reshape frame values from ``[B, F]`` for a ``[B, C, F, ...]`` sample.

    Singleton dimensions are inserted for channels and all trailing spatial or
    token axes. The operation is a reshape only; PyTorch performs the actual
    broadcasting when the result participates in tensor arithmetic.
    """
    return values.reshape(
        values.shape[0],
        1,
        values.shape[1],
        *([1] * (sample.ndim - 3)),
    )


def randn_like_va(value: VAPair) -> VAPair:
    """Create independent standard Gaussian noise for both V/A streams.

    Each returned tensor exactly matches the corresponding input tensor's shape,
    dtype, device, and layout. Video and action are sampled independently through
    separate ``torch.randn_like`` calls.
    """
    return VAPair(
        video=torch.randn_like(value.video),
        action=torch.randn_like(value.action),
    )


def apply_va_mask(
    noisy: VAPair,
    clean: VAPair,
    masks: VAMasks,
) -> VAPair:
    """Select noisy V/A values inside masks and clean values everywhere else.

    Video uses a frame mask ``[B, F]`` that is broadcast over channel, view, and
    spatial axes of ``[B, C, F, V, H, W]``. The action mask already follows the
    action tensor layout ``[B, C, F, N, 1]`` and is applied directly. This helper
    performs selection only; it does not generate noise or alter either input.
    """
    video_mask = masks.video[:, None, :, None, None, None]
    return VAPair(
        video=torch.where(video_mask, noisy.video, clean.video),
        action=torch.where(masks.action, noisy.action, clean.action),
    )


def replace_va_streams(
    base_input: dict,
    noisy: VAPair,
    clean: VAPair,
    timesteps: VATimesteps,
) -> dict:
    """Install one synthetic V/A state into the native MOT training input.

    ``_prepare_joint_input_dict(add_noise=False)`` intentionally omits the native
    autoregressive ``targets`` tensors because distillation computes its losses
    outside ``VAMOTTransformer3DModel``. The model's train forward still reads
    ``action_dict["targets"].shape`` when reshaping its output, so the adapter
    keeps that structural key as a shape carrier.

    Video values use ``[B,Cv,F,V,H,W]``, action values use
    ``[B,Ca,F,N,1]``, and both timestep tensors use ``[B,F]``. The returned
    dictionaries preserve text, masks, and attention-window metadata from
    ``base_input`` while replacing only the V/A trajectory fields.
    """
    latent_dict = {
        **base_input["latent_dict"],
        "noisy_latents": noisy.video,
        "latent": clean.video,
        "targets": base_input["latent_dict"].get("targets", clean.video),
        "timesteps": timesteps.video,
        "cond_timesteps": torch.zeros_like(timesteps.video),
    }
    action_dict = {
        **base_input["action_dict"],
        "noisy_latents": noisy.action,
        "latent": clean.action,
        "targets": base_input["action_dict"].get("targets", clean.action),
        "timesteps": timesteps.action,
        "cond_timesteps": torch.zeros_like(timesteps.action),
    }
    return {
        **base_input,
        "latent_dict": latent_dict,
        "action_dict": action_dict,
    }


def replace_text_condition(input_dict: dict, text_emb: torch.Tensor) -> dict:
    """Return a shallow copy with both V/A text-conditioning streams replaced.

    All non-text fields and top-level metadata retain their existing objects.
    Both modality dictionaries receive the same ``text_emb`` tensor so a joint
    model forward observes a consistent conditional or unconditional prompt.
    """
    return {
        **input_dict,
        "latent_dict": {
            **input_dict["latent_dict"],
            "text_emb": text_emb,
        },
        "action_dict": {
            **input_dict["action_dict"],
            "text_emb": text_emb,
        },
    }


def freeze_model(model: nn.Module) -> nn.Module:
    """Put a module in evaluation mode and disable gradients for all parameters.

    The same module object is returned so callers can compose this state change
    directly with device placement or distributed wrapping.
    """
    model.eval().requires_grad_(False)
    return model


def set_trainable(model: nn.Module) -> nn.Module:
    """Put a module in training mode and enable gradients for all parameters.

    The same module object is returned so callers can immediately apply
    activation checkpointing, parameter ownership, or distributed wrapping.
    """
    model.train().requires_grad_(True)
    return model
