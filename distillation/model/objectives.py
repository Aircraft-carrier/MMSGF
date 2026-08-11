"""Distillation losses."""
import torch
import torch.nn.functional as F

from distillation.schema import VALossWeights, VAMasks, VAPair


def _video_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Frame-balanced video MSE for ``[B,Cv,F,V,H,W]``.

    例如 B=1、两个监督 frame 的逐元素均方误差分别为 4 和 16：先把每帧的
    channel/view/space 求均值得到 [4,16]，再按该样本有效 frame 数归一，结果
    是 10。这样增加相机 view 或 latent 分辨率不会无意放大 video loss。
    target 在这里 detach，确保 EMA/score target 不形成反向路径。
    """
    loss = F.mse_loss(pred.float(), target.float().detach(), reduction="none")
    bsz, _, frames = pred.shape[:3]
    mask = mask.reshape(bsz, frames)
    loss = loss.permute(0, 2, 1, 3, 4, 5).reshape(bsz, frames, -1).mean(-1)
    loss = torch.where(mask, loss, torch.zeros_like(loss))
    return (loss.sum(1) / mask.sum(1).clamp_min(1)).mean()


def _action_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Token-mask-aware action MSE for ``[B,Ca,F,N,1]``.

    action mask 可能只让一个 frame 的部分 channel/token 有效。函数先在每个
    frame 内除以有效 token 数，再只对至少含一个有效 token 的 frame 求均值。
    因此 padding 多的样本不会因为分母包含 padding 而得到更小 loss。
    """
    loss = F.mse_loss(pred.float(), target.float().detach(), reduction="none")
    mask = mask.expand_as(loss)
    bsz, _, frames = loss.shape[:3]
    loss = loss.permute(0, 2, 1, 3, 4).reshape(bsz, frames, -1)
    mask = mask.permute(0, 2, 1, 3, 4).reshape(bsz, frames, -1)
    valid = mask.sum(-1)
    frame_loss = torch.where(
        valid > 0,
        torch.where(mask, loss, 0).sum(-1) / valid.clamp_min(1),
        0,
    )
    return frame_loss.sum() / (valid > 0).sum().clamp_min(1)


def _va_loss(
    pred: VAPair,
    target: VAPair,
    masks: VAMasks,
    weights: VALossWeights,
    name: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    video_loss = _video_mse(pred.video, target.video, masks.video)
    action_loss = _action_mse(pred.action, target.action, masks.action)
    total = weights.video * video_loss + weights.action * action_loss
    return total, {
        f"distill/{name}_video_loss": video_loss.detach(),
        f"distill/{name}_action_loss": action_loss.detach(),
        f"distill/{name}_total_loss": total.detach(),
    }


def consistency_loss(
    student_x0: VAPair,
    target_x0: VAPair,
    masks: VAMasks,
    weights: VALossWeights = VALossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _va_loss(
        student_x0,
        target_x0,
        masks,
        weights,
        "consistency",
    )


def dmd_surrogate_loss(
    generator_x0: VAPair,
    target_x0: VAPair,
    masks: VAMasks,
    weights: VALossWeights = VALossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DMD surrogate loss: ``0.5 *`` masked weighted V/A MSE.

    Reference DMD uses 0.5 * MSE so the surrogate derivative equals the
    normalized KL direction instead of twice that direction.
    """
    loss, metrics = _va_loss(generator_x0, target_x0, masks, weights, "dmd")
    return 0.5 * loss, {name: 0.5 * value for name, value in metrics.items()}


def action_aware_loss(
    student_flow: torch.Tensor,
    target_flow: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = F.mse_loss(
        student_flow.float(),
        target_flow.float().detach(),
        reduction="none",
    )
    mask = mask.expand_as(loss)
    loss = torch.where(mask, loss, 0).sum() / mask.sum().clamp_min(1)
    return loss, {"distill/action_aware_loss": loss.detach()}


def fake_score_flow_loss(
    fake_flow: VAPair,
    target_flow: VAPair,
    masks: VAMasks,
    weights: VALossWeights = VALossWeights(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    return _va_loss(
        fake_flow,
        target_flow,
        masks,
        weights,
        "fake_score",
    )
