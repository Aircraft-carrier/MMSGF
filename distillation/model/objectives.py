"""Distillation losses."""
import torch
import torch.nn.functional as F

from distillation.schema import VALossWeights, VAMasks, VAPrediction


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
    mask = mask.to(device=loss.device, dtype=torch.bool).reshape(bsz, frames)
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
    mask = mask.to(device=loss.device, dtype=torch.bool).expand_as(loss)
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
    pred: VAPrediction,
    target: VAPrediction,
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
    student_x0: VAPrediction,
    target_x0: VAPrediction,
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
    mask = mask.to(device=loss.device, dtype=torch.bool).expand_as(loss)
    loss = torch.where(mask, loss, 0).sum() / mask.sum().clamp_min(1)
    return loss, {"distill/action_aware_loss": loss.detach()}


def dmd_surrogate_loss(
    student_x0: VAPrediction,
    fake_x0: VAPrediction,
    real_x0: VAPrediction,
    masks: VAMasks,
    weights: VALossWeights = VALossWeights(),
    normalizer_eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Build a stop-gradient DMD target whose MSE has the desired score direction.

    单值示例：student=0、real=1、fake=0，normalizer=|0-1|=1，得到
    ``target = 0 - (0-1) = 1``。MSE 对 student 的梯度为负，optimizer 做
    gradient descent 后 student 会增大并朝 real 移动。代码中的 target 保留
    ``student`` 只是为了在当前点构造 surrogate；``_va_loss`` 会 detach target，
    所以不会发生 target/student 两条梯度互相抵消。

    normalizer 每个样本、每个 modality 独立计算，且只统计有效 mask；否则
    大量 clean history/padding 会稀释 ``|student-real|``，放大 DMD gradient。
    """
    def target(student, fake, real, mask):
        # expand_as 只扩展 mask view，不复制整块数据。全 False 样本的分母通过
        # clamp_min(1) 保持有限，normalizer 再由 normalizer_eps 给出安全下界；
        # 最终 loss mask 仍为 False，所以该样本对 loss/gradient 的贡献为 0。
        mask = mask.to(device=student.device, dtype=torch.bool).expand_as(student)
        error = torch.where(mask, (student.detach() - real).abs(), 0)
        normalizer = error.flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1)
        normalizer = normalizer.clamp_min(normalizer_eps)
        normalizer = normalizer.reshape(-1, *([1] * (student.ndim - 1)))
        return student - (fake - real).detach() / normalizer

    dmd_target = VAPrediction(
        video=target(
            student_x0.video,
            fake_x0.video,
            real_x0.video,
            masks.video[:, None, :, None, None, None],
        ),
        action=target(
            student_x0.action,
            fake_x0.action,
            real_x0.action,
            masks.action,
        ),
    )
    return _va_loss(
        student_x0,
        dmd_target,
        masks,
        weights,
        "dmd",
    )


def fake_score_flow_loss(
    fake_flow: VAPrediction,
    target_flow: VAPrediction,
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
