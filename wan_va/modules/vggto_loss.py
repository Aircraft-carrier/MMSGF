"""VGGT-style relative depth and local-point supervision for VGGTO."""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F


GeometryGradientLoss = Literal["grad", "grad_conf", "normal", "normal_conf", None]
_VGGT_LOSS_HARD_MAX = 100.0
VGGT_MONITOR_ERROR_HISTOGRAM_BINS = 4096
_VGGT_MONITOR_ERROR_LOG_MAX = math.log1p(torch.finfo(torch.float32).max)


def geometry_error_histogram(error: torch.Tensor) -> torch.Tensor:
    """Build a fixed log-scale histogram that can be summed across ranks."""

    log_error = torch.log1p(error.detach().float().clamp_min(0)).clamp_max(_VGGT_MONITOR_ERROR_LOG_MAX)
    return torch.histc(
        log_error,
        bins=VGGT_MONITOR_ERROR_HISTOGRAM_BINS,
        min=0.0,
        max=_VGGT_MONITOR_ERROR_LOG_MAX,
    )


def geometry_error_quantiles_from_histogram(
    histogram: torch.Tensor,
    quantiles: tuple[float, ...] = (0.5, 0.9),
) -> torch.Tensor:
    """Approximate global quantiles from a summed log-error histogram."""

    output_dtype = histogram.dtype
    histogram = histogram.to(dtype=torch.float64)
    total = histogram.sum()
    if not bool(total > 0):
        return histogram.new_full((len(quantiles),), float("nan")).to(dtype=output_dtype)

    cumulative = histogram.cumsum(dim=0)
    bin_width = _VGGT_MONITOR_ERROR_LOG_MAX / float(VGGT_MONITOR_ERROR_HISTOGRAM_BINS)
    outputs = []
    for quantile in quantiles:
        target = float(quantile) * (total - 1.0)
        lower_rank = torch.floor(target)
        upper_rank = torch.ceil(target)
        lower_index = torch.searchsorted(cumulative, lower_rank + 1.0).clamp_max(histogram.numel() - 1)
        upper_index = torch.searchsorted(cumulative, upper_rank + 1.0).clamp_max(histogram.numel() - 1)
        lower_value = torch.expm1(lower_index.to(dtype=torch.float64) * bin_width)
        upper_value = torch.expm1(upper_index.to(dtype=torch.float64) * bin_width)
        weight = target - lower_rank
        outputs.append(torch.lerp(lower_value, upper_value, weight))
    return torch.stack(outputs).to(dtype=output_dtype)


def geometry_pearson_stats(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return additive sufficient statistics for a global Pearson correlation."""

    x = x.detach().float()
    y = y.detach().float()
    return torch.stack(
        (
            x.new_tensor(x.numel()),
            x.sum(),
            y.sum(),
            x.square().sum(),
            y.square().sum(),
            (x * y).sum(),
        )
    )


def geometry_pearson_from_stats(stats: torch.Tensor) -> torch.Tensor:
    """Compute Pearson correlation from summed sufficient statistics."""

    count, sum_x, sum_y, sum_x2, sum_y2, sum_xy = stats.to(dtype=torch.float64)
    numerator = count * sum_xy - sum_x * sum_y
    denominator = torch.sqrt(
        (count * sum_x2 - sum_x.square()).clamp_min(0.0)
        * (count * sum_y2 - sum_y.square()).clamp_min(0.0)
    )
    valid = (count > 1) & (denominator > 0)
    return torch.where(valid, numerator / denominator.clamp_min(1e-12), numerator.new_full((), float("nan")))


def _relative_point_scale(
    points: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.ndim != 7 or points.shape[-1] != 3:
        raise ValueError(f"points must be [B,G,S,V,H,W,3], got {tuple(points.shape)}")
    if valid_mask.shape != points.shape[:-1]:
        raise ValueError(f"valid_mask shape {tuple(valid_mask.shape)} does not match points {tuple(points.shape)}")

    valid = valid_mask.to(device=points.device, dtype=torch.bool)
    valid = valid & torch.isfinite(points).all(dim=-1)
    safe_points = torch.where(valid[..., None], points, torch.zeros_like(points))
    dist = safe_points.norm(dim=-1)
    count = valid.sum(dim=(-1, -2), keepdim=True)
    scale = dist.sum(dim=(-1, -2), keepdim=True) / count.to(points.dtype).clamp_min(1.0)
    scale = torch.where(count > 0, scale.clamp_min(eps), torch.ones_like(scale))
    return valid, scale


def normalize_depth_targets(
    points: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize grouped depth by valid mean point norm per slot/view.

    `points` is `[B,G,S,V,H,W,3]` and `valid_mask` is `[B,G,S,V,H,W]`.
    """

    valid, scale = _relative_point_scale(points, valid_mask, eps=eps)
    safe_depth = torch.where(valid, points[..., 2], torch.zeros_like(points[..., 2]))
    depth = safe_depth / scale
    return depth, valid, scale


def normalize_point_targets(
    points: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize local point maps by the same per-slot/view scale as depth."""

    valid, scale = _relative_point_scale(points, valid_mask, eps=eps)
    safe_points = torch.where(valid[..., None], points, torch.zeros_like(points))
    normalized = safe_points / scale[..., None]
    return normalized, valid, scale


def _flatten_depth_inputs(
    pred_depth: torch.Tensor,
    pred_conf: torch.Tensor,
    points: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_depth, target_valid, _scale = normalize_depth_targets(points, valid_mask)
    bsz, groups, group_size, views, height, width = target_depth.shape
    frames = groups * group_size

    if pred_depth.shape == (bsz, frames, views, height, width, 1):
        pred = pred_depth
    elif pred_depth.shape == (bsz, frames * views, height, width, 1):
        pred = pred_depth.reshape(bsz, frames, views, height, width, 1)
    else:
        raise ValueError(
            "pred_depth must be [B,G*S,V,H,W,1] or [B,G*S*V,H,W,1], "
            f"got {tuple(pred_depth.shape)} for target {tuple(target_depth.shape)}"
        )

    if pred_conf.shape == (bsz, frames, views, height, width):
        conf = pred_conf
    elif pred_conf.shape == (bsz, frames * views, height, width):
        conf = pred_conf.reshape(bsz, frames, views, height, width)
    else:
        raise ValueError(
            "pred_conf must be [B,G*S,V,H,W] or [B,G*S*V,H,W], "
            f"got {tuple(pred_conf.shape)} for target {tuple(target_depth.shape)}"
        )

    target = target_depth.reshape(bsz, frames, views, height, width, 1).to(device=pred.device, dtype=pred.dtype)
    mask = target_valid.reshape(bsz, frames, views, height, width).to(device=pred.device, dtype=torch.bool)
    conf = conf.to(device=pred.device, dtype=pred.dtype).clamp_min(1e-6)
    return pred, conf, target, mask


def _flatten_point_inputs(
    pred_points: torch.Tensor,
    pred_conf: torch.Tensor,
    points: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    target_points, target_valid, _scale = normalize_point_targets(points, valid_mask)
    bsz, groups, group_size, views, height, width, channels = target_points.shape
    frames = groups * group_size

    if pred_points.shape == (bsz, frames, views, height, width, channels):
        pred = pred_points
    elif pred_points.shape == (bsz, frames * views, height, width, channels):
        pred = pred_points.reshape(bsz, frames, views, height, width, channels)
    else:
        raise ValueError(
            "pred_points must be [B,G*S,V,H,W,3] or [B,G*S*V,H,W,3], "
            f"got {tuple(pred_points.shape)} for target {tuple(target_points.shape)}"
        )

    if pred_conf.shape == (bsz, frames, views, height, width):
        conf = pred_conf
    elif pred_conf.shape == (bsz, frames * views, height, width):
        conf = pred_conf.reshape(bsz, frames, views, height, width)
    else:
        raise ValueError(
            "pred_conf must be [B,G*S,V,H,W] or [B,G*S*V,H,W], "
            f"got {tuple(pred_conf.shape)} for target {tuple(target_points.shape)}"
        )

    target = target_points.reshape(bsz, frames, views, height, width, channels).to(device=pred.device, dtype=pred.dtype)
    mask = target_valid.reshape(bsz, frames, views, height, width).to(device=pred.device, dtype=torch.bool)
    conf = conf.to(device=pred.device, dtype=pred.dtype).clamp_min(1e-6)
    return pred, conf, target, mask


def _gradient_loss(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, conf: torch.Tensor | None = None) -> torch.Tensor:
    mask_c = mask[..., None].expand_as(prediction)
    valid_count = mask_c.sum()
    if valid_count == 0:
        zero = prediction.sum()
        if conf is not None:
            zero = zero + conf.sum()
        return zero * 0.0

    diff = torch.where(mask_c, prediction - target, torch.zeros_like(prediction))
    grad_x = (diff[:, :, 1:] - diff[:, :, :-1]).abs()
    mask_x = mask_c[:, :, 1:] & mask_c[:, :, :-1]
    grad_y = (diff[:, 1:, :] - diff[:, :-1, :]).abs()
    mask_y = mask_c[:, 1:, :] & mask_c[:, :-1, :]

    if conf is not None:
        conf_c = conf[..., None].expand_as(prediction)
        grad_x = grad_x * conf_c[:, :, 1:]
        grad_y = grad_y * conf_c[:, 1:, :]

    grad_x = torch.where(mask_x, grad_x.clamp(max=100), torch.zeros_like(grad_x))
    grad_y = torch.where(mask_y, grad_y.clamp(max=100), torch.zeros_like(grad_y))
    return (grad_x.sum() + grad_y.sum()) / valid_count.to(prediction.dtype).clamp_min(1)


def _multi_scale_gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    conf: torch.Tensor | None,
    scales: int = 4,
) -> torch.Tensor:
    total = pred.sum() * 0.0
    for scale in range(scales):
        step = 2**scale
        total = total + _gradient_loss(
            pred[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=None if conf is None else conf[:, ::step, ::step],
        )
    return total / float(scales)


def _point_map_to_normal(point_map: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    padded_mask = F.pad(mask, (1, 1, 1, 1), mode="constant", value=False)
    safe_point_map = torch.where(mask[..., None], point_map, torch.zeros_like(point_map))
    points = F.pad(
        safe_point_map.permute(0, 3, 1, 2),
        (1, 1, 1, 1),
        mode="constant",
        value=0,
    ).permute(0, 2, 3, 1)
    center = points[:, 1:-1, 1:-1]
    up = points[:, :-2, 1:-1] - center
    left = points[:, 1:-1, :-2] - center
    down = points[:, 2:, 1:-1] - center
    right = points[:, 1:-1, 2:] - center
    normals = torch.stack(
        [
            torch.cross(up, left, dim=-1),
            torch.cross(left, down, dim=-1),
            torch.cross(down, right, dim=-1),
            torch.cross(right, up, dim=-1),
        ],
        dim=0,
    )
    valids = torch.stack(
        [
            padded_mask[:, :-2, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2],
            padded_mask[:, 1:-1, :-2] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:, 1:-1],
            padded_mask[:, 2:, 1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:],
            padded_mask[:, 1:-1, 2:] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2, 1:-1],
        ],
        dim=0,
    )
    return F.normalize(normals, p=2, dim=-1, eps=eps), valids


def _normal_loss_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    conf: torch.Tensor | None,
    gamma: float,
    alpha: float,
    compute_angle: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_normals, pred_valid = _point_map_to_normal(prediction, mask)
    target_normals, target_valid = _point_map_to_normal(target, mask)
    valid = pred_valid & target_valid
    valid_count = valid.sum()
    if int(valid_count.detach().cpu().item()) < 10:
        zero = prediction.sum()
        if conf is not None:
            zero = zero + conf.sum()
        zero = zero * 0.0
        return zero, zero.detach(), zero.detach()
    cosine = (pred_normals[valid] * target_normals[valid]).sum(dim=-1).clamp(-1 + 1e-8, 1 - 1e-8)
    if compute_angle:
        angles = torch.rad2deg(torch.acos(cosine.detach().float()))
        angle_sum = angles.sum()
        angle_count = angles.new_tensor(angles.numel())
    else:
        angle_sum = prediction.new_zeros(())
        angle_count = prediction.new_zeros(())
    loss = 1.0 - cosine
    if conf is not None:
        normal_conf = conf[None].expand(4, -1, -1, -1)[valid].clamp_min(1e-6)
        loss = gamma * loss * normal_conf - alpha * torch.log(normal_conf)
    return loss.mean(), angle_sum, angle_count


def _normal_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    conf: torch.Tensor | None,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    loss, _angle_sum, _angle_count = _normal_loss_terms(
        prediction,
        target,
        mask,
        conf=conf,
        gamma=gamma,
        alpha=alpha,
        compute_angle=False,
    )
    return loss


def _multi_scale_normal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    conf: torch.Tensor | None,
    gamma: float,
    alpha: float,
    scales: int = 3,
    return_angle: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = pred.sum() * 0.0
    angle_sum = total.detach()
    angle_count = total.detach()
    for scale in range(scales):
        step = 2**scale
        scale_loss, scale_angle_sum, scale_angle_count = _normal_loss_terms(
            pred[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=None if conf is None else conf[:, ::step, ::step],
            gamma=gamma,
            alpha=alpha,
            compute_angle=return_angle and scale == 0,
        )
        total = total + scale_loss
        if scale == 0:
            angle_sum = scale_angle_sum
            angle_count = scale_angle_count
    total = total / float(scales)
    if not return_angle:
        return total
    angle_mean = torch.where(
        angle_count > 0,
        angle_sum / angle_count.clamp_min(1),
        angle_sum.new_full((), float("nan")),
    )
    return total, angle_mean, torch.stack((angle_sum, angle_count))


def _filter_by_quantile(loss: torch.Tensor, valid_range: float) -> torch.Tensor:
    # VGGT hard-caps each per-pixel loss before optional outlier filtering.
    loss = loss.clamp(min=-_VGGT_LOSS_HARD_MAX, max=_VGGT_LOSS_HARD_MAX)
    if valid_range <= 0 or loss.numel() <= 1000:
        return loss
    threshold = torch.quantile(loss.detach().float(), float(valid_range)).to(loss.device)
    filtered = loss[loss < threshold]
    if filtered.numel() > 1000:
        return filtered
    return loss


def _compute_vggto_regression_loss(
    pred: torch.Tensor,
    conf: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    name: str,
    gradient_loss_fn: GeometryGradientLoss,
    valid_range: float = 0.98,
    gamma: float = 1.0,
    alpha: float = 0.2,
    min_valid_pixels: int = 100,
    detailed_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_pixels = mask.sum()
    pred_nonfinite = (~torch.isfinite(pred)).sum()
    conf_nonfinite = (~torch.isfinite(conf)).sum()
    target_nonfinite = (~torch.isfinite(target)).sum()
    nonfinite_metrics = {
        f"{name}_pred_nonfinite": pred_nonfinite,
        f"{name}_conf_nonfinite": conf_nonfinite,
        f"{name}_target_nonfinite": target_nonfinite,
    }
    nonfinite_total = pred_nonfinite + conf_nonfinite + target_nonfinite
    if int(nonfinite_total.detach().cpu().item()) > 0:
        nan = pred.new_full((), float("nan"))
        return nan, {
            f"loss_conf_{name}": nan,
            f"loss_reg_{name}": nan,
            f"loss_grad_{name}": nan,
            f"{name}_valid_pixels": valid_pixels.to(device=pred.device),
            f"{name}_conf_mean": nan,
            f"{name}_conf_max": nan,
            **nonfinite_metrics,
        }

    if int(valid_pixels.detach().cpu().item()) < int(min_valid_pixels):
        zero = (pred.sum() + conf.sum()) * 0.0
        return zero, {
            f"loss_conf_{name}": zero,
            f"loss_reg_{name}": zero,
            f"loss_grad_{name}": zero,
            f"{name}_valid_pixels": valid_pixels.to(device=pred.device),
            f"{name}_conf_mean": zero,
            f"{name}_conf_max": zero,
            **nonfinite_metrics,
        }

    raw_diff = torch.norm(target[mask] - pred[mask], dim=-1)
    diff = raw_diff.clamp(max=_VGGT_LOSS_HARD_MAX)
    detailed = {}
    if detailed_metrics:
        error_histogram = geometry_error_histogram(raw_diff)
        quantiles = geometry_error_quantiles_from_histogram(error_histogram)
        detailed[f"{name}_error_p50"] = quantiles[0]
        detailed[f"{name}_error_p90"] = quantiles[1]
        detailed[f"{name}_error_histogram"] = error_histogram
    reg_values = _filter_by_quantile(diff, float(valid_range))
    loss_reg = reg_values.mean() if reg_values.numel() > 0 else (pred.sum() + conf.sum()) * 0.0

    conf_values = gamma * diff * conf[mask] - alpha * torch.log(conf[mask])
    conf_values = _filter_by_quantile(conf_values, float(valid_range))
    loss_conf = conf_values.mean() if conf_values.numel() > 0 else (pred.sum() + conf.sum()) * 0.0
    valid_conf = conf[mask]
    conf_mean = valid_conf.mean() if valid_conf.numel() > 0 else (pred.sum() + conf.sum()) * 0.0
    conf_max = valid_conf.max() if valid_conf.numel() > 0 else (pred.sum() + conf.sum()) * 0.0
    if detailed_metrics:
        correlation_stats = geometry_pearson_stats(raw_diff, valid_conf)
        detailed[f"{name}_conf_error_correlation"] = geometry_pearson_from_stats(correlation_stats).to(raw_diff.dtype)
        detailed[f"{name}_conf_error_correlation_stats"] = correlation_stats

    if gradient_loss_fn is None:
        loss_grad = (pred.sum() + conf.sum()) * 0.0
    else:
        flat_pred = pred.reshape(-1, pred.shape[-3], pred.shape[-2], pred.shape[-1])
        flat_target = target.reshape_as(flat_pred)
        flat_mask = mask.reshape(-1, mask.shape[-2], mask.shape[-1])
        if "normal" in gradient_loss_fn:
            if flat_pred.shape[-1] != 3:
                raise ValueError("normal gradient loss requires 3D point predictions")
            flat_conf = conf.reshape(-1, conf.shape[-2], conf.shape[-1]) if "conf" in gradient_loss_fn else None
            normal_result = _multi_scale_normal_loss(
                flat_pred,
                flat_target,
                flat_mask,
                conf=flat_conf,
                gamma=gamma,
                alpha=alpha,
                return_angle=detailed_metrics,
            )
            if detailed_metrics:
                loss_grad, normal_angle_mean, normal_angle_stats = normal_result
                detailed[f"{name}_normal_angle_mean"] = normal_angle_mean
                detailed[f"{name}_normal_angle_stats"] = normal_angle_stats
            else:
                loss_grad = normal_result
        else:
            flat_conf = conf.reshape(-1, conf.shape[-2], conf.shape[-1]) if "conf" in gradient_loss_fn else None
            loss_grad = _multi_scale_gradient_loss(flat_pred, flat_target, flat_mask, conf=flat_conf)

    total = loss_conf + loss_reg + loss_grad
    return total, {
        f"loss_conf_{name}": loss_conf,
        f"loss_reg_{name}": loss_reg,
        f"loss_grad_{name}": loss_grad,
        f"{name}_valid_pixels": valid_pixels.to(device=pred.device),
        f"{name}_conf_mean": conf_mean,
        f"{name}_conf_max": conf_max,
        **detailed,
        **nonfinite_metrics,
    }


def compute_vggto_depth_loss(
    pred_depth: torch.Tensor,
    pred_conf: torch.Tensor,
    points: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gradient_loss_fn: GeometryGradientLoss = "grad",
    valid_range: float = 0.98,
    gamma: float = 1.0,
    alpha: float = 0.2,
    min_valid_pixels: int = 100,
    detailed_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred, conf, target, mask = _flatten_depth_inputs(pred_depth, pred_conf, points, valid_mask)
    return _compute_vggto_regression_loss(
        pred,
        conf,
        target,
        mask,
        name="depth",
        gradient_loss_fn=gradient_loss_fn,
        valid_range=valid_range,
        gamma=gamma,
        alpha=alpha,
        min_valid_pixels=min_valid_pixels,
        detailed_metrics=detailed_metrics,
    )


def compute_vggto_point_loss(
    pred_points: torch.Tensor,
    pred_conf: torch.Tensor,
    points: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gradient_loss_fn: GeometryGradientLoss = "normal",
    valid_range: float = 0.98,
    gamma: float = 1.0,
    alpha: float = 0.2,
    min_valid_pixels: int = 100,
    detailed_metrics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pred, conf, target, mask = _flatten_point_inputs(pred_points, pred_conf, points, valid_mask)
    return _compute_vggto_regression_loss(
        pred,
        conf,
        target,
        mask,
        name="point",
        gradient_loss_fn=gradient_loss_fn,
        valid_range=valid_range,
        gamma=gamma,
        alpha=alpha,
        min_valid_pixels=min_valid_pixels,
        detailed_metrics=detailed_metrics,
    )
