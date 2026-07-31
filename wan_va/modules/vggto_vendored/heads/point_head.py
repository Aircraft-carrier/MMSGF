# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""VGGT DPT point head adapted to VGGTO's cached-token contract."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .dense_head import (
    MAX_GEOMETRY_VALUE,
    _make_dense_resize_layer,
    _make_fusion_block,
    _make_scratch,
    bounded_expp1,
    custom_interpolate,
)
from .utils import create_uv_grid, position_grid_to_embed


class PointHead(nn.Module):
    """Predict signed XYZ and confidence with the original VGGT DPT layout."""

    output_dim = 3
    activation = "inv_log"
    conf_activation = "expp1"

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 16,
        features: int = 256,
        out_channels: list[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: list[int] = [4, 11, 17, 23],
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.intermediate_layer_idx = intermediate_layer_idx
        self.norm = nn.LayerNorm(dim_in, eps=1e-5)

        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, channels, kernel_size=1) for channels in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                _make_dense_resize_layer(out_channels[0], 4.0),
                _make_dense_resize_layer(out_channels[1], 2.0),
                _make_dense_resize_layer(out_channels[2], 1.0),
                _make_dense_resize_layer(out_channels[3], 0.5),
            ]
        )

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        self.scratch.output_conv1 = nn.Conv2d(features, features // 2, kernel_size=3, padding=1)
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(features // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.output_dim + 1, kernel_size=1),
        )

    def forward(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_token_start: int,
        frames_chunk_size: int | None = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_frames = images.shape[1]
        if frames_chunk_size is None or frames_chunk_size >= num_frames:
            return self._forward_impl(aggregated_tokens_list, images, patch_token_start)
        if frames_chunk_size <= 0:
            raise ValueError(f"frames_chunk_size must be positive, got {frames_chunk_size}")

        point_chunks = []
        confidence_chunks = []
        for start in range(0, num_frames, frames_chunk_size):
            end = min(start + frames_chunk_size, num_frames)
            points, confidence = self._forward_impl(
                aggregated_tokens_list,
                images,
                patch_token_start,
                start,
                end,
            )
            point_chunks.append(points)
            confidence_chunks.append(confidence)
        return torch.cat(point_chunks, dim=1), torch.cat(confidence_chunks, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: list[torch.Tensor | None],
        images: torch.Tensor,
        patch_token_start: int,
        frames_start_idx: int | None = None,
        frames_end_idx: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx].contiguous()

        batch_size, num_frames, _, height, width = images.shape
        patch_h, patch_w = height // self.patch_size, width // self.patch_size
        multi_scale_features = []
        for feature_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            tokens = aggregated_tokens_list[layer_idx]
            if tokens is None:
                raise ValueError(f"Aggregator did not cache layer {layer_idx}, which PointHead needs.")
            tokens = tokens[:, :, patch_token_start:]
            if frames_start_idx is not None and frames_end_idx is not None:
                tokens = tokens[:, frames_start_idx:frames_end_idx]
            if tokens.dtype != torch.float32:
                tokens = tokens.float()

            features = tokens.reshape(batch_size * num_frames, -1, tokens.shape[-1])
            features = self.norm(features)
            features = features.permute(0, 2, 1).reshape(batch_size * num_frames, -1, patch_h, patch_w)
            features = self.projects[feature_idx](features)
            features = self._apply_pos_embed(features, width, height)
            multi_scale_features.append(self.resize_layers[feature_idx](features))

        fused = self._scratch_forward(multi_scale_features)
        fused = custom_interpolate(fused, size=(height, width), mode="bilinear", align_corners=True)
        fused = self._apply_pos_embed(fused, width, height)
        output = self.scratch.output_conv2(fused).permute(0, 2, 3, 1)
        point_logits, confidence_logits = output[..., :3], output[..., 3]
        point_magnitude = torch.expm1(
            torch.abs(point_logits).clamp(max=math.log1p(MAX_GEOMETRY_VALUE))
        ).clamp(
            max=MAX_GEOMETRY_VALUE,
        )
        points = torch.sign(point_logits) * point_magnitude
        confidence = bounded_expp1(confidence_logits)

        points = points.view(batch_size, num_frames, height, width, 3)
        confidence = confidence.view(batch_size, num_frames, height, width)
        if points.dtype != torch.float32 or confidence.dtype != torch.float32:
            raise TypeError(f"PointHead outputs must be fp32, got points={points.dtype}, conf={confidence.dtype}")
        return points, confidence

    def _scratch_forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        fused = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        fused = self.scratch.refinenet3(fused, layer_3_rn, size=layer_2_rn.shape[2:])
        fused = self.scratch.refinenet2(fused, layer_2_rn, size=layer_1_rn.shape[2:])
        doubled_size = (2 * layer_1_rn.shape[-2], 2 * layer_1_rn.shape[-1])
        fused = self.scratch.refinenet1(fused, layer_1_rn, size=doubled_size)
        return self.scratch.output_conv1(fused)

    @staticmethod
    def _apply_pos_embed(x: torch.Tensor, width: int, height: int, ratio: float = 0.1) -> torch.Tensor:
        pos_embed = create_uv_grid(
            x.shape[-1],
            x.shape[-2],
            aspect_ratio=width / height,
            dtype=x.dtype,
            device=x.device,
        )
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1]) * ratio
        return x + pos_embed.permute(2, 0, 1)[None]
