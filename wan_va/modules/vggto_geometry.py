"""VGGTO geometry tower for 3DVA_MOT.

VGGTO is the VGGT-Omega-derived geometry path with Pi3-like scene-register
semantics: no camera token, no first-frame special register, and a shared
register table for every frame/view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn

from .fa4_attention import ChunkCausalMaskSpec
from .vggto_vendored.heads import DenseHead, PointHead
from .vggto_vendored.layers import LayerScale, Mlp, RopePositionEmbedding, SelfAttentionBlock
from .vggto_vendored.layers.vision_transformer import DinoVisionTransformer


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


@dataclass
class VGGTOGeometryState:
    tokens: torch.Tensor
    patch_hw: tuple[int, int]
    image_hw: tuple[int, int]
    patch_token_start: int

    @property
    def registers(self) -> torch.Tensor:
        return self.tokens[:, :, : self.patch_token_start]


def build_chunk_causal_visibility_mask(
    image_chunk_ids: torch.Tensor,
    tokens_per_image: int,
    *,
    image_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build chunk-causal visibility for one camera's flattened timestamps."""

    device = image_chunk_ids.device
    token_chunk_ids = image_chunk_ids.repeat_interleave(tokens_per_image)
    mask = token_chunk_ids[None, :] <= token_chunk_ids[:, None]
    if image_valid_mask is None:
        return mask

    image_valid_mask = image_valid_mask.to(device=device, dtype=torch.bool)
    if image_valid_mask.ndim == 1:
        token_valid = image_valid_mask.repeat_interleave(tokens_per_image)[None]
    elif image_valid_mask.ndim == 2:
        token_valid = image_valid_mask.repeat_interleave(tokens_per_image, dim=1)
    else:
        raise ValueError(f"image_valid_mask must be [F] or [B,F], got {tuple(image_valid_mask.shape)}")
    mask = mask[None, None] & token_valid[:, None, :, None] & token_valid[:, None, None, :]
    token_count = token_chunk_ids.numel()
    eye = torch.eye(token_count, device=device, dtype=torch.bool)[None, None]
    invalid_queries = ~token_valid[:, None, :, None]
    return torch.where(invalid_queries, eye, mask)


def _chunk_causal_spec(
    *,
    batch_size: int,
    frames: int,
    frames_per_chunk: int,
    tokens_per_frame: int,
    image_valid_mask: torch.Tensor | None,
    device: torch.device,
) -> ChunkCausalMaskSpec:
    if frames_per_chunk <= 0:
        raise ValueError(f"frames_per_chunk must be positive, got {frames_per_chunk}")

    token_valid_ids = None
    if image_valid_mask is not None:
        image_valid = image_valid_mask.to(device=device, dtype=torch.bool)
        if image_valid.ndim == 1:
            if tuple(image_valid.shape) != (frames,):
                raise ValueError(f"image_valid_mask must be [{frames}], got {tuple(image_valid.shape)}")
            image_valid = image_valid[None].expand(batch_size, -1)
        elif tuple(image_valid.shape) != (batch_size, frames):
            raise ValueError(
                f"image_valid_mask must be [B,F]=[{batch_size},{frames}], got {tuple(image_valid.shape)}"
            )
        token_valid_ids = image_valid.repeat_interleave(tokens_per_frame, dim=1)

    return ChunkCausalMaskSpec(
        frames=frames,
        frames_per_chunk=frames_per_chunk,
        tokens_per_frame=tokens_per_frame,
        token_valid_ids=token_valid_ids,
    )


def init_shared_register_from_first_other_(target: nn.Parameter, source: torch.Tensor) -> None:
    """Average VGGT-Omega first/other register tables into VGGTO shared registers."""

    if source.ndim != 4 or source.shape[1] != 2:
        raise ValueError(f"expected source register shape [1,2,R,C], got {tuple(source.shape)}")
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(f"expected target register shape [1,1,R,C], got {tuple(target.shape)}")
    if source.shape[0] != target.shape[0] or source.shape[-1] != target.shape[-1]:
        raise ValueError(f"incompatible register shapes: target={tuple(target.shape)}, source={tuple(source.shape)}")
    with torch.no_grad():
        shared = source.mean(dim=1, keepdim=True).to(device=target.device, dtype=target.dtype)
        copy_tokens = min(target.shape[2], shared.shape[2])
        target[:, :, :copy_tokens].copy_(shared[:, :, :copy_tokens])
        if target.shape[2] > copy_tokens:
            nn.init.normal_(target[:, :, copy_tokens:], std=1e-6)


class VGGTOGeometryTower(nn.Module):
    def __init__(
        self,
        *,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 30,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: Iterable[int] = tuple(range(0, 30, 2)),
        cached_layer_indices: Iterable[int] = (5, 15, 21, 29),
        dense_features: int = 256,
        dense_out_channels: Iterable[int] = (256, 512, 1024, 1024),
        masked_attn_backend: str = "dense",
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.num_register_tokens = int(num_register_tokens)
        self.patch_token_start = self.num_register_tokens
        self.patch_start_idx = self.patch_token_start
        self.cached_layer_indices = tuple(int(i) for i in cached_layer_indices)
        if len(self.cached_layer_indices) != 4 or len(set(self.cached_layer_indices)) != 4:
            raise ValueError("cached_layer_indices must contain four unique layer indices")
        invalid_cached_indices = [i for i in self.cached_layer_indices if i < 0 or i >= self.depth]
        if invalid_cached_indices:
            raise ValueError(f"cached layer indices must be in [0,{self.depth}), got {invalid_cached_indices}")
        self.cached_layer_set = set(self.cached_layer_indices)
        register_attention_indices = tuple(int(i) for i in register_attention_block_indices)
        if len(set(register_attention_indices)) != len(register_attention_indices):
            raise ValueError("register_attention_block_indices must be unique")
        invalid_indices = [i for i in register_attention_indices if i < 0 or i >= self.depth]
        if invalid_indices:
            raise ValueError(
                f"register attention indices must be in [0,{self.depth}), got {invalid_indices}"
            )
        self.register_attention_indices = set(register_attention_indices)
        if self.cached_layer_set & self.register_attention_indices:
            raise ValueError("cached_layer_indices must use native full-token inter-frame layers")
        self.masked_attn_backend = str(masked_attn_backend)

        self.patch_embed = self._build_patch_embed(self.patch_size, self.embed_dim)

        self.rope_embed = RopePositionEmbedding(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )
        self.frame_blocks = nn.ModuleList([self._new_block(mlp_ratio) for _ in range(self.depth)])
        self.cross_view_blocks = nn.ModuleList([
            nn.Identity() if layer_id in self.register_attention_indices else self._new_block(mlp_ratio)
            for layer_id in range(self.depth)
        ])
        self.inter_frame_blocks = nn.ModuleList([
            nn.Identity() if layer_id in self.register_attention_indices else self._new_block(mlp_ratio)
            for layer_id in range(self.depth)
        ])
        self.register_token = nn.Parameter(torch.empty(1, 1, self.num_register_tokens, self.embed_dim))
        self.dense_head = DenseHead(
            dim_in=2 * self.embed_dim,
            patch_size=self.patch_size,
            features=int(dense_features),
            out_channels=list(dense_out_channels),
            intermediate_layer_idx=list(self.cached_layer_indices),
        )
        self.point_head = PointHead(
            dim_in=2 * self.embed_dim,
            patch_size=self.patch_size,
            features=int(dense_features),
            out_channels=list(dense_out_channels),
            intermediate_layer_idx=list(self.cached_layer_indices),
        )

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1))

        nn.init.normal_(self.register_token, std=1e-3)
        self._init_layer_scales()
        self._init_attention_bias_masks()

    def keep_heads_fp32_(self) -> "VGGTOGeometryTower":
        """Restore the FP32 dense-head contract after casting the carrier model."""

        self.dense_head.float()
        self.point_head.float()
        return self

    def _new_block(self, mlp_ratio: float) -> SelfAttentionBlock:
        return SelfAttentionBlock(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            ffn_ratio=mlp_ratio,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            ffn_layer=Mlp,
            init_values=1e-5,
            use_qk_norm=True,
            mask_k_bias=True,
        )

    def _build_patch_embed(self, patch_size: int, embed_dim: int) -> DinoVisionTransformer:
        model = DinoVisionTransformer(
            img_size=224,
            patch_size=patch_size,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="max",
            pos_embed_rope_dtype="fp32",
            embed_dim=embed_dim,
            depth=24,
            num_heads=16,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-5,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        )
        model.init_weights()
        return model

    def _init_attention_bias_masks(self) -> None:
        for module in self.modules():
            bias_mask = getattr(module, "bias_mask", None)
            if bias_mask is None:
                continue
            out_features = module.out_features
            bias_mask.fill_(1)
            bias_mask[out_features // 3 : 2 * out_features // 3].fill_(0)

    def _init_layer_scales(self) -> None:
        for module in self.modules():
            if isinstance(module, LayerScale):
                module.reset_parameters()

    def _patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        return patch_tokens

    def encode(self, images: torch.Tensor) -> VGGTOGeometryState:
        if images.ndim != 5:
            raise ValueError(f"VGGTO images must be [B,F,3,H,W], got {tuple(images.shape)}")
        bsz, frames, channels, height, width = images.shape
        if channels != 3:
            raise ValueError(f"VGGTO image channel dimension must be 3, got {channels}")
        if height % self.patch_size != 0 or width % self.patch_size != 0:
            raise ValueError(f"VGGTO H/W must be divisible by patch_size={self.patch_size}, got {height}x{width}")

        normed = (images - self._resnet_mean.to(images)) / self._resnet_std.to(images)
        patch_tokens = self._patch_tokens(normed.reshape(bsz * frames, channels, height, width))
        register = self.register_token.expand(bsz, frames, -1, -1).reshape(
            bsz * frames,
            self.num_register_tokens,
            self.embed_dim,
        )
        tokens = torch.cat([register.to(patch_tokens.dtype), patch_tokens], dim=1)
        patch_hw = (height // self.patch_size, width // self.patch_size)
        return VGGTOGeometryState(
            tokens=tokens.reshape(bsz, frames, tokens.shape[1], self.embed_dim),
            patch_hw=patch_hw,
            image_hw=(height, width),
            patch_token_start=self.patch_token_start,
        )

    def encode_grouped(self, rgb: torch.Tensor, slot_valid_mask: Optional[torch.Tensor] = None) -> VGGTOGeometryState:
        if rgb.ndim == 7:
            bsz, groups, group_size, views, channels, height, width = rgb.shape
            if views < 2:
                raise ValueError(f"synchronized multi-view VGGTO requires V>=2, got V={views}")
            if slot_valid_mask is not None and tuple(slot_valid_mask.shape) != (bsz, groups, group_size):
                raise ValueError(
                    f"slot_valid_mask must be [B,G,S]=[{bsz},{groups},{group_size}], "
                    f"got {tuple(slot_valid_mask.shape)}"
                )
            flat = rgb.reshape(bsz, groups * group_size * views, channels, height, width)
        elif rgb.ndim == 6:
            bsz, frames, views, channels, height, width = rgb.shape
            if views < 2:
                raise ValueError(f"synchronized multi-view VGGTO requires V>=2, got V={views}")
            flat = rgb.reshape(bsz, frames * views, channels, height, width)
        else:
            raise ValueError(f"VGGTO grouped rgb must be [B,G,S,V,3,H,W] or [B,F,V,3,H,W], got {tuple(rgb.shape)}")
        del slot_valid_mask
        return self.encode(flat)

    def _frame_rope(self, patch_hw: tuple[int, int], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            sin, cos = self.rope_embed(H=patch_hw[0], W=patch_hw[1])
        return sin.to(device=device, dtype=torch.float32), cos.to(device=device, dtype=torch.float32)

    def _full_token_rope(
        self,
        patch_hw: tuple[int, int],
        images_per_sequence: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images_per_sequence <= 0:
            raise ValueError(f"images_per_sequence must be positive, got {images_per_sequence}")
        patch_sin, patch_cos = self._frame_rope(patch_hw, device)
        register_shape = (self.patch_token_start, patch_sin.shape[-1])
        image_sin = torch.cat(
            [torch.zeros(register_shape, device=device, dtype=torch.float32), patch_sin],
            dim=0,
        )
        image_cos = torch.cat(
            [torch.ones(register_shape, device=device, dtype=torch.float32), patch_cos],
            dim=0,
        )
        return image_sin.repeat(images_per_sequence, 1), image_cos.repeat(images_per_sequence, 1)

    def run_frame_block(
        self,
        tokens: torch.Tensor,
        patch_hw: tuple[int, int],
        layer_id: int,
    ) -> torch.Tensor:
        """Run one layer's per-image block without its inter-frame block."""

        bsz, frames, num_tokens, channels = tokens.shape
        return self.frame_blocks[layer_id](
            tokens.reshape(bsz * frames, num_tokens, channels),
            self._frame_rope(patch_hw, tokens.device),
        ).reshape(bsz, frames, num_tokens, channels)

    @staticmethod
    def _validate_grouped_layout(
        frame_tokens: torch.Tensor,
        *,
        groups: int,
        group_size: int,
        views: int,
        groups_per_chunk: int,
        slot_valid_mask: torch.Tensor | None,
    ) -> None:
        if frame_tokens.ndim != 4:
            raise ValueError(f"frame_tokens must be [B,F,N,C], got {tuple(frame_tokens.shape)}")
        bsz, frames = frame_tokens.shape[:2]
        if groups <= 0 or group_size <= 0:
            raise ValueError(f"groups and group_size must be positive, got G={groups}, S={group_size}")
        if views < 2:
            raise ValueError(f"synchronized multi-view VGGTO requires V>=2, got V={views}")
        if groups_per_chunk <= 0 or groups % groups_per_chunk != 0:
            raise ValueError(
                f"groups_per_chunk must positively divide G={groups}, got {groups_per_chunk}"
            )
        expected_frames = groups * group_size * views
        if frames != expected_frames:
            raise ValueError(
                f"frame_tokens F must equal G*S*V={expected_frames}, got F={frames}"
            )
        if slot_valid_mask is not None and tuple(slot_valid_mask.shape) != (bsz, groups, group_size):
            raise ValueError(
                f"slot_valid_mask must be [B,G,S]=[{bsz},{groups},{group_size}], "
                f"got {tuple(slot_valid_mask.shape)}"
            )

    def _run_cross_view_block(
        self,
        frame_tokens: torch.Tensor,
        patch_hw: tuple[int, int],
        layer_id: int,
        *,
        groups: int,
        group_size: int,
        views: int,
        slot_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, _frames, num_tokens, channels = frame_tokens.shape
        grouped = frame_tokens.reshape(bsz, groups, group_size, views, num_tokens, channels)
        values = grouped.reshape(bsz * groups * group_size, views * num_tokens, channels)
        row_valid_mask = None
        if slot_valid_mask is not None:
            row_valid_mask = slot_valid_mask.to(device=values.device, dtype=torch.bool).reshape(-1)
        backend = "fa4" if self.masked_attn_backend == "fa4" else "dense"
        values = self.cross_view_blocks[layer_id](
            values,
            self._full_token_rope(patch_hw, views, values.device),
            attention_backend=backend,
            row_valid_mask=row_valid_mask,
        )
        return values.reshape(bsz, groups, group_size, views, num_tokens, channels)

    def _run_same_view_inter_frame_block(
        self,
        grouped_tokens: torch.Tensor,
        patch_hw: tuple[int, int],
        layer_id: int,
        *,
        groups_per_chunk: int,
        slot_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        bsz, groups, group_size, views, num_tokens, channels = grouped_tokens.shape
        view_major = grouped_tokens.permute(0, 3, 1, 2, 4, 5)
        values = view_major.reshape(bsz * views, groups * group_size * num_tokens, channels)
        image_valid_mask = None
        if slot_valid_mask is not None:
            image_valid_mask = (
                slot_valid_mask.to(device=values.device, dtype=torch.bool)[:, None]
                .expand(-1, views, -1, -1)
                .reshape(bsz * views, groups * group_size)
            )
        rope = self._full_token_rope(patch_hw, groups * group_size, values.device)
        if self.masked_attn_backend == "fa4":
            spec = _chunk_causal_spec(
                batch_size=bsz * views,
                frames=groups * group_size,
                frames_per_chunk=groups_per_chunk * group_size,
                tokens_per_frame=num_tokens,
                image_valid_mask=image_valid_mask,
                device=values.device,
            )
            values = self.inter_frame_blocks[layer_id](
                values,
                rope,
                chunk_causal_spec=spec,
                attention_backend="fa4",
            )
        else:
            image_chunk_ids = (
                torch.arange(groups, device=values.device) // groups_per_chunk
            ).repeat_interleave(group_size)
            mask = build_chunk_causal_visibility_mask(
                image_chunk_ids,
                num_tokens,
                image_valid_mask=image_valid_mask,
            )
            values = self.inter_frame_blocks[layer_id](values, rope, attn_bias=mask)
        return (
            values.reshape(bsz, views, groups, group_size, num_tokens, channels)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(bsz, groups * group_size * views, num_tokens, channels)
        )

    def complete_layer(
        self,
        frame_tokens: torch.Tensor,
        patch_hw: tuple[int, int],
        layer_id: int,
        *,
        groups: int,
        group_size: int,
        views: int,
        groups_per_chunk: int,
        slot_valid_mask: torch.Tensor | None = None,
        register_override: torch.Tensor | None = None,
        cached_outputs: Optional[list[torch.Tensor | None]] = None,
    ) -> torch.Tensor:
        """Finish one layer with relation blocks or the aligned MoT registers."""

        bsz, frames, num_tokens, channels = frame_tokens.shape
        self._validate_grouped_layout(
            frame_tokens,
            groups=groups,
            group_size=group_size,
            views=views,
            groups_per_chunk=groups_per_chunk,
            slot_valid_mask=slot_valid_mask,
        )

        if register_override is not None:
            if layer_id not in self.register_attention_indices:
                raise ValueError(f"register_override is only valid for register-attention layers, got layer {layer_id}")
            expected_shape = (bsz, frames, self.patch_token_start, channels)
            if tuple(register_override.shape) != expected_shape:
                raise ValueError(
                    f"register_override must have shape {expected_shape}, got {tuple(register_override.shape)}"
                )
            inter_tokens = torch.cat([register_override, frame_tokens[:, :, self.patch_token_start :]], dim=2)
        elif layer_id in self.register_attention_indices:
            raise ValueError(
                f"register-attention layer {layer_id} requires register_override from the aligned MOT block"
            )
        else:
            cross_view_tokens = self._run_cross_view_block(
                frame_tokens,
                patch_hw,
                layer_id,
                groups=groups,
                group_size=group_size,
                views=views,
                slot_valid_mask=slot_valid_mask,
            )
            inter_tokens = self._run_same_view_inter_frame_block(
                cross_view_tokens,
                patch_hw,
                layer_id,
                groups_per_chunk=groups_per_chunk,
                slot_valid_mask=slot_valid_mask,
            )

        if cached_outputs is not None and layer_id in self.cached_layer_set:
            cached_outputs[layer_id] = torch.cat([frame_tokens, inter_tokens], dim=-1)
        return inter_tokens

    def dense_forward(
        self,
        cached_outputs: list[torch.Tensor | None],
        images: torch.Tensor,
        *,
        frames_chunk_size: int | None = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.dense_head(
            cached_outputs,
            images=images,
            patch_token_start=self.patch_token_start,
            frames_chunk_size=frames_chunk_size,
        )

    def point_forward(
        self,
        cached_outputs: list[torch.Tensor | None],
        images: torch.Tensor,
        *,
        frames_chunk_size: int | None = 8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.point_head(
            cached_outputs,
            images=images,
            patch_token_start=self.patch_token_start,
            frames_chunk_size=frames_chunk_size,
        )
