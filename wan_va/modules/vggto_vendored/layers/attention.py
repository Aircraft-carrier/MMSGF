# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Tuple

from torch import Tensor, nn
import torch
import torch.nn.functional as F

from ...fa4_attention import fa4_chunk_causal_attention, fa4_full_attention
from .utils import cat_keep_shapes, uncat_with_shapes


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        use_qk_norm: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        # VGGT-Omega change: the aggregator checkpoint was trained with Q/K
        # normalization, while upstream DINOv3 attention does not expose it.
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(head_dim, eps=1e-5)
        else:
            self.q_norm = None
            self.k_norm = None

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(
        self,
        x: Tensor,
        attn_bias=None,
        rope: Tensor | Tuple[Tensor, Tensor] = None,
        chunk_causal_spec=None,
        attention_backend: str = "dense",
        row_valid_mask: Tensor | None = None,
    ) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(
            qkv=qkv,
            attn_bias=attn_bias,
            rope=rope,
            chunk_causal_spec=chunk_causal_spec,
            attention_backend=attention_backend,
            row_valid_mask=row_valid_mask,
        )
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    @staticmethod
    def _dense_full_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2)

    def _full_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        attention_backend: str,
        row_valid_mask: Tensor | None,
    ) -> Tensor:
        if attention_backend not in {"dense", "fa4"}:
            raise ValueError(f"unsupported VGGTO relation attention backend: {attention_backend!r}")

        if row_valid_mask is None:
            if attention_backend == "fa4":
                return fa4_full_attention(q, k, v)
            return self._dense_full_attention(q, k, v)

        row_valid_mask = row_valid_mask.to(device=q.device, dtype=torch.bool)
        if tuple(row_valid_mask.shape) != (q.shape[0],):
            raise ValueError(
                f"row_valid_mask must be [{q.shape[0]}], got {tuple(row_valid_mask.shape)}"
            )

        # A fully invalid cross-view timestamp uses a per-token self diagonal.
        # Its exact pre-projection attention output is therefore V, with zero
        # Q/K gradient.  Valid timestamp rows use ordinary full attention.
        valid_indices = row_valid_mask.nonzero(as_tuple=False).flatten()
        if valid_indices.numel() == q.shape[0]:
            if attention_backend == "fa4":
                return fa4_full_attention(q, k, v)
            return self._dense_full_attention(q, k, v)
        if valid_indices.numel() == 0:
            return v

        q_valid = q.index_select(0, valid_indices)
        k_valid = k.index_select(0, valid_indices)
        v_valid = v.index_select(0, valid_indices)
        if attention_backend == "fa4":
            valid_out = fa4_full_attention(q_valid, k_valid, v_valid)
        else:
            valid_out = self._dense_full_attention(q_valid, k_valid, v_valid)
        return v.index_copy(0, valid_indices, valid_out)

    def compute_attention(
        self,
        qkv: Tensor,
        attn_bias=None,
        rope=None,
        chunk_causal_spec=None,
        attention_backend: str = "dense",
        row_valid_mask: Tensor | None = None,
    ) -> Tensor:
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if rope is not None:
            q_heads, k_heads = q.transpose(1, 2), k.transpose(1, 2)
            q_heads, k_heads = self.apply_rope(q_heads, k_heads, rope)
            q, k = q_heads.transpose(1, 2), k_heads.transpose(1, 2)

        if chunk_causal_spec is not None:
            if attention_backend != "fa4":
                raise ValueError("ChunkCausalMaskSpec requires attention_backend='fa4'")
            if attn_bias is not None or row_valid_mask is not None:
                raise ValueError("FA4 chunk-causal attention does not accept a dense mask or row_valid_mask")
            x = fa4_chunk_causal_attention(q, k, v, chunk_causal_spec)
            return x.reshape(B, N, C)

        if attn_bias is not None:
            if attention_backend != "dense" or row_valid_mask is not None:
                raise ValueError("dense attention masks cannot be combined with FA4 or row_valid_mask")
            q, k, v = [tensor.transpose(1, 2) for tensor in (q, k, v)]
            if attn_bias.dtype == torch.bool and attn_bias.ndim == 2:
                attn_bias = attn_bias[None, None]
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
            return x.transpose(1, 2).reshape(B, N, C)

        x = self._full_attention(
            q,
            k,
            v,
            attention_backend=attention_backend,
            row_valid_mask=row_valid_mask,
        )
        return x.reshape(B, N, C)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
