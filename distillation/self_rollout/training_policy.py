"""Install the strict geometry attention policy on distillation model instances."""
from __future__ import annotations

from types import MethodType

import torch

from .attention import (
    NOISE_GEOMETRY,
    SOURCE_HISTORY,
    STREAM_GEOMETRY,
    build_cache_visibility,
    build_token_metadata,
    from_mot_metadata,
    incremental_attention,
)


PROFILE_NAME = "segmented_history_strict_geometry_v1"


def _full_policy_attention(query, key, value, policy_meta, *, window_size):
    """Use sparse FlexAttention on CUDA and the dense reference elsewhere."""

    if query.device.type != "cuda":
        allow = build_cache_visibility(
            policy_meta,
            policy_meta,
            window_size=window_size,
        )
        return incremental_attention(query, key, value, allow), allow.any(dim=-1)

    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    cached = policy_meta.runtime_cache.get("flex_block_mask")
    if cached is None:
        def mask_mod(batch, head, q_idx, k_idx):
            del head
            q_seq = policy_meta.seq_ids[batch, q_idx]
            k_seq = policy_meta.seq_ids[batch, k_idx]
            q_frame = policy_meta.frame_ids[batch, q_idx]
            k_frame = policy_meta.frame_ids[batch, k_idx]
            q_order = policy_meta.order_ids[batch, q_idx]
            k_order = policy_meta.order_ids[batch, k_idx]
            q_stream = policy_meta.stream_ids[batch, q_idx]
            k_stream = policy_meta.stream_ids[batch, k_idx]
            q_noise = policy_meta.noise_ids[batch, q_idx]
            k_noise = policy_meta.noise_ids[batch, k_idx]
            valid = (
                (q_seq == k_seq)
                & (q_seq >= 0)
                & policy_meta.valid_ids[batch, q_idx]
                & policy_meta.valid_ids[batch, k_idx]
                & ((q_order - k_order).abs() <= window_size)
            )
            g_query = q_stream == STREAM_GEOMETRY
            g_key = k_stream == STREAM_GEOMETRY
            geometry = g_query & g_key & (k_frame < q_frame)
            x_query = (q_stream == 0) | (q_stream == 1)
            x_key = (k_stream == 0) | (k_stream == 1)
            clean_to_clean = (
                (q_noise == 1) & (k_noise == 1) & (k_order <= q_order)
            )
            noisy_to_clean = (
                (q_noise == 0) & (k_noise == 1) & (k_order < q_order)
            )
            noisy_to_noisy = (
                (q_noise == 0) & (k_noise == 0) & (k_order == q_order)
            )
            x_to_x = x_query & x_key & (
                clean_to_clean | noisy_to_clean | noisy_to_noisy
            )
            x_to_g = x_query & g_key & (k_order < q_order)
            return valid & (geometry | x_to_x | x_to_g)

        cached = create_block_mask(
            mask_mod,
            policy_meta.batch_size,
            1,
            policy_meta.seq_len,
            policy_meta.seq_len,
            device=policy_meta.device,
            _compile=True,
        )
        policy_meta.runtime_cache["flex_block_mask"] = cached
    output = flex_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        block_mask=cached,
    ).transpose(1, 2)
    # X queries always retain an allowed current/noise-stream key.  G has no
    # visible key for frame 0 by design; later frames have an immediate prior G.
    visible = policy_meta.valid_ids & (
        (policy_meta.stream_ids != STREAM_GEOMETRY)
        | (policy_meta.frame_ids > 0)
    )
    return output, visible


def _strict_block_forward(
    self,
    states,
    conditioning,
    text,
    video_rotary,
    action_rotary,
    meta,
    g_register=None,
    g_rotary=None,
):
    blocks = (
        self.video_block,
        self.video_block,
        self.action_block,
        self.action_block,
    )
    rotary_embeddings = (
        video_rotary,
        video_rotary,
        action_rotary,
        action_rotary,
    )
    modulations = tuple(
        self._modulation(block, temb)
        for block, temb in zip(blocks, conditioning)
    )
    qkv_parts = [
        self._self_qkv(block, hidden, modulation, rotary_emb)
        for block, hidden, modulation, rotary_emb in zip(
            blocks,
            states,
            modulations,
            rotary_embeddings,
        )
    ]
    lengths = [hidden.shape[1] for hidden in states]
    if self.geometry is None:
        if g_register is not None:
            raise ValueError("odd-layer V/A block cannot receive G registers")
    else:
        if g_register is None:
            raise ValueError("even-layer MoT block requires G registers")
        g_qkv = self.geometry.qkv_project(g_register.flatten(1, 2), g_rotary)
        qkv_parts.insert(2, g_qkv)
        lengths.insert(2, g_qkv[0].shape[1])

    query, key, value = (
        torch.cat(parts, dim=1) for parts in zip(*qkv_parts)
    )
    policy_meta = from_mot_metadata(meta)
    attention_output, visible = _full_policy_attention(
        query,
        key,
        value,
        policy_meta,
        window_size=int(meta.window_size),
    )
    attention_parts = list(torch.split(attention_output, lengths, dim=1))
    visible_parts = list(torch.split(visible, lengths, dim=1))

    g_updated = None
    if self.geometry is not None:
        g_output = attention_parts.pop(2)
        g_visible = visible_parts.pop(2)
        g_delta = self.geometry.attn_delta(g_output).reshape_as(g_register)
        g_delta = torch.where(
            g_visible.reshape_as(g_register[..., 0])[:, :, :, None],
            g_delta,
            torch.zeros_like(g_delta),
        )
        g_residual = g_register + g_delta
        g_updated = g_residual + self.geometry.ffn_delta(g_residual)

    self_outputs = []
    for block, output, visible in zip(blocks, attention_parts, visible_parts):
        projected = self._attention_output(block.attn1, output)
        projected = torch.where(
            visible[:, :, None],
            projected,
            torch.zeros_like(projected),
        )
        self_outputs.append(projected)
    text_attn = self.video_block.attn2
    text_key = text_attn.norm_k(text_attn.to_k(text)).unflatten(
        2,
        (text_attn.heads, -1),
    )
    text_value = text_attn.to_v(text).unflatten(2, (text_attn.heads, -1))
    updated_states = states.__class__(
        *(
            self._finish_block(
                block,
                hidden,
                self_output,
                modulation,
                text_key,
                text_value,
            )
            for block, hidden, self_output, modulation in zip(
                blocks,
                states,
                self_outputs,
                modulations,
            )
        )
    )
    return updated_states, g_updated


def _strict_geometry_forward(
    self,
    registers,
    rotary_emb,
    meta,
    *,
    masked_attn_backend,
):
    del masked_attn_backend
    query, key, value = self.qkv_project(registers.flatten(1, 2), rotary_emb)
    policy_meta = from_mot_metadata(meta)
    attention_output, visible = _full_policy_attention(
        query,
        key,
        value,
        policy_meta,
        window_size=int(meta.window_size),
    )
    delta = self.attn_delta(attention_output).reshape_as(registers)
    visible = visible.reshape_as(registers[..., 0])
    delta = torch.where(visible[:, :, :, None], delta, torch.zeros_like(delta))
    residual = registers + delta
    return residual + self.ffn_delta(residual)


def _strict_same_view_inter_frame(
    self,
    grouped_tokens,
    patch_hw,
    layer_id,
    *,
    groups_per_chunk,
    slot_valid_mask,
):
    del groups_per_chunk
    batch_size, groups, group_size, views, num_tokens, channels = grouped_tokens.shape
    view_major = grouped_tokens.permute(0, 3, 1, 2, 4, 5)
    values = view_major.reshape(
        batch_size * views,
        groups * group_size * num_tokens,
        channels,
    )
    block = self.inter_frame_blocks[layer_id]
    attention = block.attn
    qkv = attention.qkv(block.norm1(values)).reshape(
        values.shape[0],
        values.shape[1],
        3,
        attention.num_heads,
        channels // attention.num_heads,
    )
    query, key, value = qkv.unbind(dim=2)
    if attention.use_qk_norm:
        query = attention.q_norm(query)
        key = attention.k_norm(key)
    rope = self._full_token_rope(
        patch_hw,
        groups * group_size,
        values.device,
    )
    query_heads, key_heads = query.transpose(1, 2), key.transpose(1, 2)
    query_heads, key_heads = attention.apply_rope(query_heads, key_heads, rope)
    query, key = query_heads.transpose(1, 2), key_heads.transpose(1, 2)

    frame_ids = torch.arange(groups, device=values.device, dtype=torch.long)
    frame_ids = frame_ids[:, None].expand(groups, group_size).reshape(-1)
    frame_ids = frame_ids[None].expand(batch_size * views, -1)
    valid = None
    if slot_valid_mask is not None:
        valid = (
            slot_valid_mask.to(device=values.device, dtype=torch.bool)[:, None]
            .expand(-1, views, -1, -1)
            .reshape(batch_size * views, groups * group_size)
        )
    cache_key = (
        batch_size,
        groups,
        group_size,
        views,
        num_tokens,
        values.device,
        (
            None
            if slot_valid_mask is None
            else tuple(slot_valid_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1).tolist())
        ),
    )
    cached_meta = getattr(self, "_distillation_inter_frame_metadata", None)
    if cached_meta is None or cached_meta[0] != cache_key:
        metadata = build_token_metadata(
            batch_size=batch_size * views,
            frame_ids=frame_ids,
            tokens_per_frame=num_tokens,
            stream_id=STREAM_GEOMETRY,
            noise_id=NOISE_GEOMETRY,
            history_frames=int(getattr(self, "_distillation_history_frames")),
            chunk_size=int(getattr(self, "_distillation_chunk_size")),
            device=values.device,
            valid_ids=valid,
            committed=True,
            source_id=SOURCE_HISTORY,
        )
        self._distillation_inter_frame_metadata = (cache_key, metadata)
    else:
        metadata = cached_meta[1]
    attended, visible = _full_policy_attention(
        query,
        key,
        value,
        metadata,
        window_size=int(getattr(self, "_distillation_window_size")),
    )
    projected = attention.proj(attended.flatten(2, 3))
    projected = attention.proj_drop(projected)
    projected = torch.where(
        visible[:, :, None],
        projected,
        torch.zeros_like(projected),
    )
    values = values + block.ls1(projected)
    values = values + block.ls2(block.mlp(block.norm2(values)))
    return (
        values.reshape(
            batch_size,
            views,
            groups,
            group_size,
            num_tokens,
            channels,
        )
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(batch_size, groups * group_size * views, num_tokens, channels)
    )


def install_training_attention_policy(model, generation_shape) -> None:
    """Install the strict policy on one model instance without touching wan_va."""

    model = getattr(model, "module", model)
    profile_name = str(generation_shape.get("profile_name", PROFILE_NAME))
    if profile_name != PROFILE_NAME:
        raise ValueError(
            f"unsupported distillation attention profile {profile_name!r}"
        )
    if getattr(model, "_distillation_attention_profile", None) == PROFILE_NAME:
        return
    history_frames = int(generation_shape.get("history_frames", 4))
    chunk_size = int(generation_shape["chunk_size"])
    window_size = int(generation_shape["window_size"])
    for block in model.mot_blocks:
        block.forward = MethodType(_strict_block_forward, block)
        if block.geometry is not None:
            block.geometry.forward = MethodType(
                _strict_geometry_forward,
                block.geometry,
            )
    model.vggto._distillation_history_frames = history_frames
    model.vggto._distillation_chunk_size = chunk_size
    model.vggto._distillation_window_size = window_size
    model.vggto._run_same_view_inter_frame_block = MethodType(
        _strict_same_view_inter_frame,
        model.vggto,
    )
    model._distillation_attention_profile = PROFILE_NAME
