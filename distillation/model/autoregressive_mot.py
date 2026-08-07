"""Independent autoregressive MoT model family.

The class is parameter-compatible with ThreeDVAMOTTransformer3DModel.  Its
training path owns the AR order natively; rollout integration is added through
the normal model forward boundary so FSDP hooks remain effective.
"""
from __future__ import annotations

from typing import Any, Iterable

import torch

from distillation.self_rollout.attention import segmented_orders
from wan_va.modules.model_3dva_mot import (
    ThreeDVAMOTBlock,
    ThreeDVAMOTTransformer3DModel,
)

from .autoregressive_types import AutoregressiveProfile
from .autoregressive_types import (
    AutoregressiveGeometryJointRequest,
    AutoregressiveMOTLayerRequest,
    AutoregressiveModelOutput,
    AutoregressiveModelRequest,
    AutoregressiveStreamInput,
)
from .autoregressive_vggto import AutoregressiveVGGTOGeometryTower


class AutoregressiveThreeDVAMOTBlock(ThreeDVAMOTBlock):
    """Parameter-compatible MoT block reserved for native AR execution."""

    def forward(self, states, *args, **kwargs):
        if isinstance(states, AutoregressiveGeometryJointRequest):
            return self.forward_geometry_incremental(states)
        if isinstance(states, AutoregressiveMOTLayerRequest):
            return self.forward_incremental(states)
        return super().forward(states, *args, **kwargs)

    def forward_incremental(self, request: AutoregressiveMOTLayerRequest):
        from distillation.self_rollout.attention import (
            build_cache_selection,
            indexed_attention,
        )
        from distillation.self_rollout.cache import KVSegment

        stream = request.stream
        hidden = request.hidden_state
        block = (
            self.video_block
            if stream.block_kind == "video"
            else self.action_block
        )
        modulation = self._modulation(block, stream.conditioning)
        query, current_key, current_value = self._self_qkv(
            block,
            hidden,
            modulation,
            stream.rotary,
        )
        request.state.mot_cache.append_transaction(
            request.layer_id,
            request.transaction_id,
            KVSegment(
                current_key,
                current_value,
                stream.metadata,
                stream_id=request.stream_id,
            ),
        )
        layer_id = request.layer_id
        key, value, key_meta = request.state.mot_cache.materialize(
            layer_id,
            transaction_id=request.transaction_id,
        )
        query_valid, key_valid = build_cache_selection(stream.metadata, key_meta)
        attention_output, _ = indexed_attention(
            query,
            key,
            value,
            query_valid=query_valid,
            key_valid=key_valid,
        )
        text_attn = self.video_block.attn2
        text_key = text_attn.norm_k(text_attn.to_k(request.text)).unflatten(
            2, (text_attn.heads, -1)
        )
        text_value = text_attn.to_v(request.text).unflatten(
            2, (text_attn.heads, -1)
        )
        self_output = self._attention_output(block.attn1, attention_output)
        return self._finish_block(
            block,
            hidden,
            self_output,
            modulation,
            text_key,
            text_value,
        )

    def forward_geometry_incremental(self, request: AutoregressiveGeometryJointRequest):
        from distillation.self_rollout.attention import (
            STREAM_GEOMETRY,
            build_cache_selection,
            indexed_attention,
        )
        from distillation.self_rollout.cache import KVSegment

        geometry = self.geometry
        if geometry is None:
            raise ValueError("geometry incremental request sent to an odd MoT layer")
        registers = request.registers.to(dtype=getattr(geometry.norm1, "weight", request.registers).dtype)
        query, key, value = geometry.qkv_project(registers.flatten(1, 2), request.rotary)
        request.cache.mot_cache.append_transaction(
            request.layer_id,
            request.transaction_id,
            KVSegment(key, value, request.metadata, stream_id=STREAM_GEOMETRY),
        )
        cache_key, cache_value, cache_meta = request.cache.mot_cache.materialize(
            request.layer_id,
            transaction_id=request.transaction_id,
            stream_id=STREAM_GEOMETRY,
        )
        query_valid, key_valid = build_cache_selection(request.metadata, cache_meta)
        attended, visible = indexed_attention(
            query,
            cache_key,
            cache_value,
            query_valid=query_valid,
            key_valid=key_valid,
        )
        delta = geometry.attn_delta(attended).reshape_as(registers)
        visible = visible.reshape(
            registers.shape[0], request.groups * request.slots * request.views, registers.shape[2]
        )
        delta = torch.where(visible[:, :, :, None], delta, torch.zeros_like(delta))
        residual = registers + delta
        updated = residual + geometry.ffn_delta(residual)
        if request.slot_valid_mask is not None:
            valid = (
                request.slot_valid_mask[:, :, :, None]
                .to(device=updated.device, dtype=torch.bool)
                .expand(-1, -1, -1, request.views)
                .reshape(updated.shape[0], request.groups * request.slots * request.views)
            )
            updated = torch.where(valid[:, :, None, None], updated, registers)
        return updated


class AutoregressiveThreeDVAMOTTransformer3DModel(
    ThreeDVAMOTTransformer3DModel
):
    """MOT transformer with AR metadata and AR component construction."""

    _no_split_modules = ["AutoregressiveThreeDVAMOTBlock"]
    _repeated_blocks = ["AutoregressiveThreeDVAMOTBlock"]

    def __init__(self, *args, generation_profile=None, **kwargs):
        if isinstance(generation_profile, dict):
            generation_profile = AutoregressiveProfile.from_generation_shape(generation_profile)
        self.generation_profile = generation_profile or AutoregressiveProfile(
            profile_name="segmented_history_strict_geometry_v1",
            profile_version=2,
            order_mode="segmented",
            history_frames=4,
            chunk_size=4,
            window_size=16,
            geometry_relation="segmented_order_causal",
            x_to_g_relation="strict_order",
        )
        self.generation_profile.validate()
        super().__init__(*args, **kwargs)
        self.vggto.configure_generation_profile(self.generation_profile)
        self.register_to_config(generation_profile=self.generation_profile.as_dict())

    def _build_vggto_tower(self, **kwargs):
        return AutoregressiveVGGTOGeometryTower(
            **kwargs,
            generation_profile=self.generation_profile,
        )

    def _build_mot_block(self, **kwargs):
        return AutoregressiveThreeDVAMOTBlock(**kwargs)

    def configure_generation_profile(self, generation_shape: Any) -> None:
        profile = (
            generation_shape
            if isinstance(generation_shape, AutoregressiveProfile)
            else AutoregressiveProfile.from_generation_shape(generation_shape)
        )
        self.generation_profile = profile
        self.vggto.configure_generation_profile(profile)
        self.register_to_config(generation_profile=profile.as_dict())

    def _prepare_metadata(self, input_dict, prepared, geometry):
        x_meta, mot_meta, diagnostics = super()._prepare_metadata(
            input_dict,
            prepared,
            geometry,
        )
        profile = self.generation_profile
        chunk_size = int(input_dict.get("chunk_size", profile.chunk_size))
        return (
            self._apply_segmented_order(
                x_meta,
                history_frames=profile.history_frames,
                chunk_size=chunk_size,
            ),
            self._apply_segmented_order(
                mot_meta,
                history_frames=profile.history_frames,
                chunk_size=chunk_size,
            ),
            diagnostics,
        )

    @staticmethod
    def _segmented_frame_orders(
        *,
        num_frames: int,
        history_frames: int,
        chunk_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_order = segmented_orders(
            torch.arange(num_frames, device=device),
            history_frames=history_frames,
            chunk_size=chunk_size,
        )
        return video_order, video_order + 1

    @classmethod
    def _apply_segmented_order(
        cls,
        metadata,
        *,
        history_frames: int,
        chunk_size: int,
    ):
        if metadata.frame_ids is None:
            raise ValueError("segmented distillation order requires frame_ids")
        video_order, action_order = cls._segmented_frame_orders(
            num_frames=int(metadata.frame_ids.max().item()) + 1,
            history_frames=history_frames,
            chunk_size=chunk_size,
            device=metadata.device,
        )
        frame_order = video_order[metadata.frame_ids]
        metadata.order_ids = torch.where(
            metadata.stream_ids == 1,
            action_order[metadata.frame_ids],
            frame_order, # action 是独立的， video 和 geometry 是共享的
        )
        metadata.cache_key = None
        metadata.structure_cache_key = None
        return metadata

    def _text(self, text_emb: torch.Tensor, batch_size: int) -> torch.Tensor:
        if text_emb.shape[0] == 1 and batch_size != 1:
            text_emb = text_emb.expand(batch_size, *text_emb.shape[1:])
        if text_emb.shape[0] != batch_size:
            raise ValueError(
                f"text batch {text_emb.shape[0]} does not match {batch_size}"
            )
        embedder = self.condition_embedder.text_embedder
        dtype = embedder.linear_1.weight.dtype
        return embedder(text_emb.to(device=next(self.parameters()).device, dtype=dtype))

    @staticmethod
    def _normalize_frame_ids(
        frame_ids: Iterable[int] | torch.Tensor,
        *,
        batch_size: int,
        frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        frame_ids = torch.as_tensor(frame_ids, dtype=torch.long, device=device)
        if frame_ids.ndim == 1:
            frame_ids = frame_ids[None].expand(batch_size, -1)
        if tuple(frame_ids.shape) != (batch_size, frames):
            raise ValueError(
                f"frame_ids must be [{batch_size},{frames}], got {tuple(frame_ids.shape)}"
            )
        return frame_ids

    @staticmethod
    def _normalize_timesteps(
        timesteps: float | int | torch.Tensor,
        *,
        batch_size: int,
        frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        timesteps = torch.as_tensor(timesteps, dtype=torch.float32, device=device)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size, frames)
        elif timesteps.ndim == 1:
            if timesteps.numel() == 1:
                timesteps = timesteps.expand(batch_size, frames)
            elif timesteps.numel() == frames:
                timesteps = timesteps[None].expand(batch_size, -1)
        if tuple(timesteps.shape) != (batch_size, frames):
            raise ValueError(
                f"timesteps must broadcast to [{batch_size},{frames}], got {tuple(timesteps.shape)}"
            )
        return timesteps

    def _stream_axes(
        self,
        sample: torch.Tensor,
        *,
        frame_ids: Iterable[int] | torch.Tensor,
        timesteps: float | int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _channels, frames = sample.shape[:3]
        return (
            self._normalize_frame_ids(
                frame_ids,
                batch_size=batch_size,
                frames=frames,
                device=sample.device,
            ),
            self._normalize_timesteps(
                timesteps,
                batch_size=batch_size,
                frames=frames,
                device=sample.device,
            ),
        )

    def _video_rotary(
        self,
        frame_ids: torch.Tensor,
        *,
        views: int,
        h_tokens: int,
        w_tokens: int,
    ) -> torch.Tensor:
        # LINK: distillation/model/autoregressive_mot.py:890
        # 输入：
        # frame_ids 通常形状为 [B, F]
        # B = batch size
        # F = 当前样本包含的帧数
        #
        # frame_ids[b, f] 表示第 b 个样本中第 f 个视频位置
        # 实际对应的帧编号。
        #
        # 例如：
        # frame_ids =
        # [[0, 1, 2],
        #  [5, 6, 7]]
        #
        # 说明 batch 中第 0 个样本使用第 0、1、2 帧，
        # 第 1 个样本使用第 5、6、7 帧。
        batch_size, frames = frame_ids.shape
        device = frame_ids.device
        # 阶段 1/4：创建空间方向的 token 坐标
        #
        # height: [0, 1, ..., h_tokens-1]，形状 [H]
        # width:  [0, 1, ..., w_tokens-1]，形状 [W]
        #
        # 使用 float32，是因为后面的 self.rope 可能需要浮点位置坐标。
        height = torch.arange(h_tokens, device=device, dtype=torch.float32)
        width = torch.arange(w_tokens, device=device, dtype=torch.float32)
        grids = []
        # 阶段 2/4：为每个 batch 样本生成视频 token 坐标
        for batch in range(batch_size):
            frame = frame_ids[batch].to(dtype=torch.float32)
            # 生成四维坐标网格：
            #
            # frame  -> 帧坐标，数量 F
            # view   -> 视角坐标，数量 V
            # height -> 高度 token 坐标，数量 H
            # width  -> 宽度 token 坐标，数量 W
            #
            # ff、_vv、hh、ww 的形状都是：
            # [F, V, H, W]
            #
            # indexing="ij" 保证维度顺序与输入顺序一致：
            # [frame, view, height, width]
            ff, _vv, hh, ww = torch.meshgrid(
                frame,
                torch.arange(views, device=device, dtype=torch.float32),
                height,
                width,
                indexing="ij",
            )
            # 组合成 4 个位置坐标通道：
            #
            # 第 0 个通道：帧坐标 ff
            # 第 1 个通道：高度坐标 hh
            # 第 2 个通道：宽度坐标 ww
            # 第 3 个通道：全 0 坐标
            #
            # 注意：_vv 没有被放进 stack，
            # 所以视角编号参与了 token 数量和排列，
            # 但没有作为显式的位置坐标通道。
            #
            # stack 后：
            # [F, V, H, W] -> [4, F, V, H, W]
            #
            # reshape 后：
            # [4, F, V, H, W] -> [4, F*V*H*W]
            #
            # 每一列对应一个视频 token 的位置：
            # [frame_id, height_id, width_id, 0]
            grids.append(
                torch.stack([ff, hh, ww, torch.zeros_like(ff)], dim=0).reshape(4, -1)
            )
        # 阶段 3/4：把每个 batch 的坐标网格堆叠起来
        #
        # grids 中有 B 个张量，每个张量形状为 [4, F*V*H*W]。
        #
        # torch.stack(..., dim=0) 后：
        # [B 个 [4, N]] -> [B, 4, N]
        #
        # 其中：
        # N = F * V * H * W
        grid = torch.stack(grids, dim=0)
        expected = frames * views * h_tokens * w_tokens
        if grid.shape[-1] != expected:
            raise RuntimeError("video rotary grid length mismatch")
        
        # 阶段 4/4：将坐标网格转换为 RoPE
        #
        # grid 的形状是 [B, 4, N]。
        # self.rope 根据这 4 个坐标通道生成旋转位置编码。
        #
        # 具体输出形状取决于 self.rope 的实现。
        # 随后 [:, :, None] 在第 2 维插入一个长度为 1 的维度：
        #
        # 假设 self.rope(grid) 的形状是 [B, D, N]，
        # 那么：
        # [B, D, N] -> [B, D, 1, N]
        #
        # 这个新增维度通常用于后续与 attention head、
        # 多视角维度或其他广播维度对齐。
        return self.rope(grid)[:, :, None]

    def _action_rotary(
        self,
        frame_ids: torch.Tensor,
        *,
        tokens_per_frame: int,
    ) -> torch.Tensor:
        # LINK: wan_va/modules/model_3dva_mot.py:914
        # 输入：
        # frame_ids       通常形状为 [B, F]
        #                  B = batch size
        #                  F = frames，帧数
        #                  每个元素表示对应位置属于哪一帧
        #
        # tokens_per_frame = 每帧包含的 action token 数量
        #
        # 例如：
        # frame_ids =
        # [[0, 1, 2],
        #  [0, 1, 2]]
        # 表示 batch 中每个样本都有 3 帧。

        # 阶段 1/3：创建每一帧内部的 token slot 编号
        #
        # torch.arange(tokens_per_frame) 的形状是 [K]，
        # 其中 K = tokens_per_frame：
        #
        # K=4 时：
        # [0, 1, 2, 3]
        #
        # [None, None] 在最前面增加两个维度：
        # [K] -> [1, 1, K]
        #
        # 这样做是为了后面和 [B, F, 1] 形状的 frame_ids
        # 通过 broadcasting 相加。
        slots = torch.arange(
            tokens_per_frame,
            device=frame_ids.device,
            dtype=torch.long,
        )[None, None]
        # 阶段 2/3：将二维的“帧编号 + 帧内 slot 编号”
        # 映射成连续的一维位置编号
        #
        # frame_ids[:, :, None]：
        # [B, F] -> [B, F, 1]
        #
        # slots：
        # [1, 1, K]
        #
        # 相乘并相加时发生 broadcasting：
        # [B, F, 1] * K + [1, 1, K]
        #                         -> [B, F, K]
        #
        # 计算公式：
        # positions[b, f, k] = frame_ids[b, f] * K + k
        #
        # 例子：F=3，K=4
        #
        # frame_ids[0] = [0, 1, 2]
        #
        # frame_ids[0, :, None] * 4：
        # [[0],
        #  [4],
        #  [8]]
        #
        # 加上 slots=[[0, 1, 2, 3]] 后：
        # [[ 0,  1,  2,  3],   # 第 0 帧的 4 个 token
        #  [ 4,  5,  6,  7],   # 第 1 帧的 4 个 token
        #  [ 8,  9, 10, 11]]   # 第 2 帧的 4 个 token
        #
        # positions 的形状为 [B, F, K]。
        positions = frame_ids[:, :, None] * tokens_per_frame + slots
        # 阶段 3/3：把每帧内部的 K 个 token 展平为一个序列
        #
        # [B, F, K] -> [B, F*K]
        #
        # 例如：
        # [[0, 1, 2, 3],
        #  [4, 5, 6, 7],
        #  [8, 9, 10, 11]]
        #
        # 会变成：
        # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        #
        # 然后将这些一维位置编号传给 action_rope，
        # 由它根据位置编号生成对应的 Rotary Position Embedding。
        return self.action_rope(positions.reshape(frame_ids.shape[0], -1))

    def _video_input(
        self,
        latents: torch.Tensor,
        *,
        frame_ids: torch.Tensor,
        timesteps: torch.Tensor,
        stream_ids: torch.Tensor,
        noise_id: int,
        committed: bool,
        transaction_id: int,
        source,
        version_id: int,
        valid_frames: torch.Tensor | None,
    ) -> tuple[AutoregressiveStreamInput, tuple[int, ...]]:
        from distillation.self_rollout.attention import STREAM_VIDEO, build_token_metadata

        hidden, shape = self._embed_video(latents, stream_ids)
        batch_size, frames, views, _height, _width, h_tokens, w_tokens = shape
        tokens_per_frame = views * h_tokens * w_tokens
        _, conditioning = self._time_embed_repeated(
            timesteps,
            tokens_per_frame,
            hidden.dtype,
            action=False,
        )
        metadata = build_token_metadata(
            batch_size=batch_size,
            frame_ids=frame_ids,
            tokens_per_frame=tokens_per_frame,
            stream_id=STREAM_VIDEO,
            noise_id=noise_id,
            history_frames=self.generation_profile.history_frames,
            chunk_size=self.generation_profile.chunk_size,
            device=hidden.device,
            valid_ids=valid_frames,
            committed=committed,
            transaction_id=transaction_id,
            source_id=int(source),
            version_id=version_id,
        )
        return (
            AutoregressiveStreamInput(
                hidden=hidden,
                conditioning=conditioning,
                rotary=self._video_rotary(
                    frame_ids,
                    views=views,
                    h_tokens=h_tokens,
                    w_tokens=w_tokens,
                ),
                metadata=metadata,
                block_kind="video",
            ),
            shape,
        )

    def _action_input(
        self,
        actions: torch.Tensor,
        *,
        frame_ids: torch.Tensor,
        timesteps: torch.Tensor,
        noise_id: int,
        committed: bool,
        transaction_id: int,
        source,
        version_id: int,
        valid_mask: torch.Tensor | None,
    ) -> AutoregressiveStreamInput:
        from distillation.self_rollout.attention import STREAM_ACTION, build_token_metadata

        hidden = self._embed_action(actions)
        batch_size, _channels, frames, action_per_frame, width = actions.shape
        tokens_per_frame = action_per_frame * width
        _, conditioning = self._time_embed_repeated(
            timesteps,
            tokens_per_frame,
            hidden.dtype,
            action=True,
        )
        token_valid = None
        if valid_mask is not None:
            token_valid = valid_mask.to(device=hidden.device, dtype=torch.bool).any(dim=1)
            token_valid = token_valid.reshape(batch_size, frames * tokens_per_frame)
        metadata = build_token_metadata(
            batch_size=batch_size,
            frame_ids=frame_ids,
            tokens_per_frame=tokens_per_frame,
            stream_id=STREAM_ACTION,
            noise_id=noise_id,
            history_frames=self.generation_profile.history_frames,
            chunk_size=self.generation_profile.chunk_size,
            device=hidden.device,
            valid_ids=token_valid,
            committed=committed,
            transaction_id=transaction_id,
            source_id=int(source),
            version_id=version_id,
        )
        return AutoregressiveStreamInput(
            hidden=hidden,
            conditioning=conditioning,
            rotary=self._action_rotary(
                frame_ids,
                tokens_per_frame=tokens_per_frame,
            ),
            metadata=metadata,
            block_kind="action",
        )

    def _run_stream(
        self,
        stream: AutoregressiveStreamInput,
        *,
        text_emb: torch.Tensor,
        state,
        transaction_id: int,
    ) -> torch.Tensor:
        from distillation.self_rollout.attention import STREAM_ACTION, STREAM_VIDEO
        from distillation.self_rollout.cache import KVSegment

        # 阶段 1/3：准备一个 video 或 action stream。
        batch_size = stream.hidden.shape[0]
        text = self._text(text_emb, batch_size)
        hidden = stream.hidden
        stream_id = STREAM_VIDEO if stream.block_kind == "video" else STREAM_ACTION

        # 阶段 2/3：逐层更新当前 stream。geometry 不作为第二个输入 stream
        # 传进来；它是否可见取决于当前 layer_id 的共享 mot_cache 中是否
        # 已经提交了 geometry K/V。
        for layer_id, mot_block in enumerate(self.mot_blocks):
            if hasattr(mot_block, "forward_incremental"):
                hidden = mot_block(
                    AutoregressiveMOTLayerRequest(
                        stream=stream,
                        hidden_state=hidden,
                        text=text,
                        state=state,
                        transaction_id=transaction_id,
                        layer_id=layer_id,
                        stream_id=stream_id,
                    )
                )
                continue

            # 兼容路径：调制 -> QKV -> 共享 cache attention -> text/FFN。
            from distillation.self_rollout import attention as attention_module

            block = (
                mot_block.video_block
                if stream.block_kind == "video"
                else mot_block.action_block
            )
            modulation = mot_block._modulation(block, stream.conditioning)
            query, current_key, current_value = mot_block._self_qkv(
                block,
                hidden,
                modulation,
                stream.rotary,
            )
            state.mot_cache.append_transaction(
                layer_id,
                transaction_id,
                KVSegment(
                    current_key,
                    current_value,
                    stream.metadata,
                    stream_id=stream_id,
                ),
            )
            key, value, key_meta = state.mot_cache.materialize(
                layer_id,
                transaction_id=transaction_id,
            )
            query_valid, key_valid = attention_module.build_cache_selection(
                stream.metadata,
                key_meta,
            )
            attended, _visible = attention_module.indexed_attention(
                query,
                key,
                value,
                query_valid=query_valid,
                key_valid=key_valid,
            )
            text_attn = mot_block.video_block.attn2
            text_key = text_attn.norm_k(text_attn.to_k(text)).unflatten(
                2,
                (text_attn.heads, -1),
            )
            text_value = text_attn.to_v(text).unflatten(
                2,
                (text_attn.heads, -1),
            )
            hidden = mot_block._finish_block(
                block,
                hidden,
                mot_block._attention_output(block.attn1, attended),
                modulation,
                text_key,
                text_value,
            )

        # 阶段 3/3：返回同一个 stream 的最终 hidden states，[B,L,D]。
        return hidden

    def _run_transaction(
        self,
        stream: AutoregressiveStreamInput,
        *,
        text_emb: torch.Tensor,
        state,
        transaction_id: int,
        commit_source=None,
    ) -> torch.Tensor:
        try:
            hidden = self._run_stream(
                stream,
                text_emb=text_emb,
                state=state,
                transaction_id=transaction_id,
            )
            if commit_source is not None:
                state.mot_cache.commit_transaction(
                    transaction_id,
                    source_id=int(commit_source),
                )
            return hidden
        finally:
            state.mot_cache.discard_transaction(transaction_id)

    def _assert_clean_commit(
        self,
        state,
        *,
        frame_id: int,
        stream_id: int,
        source,
        version_id: int,
    ) -> None:
        from distillation.self_rollout.attention import NOISE_CLEAN

        state.mot_cache.assert_committed_frame(
            range(len(self.mot_blocks)),
            frame_id=frame_id,
            stream_id=stream_id,
            noise_ids=(NOISE_CLEAN,),
            source_id=int(source),
            version_id=version_id,
        )

    def predict_video(
        self,
        sample: torch.Tensor,
        *,
        timestep: float | int | torch.Tensor,
        frame_id: int,
        stream_ids: torch.Tensor,
        text_emb: torch.Tensor,
        state,
        valid_frames: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from distillation.self_rollout.attention import NOISE_NOISY
        from distillation.self_rollout.state import CacheSource

        if sample.shape[2] != 1:
            raise ValueError("predict_video expects exactly one logical frame")
        frame_ids, timesteps = self._stream_axes(
            sample,
            frame_ids=[frame_id],
            timesteps=timestep,
        )
        transaction_id = state.new_transaction_id()
        stream, shape = self._video_input(
            sample,
            frame_ids=frame_ids,
            timesteps=timesteps,
            stream_ids=stream_ids,
            noise_id=NOISE_NOISY,
            committed=False,
            transaction_id=transaction_id,
            source=CacheSource.PREDICTED,
            version_id=0,
            valid_frames=valid_frames,
        )
        hidden = self._run_transaction(
            stream,
            text_emb=text_emb,
            state=state,
            transaction_id=transaction_id,
        )
        return self._final_video(hidden, timesteps, shape)

    def predict_action(
        self,
        sample: torch.Tensor,
        *,
        timestep: float | int | torch.Tensor,
        frame_id: int,
        text_emb: torch.Tensor,
        state,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from distillation.self_rollout.attention import NOISE_NOISY
        from distillation.self_rollout.state import CacheSource

        if sample.shape[2] != 1:
            raise ValueError("predict_action expects exactly one logical frame")
        frame_ids, timesteps = self._stream_axes(
            sample,
            frame_ids=[frame_id],
            timesteps=timestep,
        )
        transaction_id = state.new_transaction_id()
        stream = self._action_input(
            sample,
            frame_ids=frame_ids,
            timesteps=timesteps,
            noise_id=NOISE_NOISY,
            committed=False,
            transaction_id=transaction_id,
            source=CacheSource.PREDICTED,
            version_id=0,
            valid_mask=valid_mask,
        )
        hidden = self._run_transaction(
            stream,
            text_emb=text_emb,
            state=state,
            transaction_id=transaction_id,
        )
        return self._final_action(hidden, timesteps, tuple(sample.shape))

    def commit_video(
        self,
        latents: torch.Tensor,
        *,
        frame_ids: Iterable[int] | torch.Tensor,
        stream_ids: torch.Tensor,
        text_emb: torch.Tensor,
        state,
        source,
        version_id: int,
        valid_frames: torch.Tensor | None = None,
    ) -> None:
        from distillation.self_rollout.attention import NOISE_CLEAN

        frame_ids, zeros = self._stream_axes(
            latents,
            frame_ids=frame_ids,
            timesteps=0,
        )
        transaction_id = state.new_transaction_id()
        clean, _ = self._video_input(
            latents,
            frame_ids=frame_ids,
            timesteps=zeros,
            stream_ids=stream_ids,
            noise_id=NOISE_CLEAN,
            committed=False,
            transaction_id=transaction_id,
            source=source,
            version_id=version_id,
            valid_frames=valid_frames,
        )
        self._run_transaction(
            clean,
            text_emb=text_emb,
            state=state,
            transaction_id=transaction_id,
            commit_source=source,
        )

    def commit_action(
        self,
        actions: torch.Tensor,
        *,
        frame_ids: Iterable[int] | torch.Tensor,
        text_emb: torch.Tensor,
        state,
        source,
        version_id: int,
        valid_mask: torch.Tensor | None = None,
    ) -> None:
        from distillation.self_rollout.attention import NOISE_CLEAN

        frame_ids, zeros = self._stream_axes(
            actions,
            frame_ids=frame_ids,
            timesteps=0,
        )
        transaction_id = state.new_transaction_id()
        clean = self._action_input(
            actions,
            frame_ids=frame_ids,
            timesteps=zeros,
            noise_id=NOISE_CLEAN,
            committed=False,
            transaction_id=transaction_id,
            source=source,
            version_id=version_id,
            valid_mask=valid_mask,
        )
        self._run_transaction(
            clean,
            text_emb=text_emb,
            state=state,
            transaction_id=transaction_id,
            commit_source=source,
        )

    def assert_video_commit(
        self,
        state,
        *,
        frame_id: int,
        source,
        version_id: int,
    ) -> None:
        from distillation.self_rollout.attention import STREAM_VIDEO

        self._assert_clean_commit(
            state,
            frame_id=frame_id,
            stream_id=STREAM_VIDEO,
            source=source,
            version_id=version_id,
        )

    def assert_action_commit(
        self,
        state,
        *,
        frame_id: int,
        source,
        version_id: int,
    ) -> None:
        from distillation.self_rollout.attention import STREAM_ACTION

        self._assert_clean_commit(
            state,
            frame_id=frame_id,
            stream_id=STREAM_ACTION,
            source=source,
            version_id=version_id,
        )

    def forward(self, input_dict, *, mode: str, return_diagnostics: bool = False, **kwargs):
        if mode != "self_rollout":
            return super().forward(
                input_dict,
                mode=mode,
                return_diagnostics=return_diagnostics,
                **kwargs,
            )
        if not isinstance(input_dict, AutoregressiveModelRequest):
            raise TypeError("mode='self_rollout' requires AutoregressiveModelRequest")
        return self.forward_autoregressive(input_dict)

    @torch.no_grad()
    def forward_autoregressive(
        self,
        request: AutoregressiveModelRequest,
    ) -> AutoregressiveModelOutput:
        """Execute one rollout operation through the AR model boundary."""

        p = request.payload
        op = request.operation
        if op == "predict_video":
            prediction = self.predict_video(**p)
            return AutoregressiveModelOutput(prediction=prediction)
        if op == "predict_action":
            prediction = self.predict_action(**p)
            return AutoregressiveModelOutput(prediction=prediction)
        if op == "commit_video":
            self.commit_video(**p)
            return AutoregressiveModelOutput()
        if op == "commit_action":
            self.commit_action(**p)
            return AutoregressiveModelOutput()
        if op == "encode_geometry_history":
            frame = self.vggto.encode_history_and_commit(self, **p)
            return AutoregressiveModelOutput(geometry_frame=frame)
        if op == "encode_geometry":
            frame = self.vggto.encode_and_commit(self, **p)
            return AutoregressiveModelOutput(geometry_frame=frame)
        raise ValueError(f"unsupported autoregressive operation {op!r}")
