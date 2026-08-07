"""Video+Action Mixture-of-Transformers model."""

import math
from typing import NamedTuple, Optional

import torch
import torch.nn.functional as F
from diffusers import WanTransformer3DModel as DiffusersWanTransformer3DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch import nn

from .model import (
    WanAttention,
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    WanTransformer3DModel,
    WanTransformerBlock,
)
from .mot_attention import (
    MOTMaskMetadata,
    apply_wan_rotary,
    attention_from_meta,
    build_x_metadata,
)
from .mot_init import (
    InitReport,
    check_diffusers_transformer_complete,
    initialize_action_expert_from_video_,
    initialize_action_io_,
    initialize_wan_patch_embedding_,
)


class VAStreamStates(NamedTuple):
    """Residual states in noisy-video, clean-video, noisy-action, clean-action order."""

    noisy_video: torch.Tensor
    clean_video: torch.Tensor
    noisy_action: torch.Tensor
    clean_action: torch.Tensor


class VAStreamConditioning(NamedTuple):
    """Timestep conditioning in the same order as :class:`VAStreamStates`."""

    noisy_video: torch.Tensor
    clean_video: torch.Tensor
    noisy_action: torch.Tensor
    clean_action: torch.Tensor


class _BlockModulation(NamedTuple):
    attention_shift: torch.Tensor
    attention_scale: torch.Tensor
    attention_gate: torch.Tensor
    ffn_shift: torch.Tensor
    ffn_scale: torch.Tensor
    ffn_gate: torch.Tensor


class _PreparedVAInputs(NamedTuple):
    states: VAStreamStates
    conditioning: VAStreamConditioning
    text: torch.Tensor
    video_rotary: torch.Tensor
    action_rotary: torch.Tensor
    shape: tuple[int, ...]
    action_tokens_per_frame: int


class ActionRotaryPosEmbed(nn.Module):
    """Independent 1D RoPE for the physical action-slot sequence."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"action attention head_dim must be even, got {head_dim}")
        self.head_dim = int(head_dim)
        self.max_seq_len = int(max_seq_len)
        self.theta = float(theta)
        freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.head_dim, 2, dtype=torch.float64)
                / self.head_dim
            )
        )
        self.register_buffer("freqs_base", freqs_base)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.ndim != 2:
            raise ValueError(
                "action RoPE positions must be [B,L], "
                f"got {tuple(positions.shape)}"
            )
        if positions.shape[1] > self.max_seq_len:
            raise ValueError(
                f"action RoPE length exceeds max_seq_len={self.max_seq_len}: "
                f"length={positions.shape[1]}"
            )
        angles = positions.to(torch.float64).unsqueeze(-1)
        angles = angles * self.freqs_base.to(positions.device)
        freqs = torch.polar(torch.ones_like(angles), angles)
        return freqs[:, :, None]


class ActionCrossAttention(nn.Module):
    """Action-owned query/output projections with shared video text K/V."""

    def __init__(self, action_dim: int, attn_dim: int, num_heads: int, eps: float):
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim={attn_dim} must be divisible by num_heads={num_heads}"
            )
        self.heads = int(num_heads)
        self.to_q = nn.Linear(action_dim, attn_dim, bias=True)
        self.to_out = nn.ModuleList(
            [nn.Linear(attn_dim, action_dim, bias=True), nn.Dropout(0.0)]
        )
        self.norm_q = nn.RMSNorm(attn_dim, eps=eps, elementwise_affine=True)


class ActionTransformerBlock(nn.Module):
    """Wan-mirrored action block whose self-attention is executed by MoT."""

    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        attn_dim: int,
        num_heads: int,
        eps: float,
    ):
        super().__init__()
        self.norm1 = FP32LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=hidden_dim,
            heads=num_heads,
            dim_head=attn_dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode="torch",
        )
        self.attn2 = ActionCrossAttention(hidden_dim, attn_dim, num_heads, eps)
        self.norm2 = FP32LayerNorm(hidden_dim, eps, elementwise_affine=True)
        self.ffn = FeedForward(
            hidden_dim,
            inner_dim=ffn_dim,
            activation_fn="gelu-approximate",
        )
        self.norm3 = FP32LayerNorm(hidden_dim, eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, hidden_dim) / hidden_dim**0.5
        )


class VAMOTBlock(nn.Module):
    """One joint Video+Action MoT layer."""

    def __init__(
        self,
        video_block: WanTransformerBlock,
        action_block: ActionTransformerBlock,
        *,
        masked_attn_backend: str,
    ):
        super().__init__()
        self.video_block = video_block
        self.action_block = action_block
        self.masked_attn_backend = masked_attn_backend

    @staticmethod
    def _modulation(block: nn.Module, temb: torch.Tensor) -> _BlockModulation:
        values = rearrange(
            block.scale_shift_table[None] + temb.float(),
            "b l n c -> b n l c",
        ).chunk(6, dim=1)
        return _BlockModulation(*(value.squeeze(1) for value in values))

    @staticmethod
    def _self_qkv(
        block: nn.Module,
        hidden: torch.Tensor,
        modulation: _BlockModulation,
        rotary_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = (
            block.norm1(hidden.float()) * (1.0 + modulation.attention_scale)
            + modulation.attention_shift
        ).type_as(hidden)
        attn = block.attn1
        query = attn.norm_q(attn.to_q(hidden)).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(attn.to_k(hidden)).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden).unflatten(2, (attn.heads, -1))
        return apply_wan_rotary(query, rotary_emb), apply_wan_rotary(key, rotary_emb), value

    @staticmethod
    def _attention_output(attn: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
        hidden = attn.to_out[0](hidden.flatten(2, 3))
        return attn.to_out[1](hidden)

    @staticmethod
    def _finish_block(
        block: nn.Module,
        hidden: torch.Tensor,
        self_output: torch.Tensor,
        modulation: _BlockModulation,
        text_key: torch.Tensor,
        text_value: torch.Tensor,
    ) -> torch.Tensor:
        hidden = (
            hidden.float() + self_output.float() * modulation.attention_gate
        ).type_as(hidden)
        attn = block.attn2
        cross_input = block.norm2(hidden.float()).type_as(hidden)
        query = attn.norm_q(attn.to_q(cross_input)).unflatten(2, (attn.heads, -1))
        cross_output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            text_key.transpose(1, 2),
            text_value.transpose(1, 2),
        ).transpose(1, 2)
        hidden = hidden + VAMOTBlock._attention_output(attn, cross_output)
        ffn_input = (
            block.norm3(hidden.float()) * (1.0 + modulation.ffn_scale)
            + modulation.ffn_shift
        ).type_as(hidden)
        ffn_output = block.ffn(ffn_input)
        return (
            hidden.float() + ffn_output.float() * modulation.ffn_gate
        ).type_as(hidden)

    def forward(
        self,
        states: VAStreamStates,
        conditioning: VAStreamConditioning,
        text: torch.Tensor,
        video_rotary: torch.Tensor,
        action_rotary: torch.Tensor,
        meta: MOTMaskMetadata,
    ) -> VAStreamStates:
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
                blocks, states, modulations, rotary_embeddings
            )
        ]
        lengths = [hidden.shape[1] for hidden in states]
        query, key, value = (torch.cat(parts, dim=1) for parts in zip(*qkv_parts))
        attention_output = attention_from_meta(
            query,
            key,
            value,
            meta,
            backend=self.masked_attn_backend,
        )
        attention_parts = torch.split(attention_output, lengths, dim=1)
        self_outputs = tuple(
            self._attention_output(block.attn1, output)
            for block, output in zip(blocks, attention_parts)
        )
        text_attn = self.video_block.attn2
        text_key = text_attn.norm_k(text_attn.to_k(text)).unflatten(
            2, (text_attn.heads, -1)
        )
        text_value = text_attn.to_v(text).unflatten(2, (text_attn.heads, -1))
        return VAStreamStates(
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
                    blocks, states, self_outputs, modulations
                )
            )
        )


class VAMOTTransformer3DModel(ModelMixin, ConfigMixin):
    """Fixed-topology Video+Action MoT model."""

    _supports_gradient_checkpointing = True
    _no_split_modules = ["VAMOTBlock"]
    _repeated_blocks = ["VAMOTBlock"]
    _NUM_STREAMS = 3
    _FIXED_TOPOLOGY = {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 24,
        "attention_head_dim": 128,
        "in_channels": 48,
        "out_channels": 48,
        "action_dim": 20,
        "action_hidden_dim": 768,
        "action_ffn_dim": 3072,
        "text_dim": 4096,
        "freq_dim": 256,
        "ffn_dim": 14336,
        "num_layers": 30,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "rope_max_seq_len": 1024,
        "pos_embed_seq_len": None,
        "attn_mode": "torch",
    }

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=20,
        action_hidden_dim=768,
        action_ffn_dim=3072,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        max_num_views=3,
        init_noise_seed=42,
        attn_mode="torch",
        masked_attn_backend="fa4",
    ):
        super().__init__()
        self._validate_fixed_topology(locals())
        if masked_attn_backend not in {"fa4", "flex", "dense"}:
            raise ValueError(f"unsupported masked attention backend: {masked_attn_backend}")
        self.patch_size = tuple(patch_size)
        self.num_attention_heads = int(num_attention_heads)
        self.attention_head_dim = int(attention_head_dim)
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.action_hidden_dim = int(action_hidden_dim)
        self.action_ffn_dim = int(action_ffn_dim)
        self.action_dim = int(action_dim)
        self.max_num_views = int(max_num_views)
        self.init_noise_seed = int(init_noise_seed)
        self.masked_attn_backend = str(masked_attn_backend)
        self.rope = WanRotaryPosEmbed(
            self.attention_head_dim,
            self.patch_size,
            rope_max_seq_len,
        )
        self.action_rope = ActionRotaryPosEmbed(
            self.attention_head_dim,
            rope_max_seq_len,
        )
        self.patch_embedding_mlp = nn.Linear(
            in_channels * math.prod(self.patch_size),
            self.inner_dim,
        )
        self.action_embedder = nn.Linear(
            self.action_dim,
            self.action_hidden_dim,
            bias=False,
        )
        self.condition_embedder = WanTimeTextImageEmbedding(
            self.inner_dim,
            freq_dim,
            self.inner_dim * 6,
            text_dim,
            pos_embed_seq_len,
        )
        self.action_condition_embedder = WanTimeTextImageEmbedding(
            self.action_hidden_dim,
            freq_dim,
            self.action_hidden_dim * 6,
            text_dim,
            pos_embed_seq_len,
        )
        del self.action_condition_embedder.text_embedder
        self.mot_blocks = nn.ModuleList(
            [
                self._build_mot_block(
                    video_block=WanTransformerBlock(
                        self.inner_dim,
                        ffn_dim,
                        self.num_attention_heads,
                        cross_attn_norm,
                        eps,
                        attn_mode="torch",
                    ),
                    action_block=ActionTransformerBlock(
                        self.action_hidden_dim,
                        self.action_ffn_dim,
                        self.inner_dim,
                        self.num_attention_heads,
                        eps,
                    ),
                    masked_attn_backend=self.masked_attn_backend,
                )
                for _ in range(num_layers)
            ]
        )
        self.stream_embedding = nn.Embedding(self._NUM_STREAMS, self.inner_dim)
        self.norm_out = FP32LayerNorm(self.inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(
            self.inner_dim,
            out_channels * math.prod(self.patch_size),
        )
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, self.inner_dim) / self.inner_dim**0.5
        )
        self.action_norm_out = FP32LayerNorm(
            self.action_hidden_dim, eps, elementwise_affine=False
        )
        self.action_proj_out = nn.Linear(self.action_hidden_dim, self.action_dim)
        self.action_scale_shift_table = nn.Parameter(
            torch.randn(1, 2, self.action_hidden_dim) / self.action_hidden_dim**0.5
        )
        self._init_stream_embedding()
        initialize_action_io_(
            self.action_embedder,
            self.action_proj_out,
            seed=self.init_noise_seed,
        )

    @classmethod
    def _validate_fixed_topology(cls, values: dict) -> None:
        mismatches = {}
        for name, expected_value in cls._FIXED_TOPOLOGY.items():
            observed = values[name]
            if isinstance(expected_value, tuple):
                observed = tuple(observed)
            if observed != expected_value:
                mismatches[name] = (observed, expected_value)
        if mismatches:
            details = ", ".join(
                f"{name}={observed!r} (expected {expected!r})"
                for name, (observed, expected) in mismatches.items()
            )
            raise ValueError(f"unsupported model topology: {details}")

    def _build_mot_block(self, **kwargs) -> VAMOTBlock:
        return VAMOTBlock(**kwargs)

    @torch.no_grad()
    def _init_stream_embedding(self) -> None:
        if not self.stream_embedding.weight.is_meta:
            self.stream_embedding.weight.zero_()

    def _embed_video(
        self,
        latents: torch.Tensor,
        stream_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, ...]]:
        if latents.ndim == 5:
            latents = latents.unsqueeze(3)
        bsz, _channels, frames, views, height, width = latents.shape
        if views > self.max_num_views:
            raise ValueError(f"views={views} exceeds max_num_views={self.max_num_views}")
        if tuple(stream_ids.shape) != (bsz, views):
            raise ValueError(
                f"stream_ids must be [B,V]=[{bsz},{views}], got {tuple(stream_ids.shape)}"
            )
        hidden = rearrange(
            latents.to(dtype=self.patch_embedding_mlp.weight.dtype),
            "b c (t p1) v (h p2) (w p3) -> b (t v h w) (c p1 p2 p3)",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
        )
        hidden = self.patch_embedding_mlp(hidden)
        h_tokens = height // self.patch_size[1]
        w_tokens = width // self.patch_size[2]
        stream_emb = self.stream_embedding(stream_ids).to(hidden.dtype)
        stream_emb = stream_emb[:, None, :, None, :].expand(
            bsz, frames, views, h_tokens * w_tokens, self.inner_dim
        )
        return hidden + stream_emb.reshape(bsz, -1, self.inner_dim), (
            bsz, frames, views, height, width, h_tokens, w_tokens
        )

    def _embed_action(self, actions: torch.Tensor) -> torch.Tensor:
        hidden = rearrange(actions, "b c t n w -> b (t n w) c")
        return self.action_embedder(hidden.to(dtype=self.action_embedder.weight.dtype))

    @staticmethod
    def _video_grid(bsz, frames, views, h_tokens, w_tokens, device):
        frame = torch.arange(frames, device=device, dtype=torch.float32)
        view = torch.arange(views, device=device, dtype=torch.float32)
        height = torch.arange(h_tokens, device=device, dtype=torch.float32)
        width = torch.arange(w_tokens, device=device, dtype=torch.float32)
        ff, _vv, hh, ww = torch.meshgrid(frame, view, height, width, indexing="ij")
        grid = torch.stack([ff, hh, ww, torch.zeros_like(ff)], dim=0).reshape(4, -1)
        return grid[None].expand(bsz, -1, -1)

    @staticmethod
    def _action_positions(bsz, frames, action_tokens_per_frame, device):
        latent_ids = torch.arange(frames, device=device, dtype=torch.long)[:, None]
        slot_ids = torch.arange(action_tokens_per_frame, device=device, dtype=torch.long)[None]
        return (latent_ids * action_tokens_per_frame + slot_ids).reshape(1, -1).expand(bsz, -1)

    def _token_valid_ids(
        self,
        input_dict,
        *,
        video_tokens_per_frame: int,
        action_tokens_per_frame: int,
        frames: int,
        device,
    ) -> torch.Tensor:
        bsz = input_dict["latent_dict"]["noisy_latents"].shape[0]
        video_valid = input_dict["latent_dict"].get("video_latent_valid_mask")
        if video_valid is None:
            video_valid = torch.ones((bsz, frames), dtype=torch.bool, device=device)
        else:
            video_valid = video_valid.to(device=device, dtype=torch.bool)
        if tuple(video_valid.shape) != (bsz, frames):
            raise ValueError(f"video_latent_valid_mask must be [B,{frames}]")
        video_tokens = video_valid[:, :, None].expand(
            -1, -1, video_tokens_per_frame
        ).reshape(bsz, -1)
        action_valid = input_dict["action_dict"].get("action_valid_mask")
        if action_valid is None:
            action_valid = torch.ones(
                (bsz, frames, action_tokens_per_frame),
                dtype=torch.bool,
                device=device,
            )
        else:
            action_valid = action_valid.to(device=device, dtype=torch.bool).any(dim=1)[..., 0]
        if tuple(action_valid.shape) != (bsz, frames, action_tokens_per_frame):
            raise ValueError(
                "action_valid_mask must produce "
                f"[B,{frames},{action_tokens_per_frame}]"
            )
        action_tokens = action_valid.reshape(bsz, -1)
        return torch.cat(
            [video_tokens, video_tokens, action_tokens, action_tokens], dim=1
        )

    def _time_embed_repeated(self, timesteps, repeats_per_frame, dtype, *, action):
        repeated = torch.repeat_interleave(timesteps, repeats_per_frame, dim=1)
        embedder = self.action_condition_embedder if action else self.condition_embedder
        temb, timestep_proj = embedder(repeated, dtype=dtype)
        return temb, timestep_proj.unflatten(2, (6, -1))

    def _prepare_va_inputs(self, input_dict) -> _PreparedVAInputs:
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        stream_ids = input_dict["stream_ids"].long().to(latent_dict["noisy_latents"].device)
        noisy_video, latent_shape = self._embed_video(latent_dict["noisy_latents"], stream_ids)
        clean_video, _ = self._embed_video(latent_dict["latent"], stream_ids)
        noisy_action = self._embed_action(action_dict["noisy_latents"])
        clean_action = self._embed_action(action_dict["latent"])
        text_dtype = self.condition_embedder.text_embedder.linear_1.weight.dtype
        text = self.condition_embedder.text_embedder(
            latent_dict["text_emb"].to(dtype=text_dtype)
        )
        bsz, frames, views, _height, _width, h_tokens, w_tokens = latent_shape
        action_shape = action_dict["noisy_latents"].shape
        if tuple(action_shape[1:]) != (self.action_dim, 8, 16, 1):
            raise ValueError(
                "current action packing requires [B,20,8,16,1], "
                f"got {tuple(action_shape)}"
            )
        if action_shape[2] != frames:
            raise ValueError(f"action F={action_shape[2]} must match video latent F={frames}")
        action_tokens_per_frame = action_shape[3] * action_shape[4]
        video_rotary = self.rope(
            self._video_grid(bsz, frames, views, h_tokens, w_tokens, noisy_video.device)
        )[:, :, None]
        action_rotary = self.action_rope(
            self._action_positions(bsz, frames, action_tokens_per_frame, noisy_action.device)
        )
        video_repeats = views * h_tokens * w_tokens
        action_repeats = action_tokens_per_frame
        _, noisy_video_temb = self._time_embed_repeated(
            latent_dict["timesteps"], video_repeats, noisy_video.dtype, action=False
        )
        _, clean_video_temb = self._time_embed_repeated(
            latent_dict["cond_timesteps"], video_repeats, clean_video.dtype, action=False
        )
        _, noisy_action_temb = self._time_embed_repeated(
            action_dict["timesteps"], action_repeats, noisy_action.dtype, action=True
        )
        _, clean_action_temb = self._time_embed_repeated(
            action_dict["cond_timesteps"], action_repeats, clean_action.dtype, action=True
        )
        return _PreparedVAInputs(
            VAStreamStates(noisy_video, clean_video, noisy_action, clean_action),
            VAStreamConditioning(
                noisy_video_temb,
                clean_video_temb,
                noisy_action_temb,
                clean_action_temb,
            ),
            text,
            video_rotary,
            action_rotary,
            latent_shape,
            action_tokens_per_frame,
        )

    def _prepare_metadata(self, input_dict, prepared: _PreparedVAInputs):
        states = prepared.states
        bsz, frames, views, _height, _width, h_tokens, w_tokens = prepared.shape
        video_tokens_per_frame = views * h_tokens * w_tokens
        valid = self._token_valid_ids(
            input_dict,
            video_tokens_per_frame=video_tokens_per_frame,
            action_tokens_per_frame=prepared.action_tokens_per_frame,
            frames=frames,
            device=states.noisy_video.device,
        )
        meta = build_x_metadata(
            batch_size=bsz,
            video_tokens_per_frame=video_tokens_per_frame,
            action_tokens_per_frame=prepared.action_tokens_per_frame,
            num_frames=frames,
            chunk_size=int(input_dict["chunk_size"]),
            window_size=int(input_dict["window_size"]),
            device=states.noisy_video.device,
            token_valid_ids=valid,
        )
        diagnostics = {
            "video_tokens_per_frame": video_tokens_per_frame,
            "action_tokens_per_frame": prepared.action_tokens_per_frame,
        }
        return meta, diagnostics

    def _run_mot_layers(self, states, *, conditioning, text, video_rotary, action_rotary, meta):
        for block in self.mot_blocks:
            states = block(states, conditioning, text, video_rotary, action_rotary, meta)
        return states

    def _project_output(self, hidden, timesteps, repeats_per_frame, *, action):
        temb, _ = self._time_embed_repeated(
            timesteps, repeats_per_frame, hidden.dtype, action=action
        )
        if action:
            scale_shift_table = self.action_scale_shift_table
            norm = self.action_norm_out
            projection = self.action_proj_out
        else:
            scale_shift_table = self.scale_shift_table
            norm = self.norm_out
            projection = self.proj_out
        shift, scale = rearrange(
            scale_shift_table[None] + temb[:, :, None, :],
            "b l n c -> b n l c",
        ).chunk(2, dim=1)
        hidden = (
            norm(hidden.float()) * (1.0 + scale.squeeze(1)) + shift.squeeze(1)
        ).type_as(hidden)
        return projection(hidden)

    def _final_video(self, hidden, timesteps, shape):
        _bsz, frames, views, _height, _width, h_tokens, w_tokens = shape
        hidden = self._project_output(
            hidden, timesteps, views * h_tokens * w_tokens, action=False
        )
        return rearrange(
            hidden,
            "b (t v h w) (p1 p2 p3 c) -> b c (t p1) v (h p2) (w p3)",
            t=frames,
            v=views,
            h=h_tokens,
            w=w_tokens,
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
        )

    def _final_action(self, hidden, timesteps, action_shape):
        _bsz, _channels, frames, action_per_frame, width = action_shape
        hidden = self._project_output(
            hidden, timesteps, action_per_frame * width, action=True
        )
        return rearrange(
            hidden,
            "b (t n w) c -> b c t n w",
            t=frames,
            n=action_per_frame,
            w=width,
        )

    def _forward_va(self, input_dict):
        prepared = self._prepare_va_inputs(input_dict)
        meta, diagnostics = self._prepare_metadata(input_dict, prepared)
        states = self._run_mot_layers(
            prepared.states,
            conditioning=prepared.conditioning,
            text=prepared.text,
            video_rotary=prepared.video_rotary,
            action_rotary=prepared.action_rotary,
            meta=meta,
        )
        return prepared, states, diagnostics

    def forward_train(self, input_dict, return_diagnostics: bool = False):
        prepared, states, diagnostics = self._forward_va(input_dict)
        out = {
            "latent_pred": self._final_video(
                states.noisy_video,
                input_dict["latent_dict"]["timesteps"],
                prepared.shape,
            ),
            "action_pred": self._final_action(
                states.noisy_action,
                input_dict["action_dict"]["timesteps"],
                tuple(input_dict["action_dict"]["targets"].shape),
            ),
            "diagnostics": diagnostics,
        }
        if return_diagnostics:
            out["states"] = states
        return out

    def forward_inference(self, input_dict, *, prediction_type: str) -> dict:
        if prediction_type not in {"video", "action"}:
            raise ValueError("prediction_type must be 'video' or 'action'")
        prepared, states, diagnostics = self._forward_va(input_dict)
        if prediction_type == "video":
            return {
                "latent_pred": self._final_video(
                    states.noisy_video,
                    input_dict["latent_dict"]["timesteps"],
                    prepared.shape,
                ),
                "diagnostics": diagnostics,
            }
        return {
            "action_pred": self._final_action(
                states.noisy_action,
                input_dict["action_dict"]["timesteps"],
                tuple(input_dict["action_dict"]["targets"].shape),
            ),
            "diagnostics": diagnostics,
        }

    def forward(self, input_dict, *, mode: str, return_diagnostics: bool = False, **_kwargs):
        if mode == "train":
            return self.forward_train(input_dict, return_diagnostics=return_diagnostics)
        if mode == "inference_video":
            return self.forward_inference(input_dict, prediction_type="video")
        if mode == "inference_action":
            return self.forward_inference(input_dict, prediction_type="action")
        raise ValueError(f"unsupported mode={mode!r}")

    @classmethod
    def from_video_backbone(
        cls,
        video_transformer_path: str,
        init_model_from_lingbot: bool = True,
        config_overrides: Optional[dict] = None,
    ):
        """Initialize the Video+Action model from LingBot or Wan2.2."""

        video_source = "LingBot" if init_model_from_lingbot else "Wan2.2"
        check_diffusers_transformer_complete(
            video_transformer_path, source_name=video_source
        )
        if init_model_from_lingbot:
            base = WanTransformer3DModel.from_pretrained(
                video_transformer_path,
                torch_dtype=torch.float32,
                attn_mode="torch",
            )
        else:
            base = DiffusersWanTransformer3DModel.from_pretrained(
                video_transformer_path,
                torch_dtype=torch.float32,
            )
        config = dict(base.config)
        for key in (
            "_class_name",
            "_diffusers_version",
            "added_kv_proj_dim",
            "image_dim",
            "qk_norm",
        ):
            config.pop(key, None)
        if config_overrides:
            config.update(config_overrides)
        model = cls(**config)
        report = InitReport(video_source=video_source, video_path=video_transformer_path)
        for name in ("rope", "condition_embedder", "norm_out", "proj_out"):
            getattr(model, name).load_state_dict(
                getattr(base, name).state_dict(), strict=True
            )
        if init_model_from_lingbot:
            model.patch_embedding_mlp.load_state_dict(
                base.patch_embedding_mlp.state_dict(), strict=True
            )
        else:
            initialize_wan_patch_embedding_(
                model.patch_embedding_mlp, base.patch_embedding
            )
        model.scale_shift_table.data.copy_(base.scale_shift_table.data)
        for layer_id, block in enumerate(model.mot_blocks):
            block.video_block.load_state_dict(
                base.blocks[layer_id].state_dict(), strict=True
            )
        report.action_init = initialize_action_expert_from_video_(model, base)
        report.loaded_video_keys = len(base.state_dict())
        report.notes.append(
            "action_init="
            f"copy:{len(report.action_init.copied)},"
            f"interpolate:{len(report.action_init.interpolated)},"
            f"scale:{len(report.action_init.scaled)},"
            f"share:{len(report.action_init.shared)},"
            f"random:{len(report.action_init.random)}"
        )
        return model, report
