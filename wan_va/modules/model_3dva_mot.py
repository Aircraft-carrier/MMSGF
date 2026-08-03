"""V/A/G Mixture-of-Transformers model."""

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
    build_geometry_metadata,
    build_mot_metadata,
    build_x_metadata,
)
from .mot_init import (
    InitReport,
    check_diffusers_transformer_complete,
    head_aware_hidden_map,
    ignored_vggto_keys,
    init_geometry_from_vggto_inter_frame_,
    initialize_action_expert_from_video_,
    initialize_action_io_,
    initialize_wan_patch_embedding_,
    load_vggt_point_head_state_dict,
    load_vggto_24_into_30_,
    load_vggto_state_dict,
    normalize_vggto_keys,
)
from .vggto_geometry import (
    VGGTOGeometryState,
    VGGTOGeometryTower,
    init_shared_register_from_first_other_,
)


class VAStreamStates(NamedTuple):
    """V/A residual states in NV, CV, NA, CA order."""

    noisy_video: torch.Tensor
    clean_video: torch.Tensor
    noisy_action: torch.Tensor
    clean_action: torch.Tensor


class VAStreamConditioning(NamedTuple):
    """Timestep conditioning in NV, CV, NA, CA order."""

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


class _GeometryContext(NamedTuple):
    state: VGGTOGeometryState
    source_images: torch.Tensor
    groups: int
    group_size: int
    views: int
    groups_per_chunk: int
    register_rotary: torch.Tensor
    cached_outputs: list[torch.Tensor | None]
    slot_valid_mask: Optional[torch.Tensor]
    image_valid_mask: Optional[torch.Tensor]


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
    """Action-owned query/output projections with no text K/V parameters."""

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
    """Wan-mirrored 768D block whose self-attention is executed by MoT."""

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


class GeometryMLP(nn.Module):
    """VGGTO-compatible geometry feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden)))


class GeometryJointStream(nn.Module):
    """Native-width VGGTO registers with a 3072D joint-attention carrier."""

    def __init__(self, g_dim: int, attn_dim: int, num_heads: int, eps: float):
        super().__init__()
        if attn_dim % num_heads != 0:
            raise ValueError(
                f"attn_dim={attn_dim} must be divisible by num_heads={num_heads}"
            )
        self.g_dim = int(g_dim)
        self.attn_dim = int(attn_dim)
        self.num_heads = int(num_heads)
        self.head_dim = attn_dim // num_heads
        self.norm1 = nn.LayerNorm(g_dim, eps=eps)
        self.qkv = nn.Linear(g_dim, attn_dim * 3)
        self.q_norm = nn.LayerNorm(self.head_dim, eps=eps)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=eps)
        self.proj = nn.Linear(attn_dim, g_dim)
        self.ls1 = nn.Parameter(torch.ones(g_dim) * 1e-5)
        self.norm2 = nn.LayerNorm(g_dim, eps=eps)
        self.mlp = GeometryMLP(g_dim, g_dim * 4)
        self.ls2 = nn.Parameter(torch.ones(g_dim) * 1e-5)

    def qkv_project(
        self,
        hidden: torch.Tensor,
        rotary_emb: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = hidden.shape
        qkv = self.qkv(self.norm1(hidden)).reshape(
            bsz,
            seq_len,
            3,
            self.num_heads,
            self.head_dim,
        )
        query, key, value = qkv.unbind(dim=2)
        query = self.q_norm(query).to(value.dtype)
        key = self.k_norm(key).to(value.dtype)
        return (
            apply_wan_rotary(query, rotary_emb),
            apply_wan_rotary(key, rotary_emb),
            value,
        )

    def attn_delta(self, attn_out: torch.Tensor) -> torch.Tensor:
        return self.proj(attn_out.flatten(2, 3)) * self.ls1

    def ffn_delta(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.norm2(hidden)) * self.ls2

    def forward(
        self,
        registers: torch.Tensor,
        rotary_emb: Optional[torch.Tensor],
        meta: MOTMaskMetadata,
        *,
        masked_attn_backend: str,
    ) -> torch.Tensor:
        query, key, value = self.qkv_project(
            registers.flatten(1, 2),
            rotary_emb,
        )
        attention_output = attention_from_meta(
            query,
            key,
            value,
            meta,
            backend=masked_attn_backend,
        )
        delta = self.attn_delta(attention_output).reshape_as(registers)
        residual = registers + delta
        return residual + self.ffn_delta(residual)


class ThreeDVAMOTBlock(nn.Module):
    """One complete V/A layer with optional even-layer G participation."""

    def __init__(
        self,
        video_block: WanTransformerBlock,
        action_block: ActionTransformerBlock,
        *,
        g_dim: Optional[int],
        attn_dim: int,
        num_heads: int,
        eps: float,
        masked_attn_backend: str,
    ):
        super().__init__()
        self.video_block = video_block
        self.action_block = action_block
        self.masked_attn_backend = masked_attn_backend
        self.geometry = (
            GeometryJointStream(g_dim, attn_dim, num_heads, eps)
            if g_dim is not None
            else None
        )

    @staticmethod
    def _modulation(
        block: nn.Module,
        temb: torch.Tensor,
    ) -> _BlockModulation:
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
        return (
            apply_wan_rotary(query, rotary_emb),
            apply_wan_rotary(key, rotary_emb),
            value,
        )

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
        query = attn.norm_q(attn.to_q(cross_input)).unflatten(
            2,
            (attn.heads, -1),
        )
        cross_output = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            text_key.transpose(1, 2),
            text_value.transpose(1, 2),
        ).transpose(1, 2)
        hidden = hidden + ThreeDVAMOTBlock._attention_output(attn, cross_output)

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
        states: VAStreamStates,              # hidden
        conditioning: VAStreamConditioning,  # time
        text: torch.Tensor,
        video_rotary: torch.Tensor,
        action_rotary: torch.Tensor,
        meta: MOTMaskMetadata,
        g_register: Optional[torch.Tensor] = None,
        g_rotary: Optional[torch.Tensor] = None,
    ) -> tuple[VAStreamStates, Optional[torch.Tensor]]:
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
            # Physical attention order is NV, CV, G, NA, CA.
            qkv_parts.insert(2, g_qkv)
            lengths.insert(2, g_qkv[0].shape[1])

        query, key, value = (
            torch.cat(parts, dim=1) for parts in zip(*qkv_parts)
        )
        attention_output = attention_from_meta(
            query,
            key,
            value,
            meta,
            backend=self.masked_attn_backend,
        )
        attention_parts = list(torch.split(attention_output, lengths, dim=1))

        g_updated = None
        if self.geometry is not None:
            g_output = attention_parts.pop(2)
            g_delta = self.geometry.attn_delta(g_output).reshape_as(g_register)
            g_residual = g_register + g_delta
            g_updated = g_residual + self.geometry.ffn_delta(g_residual)

        self_outputs = tuple(
            self._attention_output(block.attn1, output)
            for block, output in zip(blocks, attention_parts)
        )
        text_attn = self.video_block.attn2
        text_key = text_attn.norm_k(text_attn.to_k(text)).unflatten(
            2,
            (text_attn.heads, -1),
        )
        text_value = text_attn.to_v(text).unflatten(2, (text_attn.heads, -1))
        updated_states = VAStreamStates(
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


class ThreeDVAMOTTransformer3DModel(ModelMixin, ConfigMixin):
    """Fixed-topology training model for separate V/A/G MoT experts.

    Each layer jointly updates the noisy and clean V/A streams. Even layers
    additionally insert G registers in the fixed NV, CV, G, NA, CA order.
    """

    _supports_gradient_checkpointing = True
    _keep_in_fp32_modules = ["dense_head", "point_head"]
    _no_split_modules = ["ThreeDVAMOTBlock"]
    _repeated_blocks = ["ThreeDVAMOTBlock"]
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
        "vggto_patch_size": 16,
        "vggto_num_register_tokens": 16,
        "vggto_depth": 30,
        "vggto_register_attention_indices": tuple(range(0, 30, 2)),
        "vggto_cached_layer_indices": (5, 15, 21, 29),
        "vggto_pretrained_register_attention_indices": (2, 8, 12, 18, 26),
        "vggto_inserted_layer_indices": (4, 6, 10, 14, 22, 24),
        "vggto_converted_layer_indices": (0, 16, 20, 28),
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
        vggto_patch_size=16,
        vggto_num_register_tokens=16,
        vggto_depth=30,
        vggto_register_attention_indices=tuple(range(0, 30, 2)),
        vggto_cached_layer_indices=(5, 15, 21, 29),
        vggto_pretrained_register_attention_indices=(2, 8, 12, 18, 26),
        vggto_inserted_layer_indices=(4, 6, 10, 14, 22, 24),
        vggto_converted_layer_indices=(0, 16, 20, 28),
        init_noise_seed=42,
        vggto_cross_view_init_scale=1.0,
        attn_mode="torch",
        masked_attn_backend="fa4",
    ):
        super().__init__()
        self._validate_fixed_topology(locals())
        if masked_attn_backend not in {"fa4", "flex", "dense"}:
            raise ValueError(
                f"unsupported masked attention backend: {masked_attn_backend}"
            )

        self.patch_size = tuple(patch_size)
        self.num_attention_heads = int(num_attention_heads)
        self.attention_head_dim = int(attention_head_dim)
        self.inner_dim = self.num_attention_heads * self.attention_head_dim
        self.action_hidden_dim = int(action_hidden_dim)
        self.action_ffn_dim = int(action_ffn_dim)
        self.geometry_dim = 1024
        self.action_dim = int(action_dim)
        self.max_num_views = int(max_num_views)
        self.init_noise_seed = int(init_noise_seed)
        self.vggto_cross_view_init_scale = float(vggto_cross_view_init_scale)
        if (
            not math.isfinite(self.vggto_cross_view_init_scale)
            or self.vggto_cross_view_init_scale < 0
        ):
            raise ValueError(
                "vggto_cross_view_init_scale must be finite and non-negative, "
                f"got {self.vggto_cross_view_init_scale}"
            )
        self.masked_attn_backend = masked_attn_backend
        self.vggto_pretrained_register_attention_indices = set(
            int(index) for index in vggto_pretrained_register_attention_indices
        )
        self.vggto_inserted_layer_indices = set(
            int(index) for index in vggto_inserted_layer_indices
        )
        self.vggto_converted_layer_indices = set(
            int(index) for index in vggto_converted_layer_indices
        )

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

        self.vggto = VGGTOGeometryTower(
            patch_size=vggto_patch_size,
            num_register_tokens=vggto_num_register_tokens,
            depth=vggto_depth,
            register_attention_block_indices=vggto_register_attention_indices,
            cached_layer_indices=vggto_cached_layer_indices,
            masked_attn_backend=masked_attn_backend,
        )

        register_layers = self.vggto.register_attention_indices
        self.mot_blocks = nn.ModuleList(
            [
                ThreeDVAMOTBlock(
                    WanTransformerBlock(
                        self.inner_dim,
                        ffn_dim,
                        self.num_attention_heads,
                        cross_attn_norm,
                        eps,
                        attn_mode="torch",
                    ),
                    ActionTransformerBlock(
                        self.action_hidden_dim,
                        self.action_ffn_dim,
                        self.inner_dim,
                        self.num_attention_heads,
                        eps,
                    ),
                    g_dim=(self.geometry_dim if layer_id in register_layers else None),
                    attn_dim=self.inner_dim,
                    num_heads=self.num_attention_heads,
                    eps=eps,
                    masked_attn_backend=masked_attn_backend,
                )
                for layer_id in range(num_layers)
            ]
        )

        # Semantic rows are LEFT_WRIST=0, HEAD=1, RIGHT_WRIST=2.
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
            self.action_hidden_dim,
            eps,
            elementwise_affine=False,
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
                f"{name}={observed!r} (expected {expected_value!r})"
                for name, (observed, expected_value) in mismatches.items()
            )
            raise ValueError(f"unsupported model topology: {details}")

    @torch.no_grad()
    def _init_stream_embedding(self) -> None:
        if self.stream_embedding.weight.is_meta:
            return
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
            raise ValueError(
                f"views={views} exceeds max_num_views={self.max_num_views}"
            )
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
            bsz,
            frames,
            views,
            h_tokens * w_tokens,
            self.inner_dim,
        )
        return (
            hidden + stream_emb.reshape(bsz, -1, self.inner_dim),
            (bsz, frames, views, height, width, h_tokens, w_tokens),
        )

    def _embed_action(self, actions: torch.Tensor) -> torch.Tensor:
        hidden = rearrange(actions, "b c t n w -> b (t n w) c")
        return self.action_embedder(hidden.to(dtype=self.action_embedder.weight.dtype))

    def _embed_geometry(
        self,
        geometry_dict: dict,
        chunk_size: int,
    ) -> _GeometryContext:
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        rgb = geometry_dict["rgb"]
        if rgb.ndim != 7:
            raise ValueError(
                f"VGGTO geometry rgb must be [B,G,S,V,3,H,W], got {tuple(rgb.shape)}"
            )
        bsz, groups, group_size, views, channels, height, width = rgb.shape
        if views < 2:
            raise ValueError(f"synchronized multi-view VGGTO requires V>=2, got V={views}")
        if (height, width) != (224, 224):
            raise ValueError(
                f"synchronized multi-view VGGTO requires 224x224 RGB, got {height}x{width}"
            )
        if groups % chunk_size != 0:
            raise ValueError(
                f"geometry groups={groups} must be divisible by groups_per_chunk={chunk_size}"
            )
        slot_valid_mask = geometry_dict.get("slot_valid_mask")
        if slot_valid_mask is None:
            raise ValueError("VGGTO grouped geometry requires slot_valid_mask [B,G,S]")
        slot_valid_mask = slot_valid_mask.to(device=rgb.device, dtype=torch.bool)
        if tuple(slot_valid_mask.shape) != (bsz, groups, group_size):
            raise ValueError(
                f"geometry slot_valid_mask must be [B,{groups},{group_size}], "
                f"got {tuple(slot_valid_mask.shape)}"
            )
        if bool(slot_valid_mask.all().item()):
            slot_valid_mask = None
        images = rgb.reshape(
            bsz,
            groups * group_size * views,
            channels,
            height,
            width,
        )
        image_valid_mask = None
        if slot_valid_mask is not None:
            image_valid_mask = (
                slot_valid_mask[:, :, :, None]
                .expand(-1, -1, -1, views)
                .reshape(bsz, groups * group_size * views)
            )

        state = self.vggto.encode_grouped(rgb, slot_valid_mask=slot_valid_mask)
        g_rotary = self.rope(
            self._geometry_register_grid(
                bsz,
                groups,
                group_size,
                views,
                self.vggto.patch_start_idx,
                state.tokens.device,
            )
        )[:, :, None]
        return _GeometryContext(
            state=state,
            source_images=images,
            groups=groups,
            group_size=group_size,
            views=views,
            groups_per_chunk=chunk_size,
            register_rotary=g_rotary,
            cached_outputs=[None] * self.vggto.depth,
            slot_valid_mask=slot_valid_mask,
            image_valid_mask=image_valid_mask,
        )

    def _complete_vggto_layer(
        self,
        geometry: _GeometryContext,
        layer_id: int,
        frame_tokens: torch.Tensor,
        *,
        register_override: Optional[torch.Tensor] = None,
    ) -> _GeometryContext:
        tokens = self.vggto.complete_layer(
            frame_tokens,
            geometry.state.patch_hw,
            int(layer_id),
            groups=geometry.groups,
            group_size=geometry.group_size,
            views=geometry.views,
            groups_per_chunk=geometry.groups_per_chunk,
            slot_valid_mask=geometry.slot_valid_mask,
            register_override=register_override,
            cached_outputs=geometry.cached_outputs,
        )
        return geometry._replace(
            state=VGGTOGeometryState(
                tokens=tokens,
                patch_hw=geometry.state.patch_hw,
                image_hw=geometry.state.image_hw,
                patch_token_start=geometry.state.patch_token_start,
            )
        )

    @staticmethod
    def _restore_invalid_registers(
        updated: torch.Tensor,
        original: torch.Tensor,
        image_valid_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if image_valid_mask is None:
            return updated
        valid = image_valid_mask.to(device=updated.device, dtype=torch.bool)
        return torch.where(valid[:, :, None, None], updated, original)

    def _geometry_predictions(
        self,
        geometry: _GeometryContext,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        depth, depth_conf = self.vggto.dense_forward(
            geometry.cached_outputs,
            geometry.source_images,
        )
        points, points_conf = self.vggto.point_forward(
            geometry.cached_outputs,
            geometry.source_images,
        )
        bsz = depth.shape[0]
        output_prefix = (bsz, geometry.groups * geometry.group_size, geometry.views)
        depth_shape = (*output_prefix, *depth.shape[-3:-1])
        point_shape = (*output_prefix, *points.shape[-3:-1])
        return (
            depth.reshape(*depth_shape, 1),
            depth_conf.reshape(depth_shape),
            points.reshape(*point_shape, 3),
            points_conf.reshape(point_shape),
        )

    @staticmethod
    def _video_grid(bsz, frames, views, h_tokens, w_tokens, device):
        frame = torch.arange(frames, device=device, dtype=torch.float32)
        view = torch.arange(views, device=device, dtype=torch.float32)
        height = torch.arange(h_tokens, device=device, dtype=torch.float32)
        width = torch.arange(w_tokens, device=device, dtype=torch.float32)
        ff, _vv, hh, ww = torch.meshgrid(
            frame,
            view,
            height,
            width,
            indexing="ij",
        )
        grid = torch.stack([ff, hh, ww, torch.zeros_like(ff)], dim=0).reshape(4, -1)
        return grid[None].expand(bsz, -1, -1)

    @staticmethod
    def _action_positions(
        bsz: int,
        frames: int,
        action_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        latent_ids = torch.arange(frames, device=device, dtype=torch.long)[:, None]
        slot_ids = torch.arange(
            action_tokens_per_frame,
            device=device,
            dtype=torch.long,
        )[None, :]
        positions = latent_ids * action_tokens_per_frame + slot_ids
        return positions.reshape(1, -1).expand(bsz, -1)

    @staticmethod
    def _geometry_register_grid(
        bsz,
        groups,
        group_size,
        views,
        register_tokens,
        device,
    ):
        # group-level time idx
        ff = torch.arange(groups, device=device, dtype=torch.float32)[
            :, None, None, None
        ].expand(groups, group_size, views, register_tokens)
        hh = torch.full_like(ff, -1.0)
        ww = torch.full_like(ff, -1.0)
        grid = torch.stack([ff, hh, ww, torch.zeros_like(ff)], dim=0).reshape(4, -1)
        return grid[None].expand(bsz, -1, -1)

    @staticmethod
    def _geometry_token_valid_ids(
        slot_valid_mask: torch.Tensor,
        *,
        batch_size: int,
        groups: int,
        views: int,
        register_tokens: int,
        device: torch.device,
    ) -> torch.Tensor:
        slot_valid = slot_valid_mask.to(device=device, dtype=torch.bool)
        if slot_valid.ndim != 3 or tuple(slot_valid.shape[:2]) != (batch_size, groups):
            raise ValueError(
                f"geometry slot_valid_mask must be [B,{groups},S], "
                f"got {tuple(slot_valid.shape)}"
            )
        return slot_valid[:, :, :, None, None].expand(
            -1,
            -1,
            -1,
            views,
            register_tokens,
        ).reshape(batch_size, -1)

    def _mot_token_valid_ids(
        self,
        input_dict,
        *,
        video_tokens_per_frame: int,
        geometry: _GeometryContext,
        action_tokens_per_frame: int,
        frames: int,
        device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = input_dict["latent_dict"]["noisy_latents"].shape[0]
        video_latent_valid_mask = input_dict["latent_dict"].get(
            "video_latent_valid_mask"
        )
        if video_latent_valid_mask is None:
            video_valid = torch.ones((bsz, frames), dtype=torch.bool, device=device)
        else:
            video_valid = video_latent_valid_mask.to(device=device, dtype=torch.bool)
        if tuple(video_valid.shape) != (bsz, frames):
            raise ValueError(
                f"video_latent_valid_mask must be [B,{frames}], "
                f"got {tuple(video_valid.shape)}"
            )
        video_tokens = video_valid[:, :, None].expand(
            -1,
            -1,
            video_tokens_per_frame,
        ).reshape(bsz, -1)

        action_valid_mask = input_dict["action_dict"].get("action_valid_mask")
        if action_valid_mask is None:
            action_valid = torch.ones(
                (bsz, frames, action_tokens_per_frame),
                dtype=torch.bool,
                device=device,
            )
        else:
            action_valid = action_valid_mask.to(
                device=device,
                dtype=torch.bool,
            ).any(dim=1)[..., 0]
        if tuple(action_valid.shape) != (bsz, frames, action_tokens_per_frame):
            raise ValueError(
                "action_valid_mask must produce "
                f"[B,{frames},{action_tokens_per_frame}], "
                f"got {tuple(action_valid.shape)}"
            )
        action_tokens = action_valid.reshape(bsz, -1)
        x_valid = torch.cat(
            [video_tokens, video_tokens, action_tokens, action_tokens],
            dim=1,
        )

        slot_valid_mask = input_dict["geometry_dict"].get("slot_valid_mask")
        if slot_valid_mask is None:
            geometry_tokens_per_frame = (
                geometry.group_size * geometry.views * self.vggto.patch_start_idx
            )
            geometry_tokens = torch.ones(
                (bsz, frames * geometry_tokens_per_frame),
                dtype=torch.bool,
                device=device,
            )
        else:
            geometry_tokens = self._geometry_token_valid_ids(
                slot_valid_mask,
                batch_size=bsz,
                groups=frames,
                views=geometry.views,
                register_tokens=self.vggto.patch_start_idx,
                device=device,
            )
        mot_valid = torch.cat(
            [
                video_tokens,
                video_tokens,
                geometry_tokens,
                action_tokens,
                action_tokens,
            ],
            dim=1,
        )
        return x_valid, mot_valid

    def _time_embed_repeated(
        self,
        timesteps: torch.Tensor,
        repeats_per_frame: int,
        dtype: torch.dtype,
        *,
        action: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        repeated = torch.repeat_interleave(timesteps, repeats_per_frame, dim=1)
        embedder = (
            self.action_condition_embedder if action else self.condition_embedder
        )
        temb, timestep_proj = embedder(repeated, dtype=dtype)
        return temb, timestep_proj.unflatten(2, (6, -1))

    def _prepare_va_inputs(self, input_dict) -> _PreparedVAInputs:
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        geometry_dict = input_dict["geometry_dict"]
        stream_ids = geometry_dict["stream_ids"].long().to(
            latent_dict["noisy_latents"].device
        )

        noisy_video, latent_shape = self._embed_video(
            latent_dict["noisy_latents"],
            stream_ids,
        )
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
            raise ValueError(
                f"action F={action_shape[2]} must match video latent F={frames}"
            )
        action_tokens_per_frame = action_shape[3] * action_shape[4]

        video_grid = self._video_grid(
            bsz,
            frames,
            views,
            h_tokens,
            w_tokens,
            noisy_video.device,
        )
        video_rotary = self.rope(video_grid)[:, :, None]
        action_rotary = self.action_rope(
            self._action_positions(
                bsz,
                frames,
                action_tokens_per_frame,
                noisy_action.device,
            )
        )

        video_repeats = views * h_tokens * w_tokens
        action_repeats = action_tokens_per_frame
        _, noisy_video_temb = self._time_embed_repeated(
            latent_dict["timesteps"],
            video_repeats,
            noisy_video.dtype,
            action=False,
        )
        _, clean_video_temb = self._time_embed_repeated(
            latent_dict["cond_timesteps"],
            video_repeats,
            clean_video.dtype,
            action=False,
        )
        _, noisy_action_temb = self._time_embed_repeated(
            action_dict["timesteps"],
            action_repeats,
            noisy_action.dtype,
            action=True,
        )
        _, clean_action_temb = self._time_embed_repeated(
            action_dict["cond_timesteps"],
            action_repeats,
            clean_action.dtype,
            action=True,
        )
        return _PreparedVAInputs(
            states=VAStreamStates(
                noisy_video,
                clean_video,
                noisy_action,
                clean_action,
            ),
            conditioning=VAStreamConditioning(
                noisy_video_temb,
                clean_video_temb,
                noisy_action_temb,
                clean_action_temb,
            ),
            text=text,
            video_rotary=video_rotary,
            action_rotary=action_rotary,
            shape=latent_shape,
            action_tokens_per_frame=action_tokens_per_frame,
        )

    def _prepare_metadata(
        self,
        input_dict,
        prepared: _PreparedVAInputs,
        geometry: _GeometryContext,
    ) -> tuple[MOTMaskMetadata, MOTMaskMetadata, dict[str, int]]:
        states = prepared.states
        bsz, frames, views, _height, _width, h_tokens, w_tokens = prepared.shape
        action_tokens_per_frame = prepared.action_tokens_per_frame
        chunk_size = int(input_dict["chunk_size"])
        window_size = int(input_dict["window_size"])
        video_tokens_per_frame = views * h_tokens * w_tokens
        if geometry.groups != frames:
            raise ValueError(
                f"geometry groups={geometry.groups} must match VA frames={frames}"
            )
        x_valid, mot_valid = self._mot_token_valid_ids(
            input_dict,
            video_tokens_per_frame=video_tokens_per_frame,
            geometry=geometry,
            action_tokens_per_frame=action_tokens_per_frame,
            frames=frames,
            device=states.noisy_video.device,
        )
        x_meta = build_x_metadata(
            batch_size=bsz,
            video_tokens_per_frame=video_tokens_per_frame,
            action_tokens_per_frame=action_tokens_per_frame,
            num_frames=frames,
            chunk_size=chunk_size,
            window_size=window_size,
            device=states.noisy_video.device,
            token_valid_ids=x_valid,
        )
        geometry_tokens_per_frame = (
            geometry.group_size * geometry.views * self.vggto.patch_start_idx
        )
        mot_meta = build_mot_metadata(
            batch_size=bsz,
            video_tokens_per_frame=video_tokens_per_frame,
            geometry_tokens_per_frame=geometry_tokens_per_frame,
            action_tokens_per_frame=action_tokens_per_frame,
            num_frames=frames,
            chunk_size=chunk_size,
            window_size=window_size,
            device=states.noisy_video.device,
            token_valid_ids=mot_valid,
        )
        diagnostics = {
            "video_tokens_per_frame": video_tokens_per_frame,
            "geometry_tokens_per_frame": geometry_tokens_per_frame,
            "action_tokens_per_frame": action_tokens_per_frame,
        }
        return x_meta, mot_meta, diagnostics

    def _prepare_geometry_metadata(
        self,
        geometry: _GeometryContext,
        window_size: int,
    ) -> MOTMaskMetadata:
        batch_size = geometry.state.tokens.shape[0]
        geometry_tokens_per_frame = (
            geometry.group_size * geometry.views * self.vggto.patch_start_idx
        )
        token_valid_ids = None
        if geometry.slot_valid_mask is not None:
            token_valid_ids = self._geometry_token_valid_ids(
                geometry.slot_valid_mask,
                batch_size=batch_size,
                groups=geometry.groups,
                views=geometry.views,
                register_tokens=self.vggto.patch_start_idx,
                device=geometry.state.tokens.device,
            )
        return build_geometry_metadata(
            batch_size=batch_size,
            geometry_tokens_per_frame=geometry_tokens_per_frame,
            num_frames=geometry.groups,
            chunk_size=geometry.groups_per_chunk,
            window_size=int(window_size),
            device=geometry.state.tokens.device,
            token_valid_ids=token_valid_ids,
        )

    def _run_geometry_layers(
        self,
        geometry: _GeometryContext,
        meta: MOTMaskMetadata,
        *,
        capture_layer_registers: bool,
    ) -> tuple[_GeometryContext, Optional[dict[int, torch.Tensor]]]:
        layer_registers = {} if capture_layer_registers else None
        for layer_id, block in enumerate(self.mot_blocks):
            state = geometry.state
            frame_tokens = self.vggto.run_frame_block(
                state.tokens,
                state.patch_hw,
                layer_id,
            )
            register_override = None
            if layer_id in self.vggto.register_attention_indices:
                g_register = frame_tokens[:, :, : self.vggto.patch_start_idx]
                if layer_registers is not None:
                    layer_registers[layer_id] = g_register.contiguous()
                register_override = block.geometry(
                    g_register,
                    geometry.register_rotary,
                    meta,
                    masked_attn_backend=self.masked_attn_backend,
                )
                register_override = self._restore_invalid_registers(
                    register_override,
                    g_register,
                    geometry.image_valid_mask,
                )
            geometry = self._complete_vggto_layer(
                geometry,
                layer_id,
                frame_tokens,
                register_override=register_override,
            )
        return geometry, layer_registers

    def _run_training_mot_layers(
        self,
        states: VAStreamStates,
        geometry: _GeometryContext,
        *,
        conditioning: VAStreamConditioning,
        text: torch.Tensor,
        video_rotary: torch.Tensor,
        action_rotary: torch.Tensor,
        mot_meta: MOTMaskMetadata,
        x_meta: MOTMaskMetadata,
    ) -> tuple[VAStreamStates, _GeometryContext]:
        for layer_id, block in enumerate(self.mot_blocks):
            state = geometry.state
            frame_tokens = self.vggto.run_frame_block(
                state.tokens,
                state.patch_hw,
                layer_id,
            )
            is_joint_g_layer = layer_id in self.vggto.register_attention_indices
            g_register = (
                frame_tokens[:, :, : self.vggto.patch_start_idx]
                if is_joint_g_layer
                else None
            )
            states, g_updated = block(
                states,
                conditioning,
                text,
                video_rotary,
                action_rotary,
                mot_meta if is_joint_g_layer else x_meta,
                g_register,
                geometry.register_rotary if is_joint_g_layer else None,
            )
            if is_joint_g_layer:
                g_updated = self._restore_invalid_registers(
                    g_updated,
                    g_register,
                    geometry.image_valid_mask,
                )
            geometry = self._complete_vggto_layer(
                geometry,
                layer_id,
                frame_tokens,
                register_override=g_updated,
            )
        return states, geometry

    def _run_inference_mot_layers(
        self,
        states: VAStreamStates,
        geometry_condition: dict,
        *,
        conditioning: VAStreamConditioning,
        text: torch.Tensor,
        video_rotary: torch.Tensor,
        action_rotary: torch.Tensor,
        mot_meta: MOTMaskMetadata,
        x_meta: MOTMaskMetadata,
    ) -> VAStreamStates:
        layer_registers = geometry_condition["layer_registers"]
        expected_layers = self.vggto.register_attention_indices
        if set(layer_registers) != expected_layers:
            raise ValueError(
                "precomputed geometry must contain one register snapshot per "
                "joint MOT layer: "
                f"expected={sorted(expected_layers)}, got={sorted(layer_registers)}"
            )
        geometry = geometry_condition["final_geometry"]
        for layer_id, block in enumerate(self.mot_blocks):
            is_joint_g_layer = layer_id in expected_layers
            g_register = None
            g_rotary = None
            if is_joint_g_layer:
                g_register = layer_registers[layer_id].to(
                    device=states.noisy_video.device,
                    dtype=states.noisy_video.dtype,
                )
                g_rotary = geometry.register_rotary.to(
                    device=states.noisy_video.device
                )
            states, _g_updated = block(
                states,
                conditioning,
                text,
                video_rotary,
                action_rotary,
                mot_meta if is_joint_g_layer else x_meta,
                g_register,
                g_rotary,
            )
        return states

    def _project_output(
        self,
        hidden: torch.Tensor,
        timesteps: torch.Tensor,
        repeats_per_frame: int,
        *,
        action: bool,
    ) -> torch.Tensor:
        temb, _ = self._time_embed_repeated(
            timesteps,
            repeats_per_frame,
            hidden.dtype,
            action=action,
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

    def _final_video(
        self,
        hidden: torch.Tensor,
        timesteps: torch.Tensor,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        _bsz, frames, views, _height, _width, h_tokens, w_tokens = shape
        hidden = self._project_output(
            hidden,
            timesteps,
            views * h_tokens * w_tokens,
            action=False,
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

    def _final_action(
        self,
        hidden: torch.Tensor,
        timesteps: torch.Tensor,
        action_shape: tuple[int, ...],
    ) -> torch.Tensor:
        _bsz, _channels, frames, action_per_frame, width = action_shape
        hidden = self._project_output(
            hidden,
            timesteps,
            action_per_frame * width,
            action=True,
        )
        return rearrange(
            hidden,
            "b (t n w) c -> b c t n w",
            t=frames,
            n=action_per_frame,
            w=width,
        )

    def forward_train(self, input_dict, return_diagnostics: bool = False):
        """Runs one joint V/A/G training forward pass."""

        geometry_dict = input_dict["geometry_dict"]
        if "precomputed_condition" in geometry_dict:
            raise ValueError(
                "forward_train requires raw geometry and does not accept "
                "precomputed_condition"
            )
        prepared = self._prepare_va_inputs(input_dict)
        geometry = self._embed_geometry(
            geometry_dict,
            int(input_dict["chunk_size"]),
        )
        x_meta, mot_meta, diagnostics = self._prepare_metadata(
            input_dict,
            prepared,
            geometry,
        )
        states, geometry = self._run_training_mot_layers(
            prepared.states,
            geometry,
            conditioning=prepared.conditioning,
            text=prepared.text,
            video_rotary=prepared.video_rotary,
            action_rotary=prepared.action_rotary,
            mot_meta=mot_meta,
            x_meta=x_meta,
        )
        depth, depth_conf, points, points_conf = self._geometry_predictions(geometry)
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
            "depth": depth,
            "depth_conf": depth_conf,
            "points": points,
            "points_conf": points_conf,
            "geometry_valid_mask": geometry_dict.get("valid_mask"),
            "diagnostics": diagnostics,
        }
        if return_diagnostics:
            out["geometry_states"] = {"G_state_last": geometry.state.tokens}
        return out

    def _run_grouped_geometry_core(
        self,
        input_dict: dict,
        *,
        capture_layer_registers: bool,
    ) -> dict:
        chunk_size = int(input_dict["chunk_size"])
        window_size = input_dict.get("window_size")
        if window_size is None:
            window_size = 2 * math.ceil(input_dict["rgb"].shape[1] / chunk_size)
        geometry = self._embed_geometry(input_dict, chunk_size)
        geometry_meta = self._prepare_geometry_metadata(geometry, int(window_size))
        geometry, layer_registers = self._run_geometry_layers(
            geometry,
            geometry_meta,
            capture_layer_registers=capture_layer_registers,
        )
        depth, depth_conf, points, points_conf = self._geometry_predictions(
            geometry
        )
        return {
            "final_geometry": geometry,
            "layer_registers": layer_registers,
            "depth": depth,
            "depth_conf": depth_conf,
            "points": points,
            "points_conf": points_conf,
        }

    def forward_train_geometry(self, input_dict: dict) -> dict:
        required = {"rgb", "slot_valid_mask", "chunk_size", "window_size"}
        if set(input_dict) != required:
            raise ValueError(
                "train_geometry input keys must be exactly "
                f"{sorted(required)}, got {sorted(input_dict)}"
            )
        result = self._run_grouped_geometry_core(
            input_dict,
            capture_layer_registers=False,
        )
        return {
            key: result[key]
            for key in ("depth", "depth_conf", "points", "points_conf")
        }

    def precompute_geometry_condition(self, input_dict) -> dict:
        """Run the independent G stream and cache each joint layer's input."""

        result = self._run_grouped_geometry_core(
            input_dict,
            capture_layer_registers=True,
        )
        layer_registers = result["layer_registers"]
        if set(layer_registers) != self.vggto.register_attention_indices:
            raise RuntimeError("precompute_geometry did not capture every register layer")
        out = {
            "final_geometry": result["final_geometry"],
            "layer_registers": layer_registers,
            "diagnostics": {
                "register_tokens": self.vggto.patch_start_idx,
                "register_attention_indices": sorted(
                    self.vggto.register_attention_indices
                ),
            },
        }
        if bool(input_dict.get("return_points", True)):
            out.update(
                depth=result["depth"],
                depth_conf=result["depth_conf"],
                points=result["points"],
                points_conf=result["points_conf"],
            )
        return out

    def forward_inference(self, input_dict, *, prediction_type: str) -> dict:
        if prediction_type not in {"video", "action"}:
            raise ValueError(
                "prediction_type must be 'video' or 'action', "
                f"got {prediction_type!r}"
            )
        geometry_condition = input_dict["geometry_dict"].get(
            "precomputed_condition"
        )
        if geometry_condition is None:
            raise ValueError(
                "forward_inference requires "
                "geometry_dict['precomputed_condition']"
            )
        prepared = self._prepare_va_inputs(input_dict)
        geometry = geometry_condition["final_geometry"]
        x_meta, mot_meta, diagnostics = self._prepare_metadata(
            input_dict,
            prepared,
            geometry,
        )
        states = self._run_inference_mot_layers(
            prepared.states,
            geometry_condition,
            conditioning=prepared.conditioning,
            text=prepared.text,
            video_rotary=prepared.video_rotary,
            action_rotary=prepared.action_rotary,
            mot_meta=mot_meta,
            x_meta=x_meta,
        )
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

    def forward(
        self,
        input_dict,
        *,
        mode: str,
        return_diagnostics: bool = False,
        **_kwargs,
    ):
        if mode == "train":
            return self.forward_train(
                input_dict,
                return_diagnostics=return_diagnostics,
            )
        if mode == "train_geometry":
            return self.forward_train_geometry(input_dict)
        if mode == "precompute_geometry":
            return self.precompute_geometry_condition(input_dict)
        if mode == "inference_video":
            return self.forward_inference(input_dict, prediction_type="video")
        if mode == "inference_action":
            return self.forward_inference(input_dict, prediction_type="action")
        raise ValueError(f"unsupported mode={mode!r}")

    @classmethod
    def from_lingbot_and_vggto(
        cls,
        video_transformer_path: str,
        vggto_checkpoint_path: str,
        vggt_checkpoint_path: str,
        init_model_from_lingbot: bool = True,
        config_overrides: Optional[dict] = None,
    ):
        """Initialize V/A from LingBot or Wan2.2 and G from VGGTO/VGGT."""

        video_source = "LingBot" if init_model_from_lingbot else "Wan2.2"
        check_diffusers_transformer_complete(
            video_transformer_path,
            source_name=video_source,
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
        report = InitReport(
            video_source=video_source,
            video_path=video_transformer_path,
            vggto_path=vggto_checkpoint_path,
            vggt_point_path=vggt_checkpoint_path,
        )

        for name in (
            "rope",
            "condition_embedder",
            "norm_out",
            "proj_out",
        ):
            target = getattr(model, name)
            source = getattr(base, name)
            target.load_state_dict(source.state_dict(), strict=True)
        if init_model_from_lingbot:
            model.patch_embedding_mlp.load_state_dict(
                base.patch_embedding_mlp.state_dict(),
                strict=True,
            )
        else:
            initialize_wan_patch_embedding_(
                model.patch_embedding_mlp,
                base.patch_embedding,
            )
        model.scale_shift_table.data.copy_(base.scale_shift_table.data)
        for layer_id, block in enumerate(model.mot_blocks):
            block.video_block.load_state_dict(
                base.blocks[layer_id].state_dict(),
                strict=True,
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
        if init_model_from_lingbot:
            report.notes.append(
                "ignored LingBot 30D action_embedder/action_proj_out and "
                "condition_embedder_action"
            )
        else:
            report.notes.append("Wan2.2 provides video weights only")
        report.notes.append(
            "the new 20D action I/O is random and 768D conditioning is "
            "initialized from the video conditioner"
        )

        vggto_state = normalize_vggto_keys(
            load_vggto_state_dict(vggto_checkpoint_path)
        )
        remap_report = load_vggto_24_into_30_(
            model.vggto,
            vggto_state,
            cross_view_layer_scale_factor=model.vggto_cross_view_init_scale,
        )
        report.vggto_24_to_30 = remap_report
        expected_layer_categories = {
            "pretrained": set(remap_report.plan.original_register_targets),
            "inserted": set(remap_report.plan.inserted_targets),
            "converted": set(remap_report.plan.converted_mot_targets),
        }
        actual_layer_categories = {
            "pretrained": model.vggto_pretrained_register_attention_indices,
            "inserted": model.vggto_inserted_layer_indices,
            "converted": model.vggto_converted_layer_indices,
        }
        if actual_layer_categories != expected_layer_categories:
            raise ValueError(
                "MOT layer categories must match the VGGTO 24-to-30 "
                "initialization plan: "
                f"expected={expected_layer_categories}, "
                f"actual={actual_layer_categories}"
            )

        point_head_state = load_vggt_point_head_state_dict(vggt_checkpoint_path)
        model.vggto.point_head.load_state_dict(point_head_state, strict=True)
        report.loaded_vggt_point_keys = len(point_head_state)
        report.notes.append("point_head loaded strictly from original VGGT checkpoint")
        init_shared_register_from_first_other_(
            model.vggto.register_token,
            vggto_state["aggregator.register_token"],
        )
        hidden_map = head_aware_hidden_map(model.inner_dim)
        generator = torch.Generator(device="cpu").manual_seed(model.init_noise_seed)
        for layer_id in sorted(model.vggto.register_attention_indices):
            source_layer = remap_report.plan.source_for_mot(layer_id)
            init_geometry_from_vggto_inter_frame_(
                model.mot_blocks[layer_id].geometry,
                vggto_state,
                f"aggregator.inter_frame_blocks.{source_layer}",
                hidden_map=hidden_map,
                generator=generator,
                noise_scale=0.0,
            )
        report.loaded_vggto_required_keys = len(vggto_state)
        report.ignored_vggto_keys = ignored_vggto_keys(vggto_state)
        return model, report
