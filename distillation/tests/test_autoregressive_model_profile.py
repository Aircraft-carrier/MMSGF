"""Construction and profile tests for the independent AR model family."""
from __future__ import annotations

import os

import torch.distributed as dist
import torch
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from distillation.model.autoregressive_mot import (
    AutoregressiveThreeDVAMOTBlock,
    AutoregressiveThreeDVAMOTTransformer3DModel,
)
from distillation.model.autoregressive_vggto import (
    AutoregressiveVGGTOBlock,
    AutoregressiveVGGTOGeometryTower,
)
from wan_va.modules.model_3dva_mot import ThreeDVAMOTTransformer3DModel
from wan_va.modules.model_3dva_mot import ActionTransformerBlock, WanTransformerBlock
from wan_va.modules.vggto_geometry import VGGTOGeometryTower
from wan_va.train_mot import apply_ac_mot, apply_ac_vggto
from distillation.model.autoregressive_types import (
    AutoregressiveGeometryRelationRequest,
    AutoregressiveMOTLayerRequest,
    AutoregressiveStreamInput,
)
from distillation.self_rollout.attention import (
    NOISE_CLEAN,
    NOISE_GEOMETRY,
    STREAM_GEOMETRY,
    STREAM_VIDEO,
    build_token_metadata,
)
from distillation.self_rollout.cache import GeometryRolloutCache
from distillation.self_rollout.state import CacheSource, RolloutState


def _init_cpu_process_group(init_file) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1:
        dist.init_process_group(
            "gloo",
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
    else:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{init_file}",
            rank=0,
            world_size=1,
        )


def test_ar_model_constructs_specialized_components_without_monkeypatch() -> None:
    with torch.device("meta"):
        model = AutoregressiveThreeDVAMOTTransformer3DModel()

    assert isinstance(model, ThreeDVAMOTTransformer3DModel)
    assert isinstance(model.vggto, AutoregressiveVGGTOGeometryTower)
    assert isinstance(model.vggto, VGGTOGeometryTower)
    assert isinstance(model.mot_blocks[0], AutoregressiveThreeDVAMOTBlock)
    assert isinstance(model.vggto.inter_frame_blocks[1], AutoregressiveVGGTOBlock)
    assert not hasattr(model, "_distillation_order_profile")
    assert not hasattr(model.vggto, "_distillation_vggto_order_profile")


def test_ar_model_keeps_parent_parameter_key_contract() -> None:
    # Meta construction avoids allocating the 3DVA weights while still
    # exposing the complete module tree and every parameter shape.
    with torch.device("meta"):
        model = AutoregressiveThreeDVAMOTTransformer3DModel()
    keys = set(model.state_dict())
    assert "mot_blocks.0.video_block.attn1.to_q.weight" in keys
    assert "vggto.inter_frame_blocks.1.attn.qkv.weight" in keys
    assert all(not key.startswith("_autoregressive_rollout") for key in keys)


def test_ar_profile_configuration_is_explicit_and_validated() -> None:
    with torch.device("meta"):
        model = AutoregressiveThreeDVAMOTTransformer3DModel()
    model.configure_generation_profile(
        {
            "profile_name": "segmented_history_strict_geometry_v1",
            "order_mode": "segmented",
            "history_frames": 4,
            "chunk_size": 4,
            "window_size": 16,
        }
    )
    assert model.generation_profile.history_frames == 4
    assert model.vggto.generation_profile.window_size == 16
    assert model.config["generation_profile"]["order_mode"] == "segmented"


def test_activation_checkpoint_wrappers_preserve_ar_forward_capabilities() -> None:
    with torch.device("meta"):
        model = AutoregressiveThreeDVAMOTTransformer3DModel()
    apply_ac_mot(model)
    apply_ac_vggto(model)
    assert hasattr(model.mot_blocks[0], "forward_incremental")
    assert hasattr(model.vggto.inter_frame_blocks[1], "forward_incremental")
    assert hasattr(model.mot_blocks[0], "forward_geometry_incremental")


def test_ar_mot_incremental_forward_runs_through_cpu_fsdp(tmp_path) -> None:
    """Exercise the exact block forward boundary used by FSDP2 rollout.

    CUDA is not required: FSDP2's Gloo/CPU path still proves that the request
    reaches the wrapped AR block and that the cache stores ordinary tensors.
    """

    init_file = tmp_path / "fsdp_init"
    _init_cpu_process_group(init_file)
    try:
        video = WanTransformerBlock(4, 8, 1, True, 1e-6, attn_mode="torch")
        action = ActionTransformerBlock(4, 8, 4, 1, 1e-6)
        block = AutoregressiveThreeDVAMOTBlock(
            video,
            action,
            g_dim=None,
            attn_dim=4,
            num_heads=1,
            eps=1e-6,
            masked_attn_backend="dense",
        )
        fully_shard(block)
        hidden = torch.randn(1, 2, 4)
        conditioning = torch.randn(1, 2, 6, 4)
        metadata = build_token_metadata(
            batch_size=1,
            frame_ids=torch.tensor([0]),
            tokens_per_frame=2,
            stream_id=STREAM_VIDEO,
            noise_id=NOISE_CLEAN,
            history_frames=1,
            chunk_size=1,
            device=hidden.device,
            committed=False,
            transaction_id=1,
            source_id=int(CacheSource.HISTORY),
            version_id=1,
        )
        request = AutoregressiveMOTLayerRequest(
            streams=[
                AutoregressiveStreamInput(
                    hidden,
                    conditioning,
                    None,
                    metadata,
                    "video",
                )
            ],
            hidden_states=[hidden],
            text=torch.randn(1, 3, 4),
            state=RolloutState(),
            transaction_id=1,
            layer_id=0,
            stream_id=STREAM_VIDEO,
        )
        output = block(request)
        assert len(output) == 1
        assert output[0].shape == hidden.shape
        assert not isinstance(output[0], DTensor)
    finally:
        dist.destroy_process_group()


def test_ar_vggto_relation_forward_runs_through_cpu_fsdp(tmp_path) -> None:
    init_file = tmp_path / "fsdp_vggto_init"
    _init_cpu_process_group(init_file)
    try:
        from wan_va.modules.vggto_vendored.layers.ffn_layers import Mlp

        block = AutoregressiveVGGTOBlock(
            dim=4,
            num_heads=2,
            ffn_ratio=2,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
            ffn_layer=Mlp,
            init_values=1e-5,
            use_qk_norm=True,
            mask_k_bias=True,
        )
        fully_shard(block)
        values = torch.randn(1, 2, 4)
        metadata = build_token_metadata(
            batch_size=1,
            frame_ids=torch.tensor([0]),
            tokens_per_frame=2,
            stream_id=STREAM_GEOMETRY,
            noise_id=NOISE_GEOMETRY,
            history_frames=1,
            chunk_size=1,
            device=values.device,
            committed=False,
            transaction_id=3,
            source_id=int(CacheSource.HISTORY),
            version_id=1,
        )
        output = block(
            AutoregressiveGeometryRelationRequest(
                values=values,
                rope=None,
                metadata=metadata,
                cache=GeometryRolloutCache(),
                transaction_id=3,
                layer_id=0,
            )
        )
        assert output.shape == values.shape
        assert not isinstance(output, DTensor)
    finally:
        dist.destroy_process_group()
