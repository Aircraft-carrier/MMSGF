from types import SimpleNamespace

import torch

from distillation.model.autoregressive_mot import (
    AutoregressiveMOTLayerRequest,
    AutoregressiveStreamInput,
    AutoregressiveVAMOTBlock,
    AutoregressiveVAMOTTransformer3DModel,
)
from distillation.pipeline.cache import KVCache
from wan_va.modules.model import WanTransformerBlock
from wan_va.modules.model_va_mot import (
    ActionTransformerBlock,
    VAMOTTransformer3DModel,
    VAStreamConditioning,
    VAStreamStates,
)
from wan_va.modules.mot_attention import build_x_metadata


def _stream(hidden, conditioning, kind):
    return AutoregressiveStreamInput(
        hidden=hidden,
        conditioning=conditioning,
        rotary=None,
        block_kind=kind,
    )


def _run_cached(block, cache, stream, text, *, commit):
    transaction_id = cache.new_transaction_id()
    output = block(
        AutoregressiveMOTLayerRequest(
            stream=stream,
            hidden_state=stream.hidden,
            text=text,
            cache=cache,
            transaction_id=transaction_id,
            layer_id=0,
        )
    )
    if commit:
        cache.commit(transaction_id)
    else:
        cache.discard(transaction_id)
    return output


@torch.no_grad()
def test_cached_target_visibility_matches_one_fixed_window_mot_layer() -> None:
    torch.manual_seed(0)
    hidden_dim = 8
    block = AutoregressiveVAMOTBlock(
        video_block=WanTransformerBlock(
            hidden_dim,
            16,
            2,
            True,
            1e-6,
            attn_mode="torch",
        ),
        action_block=ActionTransformerBlock(
            hidden_dim,
            16,
            hidden_dim,
            2,
            1e-6,
        ),
        masked_attn_backend="dense",
    ).eval()
    states = VAStreamStates(*(torch.randn(1, 8, hidden_dim) for _ in range(4)))
    conditioning = VAStreamConditioning(
        *(torch.randn(1, 8, 6, hidden_dim) for _ in range(4))
    )
    text = torch.randn(1, 3, hidden_dim)
    video_valid = torch.ones((1, 8), dtype=torch.bool)
    action_valid = torch.tensor([[False, True, True, True, False, True, True, True]])
    metadata = build_x_metadata(
        batch_size=1,
        video_tokens_per_frame=1,
        action_tokens_per_frame=1,
        num_frames=8,
        chunk_size=4,
        window_size=16,
        device=torch.device("cpu"),
        token_valid_ids=torch.cat(
            [video_valid, video_valid, action_valid, action_valid], dim=1
        ),
    )
    fixed = block(states, conditioning, text, None, None, metadata)

    cache = KVCache()
    _run_cached(
        block,
        cache,
        _stream(states.clean_video[:, :4], conditioning.clean_video[:, :4], "video"),
        text,
        commit=True,
    )
    _run_cached(
        block,
        cache,
        _stream(states.clean_action[:, 1:4], conditioning.clean_action[:, 1:4], "action"),
        text,
        commit=True,
    )
    cached_video_target = _run_cached(
        block,
        cache,
        _stream(states.noisy_video[:, 4:8], conditioning.noisy_video[:, 4:8], "video"),
        text,
        commit=False,
    )
    torch.testing.assert_close(cached_video_target, fixed.noisy_video[:, 4:8])

    _run_cached(
        block,
        cache,
        _stream(states.clean_video[:, 4:8], conditioning.clean_video[:, 4:8], "video"),
        text,
        commit=True,
    )
    cached_action_target = _run_cached(
        block,
        cache,
        _stream(states.noisy_action[:, 5:8], conditioning.noisy_action[:, 5:8], "action"),
        text,
        commit=False,
    )
    torch.testing.assert_close(cached_action_target, fixed.noisy_action[:, 5:8])


def test_cached_rotary_positions_match_fixed_window_positions() -> None:
    identity = lambda value: value
    cached = SimpleNamespace(rope=identity, action_rope=identity)
    frame_ids = torch.arange(8)[None]

    cached_video = AutoregressiveVAMOTTransformer3DModel._video_rotary(
        cached,
        frame_ids,
        views=3,
        h_tokens=2,
        w_tokens=2,
    )
    fixed_video = VAMOTTransformer3DModel._video_grid(
        1, 8, 3, 2, 2, torch.device("cpu")
    )[:, :, None]
    torch.testing.assert_close(cached_video, fixed_video)

    cached_action = AutoregressiveVAMOTTransformer3DModel._action_rotary(
        cached,
        frame_ids,
        tokens_per_frame=16,
    )
    fixed_action = VAMOTTransformer3DModel._action_positions(
        1, 8, 16, torch.device("cpu")
    )
    torch.testing.assert_close(cached_action, fixed_action)
