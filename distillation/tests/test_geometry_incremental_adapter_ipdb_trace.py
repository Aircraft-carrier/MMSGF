"""CPU-only phase trace for the real ``GeometryIncrementalAdapter``.

No checkpoint, dataset, GPU, or production model is required.  Geometry RGB
inputs use the production rollout packing ``[B,F,S,V,3,H,W]`` while token width,
register count, and layer depth are reduced for CPU execution.  This file also
renders one grouped rectangular attention-mask PNG for every geometry phase.

Run as a test::

    PYTHONPATH=. pytest -q -s \
        distillation/tests/test_geometry_incremental_adapter_ipdb_trace.py

Run the printable trace directly::

    PYTHONPATH=. python \
        distillation/tests/test_geometry_incremental_adapter_ipdb_trace.py

Enter ipdb immediately before the first adapter phase::

    GEOMETRY_ADAPTER_IPDB=1 PYTHONPATH=. python \
        distillation/tests/test_geometry_incremental_adapter_ipdb_trace.py
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

import distillation.self_rollout.geometry_cache as geometry_cache_module
from distillation.self_rollout.attention import (
    NOISE_GEOMETRY,
    STREAM_GEOMETRY,
    TokenMetadataBatch,
)
from distillation.self_rollout.geometry_cache import (
    GeometryIncrementalAdapter,
    GeometryRolloutCache,
)
from distillation.self_rollout.state import CacheSource, RolloutState


# Production rollout geometry packing is [B,F,S,V,3,H,W].  History is frames
# 0..3, frame 4 is the known anchor, and frames 5..7 are predicted geometry.
BATCH_SIZE = 1
TOTAL_FRAMES = 8
HISTORY_FRAMES = 4
ANCHOR_FRAME = HISTORY_FRAMES
ROLLOUT_FRAME_IDS = tuple(range(ANCHOR_FRAME + 1, TOTAL_FRAMES))
GROUP_SLOTS = 4
VIEWS = 2
RGB_CHANNELS = 3
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

# Tiny model dimensions.  The fake VGGTO still exposes a register layer and a
# relation layer, which is the minimum needed to exercise both physical caches.
HIDDEN_DIM = 2
REGISTER_TOKENS = 1
PATCH_TOKENS = 1
TOKENS_PER_IMAGE = REGISTER_TOKENS + PATCH_TOKENS
DEPTH = 2
JOINT_LAYER = 0
RELATION_LAYER = 1
WINDOW_SIZE = 16
DEFAULT_MASK_DIR = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "geometry_incremental_adapter_phases"
)


class _TinyAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.num_heads = 1
        self.use_qk_norm = False
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Identity()
        self.proj_drop = nn.Identity()
        with torch.no_grad():
            self.qkv.weight.zero_()
            eye = torch.eye(dim)
            self.qkv.weight[:dim].copy_(eye)
            self.qkv.weight[dim : 2 * dim].copy_(eye)
            self.qkv.weight[2 * dim :].copy_(eye)


class _TinyRelationBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = _TinyAttention(dim)
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.ls1 = nn.Identity()
        self.ls2 = nn.Identity()
        self.mlp = nn.Identity()


class _TinyGeometryStream(nn.Module):
    def qkv_project(self, registers: torch.Tensor, rotary: torch.Tensor):
        del rotary
        values = registers[:, :, None, :]
        return values, values, values

    @staticmethod
    def attn_delta(attended: torch.Tensor) -> torch.Tensor:
        return attended.flatten(2, 3)

    @staticmethod
    def ffn_delta(registers: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(registers)


class _TinyMOTBlock(nn.Module):
    def __init__(self, geometry: nn.Module | None) -> None:
        super().__init__()
        self.geometry = geometry


class _ZeroRope(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            positions.shape[0],
            positions.shape[-1],
            self.dim,
            device=positions.device,
        )


class _TinyVGGTO(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.depth = DEPTH
        self.inter_frame_blocks = nn.ModuleList(
            [_TinyRelationBlock(HIDDEN_DIM) for _ in range(DEPTH)]
        )
        self.register_attention_indices = {JOINT_LAYER}
        self.cached_layer_set = {RELATION_LAYER}
        self.patch_start_idx = REGISTER_TOKENS

    @staticmethod
    def encode_grouped(
        rgb: torch.Tensor,
        *,
        slot_valid_mask: torch.Tensor | None,
    ):
        del slot_valid_mask
        batch, groups, slots, views, channels, height, width = rgb.shape
        expected = (
            BATCH_SIZE,
            slots,
            views,
            RGB_CHANNELS,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )
        if (batch, slots, views, channels, height, width) != expected:
            raise ValueError(
                "geometry must use production [1,F,4,2,3,224,224] packing, "
                f"got {tuple(rgb.shape)}"
            )
        image_mean = rgb.mean(dim=(-3, -2, -1)).reshape(batch, groups * slots * views, 1)
        register = torch.cat([image_mean, image_mean + 0.25], dim=-1)
        patch = torch.cat([image_mean + 0.5, image_mean + 0.75], dim=-1)
        tokens = torch.stack([register, patch], dim=2)
        return SimpleNamespace(
            tokens=tokens,
            patch_hw=(1, 1),
            image_hw=(height, width),
            patch_token_start=REGISTER_TOKENS,
        )

    @staticmethod
    def run_frame_block(
        tokens: torch.Tensor,
        patch_hw: tuple[int, int],
        layer_id: int,
    ) -> torch.Tensor:
        del patch_hw
        return tokens + float(layer_id) * 0.01

    @staticmethod
    def _run_cross_view_block(
        frame_tokens: torch.Tensor,
        patch_hw: tuple[int, int],
        layer_id: int,
        *,
        groups: int,
        group_size: int,
        views: int,
        slot_valid_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        del patch_hw, layer_id, slot_valid_mask
        batch, frames, tokens, channels = frame_tokens.shape
        if frames != groups * group_size * views:
            raise ValueError("invalid grouped geometry layout")
        return frame_tokens.reshape(batch, groups, group_size, views, tokens, channels)

    @staticmethod
    def _full_token_rope(
        patch_hw: tuple[int, int],
        images_per_sequence: int,
        device: torch.device,
    ):
        del patch_hw, images_per_sequence, device
        return None

    @staticmethod
    def dense_forward(cached_outputs, source_images):
        batch, images = source_images.shape[:2]
        depth = torch.zeros(batch, images, 1, 1, device=source_images.device)
        conf = torch.ones_like(depth)
        return depth, conf

    @staticmethod
    def point_forward(cached_outputs, source_images):
        batch, images = source_images.shape[:2]
        points = torch.zeros(batch, images, 3, 1, 1, device=source_images.device)
        conf = torch.ones(batch, images, 1, 1, device=source_images.device)
        return points, conf


class _TinyMOTModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.vggto = _TinyVGGTO()
        self.rope = _ZeroRope(HIDDEN_DIM)
        self.mot_blocks = nn.ModuleList(
            [
                _TinyMOTBlock(_TinyGeometryStream()),
                _TinyMOTBlock(None),
            ]
        )


@dataclass(frozen=True)
class _AttentionEvent:
    phase: str
    kind: str
    query: TokenMetadataBatch
    key: TokenMetadataBatch
    mask: torch.Tensor


class _VisibilityRecorder:
    """Wrap the real geometry visibility builder and retain every layer call."""

    def __init__(self) -> None:
        self.phase = "unassigned"
        self.events: list[_AttentionEvent] = []
        self._real_build = geometry_cache_module.build_cache_visibility

    def __call__(
        self,
        query: TokenMetadataBatch,
        key: TokenMetadataBatch,
        *,
        window_size: int,
    ) -> torch.Tensor:
        mask = self._real_build(query, key, window_size=window_size)
        kind = "joint_register" if query.batch_size == BATCH_SIZE else "relation"
        self.events.append(
            _AttentionEvent(
                phase=self.phase,
                kind=kind,
                query=query,
                key=key,
                mask=mask.detach().clone(),
            )
        )
        return mask


@dataclass(frozen=True)
class _TraceResult:
    state: RolloutState
    events: list[_AttentionEvent]


def _geometry(values: list[float]) -> torch.Tensor:
    """Return production-shaped synthetic [1,F,4,2,3,224,224] geometry RGB."""

    frames = torch.tensor(values, dtype=torch.float32).reshape(1, -1, 1, 1, 1, 1, 1)
    return frames.expand(
        BATCH_SIZE,
        len(values),
        GROUP_SLOTS,
        VIEWS,
        RGB_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    ).clone()


def _slot_valid(frames: int) -> torch.Tensor:
    return torch.ones(BATCH_SIZE, frames, GROUP_SLOTS, dtype=torch.bool)


def _run_trace() -> _TraceResult:
    model = _TinyMOTModel()
    adapter = GeometryIncrementalAdapter(
        model,
        history_frames=HISTORY_FRAMES,
        chunk_size=HISTORY_FRAMES,
        window_size=WINDOW_SIZE,
    )
    state = RolloutState(geometry_cache=GeometryRolloutCache())
    recorder = _VisibilityRecorder()

    if os.getenv("GEOMETRY_ADAPTER_IPDB"):
        import ipdb

        ipdb.set_trace()

    with patch.object(
        geometry_cache_module,
        "build_cache_visibility",
        new=recorder,
    ):
        recorder.phase = "history.encode_geometry"
        history = adapter.encode_history_and_commit(
            _geometry([0.0, 1.0, 2.0, 3.0]),
            frame_ids=list(range(HISTORY_FRAMES)),
            slot_valid_mask=_slot_valid(HISTORY_FRAMES),
            state=state,
            source=CacheSource.HISTORY,
            version_id=1,
        )
        assert len(history) == HISTORY_FRAMES

        recorder.phase = "anchor.encode_geometry"
        adapter.encode_and_commit(
            _geometry([4.0]),
            frame_id=ANCHOR_FRAME,
            slot_valid_mask=_slot_valid(1),
            state=state,
            source=CacheSource.ANCHOR,
            version_id=1,
            return_points=True,
        )

        for frame_id in ROLLOUT_FRAME_IDS:
            recorder.phase = f"rollout.frame{frame_id}.encode_geometry"
            adapter.encode_and_commit(
                _geometry([float(frame_id)]),
                frame_id=frame_id,
                slot_valid_mask=_slot_valid(1),
                state=state,
                source=CacheSource.PREDICTED,
                version_id=1,
            )

    return _TraceResult(state=state, events=recorder.events)


def _events_for(result: _TraceResult, phase: str) -> list[_AttentionEvent]:
    matches = [event for event in result.events if event.phase == phase]
    if len(matches) != 2:
        raise AssertionError(f"expected two events for {phase!r}, got {len(matches)}")
    return matches


def _event_for(result: _TraceResult, phase: str, kind: str) -> _AttentionEvent:
    matches = [event for event in result.events if event.phase == phase and event.kind == kind]
    if len(matches) != 1:
        raise AssertionError(f"expected one {kind} event for {phase!r}, got {len(matches)}")
    return matches[0]


def _visible_key_frames(event: _AttentionEvent) -> set[int]:
    visible = event.mask.any(dim=1)[0]
    return set(int(frame) for frame in event.key.frame_ids[0, visible].unique().tolist())


def _assert_event_geometry_only(event: _AttentionEvent) -> None:
    assert set(event.query.stream_ids.flatten().tolist()) == {STREAM_GEOMETRY}
    assert set(event.key.stream_ids.flatten().tolist()) == {STREAM_GEOMETRY}
    assert set(event.query.noise_ids.flatten().tolist()) == {NOISE_GEOMETRY}
    assert set(event.key.noise_ids.flatten().tolist()) == {NOISE_GEOMETRY}
    assert event.mask.any(), event.phase


def _assert_trace(result: _TraceResult) -> None:
    expected_phases = [
        "history.encode_geometry",
        "anchor.encode_geometry",
        *[
            f"rollout.frame{frame_id}.encode_geometry"
            for frame_id in ROLLOUT_FRAME_IDS
        ],
    ]
    assert [event.phase for event in result.events] == [
        phase
        for phase in expected_phases
        for _ in range(2)
    ]
    assert [event.kind for event in result.events] == [
        kind
        for _phase in expected_phases
        for kind in ("joint_register", "relation")
    ]

    for event in result.events:
        _assert_event_geometry_only(event)

    history_frames = set(range(HISTORY_FRAMES))
    for kind in ("joint_register", "relation"):
        event = _event_for(result, "history.encode_geometry", kind)
        assert _visible_key_frames(event) == history_frames
        assert event.mask.all()

    for frame_id in (ANCHOR_FRAME, *ROLLOUT_FRAME_IDS):
        phase = (
            "anchor.encode_geometry"
            if frame_id == ANCHOR_FRAME
            else f"rollout.frame{frame_id}.encode_geometry"
        )
        expected_visible = set(range(frame_id + 1))
        for kind in ("joint_register", "relation"):
            event = _event_for(result, phase, kind)
            assert _visible_key_frames(event) == expected_visible
            current = (
                (event.key.frame_ids == frame_id)
                & ~event.key.committed_ids
            )
            assert current.any()
            assert event.mask[:, :, current[0]].all()

    geometry_cache = result.state.geometry_cache
    assert isinstance(geometry_cache, GeometryRolloutCache)
    assert set(geometry_cache.frames) == set(range(TOTAL_FRAMES))
    result.state.mot_cache.assert_no_transactions()
    geometry_cache.relation_cache.assert_no_transactions()

    for frame_id, encoded in geometry_cache.frames.items():
        assert encoded.frame_id == frame_id
        assert tuple(encoded.rgb.shape) == (
            BATCH_SIZE,
            1,
            GROUP_SLOTS,
            VIEWS,
            RGB_CHANNELS,
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
        )
        assert tuple(encoded.final_tokens.shape) == (
            BATCH_SIZE,
            GROUP_SLOTS * VIEWS,
            TOKENS_PER_IMAGE,
            HIDDEN_DIM,
        )
        assert set(encoded.layer_registers) == {JOINT_LAYER}
        assert encoded.cached_outputs[RELATION_LAYER] is not None


def test_geometry_incremental_adapter_across_rollout_phases(tmp_path: Path) -> None:
    result = _run_trace()
    _assert_trace(result)
    mask_paths = _render_phase_masks(result.events, tmp_path)
    assert len(mask_paths) == len(result.events)
    for mask_path in mask_paths:
        with Image.open(mask_path) as image:
            assert image.format == "PNG"
            assert image.width > 100
            assert image.height > 100


def _metadata_counts(metadata: TokenMetadataBatch) -> Counter[str]:
    counts: Counter[str] = Counter()
    for frame, committed, transaction in zip(
        metadata.frame_ids[0].tolist(),
        metadata.committed_ids[0].tolist(),
        metadata.transaction_ids[0].tolist(),
    ):
        lifetime = "committed" if committed else f"tx={transaction}"
        counts[f"G/frame={int(frame)}/{lifetime}"] += 1
    return counts


def _short_token_label(metadata: TokenMetadataBatch, index: int) -> str:
    frame = int(metadata.frame_ids[0, index].item())
    if bool(metadata.committed_ids[0, index].item()):
        lifetime = "C"
    else:
        lifetime = f"T{int(metadata.transaction_ids[0, index].item())}"
    return f"G-F{frame}-{lifetime}"


def _metadata_groups(
    metadata: TokenMetadataBatch,
) -> list[tuple[str, list[int]]]:
    grouped: dict[str, list[int]] = {}
    for index in range(metadata.seq_len):
        label = _short_token_label(metadata, index)
        grouped.setdefault(label, []).append(index)
    return list(grouped.items())


def _cell_color(visible_fraction: float) -> tuple[int, int, int]:
    if visible_fraction <= 0.0:
        return (255, 255, 255)
    if visible_fraction >= 1.0:
        return (37, 99, 235)
    return (245, 158, 11)


def _render_phase_mask(event: _AttentionEvent, output_path: Path) -> Path:
    """Render a Q-group by K-group visibility matrix for one geometry event."""

    query_groups = _metadata_groups(event.query)
    key_groups = _metadata_groups(event.key)
    cell_width = 52
    cell_height = 38
    left_margin = 110
    top_margin = 150
    right_margin = 30
    bottom_margin = 60
    image = Image.new(
        "RGB",
        (
            left_margin + len(key_groups) * cell_width + right_margin,
            top_margin + len(query_groups) * cell_height + bottom_margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((15, 15), f"{event.phase} / {event.kind}", fill=(17, 24, 39), font=font)
    draw.text(
        (15, 35),
        f"raw mask [B,Q,K] = {tuple(event.mask.shape)}",
        fill=(55, 65, 81),
        font=font,
    )
    draw.text(
        (15, 52),
        "label = G-frame-lifetime; C=committed, Tn=current transaction",
        fill=(55, 65, 81),
        font=font,
    )
    draw.text(
        (15, 69),
        "blue=fully visible, white=masked, orange=partially visible",
        fill=(55, 65, 81),
        font=font,
    )
    draw.text((15, top_margin - 18), "Query (Q)", fill=(17, 24, 39), font=font)
    draw.text((left_margin, 90), "Key (K)", fill=(17, 24, 39), font=font)

    for column, (label, _indices) in enumerate(key_groups):
        label_image = Image.new("RGBA", (100, 16), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label_image)
        label_draw.text((0, 1), label, fill=(17, 24, 39), font=font)
        rotated = label_image.rotate(90, expand=True)
        x = left_margin + column * cell_width + (cell_width - rotated.width) // 2
        image.paste(rotated, (x, top_margin - rotated.height - 4), rotated)

    for row, (query_label, query_indices) in enumerate(query_groups):
        y = top_margin + row * cell_height
        draw.text((8, y + 12), query_label, fill=(17, 24, 39), font=font)
        query_selector = torch.tensor(query_indices, dtype=torch.long)
        for column, (_key_label, key_indices) in enumerate(key_groups):
            key_selector = torch.tensor(key_indices, dtype=torch.long)
            selected = event.mask[0].index_select(0, query_selector)
            selected = selected.index_select(1, key_selector)
            fraction = float(selected.float().mean().item())
            x = left_margin + column * cell_width
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                fill=_cell_color(fraction),
                outline=(156, 163, 175),
            )
            text_color = (255, 255, 255) if fraction >= 1.0 else (17, 24, 39)
            draw.text(
                (x + 18, y + 12),
                f"{fraction:.1f}",
                fill=text_color,
                font=font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def _render_phase_masks(
    events: list[_AttentionEvent],
    output_dir: Path,
) -> list[Path]:
    paths = []
    for index, event in enumerate(events, start=1):
        filename = f"{event.phase}.{event.kind}".replace(".", "_").replace("/", "_")
        paths.append(
            _render_phase_mask(
                event,
                output_dir / f"{index:02d}_{filename}.png",
            )
        )
    return paths


def main() -> None:
    result = _run_trace()
    _assert_trace(result)
    mask_dir = Path(os.getenv("GEOMETRY_ADAPTER_MASK_DIR", str(DEFAULT_MASK_DIR)))
    mask_paths = _render_phase_masks(result.events, mask_dir)

    print("GeometryIncrementalAdapter CPU rollout trace: PASS")
    print(
        "production geometry shape: "
        f"[{BATCH_SIZE},{TOTAL_FRAMES},{GROUP_SLOTS},{VIEWS},"
        f"{RGB_CHANNELS},{IMAGE_HEIGHT},{IMAGE_WIDTH}]"
    )
    print(
        "tiny VGGTO shape: "
        f"depth={DEPTH}, hidden={HIDDEN_DIM}, "
        f"register_tokens={REGISTER_TOKENS}, patch_tokens={PATCH_TOKENS}"
    )
    print("Each event below is one real geometry visibility-mask call.")
    for event in result.events:
        print(f"\n[{event.phase} / {event.kind}]")
        print(f"  query tokens: {_metadata_counts(event.query)}")
        print(f"  key tokens:   {_metadata_counts(event.key)}")
        print(
            "  mask shape:   "
            f"{tuple(event.mask.shape)}, "
            f"visible={int(event.mask.sum().item())}/{event.mask.numel()}"
        )

    geometry_cache = result.state.geometry_cache
    assert isinstance(geometry_cache, GeometryRolloutCache)
    print("\nCommitted geometry frames:")
    for frame_id in sorted(geometry_cache.frames):
        encoded = geometry_cache.frames[frame_id]
        print(
            f"  G{frame_id}: final_tokens={tuple(encoded.final_tokens.shape)}, "
            f"cached_relation_layer={encoded.cached_outputs[RELATION_LAYER] is not None}"
        )
    print(f"\nSaved {len(mask_paths)} phase masks to: {mask_dir}")


if __name__ == "__main__":
    main()
