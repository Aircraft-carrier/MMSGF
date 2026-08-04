"""Visualize original and distillation training masks as split-color PNGs.

Run directly from the repository root to regenerate the checked artifact:

    PYTHONPATH=. python distillation/tests/test_visualize_x_metadata_mask.py

The x-axis is key (K), and the y-axis is query (Q).  Every visible cell is
split diagonally: its lower-left triangle uses the query token color, and its
upper-right triangle uses the key token color.  White cells are masked.

 X-only:
      NV 8 + CV 8 + NA 8 + CA 8

Joint MOT:
    NV 8 + CV 8 + G 8 + NA 8 + CA 8

Original ``wan_va`` uses fixed-size chunks.  The distillation profile keeps the
first history chunk together, then gives each anchor/target frame its own
geometry order group.
"""

from pathlib import Path
import sys
import types

import torch
from PIL import Image, ImageDraw, ImageFont

# Direct execution should not import ``wan_va.modules.__init__`` because that
# eagerly loads optional FlashAttention model extensions unrelated to metadata
# visualization.  Register only the package path, then import the target module.
if "wan_va.modules" not in sys.modules:
    modules_package = types.ModuleType("wan_va.modules")
    modules_package.__path__ = [
        str(Path(__file__).resolve().parents[2] / "wan_va" / "modules")
    ]
    sys.modules["wan_va.modules"] = modules_package

from wan_va.modules.mot_attention import (
    NOISE_CLEAN,
    NOISE_GEOMETRY,
    NOISE_NOISY,
    STREAM_ACTION,
    STREAM_GEOMETRY,
    STREAM_VIDEO,
    MOTMaskMetadata,
    build_dense_mot_mask,
    build_geometry_metadata,
    build_mot_metadata,
    build_x_metadata,
)
from wan_va.modules.vggto_geometry import build_chunk_causal_visibility_mask
from distillation.configs.autoregressive_training import (
    autoregressive_training_cfg,
)
from distillation.configs.consistency_distillation import (
    consistency_distillation_cfg,
)
from distillation.mask_profile import (
    _apply_segmented_order,
    _build_segmented_vggto_inter_frame_mask,
    _build_segmented_vggto_inter_frame_metadata,
)


DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_FRAMES = 8
TOKENS_PER_FRAME = 1
ACTUAL_VIEWS = 2
ACTUAL_GEOMETRY_GROUP_SIZE = 4
ACTUAL_VIDEO_TOKENS_PER_FRAME = ACTUAL_VIEWS * 7 * 7
ACTUAL_ACTION_TOKENS_PER_FRAME = 16
ACTUAL_GEOMETRY_REGISTER_TOKENS = 16
ACTUAL_VGGTO_TOKENS_PER_IMAGE = ACTUAL_GEOMETRY_REGISTER_TOKENS + 14 * 14

TOKEN_COLORS = {
    "NV": (37, 99, 235),
    "CV": (13, 148, 136),
    "G": (124, 58, 237),
    "NA": (217, 119, 6),
    "CA": (225, 29, 72),
}
MASKED_COLOR = (255, 255, 255)
GRID_COLOR = (209, 213, 219)
TEXT_COLOR = (31, 41, 55)

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ORIGINAL_X_ARTIFACT_PATH = ARTIFACT_DIR / "original_chunk_x_mask.png"
ORIGINAL_MOT_ARTIFACT_PATH = ARTIFACT_DIR / "original_chunk_mot_mask.png"
DISTILLATION_X_ARTIFACT_PATH = ARTIFACT_DIR / "distillation_segmented_x_mask.png"
DISTILLATION_MOT_ARTIFACT_PATH = ARTIFACT_DIR / "distillation_segmented_mot_mask.png"
ORIGINAL_VGGTO_ARTIFACT_PATH = ARTIFACT_DIR / "original_vggto_inter_frame_mask.png"
DISTILLATION_VGGTO_ARTIFACT_PATH = (
    ARTIFACT_DIR / "distillation_vggto_inter_frame_mask.png"
)
AR_X_ARTIFACT_PATH = ARTIFACT_DIR / "ar_training_x_mask.png"
AR_MOT_ARTIFACT_PATH = ARTIFACT_DIR / "ar_training_mot_mask.png"
CONSISTENCY_X_ARTIFACT_PATH = ARTIFACT_DIR / "consistency_training_x_mask.png"
CONSISTENCY_MOT_ARTIFACT_PATH = ARTIFACT_DIR / "consistency_training_mot_mask.png"
ACTUAL_X_MASK_PNG_PATH = ARTIFACT_DIR / "actual_distillation_x_mask_binary.png"
ACTUAL_MOT_MASK_PNG_PATH = ARTIFACT_DIR / "actual_distillation_mot_mask_binary.png"
ACTUAL_VGGTO_MASK_PNG_PATH = (
    ARTIFACT_DIR / "actual_distillation_vggto_inter_frame_mask_binary.png"
)
ACTUAL_MASK_PT_PATH = ARTIFACT_DIR / "actual_distillation_training_masks.pt"


def build_x_original_metadata(generation_shape) -> MOTMaskMetadata:
    """Build X-only metadata with native ``wan_va`` chunk order."""
    chunk_size = int(generation_shape["chunk_size"])
    return build_x_metadata(
        batch_size=DEFAULT_BATCH_SIZE,
        video_tokens_per_frame=TOKENS_PER_FRAME,
        action_tokens_per_frame=TOKENS_PER_FRAME,
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=chunk_size,
        window_size=int(generation_shape["window_size"]),
        device=torch.device("cpu"),
    )


def build_mot_original_metadata(generation_shape) -> MOTMaskMetadata:
    """Build joint V/G/A metadata with native ``wan_va`` chunk order."""
    chunk_size = int(generation_shape["chunk_size"])
    return build_mot_metadata(
        batch_size=DEFAULT_BATCH_SIZE,
        video_tokens_per_frame=TOKENS_PER_FRAME,
        geometry_tokens_per_frame=TOKENS_PER_FRAME,
        action_tokens_per_frame=TOKENS_PER_FRAME,
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=chunk_size,
        window_size=int(generation_shape["window_size"]),
        device=torch.device("cpu"),
    )


def build_x_training_metadata(generation_shape) -> MOTMaskMetadata:
    """Build X-only metadata exactly as the distillation profile does."""

    chunk_size = int(generation_shape["chunk_size"])
    metadata = build_x_original_metadata(generation_shape)
    return _apply_segmented_order(
        metadata,
        history_frames=int(generation_shape.get("history_frames", 4)),
        chunk_size=chunk_size,
    )


def build_mot_training_metadata(generation_shape) -> MOTMaskMetadata:
    """Build joint V/G/A metadata exactly as the distillation profile does."""

    chunk_size = int(generation_shape["chunk_size"])
    metadata = build_mot_original_metadata(generation_shape)
    return _apply_segmented_order(
        metadata,
        history_frames=int(generation_shape.get("history_frames", 4)),
        chunk_size=chunk_size,
    )


def build_dense_training_mask(
    metadata: MOTMaskMetadata,
) -> torch.Tensor:
    """Build the dense reference mask used by native ``wan_va`` training."""

    return build_dense_mot_mask(metadata)


def build_metadata_and_masks(
    generation_shape,
    *,
    segmented: bool,
) -> tuple[
    MOTMaskMetadata,
    torch.Tensor,
    MOTMaskMetadata,
    torch.Tensor,
]:
    """Return X-only and joint metadata/masks for one order profile."""

    if segmented:
        x_metadata = build_x_training_metadata(generation_shape)
        mot_metadata = build_mot_training_metadata(generation_shape)
    else:
        x_metadata = build_x_original_metadata(generation_shape)
        mot_metadata = build_mot_original_metadata(generation_shape)
    return (
        x_metadata,
        build_dense_training_mask(x_metadata)[0],
        mot_metadata,
        build_dense_training_mask(mot_metadata)[0],
    )


def build_training_metadata_and_masks(
    generation_shape,
) -> tuple[
    MOTMaskMetadata,
    torch.Tensor,
    MOTMaskMetadata,
    torch.Tensor,
]:
    """Return X-only and joint metadata/masks for one training configuration."""

    return build_metadata_and_masks(generation_shape, segmented=True)


def build_actual_training_metadata_and_masks(
    generation_shape,
) -> tuple[
    MOTMaskMetadata,
    torch.Tensor,
    MOTMaskMetadata,
    torch.Tensor,
    MOTMaskMetadata,
    torch.Tensor,
]:
    """Return full token-level masks for the current 8-frame V=2 training pack."""

    chunk_size = int(generation_shape["chunk_size"])
    history_frames = int(generation_shape.get("history_frames", 4))
    device = torch.device("cpu")
    x_metadata = build_x_metadata(
        batch_size=DEFAULT_BATCH_SIZE,
        video_tokens_per_frame=ACTUAL_VIDEO_TOKENS_PER_FRAME,
        action_tokens_per_frame=ACTUAL_ACTION_TOKENS_PER_FRAME,
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=chunk_size,
        window_size=int(generation_shape["window_size"]),
        device=device,
    )
    mot_metadata = build_mot_metadata(
        batch_size=DEFAULT_BATCH_SIZE,
        video_tokens_per_frame=ACTUAL_VIDEO_TOKENS_PER_FRAME,
        geometry_tokens_per_frame=(
            ACTUAL_GEOMETRY_GROUP_SIZE
            * ACTUAL_VIEWS
            * ACTUAL_GEOMETRY_REGISTER_TOKENS
        ),
        action_tokens_per_frame=ACTUAL_ACTION_TOKENS_PER_FRAME,
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=chunk_size,
        window_size=int(generation_shape["window_size"]),
        device=device,
    )
    vggto_metadata = build_geometry_metadata(
        batch_size=DEFAULT_BATCH_SIZE,
        geometry_tokens_per_frame=(
            ACTUAL_GEOMETRY_GROUP_SIZE * ACTUAL_VGGTO_TOKENS_PER_IMAGE
        ),
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=chunk_size,
        window_size=int(generation_shape["window_size"]),
        device=device,
    )
    x_metadata = _apply_segmented_order(
        x_metadata,
        history_frames=history_frames,
        chunk_size=chunk_size,
    )
    mot_metadata = _apply_segmented_order(
        mot_metadata,
        history_frames=history_frames,
        chunk_size=chunk_size,
    )
    vggto_metadata = _apply_segmented_order(
        vggto_metadata,
        history_frames=history_frames,
        chunk_size=chunk_size,
    )
    vggto_mask = _build_segmented_vggto_inter_frame_mask(
        groups=DEFAULT_NUM_FRAMES,
        group_size=ACTUAL_GEOMETRY_GROUP_SIZE,
        tokens_per_image=ACTUAL_VGGTO_TOKENS_PER_IMAGE,
        history_frames=history_frames,
        chunk_size=chunk_size,
        device=device,
        image_valid_mask=None,
    )
    return (
        x_metadata,
        build_dense_mot_mask(x_metadata)[0],
        mot_metadata,
        build_dense_mot_mask(mot_metadata)[0],
        vggto_metadata,
        vggto_mask,
    )


def build_vggto_inter_frame_metadata_and_masks(
    generation_shape,
    *,
    segmented: bool,
) -> tuple[MOTMaskMetadata, torch.Tensor]:
    """Return one-view VGGTO inter-frame metadata and same-view relation mask."""

    chunk_size = int(generation_shape["chunk_size"])
    metadata = build_geometry_metadata(
        batch_size=DEFAULT_BATCH_SIZE,
        geometry_tokens_per_frame=TOKENS_PER_FRAME,
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=chunk_size,
        window_size=int(generation_shape["window_size"]),
        device=torch.device("cpu"),
    )
    if segmented:
        metadata = _apply_segmented_order(
            metadata,
            history_frames=int(generation_shape.get("history_frames", 4)),
            chunk_size=chunk_size,
        )
        mask = _build_segmented_vggto_inter_frame_mask(
            groups=DEFAULT_NUM_FRAMES,
            group_size=1,
            tokens_per_image=TOKENS_PER_FRAME,
            history_frames=int(generation_shape.get("history_frames", 4)),
            chunk_size=chunk_size,
            device=torch.device("cpu"),
            image_valid_mask=None,
        )
    else:
        mask = build_chunk_causal_visibility_mask(
            torch.arange(DEFAULT_NUM_FRAMES, dtype=torch.long) // chunk_size,
            TOKENS_PER_FRAME,
        )
    return metadata, mask


def save_binary_mask(mask: torch.Tensor, output_path: Path) -> Path:
    """Save a complete token-level boolean mask as a compact 1-pixel PNG."""

    image = Image.fromarray(
        mask.detach().to(device="cpu", dtype=torch.uint8).mul(255).numpy(),
        mode="L",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def save_actual_training_masks(generation_shape) -> dict[str, Path]:
    """Save full token-level actual training masks and their metadata."""

    x_meta, x_mask, mot_meta, mot_mask, vggto_meta, vggto_mask = (
        build_actual_training_metadata_and_masks(generation_shape)
    )
    torch.save(
        {
            "description": (
                "Full token-level distillation training masks for "
                "G=8, S=4, V=2, history=4."
            ),
            "token_counts": {
                "x_seq_len": int(x_meta.seq_len),
                "mot_seq_len": int(mot_meta.seq_len),
                "vggto_inter_frame_seq_len": int(vggto_mask.shape[0]),
                "video_tokens_per_frame": ACTUAL_VIDEO_TOKENS_PER_FRAME,
                "action_tokens_per_frame": ACTUAL_ACTION_TOKENS_PER_FRAME,
                "mot_geometry_tokens_per_frame": (
                    ACTUAL_GEOMETRY_GROUP_SIZE
                    * ACTUAL_VIEWS
                    * ACTUAL_GEOMETRY_REGISTER_TOKENS
                ),
                "vggto_tokens_per_image": ACTUAL_VGGTO_TOKENS_PER_IMAGE,
            },
            "x_order_ids": x_meta.order_ids.cpu(),
            "mot_order_ids": mot_meta.order_ids.cpu(),
            "vggto_order_ids": vggto_meta.order_ids.cpu(),
            "x_mask": x_mask.cpu(),
            "mot_mask": mot_mask.cpu(),
            "vggto_inter_frame_mask": vggto_mask.cpu(),
        },
        ACTUAL_MASK_PT_PATH,
    )
    return {
        "x_binary": save_binary_mask(x_mask, ACTUAL_X_MASK_PNG_PATH),
        "mot_binary": save_binary_mask(mot_mask, ACTUAL_MOT_MASK_PNG_PATH),
        "vggto_binary": save_binary_mask(vggto_mask, ACTUAL_VGGTO_MASK_PNG_PATH),
        "pt": ACTUAL_MASK_PT_PATH,
    }


def render_metadata_mask(
    metadata: MOTMaskMetadata,
    mask: torch.Tensor,
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Render one sample's allowed-form mask and return the written PNG path."""

    if metadata.batch_size != 1:
        raise ValueError(
            f"visualization expects batch_size=1, got {metadata.batch_size}"
        )
    if tuple(mask.shape) != (metadata.seq_len, metadata.seq_len):
        raise ValueError(
            f"mask must be [{metadata.seq_len},{metadata.seq_len}], "
            f"got {tuple(mask.shape)}"
        )

    labels = _token_labels(metadata)
    kinds = [_token_kind(metadata, index) for index in range(metadata.seq_len)]

    cell_size = 22
    left_margin = 105
    top_margin = 135
    right_margin = 35
    bottom_margin = 115
    grid_size = metadata.seq_len * cell_size
    image = Image.new(
        "RGB",
        (
            left_margin + grid_size + right_margin,
            top_margin + grid_size + bottom_margin,
        ),
        MASKED_COLOR,
    )
    draw = ImageDraw.Draw(image)
    title_font = _load_font(21)
    label_font = _load_font(12)
    text_font = _load_font(14)

    draw.text(
        (left_margin + grid_size // 2, 20),
        title,
        fill=TEXT_COLOR,
        font=title_font,
        anchor="ma",
    )
    _draw_legend(draw, text_font, left_margin, 53, kinds)
    draw.text(
        (left_margin + grid_size // 2, top_margin - 25),
        "Key tokens (K)",
        fill=TEXT_COLOR,
        font=text_font,
        anchor="mm",
    )
    draw.text(
        (8, top_margin - 25),
        "Query tokens (Q)",
        fill=TEXT_COLOR,
        font=text_font,
    )

    visible = mask.detach().to(device="cpu", dtype=torch.bool)
    for query in range(metadata.seq_len):
        for key in range(metadata.seq_len):
            left = left_margin + key * cell_size
            top = top_margin + query * cell_size
            right = left + cell_size
            bottom = top + cell_size
            if bool(visible[query, key]):
                draw.polygon(
                    [(left, top), (left, bottom), (right, bottom)],
                    fill=TOKEN_COLORS[kinds[query]],
                )
                draw.polygon(
                    [(left, top), (right, top), (right, bottom)],
                    fill=TOKEN_COLORS[kinds[key]],
                )
            draw.rectangle(
                [(left, top), (right, bottom)],
                outline=GRID_COLOR,
                width=1,
            )

    for index, (label, kind) in enumerate(zip(labels, kinds)):
        row_y = top_margin + index * cell_size + cell_size // 2
        draw.text(
            (left_margin - 8, row_y),
            label,
            fill=TOKEN_COLORS[kind],
            font=label_font,
            anchor="rm",
        )
        column_x = left_margin + index * cell_size + cell_size // 2
        _draw_rotated_label(
            image,
            label,
            TOKEN_COLORS[kind],
            label_font,
            column_x,
            top_margin + grid_size + 7,
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG")
    return output_path


def _token_kind(metadata: MOTMaskMetadata, index: int) -> str:
    stream = int(metadata.stream_ids[0, index].item())
    noise = int(metadata.noise_ids[0, index].item())
    kinds = {
        (STREAM_VIDEO, NOISE_NOISY): "NV",
        (STREAM_VIDEO, NOISE_CLEAN): "CV",
        (STREAM_GEOMETRY, NOISE_GEOMETRY): "G",
        (STREAM_ACTION, NOISE_NOISY): "NA",
        (STREAM_ACTION, NOISE_CLEAN): "CA",
    }
    try:
        return kinds[(stream, noise)]
    except KeyError as exc:
        raise ValueError(
            f"unsupported MOT token stream/noise pair {(stream, noise)}"
        ) from exc


def _token_labels(metadata: MOTMaskMetadata) -> list[str]:
    if metadata.frame_ids is None:
        raise ValueError("metadata.frame_ids is required for visualization")
    return [
        f"{_token_kind(metadata, index)}:F{int(metadata.frame_ids[0, index]):02d}"
        for index in range(metadata.seq_len)
    ]


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    start_x: int,
    y: int,
    kinds: list[str],
) -> None:
    x = start_x
    for kind, color in TOKEN_COLORS.items():
        if kind not in kinds:
            continue
        draw.rectangle([(x, y), (x + 14, y + 14)], fill=color)
        draw.text((x + 20, y + 7), kind, fill=TEXT_COLOR, font=font, anchor="lm")
        x += 62
    draw.text(
        (start_x, y + 30),
        "visible: lower-left=Q, upper-right=K; white=masked",
        fill=TEXT_COLOR,
        font=font,
    )


def _draw_rotated_label(
    image: Image.Image,
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
    center_x: int,
    top: int,
) -> None:
    label = Image.new("RGBA", (80, 18), (255, 255, 255, 0))
    ImageDraw.Draw(label).text((0, 1), text, fill=color, font=font)
    label = label.rotate(90, expand=True)
    image.paste(label, (center_x - label.width // 2, top), label)


def test_visualize_ar_training_masks(tmp_path):
    metadata = build_training_metadata_and_masks(
        autoregressive_training_cfg.distill.generation_shape
    )
    _assert_training_mask_semantics(*metadata)

    x_output = render_metadata_mask(
        metadata[0],
        metadata[1],
        tmp_path / "ar_training_x_mask.png",
        title="AR training X-only attention mask",
    )
    mot_output = render_metadata_mask(
        metadata[2],
        metadata[3],
        tmp_path / "ar_training_mot_mask.png",
        title="AR training joint V/G/A attention mask",
    )
    _assert_png_colors(x_output, {"NV", "CV", "NA", "CA"})
    _assert_png_colors(mot_output, {"NV", "CV", "G", "NA", "CA"})


def test_visualize_original_and_distillation_masks(tmp_path):
    generation_shape = consistency_distillation_cfg.distill.generation_shape
    original = build_metadata_and_masks(generation_shape, segmented=False)
    distillation = build_metadata_and_masks(generation_shape, segmented=True)

    _assert_original_mask_semantics(*original)
    _assert_training_mask_semantics(*distillation)

    original_x_output = render_metadata_mask(
        original[0],
        original[1],
        tmp_path / "original_chunk_x_mask.png",
        title="Original wan_va chunk X-only attention mask",
    )
    original_mot_output = render_metadata_mask(
        original[2],
        original[3],
        tmp_path / "original_chunk_mot_mask.png",
        title="Original wan_va chunk joint V/G/A attention mask",
    )
    distillation_x_output = render_metadata_mask(
        distillation[0],
        distillation[1],
        tmp_path / "distillation_segmented_x_mask.png",
        title="Distillation segmented X-only attention mask",
    )
    distillation_mot_output = render_metadata_mask(
        distillation[2],
        distillation[3],
        tmp_path / "distillation_segmented_mot_mask.png",
        title="Distillation segmented joint V/G/A attention mask",
    )

    _assert_png_colors(original_x_output, {"NV", "CV", "NA", "CA"})
    _assert_png_colors(original_mot_output, {"NV", "CV", "G", "NA", "CA"})
    _assert_png_colors(distillation_x_output, {"NV", "CV", "NA", "CA"})
    _assert_png_colors(distillation_mot_output, {"NV", "CV", "G", "NA", "CA"})


def test_visualize_original_and_distillation_vggto_inter_frame_masks(tmp_path):
    generation_shape = consistency_distillation_cfg.distill.generation_shape
    original = build_vggto_inter_frame_metadata_and_masks(
        generation_shape,
        segmented=False,
    )
    distillation = build_vggto_inter_frame_metadata_and_masks(
        generation_shape,
        segmented=True,
    )

    _assert_original_geometry_visibility(original[1])
    _assert_distillation_geometry_visibility(distillation[1])

    original_output = render_metadata_mask(
        original[0],
        original[1],
        tmp_path / "original_vggto_inter_frame_mask.png",
        title="Original VGGTO same-view inter-frame mask",
    )
    distillation_output = render_metadata_mask(
        distillation[0],
        distillation[1],
        tmp_path / "distillation_vggto_inter_frame_mask.png",
        title="Distillation VGGTO same-view inter-frame mask",
    )

    _assert_png_colors(original_output, {"G"})
    _assert_png_colors(distillation_output, {"G"})


def test_segmented_vggto_fa4_metadata_matches_dense_reference():
    generation_shape = consistency_distillation_cfg.distill.generation_shape
    metadata = _build_segmented_vggto_inter_frame_metadata(
        batch_size=DEFAULT_BATCH_SIZE,
        groups=DEFAULT_NUM_FRAMES,
        group_size=1,
        tokens_per_image=TOKENS_PER_FRAME,
        history_frames=int(generation_shape.get("history_frames", 4)),
        chunk_size=int(generation_shape["chunk_size"]),
        window_size=int(generation_shape["window_size"]),
        device=torch.device("cpu"),
        image_valid_mask=None,
    )
    dense_from_metadata = build_dense_mot_mask(metadata)[0]
    dense_reference = _build_segmented_vggto_inter_frame_mask(
        groups=DEFAULT_NUM_FRAMES,
        group_size=1,
        tokens_per_image=TOKENS_PER_FRAME,
        history_frames=int(generation_shape.get("history_frames", 4)),
        chunk_size=int(generation_shape["chunk_size"]),
        device=torch.device("cpu"),
        image_valid_mask=None,
    )

    assert torch.equal(dense_from_metadata, dense_reference)
    _assert_distillation_geometry_visibility(dense_from_metadata)


def test_original_vggto_fa4_spec_matches_dense_reference():
    from wan_va.modules.fa4_attention import (
        ChunkCausalMaskSpec,
        chunk_causal_mask_from_spec,
    )

    generation_shape = consistency_distillation_cfg.distill.generation_shape
    spec = ChunkCausalMaskSpec(
        frames=DEFAULT_NUM_FRAMES,
        frames_per_chunk=int(generation_shape["chunk_size"]),
        tokens_per_frame=TOKENS_PER_FRAME,
    )
    dense_from_spec = chunk_causal_mask_from_spec(
        DEFAULT_BATCH_SIZE,
        spec,
        torch.device("cpu"),
    )[0, 0]
    dense_reference = build_dense_mot_mask(
        build_geometry_metadata(
            batch_size=DEFAULT_BATCH_SIZE,
            geometry_tokens_per_frame=TOKENS_PER_FRAME,
            num_frames=DEFAULT_NUM_FRAMES,
            chunk_size=int(generation_shape["chunk_size"]),
            window_size=int(generation_shape["window_size"]),
            device=torch.device("cpu"),
        )
    )[0]

    assert torch.equal(dense_from_spec, dense_reference)
    _assert_original_geometry_visibility(dense_from_spec)


def test_actual_training_masks_match_segmented_geometry_semantics():
    generation_shape = consistency_distillation_cfg.distill.generation_shape
    x_meta, x_mask, mot_meta, mot_mask, vggto_meta, vggto_mask = (
        build_actual_training_metadata_and_masks(generation_shape)
    )

    assert tuple(x_mask.shape) == (1824, 1824)
    assert tuple(mot_mask.shape) == (2848, 2848)
    assert tuple(vggto_mask.shape) == (6784, 6784)

    assert x_meta.order_ids[0, :ACTUAL_VIDEO_TOKENS_PER_FRAME].unique().tolist() == [0]
    assert mot_meta.order_ids[0, 1568:1696].unique().tolist() == [0]
    assert vggto_meta.order_ids[0, :848].unique().tolist() == [0]

    _assert_distillation_group_visibility(
        mot_mask[
            2 * DEFAULT_NUM_FRAMES * ACTUAL_VIDEO_TOKENS_PER_FRAME :
            2 * DEFAULT_NUM_FRAMES * ACTUAL_VIDEO_TOKENS_PER_FRAME
            + DEFAULT_NUM_FRAMES
            * ACTUAL_GEOMETRY_GROUP_SIZE
            * ACTUAL_VIEWS
            * ACTUAL_GEOMETRY_REGISTER_TOKENS,
            2 * DEFAULT_NUM_FRAMES * ACTUAL_VIDEO_TOKENS_PER_FRAME :
            2 * DEFAULT_NUM_FRAMES * ACTUAL_VIDEO_TOKENS_PER_FRAME
            + DEFAULT_NUM_FRAMES
            * ACTUAL_GEOMETRY_GROUP_SIZE
            * ACTUAL_VIEWS
            * ACTUAL_GEOMETRY_REGISTER_TOKENS,
        ],
        tokens_per_group=(
            ACTUAL_GEOMETRY_GROUP_SIZE
            * ACTUAL_VIEWS
            * ACTUAL_GEOMETRY_REGISTER_TOKENS
        ),
    )
    _assert_distillation_group_visibility(
        vggto_mask,
        tokens_per_group=ACTUAL_GEOMETRY_GROUP_SIZE
        * ACTUAL_VGGTO_TOKENS_PER_IMAGE,
    )


def test_visualize_consistency_training_masks(tmp_path):
    metadata = build_training_metadata_and_masks(
        consistency_distillation_cfg.distill.generation_shape
    )
    _assert_training_mask_semantics(*metadata)

    x_output = render_metadata_mask(
        metadata[0],
        metadata[1],
        tmp_path / "consistency_training_x_mask.png",
        title="Consistency training X-only attention mask",
    )
    mot_output = render_metadata_mask(
        metadata[2],
        metadata[3],
        tmp_path / "consistency_training_mot_mask.png",
        title="Consistency training joint V/G/A attention mask",
    )
    _assert_png_colors(x_output, {"NV", "CV", "NA", "CA"})
    _assert_png_colors(mot_output, {"NV", "CV", "G", "NA", "CA"})


def test_ar_and_consistency_training_masks_are_identical():
    ar = build_training_metadata_and_masks(
        autoregressive_training_cfg.distill.generation_shape
    )
    consistency = build_training_metadata_and_masks(
        consistency_distillation_cfg.distill.generation_shape
    )

    assert torch.equal(ar[0].order_ids, consistency[0].order_ids)
    assert torch.equal(ar[1], consistency[1])
    assert torch.equal(ar[2].order_ids, consistency[2].order_ids)
    assert torch.equal(ar[3], consistency[3])


def _assert_training_mask_semantics(
    x_metadata: MOTMaskMetadata,
    x_mask: torch.Tensor,
    mot_metadata: MOTMaskMetadata,
    mot_mask: torch.Tensor,
) -> None:
    expected_video_order = [0, 0, 0, 0, 2, 4, 6, 8]
    expected_action_order = [1, 1, 1, 1, 3, 5, 7, 9]

    assert tuple(x_mask.shape) == (32, 32)
    assert x_metadata.order_ids[0, :8].tolist() == expected_video_order
    assert x_metadata.order_ids[0, 16:24].tolist() == expected_action_order

    # X-only packing: NV[0:8], CV[8:16], NA[16:24], CA[24:32].
    assert bool(x_mask[0, 0])
    assert bool(x_mask[0, 3])
    assert not bool(x_mask[0, 8])
    assert bool(x_mask[4, 8])
    assert bool(x_mask[8, 11])
    assert bool(x_mask[16, 8])

    assert tuple(mot_mask.shape) == (40, 40)
    assert mot_metadata.order_ids[0, 16:24].tolist() == expected_video_order

    # Joint packing: NV[0:8], CV[8:16], G[16:24], NA[24:32], CA[32:40].
    _assert_distillation_geometry_visibility(mot_mask[16:24, 16:24])
    assert not bool(mot_mask[16:24, :16].any())
    assert not bool(mot_mask[16:24, 24:].any())

    assert not bool(mot_mask[0, 16])
    assert bool(mot_mask[4, 16])
    assert not bool(mot_mask[4, 20])
    assert bool(mot_mask[24, 16])


def _assert_original_mask_semantics(
    x_metadata: MOTMaskMetadata,
    x_mask: torch.Tensor,
    mot_metadata: MOTMaskMetadata,
    mot_mask: torch.Tensor,
) -> None:
    expected_video_order = [0, 0, 0, 0, 2, 2, 2, 2]
    expected_action_order = [1, 1, 1, 1, 3, 3, 3, 3]

    assert tuple(x_mask.shape) == (32, 32)
    assert x_metadata.order_ids[0, :8].tolist() == expected_video_order
    assert x_metadata.order_ids[0, 16:24].tolist() == expected_action_order

    assert tuple(mot_mask.shape) == (40, 40)
    assert mot_metadata.order_ids[0, 16:24].tolist() == expected_video_order

    # Original packing: G[16:24].  Frames 0..3 are chunk 0 and frames 4..7 are
    # chunk 1, so each 4-frame chunk is internally bidirectional.
    _assert_original_geometry_visibility(mot_mask[16:24, 16:24])


def _assert_original_geometry_visibility(geometry: torch.Tensor) -> None:
    assert bool(geometry[:4, :4].all())
    assert not bool(geometry[:4, 4:].any())
    assert bool(geometry[4:, :].all())


def _assert_distillation_geometry_visibility(geometry: torch.Tensor) -> None:
    # History G0..G3 is one order group, so visibility is bidirectional.
    assert bool(geometry[:4, :4].all())
    assert not bool(geometry[:4, 4:].any())
    # Anchor G4 and each later target are separate groups: past + self only.
    for frame in range(4, DEFAULT_NUM_FRAMES):
        assert bool(geometry[frame, : frame + 1].all())
        assert not bool(geometry[frame, frame + 1 :].any())


def _assert_distillation_group_visibility(
    mask: torch.Tensor,
    *,
    tokens_per_group: int,
) -> None:
    for query_group in range(DEFAULT_NUM_FRAMES):
        q = slice(
            query_group * tokens_per_group,
            (query_group + 1) * tokens_per_group,
        )
        for key_group in range(DEFAULT_NUM_FRAMES):
            k = slice(
                key_group * tokens_per_group,
                (key_group + 1) * tokens_per_group,
            )
            visible = bool(mask[q, k].all())
            if query_group < 4:
                expected = key_group < 4
            else:
                expected = key_group <= query_group
            assert visible is expected


def _assert_png_colors(output_path: Path, expected_kinds: set[str]) -> None:
    with Image.open(output_path) as image:
        assert image.format == "PNG"
        assert image.width > 40
        assert image.height > 40
        colors = image.convert("RGB").getcolors(
            maxcolors=image.width * image.height
        )
        assert colors is not None
        pixels = {color for _count, color in colors}
    assert {TOKEN_COLORS[kind] for kind in expected_kinds}.issubset(pixels)


# 直接在终端运行的生成指令：
# PYTHONPATH=. python distillation/tests/test_visualize_x_metadata_mask.py
if __name__ == "__main__":
    generation_shape = consistency_distillation_cfg.distill.generation_shape
    comparisons = (
        (
            "Original wan_va chunk",
            build_metadata_and_masks(generation_shape, segmented=False),
            ORIGINAL_X_ARTIFACT_PATH,
            ORIGINAL_MOT_ARTIFACT_PATH,
        ),
        (
            "Distillation segmented",
            build_metadata_and_masks(generation_shape, segmented=True),
            DISTILLATION_X_ARTIFACT_PATH,
            DISTILLATION_MOT_ARTIFACT_PATH,
        ),
    )
    for stage_name, metadata, x_path, mot_path in comparisons:
        x_meta, x_mask, mot_meta, mot_mask = metadata
        print(
            render_metadata_mask(
                x_meta,
                x_mask,
                x_path,
                title=f"{stage_name} X-only attention mask",
            )
        )
        print(
            render_metadata_mask(
                mot_meta,
                mot_mask,
                mot_path,
                title=f"{stage_name} joint V/G/A attention mask",
            )
        )

    vggto_masks = (
        (
            "Original VGGTO same-view inter-frame",
            build_vggto_inter_frame_metadata_and_masks(
                generation_shape,
                segmented=False,
            ),
            ORIGINAL_VGGTO_ARTIFACT_PATH,
        ),
        (
            "Distillation VGGTO same-view inter-frame",
            build_vggto_inter_frame_metadata_and_masks(
                generation_shape,
                segmented=True,
            ),
            DISTILLATION_VGGTO_ARTIFACT_PATH,
        ),
    )
    for stage_name, metadata, output_path in vggto_masks:
        meta, mask = metadata
        print(
            render_metadata_mask(
                meta,
                mask,
                output_path,
                title=f"{stage_name} mask",
            )
        )

    for name, path in save_actual_training_masks(generation_shape).items():
        print(f"{name}: {path}")

    stages = (
        (
            "AR training",
            autoregressive_training_cfg.distill.generation_shape,
            AR_X_ARTIFACT_PATH,
            AR_MOT_ARTIFACT_PATH,
        ),
        (
            "Consistency training",
            consistency_distillation_cfg.distill.generation_shape,
            CONSISTENCY_X_ARTIFACT_PATH,
            CONSISTENCY_MOT_ARTIFACT_PATH,
        ),
    )
    for stage_name, generation_shape, x_path, mot_path in stages:
        x_meta, x_mask, mot_meta, mot_mask = build_training_metadata_and_masks(
            generation_shape
        )
        print(
            render_metadata_mask(
                x_meta,
                x_mask,
                x_path,
                title=f"{stage_name} X-only attention mask",
            )
        )
        print(
            render_metadata_mask(
                mot_meta,
                mot_mask,
                mot_path,
                title=f"{stage_name} joint V/G/A attention mask",
            )
        )
