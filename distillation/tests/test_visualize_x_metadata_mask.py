"""Visualize AR and consistency training masks as split-color PNGs.

Run directly from the repository root to regenerate the checked artifact:

    PYTHONPATH=. python distillation/tests/test_visualize_x_metadata_mask.py

The x-axis is key (K), and the y-axis is query (Q).  Every visible cell is
split diagonally: its lower-left triangle uses the query token color, and its
upper-right triangle uses the key token color.  White cells are masked.

 X-only:
      NV 8 + CV 8 + NA 8 + CA 8

Joint MOT:
    NV 8 + CV 8 + G 8 + NA 8 + CA 8

Both stages currently install ``segmented_history_strict_geometry_v1`` and
therefore use identical native ``wan_va`` training masks.  History geometry
shares one mutually visible order group; anchor and later geometry groups can
read all earlier groups plus their own group.
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
    build_mot_metadata,
    build_x_metadata,
)
from distillation.configs.autoregressive_training import (
    autoregressive_training_cfg,
)
from distillation.configs.consistency_distillation import (
    consistency_distillation_cfg,
)
from distillation.mask_profile import _apply_segmented_order


DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_FRAMES = 8
TOKENS_PER_FRAME = 1

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
AR_X_ARTIFACT_PATH = ARTIFACT_DIR / "ar_training_x_mask.png"
AR_MOT_ARTIFACT_PATH = ARTIFACT_DIR / "ar_training_mot_mask.png"
CONSISTENCY_X_ARTIFACT_PATH = ARTIFACT_DIR / "consistency_training_x_mask.png"
CONSISTENCY_MOT_ARTIFACT_PATH = ARTIFACT_DIR / "consistency_training_mot_mask.png"


def build_x_training_metadata(generation_shape) -> MOTMaskMetadata:
    """Build X-only metadata exactly as the configured training profile does."""

    chunk_size = int(generation_shape["chunk_size"])
    metadata = build_x_metadata(
        batch_size=DEFAULT_BATCH_SIZE,
        video_tokens_per_frame=TOKENS_PER_FRAME,
        action_tokens_per_frame=TOKENS_PER_FRAME,
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=chunk_size,
        window_size=int(generation_shape["window_size"]),
        device=torch.device("cpu"),
    )
    return _apply_segmented_order(
        metadata,
        history_frames=int(generation_shape.get("history_frames", 4)),
        chunk_size=chunk_size,
    )


def build_mot_training_metadata(generation_shape) -> MOTMaskMetadata:
    """Build joint V/G/A metadata as the configured training profile does."""

    chunk_size = int(generation_shape["chunk_size"])
    metadata = build_mot_metadata(
        batch_size=DEFAULT_BATCH_SIZE,
        video_tokens_per_frame=TOKENS_PER_FRAME,
        geometry_tokens_per_frame=TOKENS_PER_FRAME,
        action_tokens_per_frame=TOKENS_PER_FRAME,
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=chunk_size,
        window_size=int(generation_shape["window_size"]),
        device=torch.device("cpu"),
    )
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


def build_training_metadata_and_masks(
    generation_shape,
) -> tuple[
    MOTMaskMetadata,
    torch.Tensor,
    MOTMaskMetadata,
    torch.Tensor,
]:
    """Return X-only and joint metadata/masks for one training configuration."""

    x_metadata = build_x_training_metadata(generation_shape)
    mot_metadata = build_mot_training_metadata(generation_shape)
    return (
        x_metadata,
        build_dense_training_mask(x_metadata)[0],
        mot_metadata,
        build_dense_training_mask(mot_metadata)[0],
    )


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
    geometry = mot_mask[16:24, 16:24]
    # History G0..G3 is one order group, so visibility is bidirectional.
    assert bool(geometry[:4, :4].all())
    assert not bool(geometry[:4, 4:].any())
    # Anchor G4 and each later target are separate groups: past + self only.
    for frame in range(4, DEFAULT_NUM_FRAMES):
        assert bool(geometry[frame, : frame + 1].all())
        assert not bool(geometry[frame, frame + 1 :].any())
    assert not bool(mot_mask[16:24, :16].any())
    assert not bool(mot_mask[16:24, 24:].any())

    assert not bool(mot_mask[0, 16])
    assert bool(mot_mask[4, 16])
    assert not bool(mot_mask[4, 20])
    assert bool(mot_mask[24, 16])


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
