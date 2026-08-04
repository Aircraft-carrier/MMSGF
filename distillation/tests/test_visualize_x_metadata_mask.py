"""Visualize default X-only and joint MOT masks as split-color PNGs.

Run directly from the repository root to regenerate the checked artifact:

    PYTHONPATH=. python distillation/tests/test_visualize_x_metadata_mask.py

The x-axis is key (K), and the y-axis is query (Q).  Every visible cell is
split diagonally: its lower-left triangle uses the query token color, and its
upper-right triangle uses the key token color.  White cells are masked.

 X-only:
      NV 8 + CV 8 + NA 8 + CA 8

Joint:
    NV 8 + CV 8 + G 8 + NA 8 + CA 8

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
from distillation.self_rollout.attention import (
    build_cache_visibility,
    from_mot_metadata,
    segmented_orders,
)


DEFAULT_BATCH_SIZE = 1
DEFAULT_NUM_FRAMES = 8
DEFAULT_HISTORY_FRAMES = 4
DEFAULT_CHUNK_SIZE = 4
DEFAULT_WINDOW_SIZE = 10
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

X_ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "build_x_metadata_dense_mask.png"
)
MOT_ARTIFACT_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "build_mot_metadata_dense_mask.png"
)

def build_x_metadata4sgf() -> MOTMaskMetadata:
    """Build the X-only metadata with the two-stage SGF order."""
    metadata = build_x_metadata(
        batch_size=DEFAULT_BATCH_SIZE,
        video_tokens_per_frame=TOKENS_PER_FRAME,
        action_tokens_per_frame=TOKENS_PER_FRAME,
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=DEFAULT_CHUNK_SIZE,
        window_size=DEFAULT_WINDOW_SIZE,
        device=torch.device("cpu"),
    )
    video_order, action_order = _build_sgf_orders(
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=DEFAULT_CHUNK_SIZE,
        device=metadata.device,
    )
    metadata.order_ids = torch.cat(
        [video_order, video_order, action_order, action_order], dim=1
    )
    metadata.cache_key = None
    metadata.structure_cache_key = None
    return metadata


def build_mot_metadata4sgf() -> MOTMaskMetadata:
    """Build the joint V/G/A metadata with the two-stage SGF order."""
    metadata = build_mot_metadata(
        batch_size=DEFAULT_BATCH_SIZE,
        video_tokens_per_frame=TOKENS_PER_FRAME,
        geometry_tokens_per_frame=TOKENS_PER_FRAME,
        action_tokens_per_frame=TOKENS_PER_FRAME,
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=DEFAULT_CHUNK_SIZE,
        window_size=DEFAULT_WINDOW_SIZE,
        device=torch.device("cpu"),
    )
    video_order, action_order = _build_sgf_orders(
        num_frames=DEFAULT_NUM_FRAMES,
        chunk_size=DEFAULT_CHUNK_SIZE,
        device=metadata.device,
    )
    metadata.order_ids = torch.cat(
        [video_order, video_order, video_order, action_order, action_order], dim=1
    )
    metadata.cache_key = None
    metadata.structure_cache_key = None
    return metadata


def _build_sgf_orders(
    *, num_frames: int, chunk_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return SGF video/geometry and action clocks for one frame sequence."""
    video_order = segmented_orders(
        torch.arange(num_frames, device=device),
        history_frames=DEFAULT_HISTORY_FRAMES,
        chunk_size=chunk_size,
    )
    action_order = video_order + 1
    return video_order[None, :], action_order[None, :]



def build_default_x_metadata_and_mask() -> tuple[MOTMaskMetadata, torch.Tensor]:
    """Build the configured 8-frame X-only mask with one V/A token per frame."""

    metadata = build_x_metadata4sgf()
    return metadata, build_dense_mot_mask(metadata)[0]


def build_default_mot_metadata_and_mask() -> tuple[MOTMaskMetadata, torch.Tensor]:
    """Build the configured 8-frame joint mask with one V/G/A token per frame."""

    metadata = build_mot_metadata4sgf()
    policy_metadata = from_mot_metadata(metadata)
    return metadata, build_cache_visibility(
        policy_metadata,
        policy_metadata,
        window_size=metadata.window_size,
    )[0]




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


def test_visualize_default_x_metadata_mask(tmp_path):
    metadata, mask = build_default_x_metadata_and_mask()

    assert tuple(mask.shape) == (32, 32)
    expected_video_order = [0, 0, 0, 0, 2, 4, 6, 8]
    expected_action_order = [1, 1, 1, 1, 3, 5, 7, 9]
    assert metadata.order_ids[0, :8].tolist() == expected_video_order
    assert metadata.order_ids[0, 16:24].tolist() == expected_action_order
    assert bool(mask[0, 0])
    assert not bool(mask[0, 4])
    assert not bool(mask[8, 4])
    assert not bool(mask[8, 12])

    output_path = render_metadata_mask(
        metadata,
        mask,
        tmp_path / "x_mask.png",
        title="X-only MOT dense attention mask",
    )
    _assert_png_colors(output_path, {"NV", "CV", "NA", "CA"})


def test_visualize_default_mot_metadata_mask(tmp_path):
    metadata, mask = build_default_mot_metadata_and_mask()

    assert tuple(mask.shape) == (40, 40)
    assert metadata.order_ids[0, 16:24].tolist() == [0, 0, 0, 0, 2, 4, 6, 8]
    assert not bool(mask[0, 8])
    assert bool(mask[12, 8])
    assert bool(mask[12, 16])
    assert bool(mask[8, 9])
    assert not bool(mask[8, 0])
    assert not bool(mask[16, 16])
    assert not bool(mask[16, 17])
    assert bool(mask[20, 19])

    output_path = render_metadata_mask(
        metadata,
        mask,
        tmp_path / "mot_mask.png",
        title="Joint V/A/G MOT dense attention mask",
    )
    _assert_png_colors(output_path, {"NV", "CV", "G", "NA", "CA"})


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
    x_meta, x_dense_mask = build_default_x_metadata_and_mask()
    x_saved_path = render_metadata_mask(
        x_meta,
        x_dense_mask,
        X_ARTIFACT_PATH,
        title="X-only MOT dense attention mask",
    )
    mot_meta, mot_dense_mask = build_default_mot_metadata_and_mask()
    mot_saved_path = render_metadata_mask(
        mot_meta,
        mot_dense_mask,
        MOT_ARTIFACT_PATH,
        title="Joint V/A/G MOT dense attention mask",
    )
    print(x_saved_path)
    print(mot_saved_path)
