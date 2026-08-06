"""Visualize MOT masks after dataset loading with one token per frame.

Run directly from the repository root:

    PYTHONPATH=. python distillation/tests/test_visualize_dataset_loaded_masks.py

This follows the compact style of ``test_visualize_x_metadata_mask.py`` for the
V=2 dataset case: each frame contributes one video token, one geometry token,
and one action token.  Dataset masks are folded to frame-level validity before
they are passed to the attention metadata.
"""

import importlib.util
from pathlib import Path

import torch


_BASE_VIZ_PATH = Path(__file__).with_name("test_visualize_x_metadata_mask.py")
_BASE_VIZ_SPEC = importlib.util.spec_from_file_location(
    "base_visualize_x_metadata_mask",
    _BASE_VIZ_PATH,
)
if _BASE_VIZ_SPEC is None or _BASE_VIZ_SPEC.loader is None:
    raise RuntimeError(f"failed to load base visualization module: {_BASE_VIZ_PATH}")
base_viz = importlib.util.module_from_spec(_BASE_VIZ_SPEC)
_BASE_VIZ_SPEC.loader.exec_module(base_viz)
_apply_autoregressive_order = base_viz._apply_segmented_order


DATASET_LOADED_ARTIFACT_DIR = base_viz.ARTIFACT_DIR / "dataset_loaded_masks"
DATASET_VIEW_COUNT = 2
AUTOREGRESSIVE_GENERATION_SHAPE = {
    "profile_name": "autoregressive_history_strict_geometry_v1",
    "order_mode": "autoregressive",
    "history_frames": 4,
    "chunk_size": 4,
    "window_size": 16,
}


def build_dataset_loaded_metadata_and_masks(
    generation_shape,
    *,
    video_latent_valid_mask: torch.Tensor | None = None,
    action_valid_mask: torch.Tensor | None = None,
    geometry_group_valid_mask: torch.Tensor | None = None,
    autoregressive_order: bool = True,
) -> tuple[
    base_viz.MOTMaskMetadata,
    torch.Tensor,
    base_viz.MOTMaskMetadata,
    torch.Tensor,
    base_viz.MOTMaskMetadata,
    torch.Tensor,
]:
    """Build compact one-token-per-frame masks with dataset validity applied."""

    batch_size = base_viz.DEFAULT_BATCH_SIZE
    frames = base_viz.DEFAULT_NUM_FRAMES
    chunk_size = int(generation_shape["chunk_size"])
    window_size = int(generation_shape["window_size"])
    history_frames = int(generation_shape.get("history_frames", 4))
    device = torch.device("cpu")

    video_valid = _frame_valid(
        video_latent_valid_mask,
        name="video_latent_valid_mask",
        batch_size=batch_size,
        frames=frames,
    )
    action_valid = _frame_valid(
        action_valid_mask,
        name="action_valid_mask",
        batch_size=batch_size,
        frames=frames,
    )
    geometry_valid = _frame_valid(
        geometry_group_valid_mask,
        name="geometry_group_valid_mask",
        batch_size=batch_size,
        frames=frames,
    )

    x_valid = torch.cat(
        [video_valid, video_valid, action_valid, action_valid],
        dim=1,
    )
    mot_valid = torch.cat(
        [video_valid, video_valid, geometry_valid, action_valid, action_valid],
        dim=1,
    )

    x_metadata = base_viz.build_x_metadata(
        batch_size=batch_size,
        video_tokens_per_frame=base_viz.TOKENS_PER_FRAME,
        action_tokens_per_frame=base_viz.TOKENS_PER_FRAME,
        num_frames=frames,
        chunk_size=chunk_size,
        window_size=window_size,
        device=device,
        token_valid_ids=x_valid,
    )
    mot_metadata = base_viz.build_mot_metadata(
        batch_size=batch_size,
        video_tokens_per_frame=base_viz.TOKENS_PER_FRAME,
        geometry_tokens_per_frame=base_viz.TOKENS_PER_FRAME,
        action_tokens_per_frame=base_viz.TOKENS_PER_FRAME,
        num_frames=frames,
        chunk_size=chunk_size,
        window_size=window_size,
        device=device,
        token_valid_ids=mot_valid,
    )
    geometry_metadata = base_viz.build_geometry_metadata(
        batch_size=batch_size,
        geometry_tokens_per_frame=base_viz.TOKENS_PER_FRAME,
        num_frames=frames,
        chunk_size=chunk_size,
        window_size=window_size,
        device=device,
        token_valid_ids=geometry_valid,
    )

    if autoregressive_order:
        x_metadata = _apply_autoregressive_order(
            x_metadata,
            history_frames=history_frames,
            chunk_size=chunk_size,
        )
        mot_metadata = _apply_autoregressive_order(
            mot_metadata,
            history_frames=history_frames,
            chunk_size=chunk_size,
        )
        geometry_metadata = _apply_autoregressive_order(
            geometry_metadata,
            history_frames=history_frames,
            chunk_size=chunk_size,
        )
    geometry_mask = base_viz.build_dense_mot_mask(geometry_metadata)[0]

    return (
        x_metadata,
        base_viz.build_dense_mot_mask(x_metadata)[0],
        mot_metadata,
        base_viz.build_dense_mot_mask(mot_metadata)[0],
        geometry_metadata,
        geometry_mask,
    )


def dataset_loaded_mask_scenarios():
    """Return one-token-per-frame V=2 scenarios for dataset loading outcomes."""

    all_frames = torch.ones(
        (base_viz.DEFAULT_BATCH_SIZE, base_viz.DEFAULT_NUM_FRAMES),
        dtype=torch.bool,
    )

    left_padding = all_frames.clone()
    left_padding[:, 0] = False

    right_padding = all_frames.clone()
    right_padding[:, -1] = False

    missing_geometry = all_frames.clone()
    missing_geometry[:, 5] = False
    missing_geometry[:, 6] = False

    sparse_actions = all_frames.clone()
    sparse_actions[:, 2] = False
    sparse_actions[:, 5] = False

    data_variants = (
        (
            "all_valid",
            "all-valid",
            dict(),
        ),
        (
            "left_padding",
            "left-padding",
            dict(
                video_latent_valid_mask=left_padding,
                action_valid_mask=left_padding,
                geometry_group_valid_mask=left_padding,
            ),
        ),
        (
            "right_padding",
            "right-padding",
            dict(
                video_latent_valid_mask=right_padding,
                action_valid_mask=right_padding,
                geometry_group_valid_mask=right_padding,
            ),
        ),
        (
            "missing_geometry",
            "missing-geometry",
            dict(geometry_group_valid_mask=missing_geometry),
        ),
        (
            "sparse_actions",
            "sparse-action",
            dict(action_valid_mask=sparse_actions),
        ),
    )
    order_profiles = (
        ("wan_va_original", "wan_va original chunk", False),
        ("autoregressive_order", "Autoregressive order", True),
    )

    scenarios = []
    for profile_slug, profile_title, autoregressive_order in order_profiles:
        for variant_slug, variant_title, variant_kwargs in data_variants:
            kwargs = dict(variant_kwargs)
            kwargs["autoregressive_order"] = autoregressive_order
            scenarios.append(
                (
                    f"{profile_slug}_v{DATASET_VIEW_COUNT}_{variant_slug}",
                    (
                        f"{profile_title} V={DATASET_VIEW_COUNT} "
                        f"{variant_title} dataset mask"
                    ),
                    kwargs,
                )
            )
    return tuple(scenarios)


def render_dataset_loaded_scenario(
    scenario_name: str,
    title: str,
    kwargs: dict,
    output_dir: Path,
) -> dict[str, Path]:
    x_meta, x_mask, mot_meta, mot_mask, geometry_meta, geometry_mask = (
        build_dataset_loaded_metadata_and_masks(
            AUTOREGRESSIVE_GENERATION_SHAPE,
            **kwargs,
        )
    )
    scenario_dir = Path(output_dir) / scenario_name
    return {
        "x": base_viz.render_metadata_mask(
            x_meta,
            x_mask,
            scenario_dir / "x_mask.png",
            title=f"{title}: X-only",
        ),
        "mot": base_viz.render_metadata_mask(
            mot_meta,
            mot_mask,
            scenario_dir / "mot_mask.png",
            title=f"{title}: joint V/G/A",
        ),
        "geometry": base_viz.render_metadata_mask(
            geometry_meta,
            geometry_mask,
            scenario_dir / "geometry_mask.png",
            title=f"{title}: geometry-only",
        ),
    }


def _frame_valid(
    value: torch.Tensor | None,
    *,
    name: str,
    batch_size: int,
    frames: int,
) -> torch.Tensor:
    if value is None:
        return torch.ones((batch_size, frames), dtype=torch.bool)

    valid = value.to(dtype=torch.bool)
    if valid.ndim == 2:
        frame_valid = valid
    elif valid.ndim > 2:
        frame_valid = valid.flatten(start_dim=2).any(dim=2)
    else:
        raise ValueError(f"{name} must include batch and frame axes")

    if tuple(frame_valid.shape) != (batch_size, frames):
        raise ValueError(
            f"{name} must collapse to [{batch_size},{frames}], "
            f"got {tuple(frame_valid.shape)} from {tuple(valid.shape)}"
        )
    return frame_valid


def _assert_invalid_tokens_have_no_dense_attention(
    metadata: base_viz.MOTMaskMetadata,
    mask: torch.Tensor,
) -> None:
    if metadata.token_valid_ids is None:
        return
    invalid = ~metadata.token_valid_ids[0]
    if not bool(invalid.any().item()):
        return
    assert not bool(mask[invalid, :].any().item())
    assert not bool(mask[:, invalid].any().item())


def _assert_invalid_geometry_tokens_have_no_attention(
    metadata: base_viz.MOTMaskMetadata,
    mask: torch.Tensor,
) -> None:
    _assert_invalid_tokens_have_no_dense_attention(metadata, mask)


def test_visualize_dataset_loaded_masks(tmp_path):
    for scenario_name, title, kwargs in dataset_loaded_mask_scenarios():
        paths = render_dataset_loaded_scenario(
            scenario_name,
            title,
            kwargs,
            tmp_path,
        )
        base_viz._assert_png_colors(paths["x"], {"NV", "CV", "NA", "CA"})
        base_viz._assert_png_colors(paths["mot"], {"NV", "CV", "G", "NA", "CA"})
        base_viz._assert_png_colors(paths["geometry"], {"G"})


def test_dataset_loaded_validity_masks_are_applied():
    for _scenario_name, _title, kwargs in dataset_loaded_mask_scenarios():
        x_meta, x_mask, mot_meta, mot_mask, geometry_meta, geometry_mask = (
            build_dataset_loaded_metadata_and_masks(
                AUTOREGRESSIVE_GENERATION_SHAPE,
                **kwargs,
            )
        )
        _assert_invalid_tokens_have_no_dense_attention(x_meta, x_mask)
        _assert_invalid_tokens_have_no_dense_attention(mot_meta, mot_mask)
        _assert_invalid_geometry_tokens_have_no_attention(geometry_meta, geometry_mask)


if __name__ == "__main__":
    for scenario_name, title, kwargs in dataset_loaded_mask_scenarios():
        paths = render_dataset_loaded_scenario(
            scenario_name,
            title,
            kwargs,
            DATASET_LOADED_ARTIFACT_DIR,
        )
        print(f"{scenario_name}:")
        for key, path in paths.items():
            print(f"  {key}: {path}")
