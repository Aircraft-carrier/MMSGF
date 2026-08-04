from __future__ import annotations

from pathlib import Path

from torch import nn

from distillation.mask_profile import install_order_profile
from distillation.self_rollout.training_policy import PROFILE_NAME


class _Geometry(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("not called")


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.geometry = _Geometry()

    def forward(self, *args, **kwargs):
        raise AssertionError("not called")


class _VGGTO(nn.Module):
    def _run_same_view_inter_frame_block(self, *args, **kwargs):
        raise AssertionError("not called")


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mot_blocks = nn.ModuleList([_Block()])
        self.vggto = _VGGTO()

    @staticmethod
    def _prepare_metadata(input_dict, prepared, geometry):
        del input_dict, prepared, geometry
        raise AssertionError("not called")


def test_segmented_profile_installs_instance_local_strict_attention_policy() -> None:
    model = _Model()
    install_order_profile(
        model,
        {"order_mode": "segmented", "chunk_size": 4, "window_size": 16},
    )
    assert model._distillation_order_profile == "segmented"
    assert model._distillation_attention_profile == PROFILE_NAME
    assert model.mot_blocks[0].forward.__func__.__name__ == "_strict_block_forward"
    assert model.mot_blocks[0].geometry.forward.__func__.__name__ == "_strict_geometry_forward"
    assert model.vggto._run_same_view_inter_frame_block.__func__.__name__ == "_strict_same_view_inter_frame"


def test_new_rollout_import_boundary_is_distillation_owned() -> None:
    root = Path(__file__).resolve().parents[1]
    rollout_sources = "\n".join(
        path.read_text()
        for path in (root / "self_rollout").glob("*.py")
    )
    assert "inference.mot_inference" not in rollout_sources
    trainer_source = (root / "trainer" / "consistency.py").read_text()
    assert "distillation.self_rollout" in trainer_source
    assert "autoregressive_rollout" not in trainer_source
    assert "run_mot_inference" not in trainer_source
