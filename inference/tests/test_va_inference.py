import pytest
import torch

from inference.mot_inference import decoded_rgb_to_latent_frames
from wan_va.configs.mot_eval_cfg import make_mot_eval_cfg


def test_decoded_rgb_selects_one_frame_per_latent() -> None:
    decoded = torch.arange(13).reshape(1, 13, 1, 1, 1, 1)
    selected = decoded_rgb_to_latent_frames(decoded, latent_frames=4)
    assert selected.flatten().tolist() == [0, 1, 5, 9]


def test_eval_modes_are_video_or_full(tmp_path) -> None:
    assert make_mot_eval_cfg(dataset_root=tmp_path, wan22_model_root=tmp_path).mode == "video"
    with pytest.raises(ValueError):
        make_mot_eval_cfg(dataset_root=tmp_path, wan22_model_root=tmp_path, mode="invalid")
