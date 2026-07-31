import ast
import json
import random
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from inference import mot_chunk_infer
from wan_va.dataset.mot_dataset import mot_action_per_frame, relative_20d_to_absolute_actions


_REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeTransformer:
    def __init__(self):
        self.events = []
        self.masked_attn_backend = "fa4"
        self.weight = torch.zeros(1)
        self.vggto = SimpleNamespace(keep_heads_fp32_=self._keep_heads_fp32)
        self.mot_blocks = [SimpleNamespace(masked_attn_backend="fa4")]

    def to(self, *, device=None, dtype=None):
        self.events.append(("to", device, dtype))
        return self

    def _keep_heads_fp32(self):
        self.events.append("keep_heads_fp32")

    def eval(self):
        self.events.append("eval")
        return self

    def requires_grad_(self, requires_grad):
        self.events.append(("requires_grad", requires_grad))
        return self

    def state_dict(self):
        return {"weight": self.weight}

    def named_parameters(self):
        return iter(())

    def load_state_dict(self, state_dict, strict):
        self.events.append(("load_state_dict", strict))
        self.weight.copy_(state_dict["weight"])


def test_build_runner_loads_and_prepares_transformer(tmp_path, monkeypatch) -> None:
    transformer_dir = tmp_path / "checkpoint" / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / mot_chunk_infer.SAFETENSORS_NAME).touch()
    transformer = _FakeTransformer()

    loaded = []

    def load_transformer(path, **kwargs):
        loaded.append((Path(path), kwargs))
        return transformer

    monkeypatch.setattr(
        mot_chunk_infer.ThreeDVAMOTTransformer3DModel,
        "from_pretrained",
        load_transformer,
    )
    fa4_devices = []
    monkeypatch.setattr(
        mot_chunk_infer,
        "validate_fa4_training_environment",
        lambda device: fa4_devices.append(device),
    )
    config = SimpleNamespace(device="cpu", masked_attn_backend="auto")

    runner = mot_chunk_infer._build_runner(transformer_dir.parent, config)

    assert isinstance(runner, mot_chunk_infer.MOTInferenceSession)
    assert loaded == [(transformer_dir, {"torch_dtype": torch.bfloat16})]
    assert fa4_devices == []
    assert runner.transformer is transformer
    assert runner.dtype == torch.bfloat16
    assert runner.config.masked_attn_backend == "dense"
    assert transformer.masked_attn_backend == "dense"
    assert transformer.vggto.masked_attn_backend == "dense"
    assert transformer.mot_blocks[0].masked_attn_backend == "dense"
    assert transformer.events == [
        ("to", torch.device("cpu"), None),
        "keep_heads_fp32",
        "eval",
        ("requires_grad", False),
    ]


def test_build_runner_uses_fa4_for_manual_gpu_eval(tmp_path, monkeypatch) -> None:
    transformer_dir = tmp_path / "checkpoint" / "transformer"
    transformer_dir.mkdir(parents=True)
    (transformer_dir / mot_chunk_infer.SAFETENSORS_NAME).touch()
    transformer = _FakeTransformer()
    monkeypatch.setattr(
        mot_chunk_infer.ThreeDVAMOTTransformer3DModel,
        "from_pretrained",
        lambda *_args, **_kwargs: transformer,
    )
    fa4_devices = []
    monkeypatch.setattr(
        mot_chunk_infer,
        "validate_fa4_training_environment",
        lambda device: fa4_devices.append(device),
    )
    config = SimpleNamespace(device="cuda:0", masked_attn_backend="auto")

    mot_chunk_infer._build_runner(transformer_dir.parent, config)

    assert fa4_devices == [torch.device("cuda:0")]
    assert config.masked_attn_backend == "fa4"
    assert transformer.masked_attn_backend == "fa4"


def test_load_transformer_checkpoint_reads_current_dcp_model_state(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "checkpoint"
    transformer_dir = checkpoint_dir / "transformer"
    dcp_dir = checkpoint_dir / mot_chunk_infer.DCP_DIR_NAME
    transformer_dir.mkdir(parents=True)
    dcp_dir.mkdir()
    (checkpoint_dir / "_SUCCESS").touch()
    (dcp_dir / ".metadata").touch()
    transformer = _FakeTransformer()
    monkeypatch.setattr(
        mot_chunk_infer.ThreeDVAMOTTransformer3DModel,
        "from_config",
        lambda path: transformer,
    )
    calls = []

    def load(state, *, checkpoint_id, no_dist):
        calls.append((checkpoint_id, no_dist))
        state["model"]["weight"].fill_(2)

    monkeypatch.setattr(mot_chunk_infer.dcp, "load", load)

    loaded = mot_chunk_infer._load_transformer_checkpoint(transformer_dir)

    assert loaded is transformer
    assert calls == [(dcp_dir, True)]
    assert transformer.weight.item() == 2
    assert transformer.events == [
        ("to", None, torch.bfloat16),
        "keep_heads_fp32",
        ("load_state_dict", True),
    ]


def test_transformer_dtype_contract_is_bf16_carrier_with_fp32_geometry_heads() -> None:
    transformer = torch.nn.Module()
    transformer.carrier = torch.nn.Linear(2, 2).to(torch.bfloat16)
    transformer.vggto = torch.nn.Module()
    transformer.vggto.dense_head = torch.nn.Linear(2, 2).float()
    transformer.vggto.point_head = torch.nn.Linear(2, 2).float()

    mot_chunk_infer._validate_transformer_dtypes(transformer)

    transformer.vggto.point_head.to(torch.bfloat16)
    with pytest.raises(TypeError, match="vggto.point_head.weight"):
        mot_chunk_infer._validate_transformer_dtypes(transformer)


def _inference_session() -> mot_chunk_infer.MOTInferenceSession:
    config = SimpleNamespace(
        action_chunk_size=48,
        video_downsample_ratio=4,
        vae_temporal_factor=4,
        norm_stat={"q01": [100.0] * 20, "q99": [200.0] * 20},
    )
    return mot_chunk_infer.MOTInferenceSession(
        config=config,
        transformer=object(),
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize("mode", ["video", "geometry", "full"])
def test_single_entrypoint_accepts_each_inference_mode(mode: str) -> None:
    args = mot_chunk_infer.parse_args(
        ["--checkpoint-root", "/checkpoint", "--mode", mode]
    )

    assert args.mode == mode


def test_inference_session_dispatches_explicit_mode(monkeypatch) -> None:
    runner = _inference_session()
    calls = []
    monkeypatch.setattr(
        runner,
        "run_video",
        lambda batch, frame_count: calls.append(("video", batch, frame_count)),
    )
    monkeypatch.setattr(
        runner,
        "run_geometry",
        lambda batch, frame_count: calls.append(("geometry", batch, frame_count)),
    )
    monkeypatch.setattr(
        runner,
        "run_full",
        lambda batch, frame_count: calls.append(("full", batch, frame_count)),
    )
    batch = {"sample": 1}

    for mode in ("video", "geometry", "full"):
        runner.run(batch, 8, mode=mode)

    assert calls == [
        ("video", batch, 8),
        ("geometry", batch, 8),
        ("full", batch, 8),
    ]


def test_geometry_sample_path_never_materializes_vae_latents(tmp_path) -> None:
    class Dataset:
        rows = [
            {
                "inference_start_frame": 0,
                "source_dataset": "source",
                "task_name": "task",
                "episode_index": 0,
            }
        ]

        @staticmethod
        def get_window(_sample_idx, _start_frame):
            return {
                "geometry_rgb": torch.zeros(8, 4, 2, 3, 2, 2),
                "geometry_group_valid_mask": torch.ones(8, 4, dtype=torch.bool),
                "has_pointcloud": torch.tensor(False),
            }

    class Runner:
        config = SimpleNamespace(
            action_chunk_size=48,
            video_downsample_ratio=4,
            vae_temporal_factor=4,
        )

        def __init__(self):
            self.materialize_calls = 0

        @staticmethod
        def move_batch_to_device(batch):
            return batch

        def materialize_batch_latents(self, _batch):
            self.materialize_calls += 1
            raise AssertionError("geometry mode must not load the VAE")

        @staticmethod
        def run(batch, frame_count, *, mode):
            assert mode == "geometry"
            assert "latents" not in batch
            return SimpleNamespace(
                pred_depth=torch.zeros(1, frame_count, 2, 2, 2, 1),
                pred_depth_conf=torch.ones(1, frame_count, 2, 2, 2),
                pred_points=torch.zeros(1, frame_count * 4, 2, 2, 2, 3),
                geometry_rgb=batch["geometry_rgb"],
            )

        @staticmethod
        def save_depth_video(_pred_depth, _pred_conf, _geometry_rgb, _batch, _sample_dir):
            return "depth_rgb_pred_conf.mp4"

    runner = Runner()
    mot_chunk_infer._run_one_sample(
        runner,
        Dataset(),
        0,
        tmp_path,
        mode="geometry",
    )

    sample_dir = next(tmp_path.iterdir())
    assert runner.materialize_calls == 0
    assert (sample_dir / "pred_depth.pt").is_file()
    assert (sample_dir / "pred_points.pt").is_file()
    assert not (sample_dir / "pred_latents.pt").exists()
    assert not (sample_dir / "pred_actions_norm.pt").exists()
    metadata = json.loads((sample_dir / "metadata.json").read_text())
    assert metadata["inference_mode"] == "geometry"
    assert metadata["gt_pointcloud_plys"] == []
    assert len(metadata["pred_pointcloud_plys"]) == 8 * 4 * 2
    assert all((sample_dir / record["path"]).is_file() for record in metadata["pred_pointcloud_plys"])


def _action_artifact_batch() -> tuple[dict, torch.Tensor]:
    tokens_per_frame = mot_action_per_frame()
    relative = torch.zeros(20)
    relative[0:3] = torch.tensor([0.1, 0.2, 0.3])
    relative[3] = relative[7] = 1.0
    relative[9] = 0.4
    relative[10:13] = torch.tensor([-0.1, -0.2, -0.3])
    relative[13] = relative[17] = 1.0
    relative[19] = 0.6
    actions = torch.zeros(1, 20, 8, tokens_per_frame, 1)
    action_valid_mask = torch.zeros_like(actions, dtype=torch.bool)
    action_valid_mask[:, :, 1:4] = True
    action_valid_mask[:, :, 5:8] = True
    action_loss_mask = torch.zeros_like(action_valid_mask)
    action_loss_mask[:, :, 5:8] = True
    refs = torch.zeros(1, 16, 8, tokens_per_frame, 1)
    refs[:, 0:3] = torch.tensor([10.0, 20.0, 30.0]).view(1, 3, 1, 1, 1)
    refs[:, 3] = 1.0
    refs[:, 8:11] = torch.tensor([-10.0, -20.0, -30.0]).view(1, 3, 1, 1, 1)
    refs[:, 11] = 1.0
    return {
        "action_loss_mask": action_loss_mask,
        "action_valid_mask": action_valid_mask,
        "action_q01": (relative - 1.0).unsqueeze(0),
        "action_q99": (relative + 1.0).unsqueeze(0),
        "action_reference_states": refs,
    }, relative


def test_inference_action_plots_use_sample_stats_and_absolute_actions(tmp_path, monkeypatch) -> None:
    runner = _inference_session()
    batch, expected_relative = _action_artifact_batch()
    actions = torch.zeros_like(batch["action_valid_mask"], dtype=torch.float32)
    saved_plots = {}

    def save_plot(*, pred, gt, labels, history_steps, break_at_history, path):
        del labels
        saved_plots[path.name] = (
            torch.from_numpy(pred.copy()),
            torch.from_numpy(gt.copy()),
            history_steps,
            break_at_history,
        )
        path.touch()

    monkeypatch.setattr(mot_chunk_infer, "_save_action_20d_plot", save_plot)

    runner.save_action_plot(actions, actions, tmp_path, batch=batch)

    relative_plot, relative_gt, history_steps, break_at_history = saved_plots[
        "action_plot_denorm_20d_relative.png"
    ]
    assert relative_plot.shape == relative_gt.shape == (96, 20)
    assert history_steps == 48
    assert break_at_history
    torch.testing.assert_close(
        relative_plot,
        expected_relative.expand_as(relative_plot),
        atol=1e-5,
        rtol=0,
    )
    token_valid = batch["action_valid_mask"][0, :, :, :, 0].all(dim=0).reshape(-1)
    refs = batch["action_reference_states"][0, :, :, :, 0].permute(1, 2, 0).reshape(-1, 16)[token_valid].numpy()
    expected_absolute = torch.from_numpy(relative_20d_to_absolute_actions(refs, relative_plot.numpy()))
    expected_absolute_plot = mot_chunk_infer._absolute_16d_to_plot_20d(expected_absolute)
    absolute_plot, absolute_gt, absolute_history_steps, absolute_break = saved_plots[
        "action_plot_denorm_20d_absolute.png"
    ]
    assert absolute_plot.shape == absolute_gt.shape == (96, 20)
    assert absolute_history_steps == 48
    assert not absolute_break
    assert torch.isfinite(absolute_plot).all()
    torch.testing.assert_close(
        absolute_plot,
        expected_absolute_plot,
    )
    assert (tmp_path / "action_plot_denorm_20d_relative.png").is_file()
    assert (tmp_path / "action_plot_denorm_20d_absolute.png").is_file()


def test_inference_action_plots_require_per_sample_metadata(tmp_path, monkeypatch) -> None:
    runner = _inference_session()
    batch, _ = _action_artifact_batch()
    del batch["action_q99"]
    actions = torch.zeros_like(batch["action_valid_mask"], dtype=torch.float32)
    monkeypatch.setattr(
        mot_chunk_infer,
        "_save_action_20d_plot",
        lambda **_kwargs: pytest.fail("metadata validation must run before artifact export"),
    )

    with pytest.raises(KeyError, match="action_q99"):
        runner.save_action_plot(actions, actions, tmp_path, batch=batch)


def test_inference_config_is_fixed_to_current_training_protocol(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    mot_config = {
        "action_dim": 20,
        "norm_stat": {"q01": [0.0] * 20, "q99": [1.0] * 20},
        "norm_stats_by_task": {"task": {"q01": [0.0] * 20, "q99": [1.0] * 20}},
        "action_chunk_size": 48,
        "video_downsample_ratio": 4,
        "vae_temporal_factor": 4,
        "empty_emb_path": "/fallback/empty_emb.pt",
    }
    (dataset_root / "meta" / "mot_config.json").write_text(json.dumps(mot_config), encoding="utf-8")
    config = mot_chunk_infer._make_config(
        SimpleNamespace(
            dataset_root=str(dataset_root),
            wan22_model_root=str(tmp_path / "wan"),
            device="cuda:0",
            masked_attn_backend="auto",
            num_inference_steps=25,
            action_num_inference_steps=50,
            guidance_scale=5.0,
            action_guidance_scale=1.0,
            inference_video_fps=10,
        )
    )

    assert config.num_inference_steps == 25
    assert config.action_num_inference_steps == 50
    assert config.guidance_scale == 5
    assert config.action_guidance_scale == 1
    assert config.inference_video_fps == 10
    assert config.masked_attn_backend == "auto"
    assert config.empty_emb_path == str(dataset_root / "empty_emb.pt")
    assert not hasattr(config, "eval_freq")


def test_checkpoint_evaluation_converts_before_inference(tmp_path, monkeypatch) -> None:
    checkpoint_dir = tmp_path / "checkpoint_step_7"
    transformer_dir = checkpoint_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    (checkpoint_dir / "_SUCCESS").touch()
    output_dir = tmp_path / "eval"
    events = []

    def convert(path):
        assert Path(path) == checkpoint_dir
        events.append("convert")
        weight_path = transformer_dir / mot_chunk_infer.SAFETENSORS_NAME
        weight_path.touch()
        return weight_path

    def run_artifacts(path, eval_cfg, destination):
        assert Path(path) == checkpoint_dir
        assert eval_cfg.mode == "video"
        assert Path(destination) == output_dir
        assert (transformer_dir / mot_chunk_infer.SAFETENSORS_NAME).is_file()
        events.append("inference")

    monkeypatch.setattr(mot_chunk_infer, "convert_dcp_to_safetensors", convert)
    monkeypatch.setattr(mot_chunk_infer, "_run_evaluation_artifacts", run_artifacts)

    result = mot_chunk_infer.run_checkpoint_evaluation(
        checkpoint_dir,
        SimpleNamespace(mode="video"),
        output_dir=output_dir,
    )

    assert events == ["convert", "inference"]
    assert result == output_dir
    assert (output_dir / "_SUCCESS").is_file()


def _manifest_row(
    episode_index: int,
    *,
    has_pointcloud: bool,
    source_dataset: str = "lumos_lerobot",
) -> dict:
    return {
        "data_file": f"/data/{episode_index}.parquet",
        "dataset_from_index": 0,
        "dataset_to_index": 300,
        "episode_index": episode_index,
        "fps": 10,
        "has_pointcloud": has_pointcloud,
        "norm_stats_key": "task",
        "segment": {"action_text": "task", "start_frame": 0, "end_frame": 300},
        "source_dataset": source_dataset,
        "source_lerobot_task_dir": "/data/task",
        "task_uid": "task",
        "timestamp_policy": "episode_local_frame_over_fps_v1",
        "valid_start_range": [0, 267],
        "video_downsample_ratio": 4,
        "views": [],
    }


def test_selects_one_pointcloud_and_one_non_pointcloud_episode_per_source(tmp_path) -> None:
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    for filename, has_pointcloud in (
        ("mot_final_training_pointcloud_manifest.jsonl", True),
        ("mot_final_training_non_pointcloud_manifest.jsonl", False),
    ):
        rows = []
        for source_idx, source_dataset in enumerate(mot_chunk_infer.EVAL_SOURCE_DATASETS):
            rows.extend(
                _manifest_row(
                    source_idx * 10 + idx,
                    has_pointcloud=has_pointcloud,
                    source_dataset=source_dataset,
                )
                for idx in range(8)
            )
            padded_only = _manifest_row(
                99 + source_idx * 100,
                has_pointcloud=has_pointcloud,
                source_dataset=source_dataset,
            )
            padded_only["segment"]["end_frame"] = 80
            padded_only["valid_start_range"] = [0, 47]
            rows.append(padded_only)
        (meta_dir / filename).write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    selected = mot_chunk_infer._select_inference_rows(
        dataset_root=tmp_path,
        rng=random.Random(42),
    )
    repeated = mot_chunk_infer._select_inference_rows(
        dataset_root=tmp_path,
        rng=random.Random(42),
    )

    assert selected == repeated
    assert len(selected) == 4
    assert {
        (row["source_dataset"], bool(row["has_pointcloud"]))
        for row in selected
    } == {
        (source_dataset, has_pointcloud)
        for source_dataset in mot_chunk_infer.EVAL_SOURCE_DATASETS
        for has_pointcloud in (True, False)
    }
    assert all(row["episode_index"] not in {99, 199} for row in selected)


def test_random_start_excludes_all_history_and_target_padding() -> None:
    row = _manifest_row(0, has_pointcloud=True)
    row["segment"] = {"action_text": "task", "start_frame": 10, "end_frame": 200}
    row["valid_start_range"] = [20, 180]

    starts = {
        mot_chunk_infer._sample_full_window_start(
            row,
            action_chunk_size=48,
            rng=random.Random(seed),
        )
        for seed in range(20)
    }

    assert min(starts) >= 59
    assert max(starts) <= 151
    assert len(starts) > 1


def test_save_video_decodes_history_and_target_separately(tmp_path, monkeypatch) -> None:
    runner = _inference_session()
    runner.config.inference_video_fps = 10
    decode_sizes = []

    def decode(latents):
        decode_sizes.append(int(latents.shape[2]))
        return torch.zeros(1, 13, 2, 3, 2, 2)

    saved = {}
    labels = []
    monkeypatch.setattr(runner, "decode_latents_to_rgb_views", decode)
    monkeypatch.setattr(
        mot_chunk_infer,
        "_annotate_panel",
        lambda frame, label, _color: labels.append(label) or frame,
    )
    monkeypatch.setattr(
        mot_chunk_infer.imageio,
        "mimsave",
        lambda path, frames, fps: saved.update(path=path, frames=frames, fps=fps),
    )

    runner.save_video(
        torch.zeros(1, 1, 8, 2, 1, 1),
        torch.zeros(1, 26, 2, 3, 2, 2),
        tmp_path,
    )

    assert decode_sizes == [4, 4]
    assert len(saved["frames"]) == 26
    assert saved["fps"] == 10
    assert sum(label.startswith("GT |") for label in labels) == 52
    assert sum(label.startswith("PRED |") for label in labels) == 52
    assert any(label.endswith("HISTORY") for label in labels)
    assert any(label.endswith("TARGET") for label in labels)


def test_depth_video_layout_depends_on_gt_pointcloud_availability(tmp_path, monkeypatch) -> None:
    runner = _inference_session()
    runner.config.inference_video_fps = 10
    points = torch.zeros(1, 8, 4, 2, 4, 4, 3)
    points[..., 2] = 2.0
    valid = torch.ones(1, 8, 4, 2, 4, 4, dtype=torch.bool)
    geometry_rgb = torch.full((1, 8, 4, 2, 3, 4, 4), 0.5)
    batch = {
        "has_pointcloud": torch.tensor([True]),
        "geometry_pts3d": points,
        "geometry_point_valid_mask": valid,
    }
    pred_depth = torch.full((1, 8, 2, 4, 4, 1), 1.5)
    pred_conf = torch.full((1, 8, 2, 4, 4), 5.0)
    labels = []
    saved = {}
    monkeypatch.setattr(
        mot_chunk_infer,
        "_colorize_depth",
        lambda values, valid, **_kwargs: torch.zeros(*values.shape, 3, dtype=torch.uint8).numpy(),
    )
    monkeypatch.setattr(
        mot_chunk_infer,
        "_annotate_panel",
        lambda frame, label, _color: labels.append(label) or frame,
    )
    monkeypatch.setattr(
        mot_chunk_infer.imageio,
        "mimsave",
        lambda path, frames, fps: saved.update(path=path, frames=frames, fps=fps),
    )

    artifact = runner.save_depth_video(pred_depth, pred_conf, geometry_rgb, batch, tmp_path)

    assert artifact == "depth_rgb_gt_pred_diff_conf.mp4"
    assert saved["path"] == tmp_path / artifact
    assert saved["frames"].shape == (8, 8, 20, 3)
    assert saved["fps"] == 10
    assert sum(label.startswith("RGB |") for label in labels) == 16
    assert sum(label.startswith("GT DEPTH |") for label in labels) == 16
    assert sum(label.startswith("PRED DEPTH |") for label in labels) == 16
    assert sum(label.startswith("DEPTH DIFF |") for label in labels) == 16
    assert sum(label.startswith("PRED CONF |") for label in labels) == 16

    labels.clear()
    saved.clear()
    artifact = runner.save_depth_video(
        pred_depth,
        pred_conf,
        geometry_rgb,
        {**batch, "has_pointcloud": torch.tensor([False])},
        tmp_path,
    )

    assert artifact == "depth_rgb_pred_conf.mp4"
    assert saved["path"] == tmp_path / artifact
    assert saved["frames"].shape == (8, 8, 12, 3)
    assert saved["fps"] == 10
    assert sum(label.startswith("RGB |") for label in labels) == 16
    assert sum(label.startswith("PRED DEPTH |") for label in labels) == 16
    assert sum(label.startswith("PRED CONF |") for label in labels) == 16
    assert not any(label.startswith("GT DEPTH |") for label in labels)
    assert not any(label.startswith("DEPTH DIFF |") for label in labels)


def test_pointcloud_samples_export_colored_gt_ply(tmp_path) -> None:
    points = torch.arange(8 * 4 * 2 * 2 * 2 * 3, dtype=torch.float32).reshape(8, 4, 2, 2, 2, 3)
    valid = torch.ones(8, 4, 2, 2, 2, dtype=torch.bool)
    valid[0, 0, :, 0, 0] = False
    rgb = torch.ones(8, 4, 2, 3, 2, 2)
    group_valid = torch.zeros(8, 4, dtype=torch.bool)
    group_valid[0, 0] = True
    group_valid[1, 1] = True
    sample = {
        "has_pointcloud": torch.tensor(True),
        "geometry_pts3d": points,
        "geometry_point_valid_mask": valid,
        "geometry_rgb": rgb,
        "geometry_group_valid_mask": group_valid,
    }

    records = mot_chunk_infer._save_gt_pointcloud_plys(sample, tmp_path)

    assert len(records) == 4
    assert {record["num_vertices"] for record in records} == {3, 4}
    for record in records:
        payload = (tmp_path / record["path"]).read_bytes()
        assert payload.startswith(b"ply\nformat binary_little_endian 1.0\n")
    assert mot_chunk_infer._save_gt_pointcloud_plys(
        {**sample, "has_pointcloud": torch.tensor(False)},
        tmp_path / "non_pointcloud",
    ) == []


def test_pred_pointcloud_uses_action_geometry_points_rgb_and_gt_scale(tmp_path, monkeypatch) -> None:
    gt_points = torch.zeros(1, 8, 4, 1, 2, 2, 3)
    gt_points[..., 2] = 2.0
    point_valid = torch.ones(1, 8, 4, 1, 2, 2, dtype=torch.bool)
    group_valid = torch.zeros(1, 8, 4, dtype=torch.bool)
    group_valid[:, 0, 0] = True
    pred_points = torch.ones(1, 32, 1, 2, 2, 3)
    pred_rgb = torch.zeros(1, 8, 4, 1, 3, 2, 2)
    pred_rgb[:, :, :, :, 0] = 1.0
    batch = {
        "has_pointcloud": torch.tensor([True]),
        "geometry_pts3d": gt_points,
        "geometry_point_valid_mask": point_valid,
        "geometry_group_valid_mask": group_valid,
    }
    writes = []

    def write(path, points, colors, valid):
        writes.append((path, points.copy(), colors.copy(), valid.copy()))
        return int(valid.sum())

    monkeypatch.setattr(mot_chunk_infer, "write_ply", write)

    records = mot_chunk_infer._save_pred_pointcloud_plys(pred_points, pred_rgb, batch, tmp_path)

    assert len(records) == len(writes) == 1
    assert records[0]["path"].startswith("pred_pointcloud/")
    _path, saved_points, saved_colors, saved_valid = writes[0]
    torch.testing.assert_close(torch.from_numpy(saved_points), torch.full((2, 2, 3), 2.0))
    assert bool(saved_valid.all())
    assert bool((saved_colors[..., 0] == 255).all())
    assert bool((saved_colors[..., 1:] == 0).all())


def test_pred_pointcloud_without_gt_exports_normalized_finite_points(tmp_path, monkeypatch) -> None:
    group_valid = torch.zeros(1, 2, 2, dtype=torch.bool)
    group_valid[:, 0, 1] = True
    pred_points = torch.arange(4 * 1 * 2 * 2 * 3, dtype=torch.float32).reshape(1, 4, 1, 2, 2, 3)
    pred_points[:, 1, :, 0, 0] = float("nan")
    pred_rgb = torch.zeros(1, 2, 2, 1, 3, 2, 2)
    pred_rgb[:, :, :, :, 1] = 1.0
    batch = {
        "has_pointcloud": torch.tensor([False]),
        "geometry_group_valid_mask": group_valid,
    }
    writes = []

    def write(path, points, colors, valid):
        writes.append((path, points.copy(), colors.copy(), valid.copy()))
        return int(valid.sum())

    monkeypatch.setattr(mot_chunk_infer, "write_ply", write)

    records = mot_chunk_infer._save_pred_pointcloud_plys(pred_points, pred_rgb, batch, tmp_path)

    assert len(records) == len(writes) == 1
    assert records[0]["path"] == "pred_pointcloud/group_00_slot_1_view_0.ply"
    _path, saved_points, saved_colors, saved_valid = writes[0]
    torch.testing.assert_close(torch.from_numpy(saved_points), pred_points[0, 1, 0], equal_nan=True)
    assert not bool(saved_valid[0, 0])
    assert int(saved_valid.sum()) == 3
    assert bool((saved_colors[..., 1] == 255).all())
    assert bool((saved_colors[..., (0, 2)] == 0).all())


def _import_targets(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets.add(node.module)
            targets.update(f"{node.module}.{alias.name}" for alias in node.names)
    return targets


def _imports_module(targets: set[str], module: str) -> bool:
    return any(target == module or target.startswith(f"{module}.") for target in targets)


def test_training_and_inference_import_boundaries() -> None:
    train_path = _REPO_ROOT / "wan_va" / "train_mot.py"
    train_source = train_path.read_text(encoding="utf-8")
    train_targets = _import_targets(train_path)

    assert not _imports_module(train_targets, "inference")
    assert not _imports_module(train_targets, "wan_va.mot_inference")
    assert "run_mot_inference" not in train_source
    assert "evaluate_batch" not in train_source
    assert "eval_freq" not in train_source
    assert not (_REPO_ROOT / "wan_va" / "mot_inference.py").exists()

    for path in (_REPO_ROOT / "inference").rglob("*.py"):
        assert not _imports_module(_import_targets(path), "wan_va.train_mot"), path
