import json
from types import SimpleNamespace

import torch
from torch import nn

from distillation.trainer.base import DistillationTrainerBase


class _ExportModel(nn.Linear):
    def __init__(self) -> None:
        super().__init__(2, 2)
        self.config = {"in_features": 2, "out_features": 2}


class _CheckpointTrainer(DistillationTrainerBase):
    method = "self_gradient_forcing_dmd"

    def _trainable_model(self):
        return self.network

    def get_method_state_dict(self):
        return {"method": self.method, "marker": self.marker}

    def load_method_state_dict(self, state):
        assert state["method"] == self.method
        self.marker = state["marker"]


def _trainer(tmp_path) -> _CheckpointTrainer:
    trainer = _CheckpointTrainer.__new__(_CheckpointTrainer)
    trainer.network = _ExportModel()
    trainer.optimizer = torch.optim.AdamW(trainer.network.parameters(), lr=1e-3)
    trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        trainer.optimizer,
        lr_lambda=lambda _step: 1.0,
    )
    trainer.device = torch.device("cpu")
    trainer.step = 5
    trainer.optimizer_step = 3
    trainer.marker = "saved"
    trainer.save_dir = tmp_path
    trainer._last_checkpoint_step = None
    trainer.config = SimpleNamespace(
        rank=0,
        distill=SimpleNamespace(
            model_architecture="autoregressive_va_mot_v1",
            generation_shape={
                "profile_name": "segmented_history_va_v1",
                "order_mode": "segmented",
                "history_frames": 4,
                "chunk_size": 4,
                "window_size": 16,
            },
        ),
    )
    return trainer


def test_checkpoint_is_resumable_and_loadable_by_next_stage(tmp_path) -> None:
    trainer = _trainer(tmp_path)
    loss = trainer.network(torch.ones(1, 2)).sum()
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad(set_to_none=True)
    saved_weight = trainer.network.weight.detach().clone()
    torch.manual_seed(123)

    checkpoint = trainer.save_checkpoint()
    expected_next_random = torch.rand(())

    assert (checkpoint / "_SUCCESS").is_file()
    assert (checkpoint / "training_state.pt").is_file()
    assert (checkpoint / "distributed_state" / ".metadata").is_file()
    assert (checkpoint / "transformer" / "config.json").is_file()
    assert (
        checkpoint
        / "transformer"
        / "diffusion_pytorch_model.safetensors"
    ).is_file()
    metadata = json.loads(
        (checkpoint / "checkpoint_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["model_architecture"] == "autoregressive_va_mot_v1"
    assert metadata["exported_model"] == "student"

    with torch.no_grad():
        trainer.network.weight.zero_()
    trainer.marker = "changed"
    torch.manual_seed(999)
    trainer.load_checkpoint(checkpoint)

    torch.testing.assert_close(trainer.network.weight, saved_weight)
    assert trainer.marker == "saved"
    assert trainer.step == 5
    assert trainer.optimizer_step == 3
    torch.testing.assert_close(torch.rand(()), expected_next_random)
