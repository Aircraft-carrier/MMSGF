"""Autoregressive training adapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from distillation.mask_profile import generation_profile_contract
from distillation.model.autoregressive_mot import (
    AutoregressiveVAMOTTransformer3DModel,
)
from wan_va.train_mot import MOTTrainer


class AutoregressiveTrainer(MOTTrainer):
    transformer_model_cls = AutoregressiveVAMOTTransformer3DModel
    checkpoint_model_architecture = "autoregressive_va_mot_v1"

    @classmethod
    def _validate_transformer_checkpoint_layout(cls, checkpoint_path: Path):
        """Allow parameter-compatible native VA exports for fresh initialization."""

        try:
            return super()._validate_transformer_checkpoint_layout(checkpoint_path)
        except ValueError as autoregressive_error:
            try:
                return MOTTrainer._validate_transformer_checkpoint_layout(checkpoint_path)
            except ValueError:
                raise autoregressive_error

    def __init__(self, config: Any):
        if config.distill.resume_from is not None:
            config.resume_from = str(config.distill.resume_from)
            config.initialize_from = None
        elif config.distill.student_init is not None:
            config.initialize_from = str(config.distill.student_init)
        super().__init__(config)

    def _write_checkpoint_metadata(self, checkpoint_dir: Path, *, has_full_state: bool) -> None:
        super()._write_checkpoint_metadata(checkpoint_dir, has_full_state=has_full_state)
        metadata_path = Path(checkpoint_dir) / "checkpoint_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "distill_method": self.config.distill.method,
                "exported_model": "student",
                "step": int(self.step),
                "optimizer_step": int(self.optimizer_step),
                "model_architecture": self.checkpoint_model_architecture,
                "generation_profile": generation_profile_contract(
                    self.config.distill.generation_shape
                ),
            }
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _prepare_joint_input_dict(self, batch_dict: dict) -> dict:
        input_dict = super()._prepare_joint_input_dict(batch_dict)
        input_dict["chunk_size"] = self.config.distill.generation_shape["chunk_size"]
        input_dict["window_size"] = self.config.distill.generation_shape["window_size"]
        return input_dict
