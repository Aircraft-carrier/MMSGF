"""Autoregressive training adapter."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from distillation.mask_profile import generation_profile_contract, install_order_profile
from wan_va.train_mot import MOTTrainer


class AutoregressiveTrainer(MOTTrainer):
    def _load_transformer(self):
        model = super()._load_transformer()
        install_order_profile(model, self.config.distill.generation_shape)
        return model

    def __init__(self, config: Any):
        if config.distill.resume_from is not None:
            config.resume_from = str(config.distill.resume_from)
            config.initialize_from = None
        elif config.distill.student_init is not None:
            config.initialize_from = str(config.distill.student_init)
        super().__init__(config)
        install_order_profile(self.transformer, config.distill.generation_shape)

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
