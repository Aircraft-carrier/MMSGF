"""Autoregressive training adapter."""
from __future__ import annotations

from typing import Any

from distillation.mask_profile import install_order_profile
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

    def _prepare_joint_input_dict(self, batch_dict: dict) -> dict:
        input_dict = super()._prepare_joint_input_dict(batch_dict)
        input_dict["chunk_size"] = self.config.distill.generation_shape["chunk_size"]
        input_dict["window_size"] = self.config.distill.generation_shape["window_size"]
        return input_dict
