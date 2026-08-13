import numpy as np
import pytest

from distillation.eval.protocol import encode_jpeg
from distillation.eval.server import PolicyService, ServiceError


class _ParameterModel:
    def parameters(self):
        import torch

        yield torch.nn.Parameter(torch.zeros(()))


class _Pipeline:
    def __init__(self):
        self.model = _ParameterModel()
        self.calls = 0

    def reset(self, *, task_name, instruction, seed):
        self.reset_args = (task_name, instruction, seed)

    def infer(self, **kwargs):
        self.calls += 1
        return {
            "observation_step": kwargs["observations"][-1].step,
            "actions": [[0.0] * 16] * 16,
            "action_type": "ee",
            "predicted_video": None,
            "timings_ms": {},
        }


def _wire_observation(step=0):
    jpeg = encode_jpeg(np.zeros((4, 5, 3), dtype=np.uint8))
    return {
        "step": step,
        "images": {
            "cam_high": jpeg,
            "cam_left_wrist": jpeg,
            "cam_right_wrist": jpeg,
        },
        "state": [0.0] * 16,
    }


def _ready_service():
    return PolicyService(pipeline=_Pipeline(), checkpoint="ckpt", ready=True)


def test_service_health_reset_actions_and_idempotency() -> None:
    service = _ready_service()
    assert service.health()["architecture"] == "autoregressive_va_mot_v1"
    reset = service.reset(
        {
            "session_id": "episode",
            "task_name": "task",
            "instruction": "do it",
            "seed": 7,
        }
    )
    assert reset["next_observation_step"] == 0
    payload = {
        "session_id": "episode",
        "request_id": 0,
        "observations": [_wire_observation()],
        "executed_actions": [],
    }
    first = service.actions(payload)
    second = service.actions(payload)
    assert second == first
    assert service.pipeline.calls == 1


def test_service_rejects_out_of_order_request() -> None:
    service = _ready_service()
    service.reset(
        {
            "session_id": "episode",
            "task_name": "task",
            "instruction": "do it",
        }
    )
    with pytest.raises(ServiceError) as error:
        service.actions(
            {
                "session_id": "episode",
                "request_id": 2,
                "observations": [_wire_observation()],
                "executed_actions": [],
            }
        )
    assert error.value.status == 409
