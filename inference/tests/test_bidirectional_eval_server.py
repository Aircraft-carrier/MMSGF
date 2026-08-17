import numpy as np

from distillation.eval.protocol import encode_jpeg
from inference.eval.server import BidirectionalPolicyService


class _Model:
    def parameters(self):
        import torch

        yield torch.nn.Parameter(torch.zeros(()))


class _Pipeline:
    def __init__(self):
        self.model = _Model()
        self.calls = 0

    def reset(self, **_kwargs):
        return None

    def infer(self, **kwargs):
        self.calls += 1
        return {
            "observation_step": kwargs["observations"][-1].step,
            "actions": [[0.0] * 16] * 48,
            "action_type": "ee",
            "predicted_video": None,
            "timings_ms": {},
        }


def _wire_observation():
    jpeg = encode_jpeg(np.zeros((4, 5, 3), dtype=np.uint8))
    return {
        "step": 0,
        "images": {
            key: jpeg for key in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
        "state": [0.0] * 16,
    }


def test_health_and_duplicate_request_reuse_shared_service_protocol() -> None:
    service = BidirectionalPolicyService(
        pipeline=_Pipeline(), checkpoint="checkpoint", ready=True
    )
    assert service.health()["architecture"] == "va_mot_v1"
    service.reset(
        {
            "session_id": "episode",
            "task_name": "task",
            "instruction": "do it",
            "seed": 3,
        }
    )
    payload = {
        "session_id": "episode",
        "request_id": 0,
        "observations": [_wire_observation()],
        "executed_actions": [],
    }
    first = service.actions(payload)
    assert service.actions(payload) == first
    assert service.pipeline.calls == 1
