import base64
import io
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from distillation.eval import robotwin_client


def _jpeg(value: int) -> str:
    image = Image.fromarray(np.full((2, 3, 3), value, dtype=np.uint8))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_write_predicted_video_joins_views_and_writes_every_frame(
    tmp_path, monkeypatch
) -> None:
    class Stdin:
        def __init__(self):
            self.payload = bytearray()

        def write(self, value):
            self.payload.extend(value)

        def close(self):
            return None

    class Process:
        def __init__(self):
            self.stdin = Stdin()

        def wait(self):
            return 0

    process = Process()
    command = None

    def popen(value, *, stdin):
        nonlocal command
        command = value
        assert stdin is robotwin_client.subprocess.PIPE
        return process

    monkeypatch.setattr(robotwin_client.subprocess, "Popen", popen)
    payload = {
        "fps": 10,
        "camera_keys": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
        "frames": [
            [_jpeg(10), _jpeg(20), _jpeg(30)],
            [_jpeg(40), _jpeg(50), _jpeg(60)],
        ],
    }

    robotwin_client.write_predicted_video(payload, tmp_path / "generated.mp4")

    assert command[command.index("-video_size") + 1] == "9x2"
    assert command[command.index("-framerate") + 1] == "10"
    assert len(process.stdin.payload) == 2 * 2 * 9 * 3


def test_encode_observation_uses_training_camera_and_eef_order(monkeypatch) -> None:
    monkeypatch.setattr(
        robotwin_client,
        "encode_jpeg",
        lambda image: str(int(np.asarray(image)[0, 0, 0])),
    )
    observation = {
        "observation": {
            "head_camera": {"rgb": np.full((1, 1, 3), 1, np.uint8)},
            "left_camera": {"rgb": np.full((1, 1, 3), 2, np.uint8)},
            "right_camera": {"rgb": np.full((1, 1, 3), 3, np.uint8)},
        },
        "endpose": {
            "left_endpose": list(range(7)),
            "left_gripper": 7,
            "right_endpose": list(range(8, 15)),
            "right_gripper": 15,
        },
    }

    encoded = robotwin_client.encode_observation(observation, step=4)

    assert encoded["step"] == 4
    assert encoded["images"] == {
        "cam_high": "1",
        "cam_left_wrist": "2",
        "cam_right_wrist": "3",
    }
    assert encoded["state"] == list(range(16))


def test_instruction_selection_is_deterministic_for_episode_seed(monkeypatch) -> None:
    choices = ["first", "second", "third"]
    monkeypatch.setitem(
        sys.modules,
        "generate_episode_instructions",
        SimpleNamespace(
            generate_episode_descriptions=lambda *_args: [{"unseen": choices}]
        ),
    )

    selected = robotwin_client.select_instruction("task", {}, "unseen", seed=17)

    assert selected == str(np.random.default_rng(17).choice(choices))


def test_episode_sends_executed_action_with_the_resulting_observation(
    tmp_path, monkeypatch
) -> None:
    class Policy:
        def __init__(self):
            self.requests = []

        def reset(self, *args):
            self.reset_args = args

        def actions(
            self,
            session_id,
            request_id,
            observations,
            executed_actions,
            return_video=False,
        ):
            self.requests.append(
                (session_id, request_id, observations, executed_actions, return_video)
            )
            count = 2 if request_id == 0 else 1
            return {
                "actions": [([request_id] * 16) for _ in range(count)],
                "action_type": "ee",
            }

    class Environment:
        take_action_cnt = 0
        step_lim = 3
        eval_success = False
        eval_video_path = None

        def setup_demo(self, **_kwargs):
            return None

        def set_instruction(self, **_kwargs):
            return None

        def get_obs(self):
            value = self.take_action_cnt
            image = np.full((1, 1, 3), value, np.uint8)
            return {
                "observation": {
                    "head_camera": {"rgb": image},
                    "left_camera": {"rgb": image},
                    "right_camera": {"rgb": image},
                },
                "endpose": {
                    "left_endpose": [0] * 7,
                    "left_gripper": 0,
                    "right_endpose": [0] * 7,
                    "right_gripper": 0,
                },
            }

        def take_action(self, action, action_type):
            assert action_type == "ee"
            self.take_action_cnt += 1

        def close_env(self, **_kwargs):
            return None

    monkeypatch.setattr(
        robotwin_client,
        "select_instruction",
        lambda *_args: "instruction",
    )
    policy = Policy()
    success = robotwin_client.run_episode(
        Environment(),
        {"eval_video_log": False, "clear_cache_freq": 5},
        policy,
        "task",
        "unseen",
        episode_id=0,
        seed=11,
        episode_info={},
        output_dir=tmp_path,
        save_predicted_videos=False,
    )

    assert not success
    assert [request[1] for request in policy.requests] == [0, 1]
    assert [item["step"] for item in policy.requests[0][2]] == [0]
    assert policy.requests[0][3] == []
    assert [item["step"] for item in policy.requests[1][2]] == [1, 2]
    assert policy.requests[1][3] == [[0.0] * 16, [0.0] * 16]
