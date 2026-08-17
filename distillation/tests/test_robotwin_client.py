import base64
import io

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
