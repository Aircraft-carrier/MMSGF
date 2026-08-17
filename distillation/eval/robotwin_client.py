"""RoboTwin rollout client for the distillation HTTP policy server."""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


CAMERA_MAP = {
    "cam_high": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes", "y"}:
        return True
    if value in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def encode_jpeg(image: np.ndarray, quality: int = 90) -> str:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(
        buffer, format="JPEG", quality=quality
    )
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_jpeg(value: str) -> np.ndarray:
    with Image.open(io.BytesIO(base64.b64decode(value))) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def write_predicted_video(payload: dict[str, Any], output_path: Path) -> None:
    frames = [
        np.concatenate([decode_jpeg(image) for image in frame], axis=1)
        for frame in payload["frames"]
    ]
    if not frames:
        raise ValueError("predicted video contains no generated frames")
    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(int(payload["fps"])),
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )
    for frame in frames:
        process.stdin.write(frame.tobytes())
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to save predicted video: {output_path}")


def encode_observation(observation: dict[str, Any], step: int) -> dict[str, Any]:
    images = {
        wire_name: encode_jpeg(observation["observation"][robotwin_name]["rgb"])
        for wire_name, robotwin_name in CAMERA_MAP.items()
    }
    endpose = observation["endpose"]
    state = np.concatenate(
        [
            np.asarray(endpose["left_endpose"], dtype=np.float32),
            np.asarray([endpose["left_gripper"]], dtype=np.float32),
            np.asarray(endpose["right_endpose"], dtype=np.float32),
            np.asarray([endpose["right_gripper"]], dtype=np.float32),
        ]
    )
    if state.shape != (16,):
        raise ValueError(f"RoboTwin endpose state must have shape (16,), got {state.shape}")
    return {"step": int(step), "images": images, "state": state.tolist()}


class PolicyClient:
    def __init__(self, host: str, port: int, timeout: float, retries: int) -> None:
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        self.retries = retries

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"policy server returned HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(1)
        raise ConnectionError(f"policy request failed after retries: {last_error}")

    def reset(self, session_id: str, task_name: str, instruction: str, seed: int) -> None:
        self.post(
            "/v1/reset",
            {
                "session_id": session_id,
                "task_name": task_name,
                "instruction": instruction,
                "seed": seed,
            },
        )

    def actions(
        self,
        session_id: str,
        request_id: int,
        observations: list[dict[str, Any]],
        executed_actions: list[list[float]],
        return_video: bool = False,
    ) -> dict[str, Any]:
        return self.post(
            "/v1/actions",
            {
                "session_id": session_id,
                "request_id": request_id,
                "observations": observations,
                "executed_actions": executed_actions,
                "return_video": bool(return_video),
            },
        )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_progress(
    path: Path,
    task_name: str,
    task_config: str,
    initial_seed: int,
    resume: bool,
) -> dict[str, Any]:
    if resume and path.is_file():
        progress = json.loads(path.read_text(encoding="utf-8"))
        if progress.get("task_name") != task_name:
            raise ValueError(f"progress task mismatch in {path}")
        if progress.get("task_config") != task_config:
            raise ValueError(f"progress task config mismatch in {path}")
        return progress
    return {
        "task_name": task_name,
        "task_config": task_config,
        "completed_episodes": 0,
        "success_count": 0,
        "next_seed": initial_seed,
    }


def resolve_task_config(robotwin_repo: Path, value: str) -> tuple[str, Path]:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.stem, candidate.resolve()
    if candidate.suffix in {".yml", ".yaml"}:
        config_path = robotwin_repo / "task_config" / candidate.name
    else:
        config_path = robotwin_repo / "task_config" / f"{value}.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"task config not found: {config_path}")
    return config_path.stem, config_path.resolve()


def prepare_config(robotwin_repo: Path, task_name: str, task_config_path: Path) -> dict[str, Any]:
    from envs import CONFIGS_PATH

    args = yaml.safe_load(task_config_path.read_text(encoding="utf-8"))
    args["task_name"] = task_name
    args["task_config"] = task_config_path.stem
    args["eval_mode"] = True
    args["save_data"] = False

    config_root = Path(CONFIGS_PATH)
    embodiment_types = yaml.safe_load(
        (config_root / "_embodiment_config.yml").read_text(encoding="utf-8")
    )
    camera_config = yaml.safe_load(
        (config_root / "_camera_config.yml").read_text(encoding="utf-8")
    )
    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        left_file = right_file = embodiment_types[embodiment[0]]["file_path"]
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        left_file = embodiment_types[embodiment[0]]["file_path"]
        right_file = embodiment_types[embodiment[1]]["file_path"]
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("embodiment must contain one or three items")

    def embodiment_config(path: str) -> dict[str, Any]:
        path_obj = Path(path)
        if not path_obj.is_absolute():
            path_obj = robotwin_repo / path_obj
        return yaml.safe_load((path_obj / "config.yml").read_text(encoding="utf-8"))

    args["left_robot_file"] = left_file
    args["right_robot_file"] = right_file
    args["left_embodiment_config"] = embodiment_config(left_file)
    args["right_embodiment_config"] = embodiment_config(right_file)
    head_camera = camera_config[args["camera"]["head_camera_type"]]
    args["head_camera_h"] = head_camera["h"]
    args["head_camera_w"] = head_camera["w"]
    return args


def create_task(task_name: str):
    import importlib

    module = importlib.import_module(f"envs.{task_name}")
    return getattr(module, task_name)()


def select_instruction(task_name: str, episode_info: dict[str, Any], instruction_type: str) -> str:
    from generate_episode_instructions import generate_episode_descriptions

    descriptions = generate_episode_descriptions(task_name, [episode_info], 100)
    choices = descriptions[0][instruction_type]
    if not choices:
        raise ValueError(
            f"task {task_name!r} generated no {instruction_type!r} instructions"
        )
    return str(np.random.choice(choices))


def start_video(task_env, output_path: Path, width: int, height: int):
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            "10",
            "-i",
            "-",
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            "-crf",
            "23",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )
    task_env._set_eval_video_ffmpeg(process)


def find_stable_episode(task_env, args: dict[str, Any], episode_id: int, seed: int):
    from envs.utils.create_actor import UnStableError

    while True:
        try:
            task_env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **args)
            episode_info = task_env.play_once()["info"]
            valid = task_env.plan_success and task_env.check_success()
            task_env.close_env()
            if valid:
                return seed, episode_info
        except UnStableError as exc:
            print(f"Skipping unstable expert seed {seed}: {exc}")
            task_env.close_env()
        except Exception:
            task_env.close_env()
            raise
        seed += 1


def run_episode(
    task_env,
    args: dict[str, Any],
    policy: PolicyClient,
    task_name: str,
    instruction_type: str,
    episode_id: int,
    seed: int,
    episode_info: dict[str, Any],
    output_dir: Path,
    save_predicted_videos: bool,
) -> bool:
    instruction = select_instruction(task_name, episode_info, instruction_type)
    args["eval_video_save_dir"] = str(output_dir) if args.get("eval_video_log") else None
    task_env.setup_demo(now_ep_num=episode_id, seed=seed, is_test=True, **args)
    task_env.set_instruction(instruction=instruction)

    if args.get("eval_video_log", False):
        start_video(
            task_env,
            output_dir / f"episode{episode_id}.mp4",
            int(args["head_camera_w"]),
            int(args["head_camera_h"]),
        )

    session_id = f"{task_name}-seed-{seed}-{os.getpid()}"
    policy.reset(session_id, task_name, instruction, seed)
    observation = task_env.get_obs()
    pending_observations = [encode_observation(observation, 0)]
    pending_actions: list[list[float]] = []
    next_step = 1
    request_id = 0

    try:
        while task_env.take_action_cnt < task_env.step_lim and not task_env.eval_success:
            response = policy.actions(
                session_id,
                request_id,
                pending_observations,
                pending_actions,
                return_video=save_predicted_videos,
            )
            if save_predicted_videos:
                write_predicted_video(
                    response["predicted_video"],
                    output_dir
                    / "generated_videos"
                    / f"episode{episode_id:03d}_request{request_id:04d}.mp4",
                )
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 16:
                raise ValueError(f"server actions must have shape (N, 16), got {actions.shape}")
            pending_observations = []
            pending_actions = []
            for action in actions:
                task_env.take_action(action, action_type=response.get("action_type", "ee"))
                observation = task_env.get_obs()
                pending_actions.append(action.tolist())
                pending_observations.append(encode_observation(observation, next_step))
                next_step += 1
                if task_env.eval_success or task_env.take_action_cnt >= task_env.step_lim:
                    break
            request_id += 1
        return bool(task_env.eval_success)
    finally:
        if task_env.eval_video_path is not None:
            task_env._del_eval_video_ffmpeg()
        task_env.close_env(clear_cache=((episode_id + 1) % args["clear_cache_freq"] == 0))


def write_result(path: Path, progress: dict[str, Any], instruction_type: str) -> None:
    completed = int(progress["completed_episodes"])
    success_count = int(progress["success_count"])
    success_rate = success_count / completed if completed else 0.0
    path.write_text(
        "\n".join(
            [
                f"Timestamp: {datetime.now().astimezone().isoformat(timespec='seconds')}",
                f"Task: {progress['task_name']}",
                f"Task Config: {progress['task_config']}",
                f"Instruction Type: {instruction_type}",
                f"Success: {success_count}/{completed}",
                str(success_rate),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin-repo", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--eval-num-episodes", type=int, default=100)
    parser.add_argument("--eval-output-dir", type=Path, required=True)
    parser.add_argument("--result-file-name", required=True)
    parser.add_argument("--resume-from-progress", type=parse_bool, default=True)
    parser.add_argument("--resume-progress-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction-type", choices=("seen", "unseen"), default="unseen")
    parser.add_argument("--request-timeout", type=float, default=1800)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--save-predicted-video", type=parse_bool, default=False)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    cli = parse_args(argv)
    robotwin_repo = cli.robotwin_repo.resolve()
    os.chdir(robotwin_repo)
    sys.path[:0] = [
        str(robotwin_repo),
        str(robotwin_repo / "policy"),
        str(robotwin_repo / "description" / "utils"),
    ]

    from script.test_render import Sapien_TEST

    Sapien_TEST()
    task_config, task_config_path = resolve_task_config(robotwin_repo, cli.task_config)
    args = prepare_config(robotwin_repo, cli.task_name, task_config_path)
    output_dir = cli.eval_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = cli.resume_progress_path.resolve()
    result_path = output_dir / cli.result_file_name
    initial_seed = 100000 * (1 + cli.seed)
    progress = load_progress(
        progress_path,
        cli.task_name,
        task_config,
        initial_seed,
        cli.resume_from_progress,
    )
    policy = PolicyClient(cli.host, cli.port, cli.request_timeout, cli.request_retries)
    task_env = create_task(cli.task_name)

    while int(progress["completed_episodes"]) < cli.eval_num_episodes:
        episode_id = int(progress["completed_episodes"])
        seed, episode_info = find_stable_episode(
            task_env, args, episode_id, int(progress["next_seed"])
        )
        success = run_episode(
            task_env,
            args,
            policy,
            cli.task_name,
            cli.instruction_type,
            episode_id,
            seed,
            episode_info,
            output_dir,
            cli.save_predicted_video,
        )
        progress["completed_episodes"] = episode_id + 1
        progress["success_count"] = int(progress["success_count"]) + int(success)
        progress["next_seed"] = seed + 1
        progress["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        atomic_write_json(progress_path, progress)
        rate = int(progress["success_count"]) / int(progress["completed_episodes"])
        print(
            f"{cli.task_name}: {progress['success_count']}/{progress['completed_episodes']} "
            f"({rate:.1%}), next_seed={progress['next_seed']}"
        )

    write_result(result_path, progress, cli.instruction_type)
    print(f"Result written to {result_path}")


if __name__ == "__main__":
    main()
