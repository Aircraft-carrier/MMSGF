"""Single-session HTTP server for native autoregressive MOT evaluation."""
from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import torch

from .infer_pipeline import load_pipeline
from .protocol import parse_observation


class ServiceError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)


@dataclass
class PolicyService:
    pipeline: Any = None
    checkpoint: str = ""
    session_id: str | None = None
    last_request_id: int = -1
    last_response: dict[str, Any] | None = None
    ready: bool = False
    error: str | None = None

    def __post_init__(self) -> None:
        self.lock = threading.Lock()

    def health(self) -> dict[str, Any]:
        if not self.ready or self.pipeline is None:
            raise ServiceError(503, self.error or "model is not ready")
        parameter = next(self.pipeline.model.parameters())
        return {
            "ready": True,
            "checkpoint": self.checkpoint,
            "architecture": "autoregressive_va_mot_v1",
            "device": str(parameter.device),
            "dtype": str(parameter.dtype),
        }

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.ready or self.pipeline is None:
            raise ServiceError(503, "model is not ready")
        try:
            session_id = str(payload["session_id"])
            task_name = str(payload["task_name"])
            instruction = str(payload["instruction"])
            seed = int(payload.get("seed", 0))
            if not session_id or not instruction:
                raise ValueError("session_id and instruction must be non-empty")
            with self.lock:
                self.pipeline.reset(
                    task_name=task_name,
                    instruction=instruction,
                    seed=seed,
                )
                self.session_id = session_id
                self.last_request_id = -1
                self.last_response = None
            return {"session_id": session_id, "next_observation_step": 0}
        except KeyError as exc:
            raise ServiceError(422, str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise ServiceError(400, str(exc)) from exc

    def actions(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.ready or self.pipeline is None:
            raise ServiceError(503, "model is not ready")
        try:
            session_id = str(payload["session_id"])
            request_id = int(payload["request_id"])
            with self.lock:
                if session_id != self.session_id:
                    raise ServiceError(404, "unknown session")
                if request_id == self.last_request_id and self.last_response is not None:
                    return self.last_response
                if request_id != self.last_request_id + 1:
                    raise ServiceError(
                        409,
                        f"expected request_id {self.last_request_id + 1}, got {request_id}",
                    )
                observations = [
                    parse_observation(item) for item in payload["observations"]
                ]
                executed = [
                    torch.as_tensor(action, dtype=torch.float32).numpy()
                    for action in payload.get("executed_actions", [])
                ]
                result = self.pipeline.infer(
                    observations=observations,
                    executed_actions=executed,
                    request_id=request_id,
                    return_video=bool(payload.get("return_video", False)),
                )
                response = {
                    "session_id": session_id,
                    "request_id": request_id,
                    **result,
                }
                self.last_request_id = request_id
                self.last_response = response
                return response
        except ServiceError:
            raise
        except KeyError as exc:
            raise ServiceError(422, f"missing field {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ServiceError(400, str(exc)) from exc


def make_handler(service: PolicyService):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length))

        def do_GET(self) -> None:
            if self.path != "/healthz":
                self._send(404, {"error": "not found"})
                return
            try:
                self._send(200, service.health())
            except ServiceError as exc:
                self._send(exc.status, {"error": str(exc)})

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                if self.path == "/v1/reset":
                    result = service.reset(payload)
                elif self.path == "/v1/actions":
                    result = service.actions(payload)
                else:
                    raise ServiceError(404, "not found")
                self._send(200, result)
            except ServiceError as exc:
                self._send(exc.status, {"error": str(exc)})
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                self._send(400, {"error": str(exc)})

        def log_message(self, format: str, *args) -> None:
            return

    return Handler


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--video-num-steps", type=int, default=2)
    parser.add_argument("--action-num-steps", type=int, default=4)
    parser.add_argument(
        "--prediction-chunks", type=int, choices=range(1, 4), default=1
    )
    parser.add_argument("--video-snr-shift", type=float, default=5.0)
    parser.add_argument("--action-snr-shift", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    pipeline = load_pipeline(
        checkpoint_root=args.checkpoint_root,
        dataset_root=args.dataset_root,
        model_root=args.model_root,
        device=args.device,
        dtype=dtype,
        video_num_steps=args.video_num_steps,
        action_num_steps=args.action_num_steps,
        prediction_chunks=args.prediction_chunks,
        video_snr_shift=args.video_snr_shift,
        action_snr_shift=args.action_snr_shift,
    )
    service = PolicyService(
        pipeline=pipeline,
        checkpoint=args.checkpoint_root,
        ready=True,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
