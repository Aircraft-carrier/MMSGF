"""Single-session HTTP server for cached bidirectional VA-MOT evaluation."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from http.server import ThreadingHTTPServer

import torch

from distillation.eval.server import PolicyService, make_handler

from .pipeline import load_pipeline


@dataclass
class BidirectionalPolicyService(PolicyService):
    def health(self):
        health = super().health()
        health["architecture"] = "va_mot_v1"
        return health


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--video-num-steps", type=int, default=25)
    parser.add_argument("--action-num-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
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
        guidance_scale=args.guidance_scale,
        video_snr_shift=args.video_snr_shift,
        action_snr_shift=args.action_snr_shift,
    )
    service = BidirectionalPolicyService(
        pipeline=pipeline, checkpoint=args.checkpoint_root, ready=True
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(service))
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
