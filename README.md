# UMI Video+Action MOT

This repository trains and evaluates a fixed-window Mixture-of-Transformers model
with two streams: video latents and robot actions.

## Model

`wan_va/modules/model_va_mot.py` contains the joint Video+Action transformer.
Every layer has video and action blocks that communicate through the shared MOT
attention policy in `wan_va/modules/mot_attention.py`. The model can initialize
the video backbone from LingBot-VA or Wan2.2 and initializes the action input,
output, and expert parameters locally.

Published checkpoints use `model_architecture=va_mot_v1`. Checkpoints from older
architectures are intentionally rejected instead of being partially loaded.

## Dataset

The runtime dataset is represented by one manifest:

```text
<dataset-root>/
├── meta/
│   ├── mot_config.json
│   └── mot_final_training_manifest.jsonl
├── empty_emb.pt
├── text_emb_cache.pt
└── cache/actions/action_cache_manifest.jsonl
```

Build a frozen selection and the final runtime manifest with:

```bash
python -m wan_va.dataset.build_training_selection \
  --dataset-root /path/to/source \
  --output-root /path/to/prepared

python -m wan_va.dataset.build_training_dataset \
  --dataset-root /path/to/source \
  --selection-root /path/to/prepared \
  --output-root /path/to/prepared
```

`MotTrainData` loads RGB frames, cached actions, masks, text embeddings, and
stream IDs. The VAE input is split into history and target windows and encoded
during training unless latents are already cached.

## Training

Set the prepared dataset and the video backbone, then launch the trainer:

```bash
export MOT_DATASET_ROOT=/path/to/prepared
export WAN22_PRETRAINED_MODEL_PATH=/path/to/lingbot-va-base
torchrun --nproc_per_node=4 wan_va/train_mot.py \
  --config-name umi_3dwam_train \
  --save-root train_logs/va_mot
```

The optimizer has one `video_action_mot` parameter group. Training computes only
video flow loss and action flow loss.

## Distillation

The `distillation/` package provides:

- autoregressive Video+Action training;
- consistency distillation;
- self-gradient-forcing DMD;
- incremental Video→Action rollout with a MOT KV cache.

All stages use the `segmented_history_va_v1` generation profile and the same
Video+Action checkpoint contract.

## Inference

Checkpoint evaluation supports `video` and `full` modes:

```bash
python inference/mot_chunk_infer.py \
  --checkpoint-root /path/to/checkpoint_step_10000 \
  --mode full
```

`video` samples the target video stream. `full` samples video first and then
actions from the resulting Video+Action context.

## Verification

```bash
python -m compileall -q wan_va distillation inference
pytest -q wan_va/tests distillation/tests inference/tests
```
