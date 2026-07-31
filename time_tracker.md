# MOT 8-GPU Batch Size 2 Time Tracker

Date: 2026-06-24, Asia/Shanghai

## Latest Update - Mask-Only Block-Sparse Cache

Run stamp: `codex_bs2_chunk16_100step_timing_maskonlycache_0624_175615`

Command:

```bash
RUN_STAMP=codex_bs2_chunk16_100step_timing_maskonlycache_0624_175615 \
BATCH_SIZE=2 NUM_STEPS=100 LOG_INTERVAL=100 SAVE_INTERVAL=999999 EVAL_FREQ=999999 \
DISABLE_WANDB=1 SHOW_ALL_RANK_LOGS=0 \
MOT_PROFILE_TIMING=1 MOT_PROFILE_PHASE_EVENTS=0 MOT_PROFILE_FORWARD_SYNC=0 \
MOT_PROFILE_FORWARD_EVENTS=0 MOT_PROFILE_STEP_SYNC=0 MOT_DEBUG_PHASES=0 \
MASKED_ATTN_BACKEND=fa4 MASTER_PORT=29673 \
bash 1shell/train_mot_mixed_8gpu.sh --action-chunk-size 16
```

Log directory:

```text
train_logs/mix_train_real25_preprocess/codex_bs2_chunk16_100step_timing_maskonlycache_0624_175615/torchrun_logs/none_pr_6ipw2/attempt_0
```

Result: completed `100/100` steps on all 8 ranks. Timing coverage is `800` entries = `100` steps x `8` ranks. No OOM, CUDA illegal memory access, traceback, or NCCL failure was found; only the existing NCCL device-id warning appeared. `nvidia-smi` showed no remaining compute process after the run.

### Optimization Applied

- Collapsed all-true `token_valid_ids` to `None`, so those batches use the exact cached FA4 block-sparse path.
- Added a stable `structure_cache_key` for partial-valid MoT metadata.
- For partial-valid FA4 masks, cached a structural superset as mask-only block-sparse metadata. Full blocks are merged into mask blocks, and `full_block_cnt` is forced to zero so FA4 still calls `mask_mod` and filters actual invalid tokens at element level.
- Applied the same mask-only cache idea to Pi3 frame-causal FA4 blocks when `FrameCausalMaskSpec.token_valid_ids` is present.

### Run Comparison

| run | avg 1-99 | median 1-99 | p95 1-99 | avg 50-99 | avg 90-99 | min/max 1-99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 1.764s | 1.609s | 2.777s | 1.649s | 1.425s | 1.228s / 4.086s |
| all-valid cache only | 1.850s | 1.627s | 3.325s | 1.692s | 1.455s | 1.248s / 4.148s |
| mask-only cache | 1.795s | 1.553s | 3.238s | 1.669s | 1.636s | 1.191s / 3.541s |

Interpretation: the useful improvement is in the normal/fast path. Compared with the original baseline, the mask-only cache improves the steady-state median from `1.609s` to `1.553s`, the last-10-step median from about `1.243s` to `1.208s`, and the best repeated region reaches about `1.20s/step` (steps 71-77 and 82-89). The 1-99 average changes less because several remaining outlier steps still dominate the mean.

### Latest Overall Timing

| scope | avg_total_s | median_s | p95_s | min_s | max_s |
| --- | ---: | ---: | ---: | ---: | ---: |
| steps 0-99 | 2.665 | 1.555 | 3.340 | 1.191 | 88.797 |
| steps 1-99 | 1.795 | 1.553 | 3.238 | 1.191 | 3.541 |
| steps 50-99 | 1.669 | 1.412 | 3.184 | 1.191 | 3.481 |
| steps 70-89 | 1.468 | 1.203 | 3.136 | 1.191 | 3.227 |
| steps 71-77 | 1.199 | 1.198 | 1.207 | 1.194 | 1.210 |
| steps 82-89 | 1.210 | 1.203 | 1.252 | 1.191 | 1.272 |
| steps 90-99 | 1.636 | 1.208 | 3.006 | 1.197 | 3.339 |

Step 0 remains warmup-heavy: `88.797s` average. This includes checkpoint/data/CUDA kernel warmup and should not be used as steady-state throughput.

### Latest Per-Step Average

| step | avg_total_s | min_rank_s | max_rank_s |
| ---: | ---: | ---: | ---: |
| 0 | 88.797 | 82.185 | 92.120 |
| 1 | 1.587 | 1.583 | 1.600 |
| 2 | 1.430 | 1.416 | 1.452 |
| 3 | 3.387 | 3.287 | 3.627 |
| 4 | 1.724 | 1.719 | 1.735 |
| 5 | 1.507 | 1.495 | 1.540 |
| 6 | 1.435 | 1.408 | 1.459 |
| 7 | 1.600 | 1.577 | 1.636 |
| 8 | 2.076 | 1.919 | 2.339 |
| 9 | 2.285 | 2.220 | 2.698 |
| 10 | 1.532 | 1.526 | 1.538 |
| 11 | 1.482 | 1.476 | 1.498 |
| 12 | 1.609 | 1.592 | 1.687 |
| 13 | 1.653 | 1.584 | 2.037 |
| 14 | 2.147 | 1.933 | 2.359 |
| 15 | 2.539 | 2.527 | 2.594 |
| 16 | 1.506 | 1.497 | 1.525 |
| 17 | 1.513 | 1.492 | 1.583 |
| 18 | 1.655 | 1.644 | 1.677 |
| 19 | 1.653 | 1.582 | 2.039 |
| 20 | 2.608 | 2.423 | 2.916 |
| 21 | 2.611 | 2.597 | 2.634 |
| 22 | 1.474 | 1.461 | 1.508 |
| 23 | 1.570 | 1.542 | 1.627 |
| 24 | 1.714 | 1.708 | 1.738 |
| 25 | 1.468 | 1.448 | 1.502 |
| 26 | 2.584 | 2.572 | 2.598 |
| 27 | 3.541 | 3.478 | 3.944 |
| 28 | 1.865 | 1.854 | 1.886 |
| 29 | 1.603 | 1.580 | 1.659 |
| 30 | 1.446 | 1.429 | 1.482 |
| 31 | 1.596 | 1.581 | 1.611 |
| 32 | 1.575 | 1.567 | 1.592 |
| 33 | 2.483 | 2.470 | 2.512 |
| 34 | 2.976 | 2.902 | 3.401 |
| 35 | 2.630 | 2.621 | 2.658 |
| 36 | 1.453 | 1.447 | 1.461 |
| 37 | 1.525 | 1.513 | 1.565 |
| 38 | 1.474 | 1.463 | 1.516 |
| 39 | 1.553 | 1.544 | 1.567 |
| 40 | 1.615 | 1.602 | 1.672 |
| 41 | 3.172 | 3.163 | 3.196 |
| 42 | 3.366 | 3.211 | 3.900 |
| 43 | 1.333 | 1.326 | 1.345 |
| 44 | 1.461 | 1.455 | 1.480 |
| 45 | 1.636 | 1.628 | 1.652 |
| 46 | 1.572 | 1.564 | 1.591 |
| 47 | 1.462 | 1.450 | 1.501 |
| 48 | 1.669 | 1.654 | 1.702 |
| 49 | 2.923 | 2.844 | 3.399 |
| 50 | 2.173 | 2.013 | 2.641 |
| 51 | 2.696 | 2.680 | 2.766 |
| 52 | 1.440 | 1.434 | 1.448 |
| 53 | 1.384 | 1.367 | 1.411 |
| 54 | 1.612 | 1.597 | 1.650 |
| 55 | 1.548 | 1.533 | 1.578 |
| 56 | 1.557 | 1.533 | 1.569 |
| 57 | 1.480 | 1.447 | 1.522 |
| 58 | 3.481 | 3.477 | 3.486 |
| 59 | 2.804 | 2.726 | 3.300 |
| 60 | 2.031 | 1.951 | 2.061 |
| 61 | 1.492 | 1.487 | 1.496 |
| 62 | 1.461 | 1.454 | 1.477 |
| 63 | 1.557 | 1.547 | 1.603 |
| 64 | 1.542 | 1.537 | 1.545 |
| 65 | 1.540 | 1.524 | 1.568 |
| 66 | 1.546 | 1.530 | 1.560 |
| 67 | 2.490 | 2.319 | 2.978 |
| 68 | 1.365 | 1.292 | 1.852 |
| 69 | 2.551 | 2.468 | 3.109 |
| 70 | 1.813 | 1.811 | 1.818 |
| 71 | 1.197 | 1.194 | 1.200 |
| 72 | 1.195 | 1.192 | 1.200 |
| 73 | 1.198 | 1.193 | 1.205 |
| 74 | 1.210 | 1.201 | 1.242 |
| 75 | 1.198 | 1.194 | 1.204 |
| 76 | 1.201 | 1.198 | 1.203 |
| 77 | 1.194 | 1.192 | 1.202 |
| 78 | 3.132 | 3.126 | 3.136 |
| 79 | 1.209 | 1.206 | 1.211 |
| 80 | 3.227 | 3.134 | 3.844 |
| 81 | 1.908 | 1.901 | 1.928 |
| 82 | 1.198 | 1.192 | 1.205 |
| 83 | 1.204 | 1.199 | 1.208 |
| 84 | 1.214 | 1.209 | 1.220 |
| 85 | 1.194 | 1.191 | 1.198 |
| 86 | 1.204 | 1.199 | 1.209 |
| 87 | 1.191 | 1.185 | 1.195 |
| 88 | 1.202 | 1.196 | 1.207 |
| 89 | 1.272 | 1.187 | 1.852 |
| 90 | 2.599 | 2.588 | 2.621 |
| 91 | 1.284 | 1.194 | 1.872 |
| 92 | 3.339 | 3.328 | 3.368 |
| 93 | 1.920 | 1.915 | 1.925 |
| 94 | 1.204 | 1.199 | 1.207 |
| 95 | 1.206 | 1.204 | 1.210 |
| 96 | 1.206 | 1.199 | 1.212 |
| 97 | 1.209 | 1.205 | 1.213 |
| 98 | 1.200 | 1.198 | 1.202 |
| 99 | 1.197 | 1.192 | 1.203 |

### Latest Top-Level Components

Steady-state averages across steps 1-99 and all 8 ranks.

| component | avg_s | p95_s | max_s |
| --- | ---: | ---: | ---: |
| `data_load` | 0.0013 | 0.0026 | 0.0191 |
| `data_barrier` | 0.0026 | 0.0040 | 0.0882 |
| `prepare_input` | 0.0028 | 0.0038 | 0.0287 |
| `to_device` | 0.0052 | 0.0068 | 0.6643 |
| `materialize_total` | 0.0949 | 0.1354 | 0.6083 |
| `set_gradient_sync` | 0.0026 | 0.0035 | 0.0070 |
| `transformer_forward` | 0.7943 | 2.0454 | 2.5349 |
| `compute_loss` | 0.0010 | 0.0013 | 0.0027 |
| `loss_finite_reduce` | 0.0226 | 0.0248 | 0.6047 |
| `backward` | 0.7403 | 0.8573 | 1.4270 |
| `clip_grad_norm` | 0.0455 | 0.0543 | 0.7673 |
| `grad_finite_reduce` | 0.0098 | 0.0166 | 0.7056 |
| `optimizer_step` | 0.0683 | 0.0702 | 0.7353 |
| `lr_scheduler_step` | 0.0000 | 0.0001 | 0.0001 |
| `zero_grad` | 0.0042 | 0.0124 | 0.0426 |

### Latest Forward Breakdown

| forward component | avg_s | p95_s | max_s |
| --- | ---: | ---: | ---: |
| `model_mot_pi3_decode` | 0.3562 | 1.2411 | 1.9319 |
| `model_mot_joint_block` | 0.1112 | 0.1357 | 0.8297 |
| `model_point_forward` | 0.0918 | 0.5155 | 0.9729 |
| `model_prepare_train_inputs` | 0.0882 | 0.1579 | 0.8603 |
| `model_build_mot_meta` | 0.0842 | 0.6225 | 0.6944 |
| `model_front_block` | 0.0510 | 0.0668 | 0.7332 |
| `vae_history` | 0.0488 | 0.0704 | 0.5671 |
| `model_embed_geometry` | 0.0479 | 0.0530 | 0.7160 |
| `vae_target` | 0.0460 | 0.0707 | 0.1351 |
| `model_build_x_rotary` | 0.0381 | 0.0996 | 0.7352 |
| `model_pi3_point_decoder` | 0.0189 | 0.0366 | 0.4892 |
| `model_pi3_point_conv_head` | 0.0163 | 0.0131 | 0.6842 |
| `model_mot_writeback` | 0.0029 | 0.0035 | 0.0279 |
| `model_mot_pool_registers` | 0.0025 | 0.0030 | 0.0304 |
| `model_build_timestep_emb` | 0.0013 | 0.0019 | 0.0048 |
| `model_final_video` | 0.0008 | 0.0011 | 0.0031 |
| `model_final_action` | 0.0006 | 0.0008 | 0.0023 |
| `model_build_front_meta` | 0.0004 | 0.0005 | 0.0013 |
| `model_embed_noisy_video` | 0.0003 | 0.0004 | 0.0009 |
| `model_pi3_point_output` | 0.0002 | 0.0002 | 0.0005 |
| `model_forward_untracked` | 0.0000 | 0.0000 | 0.0000 |

### Remaining Slow Steps

| step | avg_total_s | main averaged contributors |
| ---: | ---: | --- |
| 27 | 3.541 | `transformer_forward` 2.136s, `backward` 0.759s, `model_mot_pi3_decode` 0.732s, `model_point_forward` 0.529s |
| 58 | 3.481 | `transformer_forward` 1.992s, `model_mot_pi3_decode` 0.859s, `backward` 0.747s, `model_build_mot_meta` 0.593s |
| 3 | 3.387 | `backward` 1.396s, `transformer_forward` 1.378s, `model_mot_pi3_decode` 0.516s, `model_build_mot_meta` 0.406s |
| 42 | 3.366 | `transformer_forward` 2.249s, `model_mot_pi3_decode` 1.709s, `backward` 0.740s, `model_mot_joint_block` 0.310s |
| 92 | 3.339 | `transformer_forward` 1.796s, `model_mot_pi3_decode` 0.786s, `backward` 0.651s, `grad_finite_reduce` 0.617s |
| 80 | 3.227 | `transformer_forward` 2.288s, `model_mot_pi3_decode` 1.279s, `backward` 0.660s, `model_build_mot_meta` 0.546s |
| 41 | 3.172 | `transformer_forward` 2.220s, `model_mot_pi3_decode` 1.261s, `backward` 0.723s, `model_point_forward` 0.561s |
| 78 | 3.132 | `transformer_forward` 2.298s, `model_point_forward` 0.731s, `model_mot_pi3_decode` 0.722s, `backward` 0.651s |
| 34 | 2.976 | `backward` 1.179s, `transformer_forward` 1.026s, `model_prepare_train_inputs` 0.486s, `model_build_x_rotary` 0.441s |
| 49 | 2.923 | `transformer_forward` 1.848s, `model_mot_pi3_decode` 1.332s, `backward` 0.768s, `model_mot_joint_block` 0.181s |
| 59 | 2.804 | `transformer_forward` 1.757s, `model_mot_pi3_decode` 1.322s, `backward` 0.774s, `model_mot_joint_block` 0.253s |
| 51 | 2.696 | `transformer_forward` 1.740s, `backward` 0.735s, `model_mot_pi3_decode` 0.722s, `model_build_mot_meta` 0.585s |

### Current Conclusion

The repeated mask-construction issue is optimized, but not every slow step disappears. The latest run still has 15 steps with averaged `transformer_forward > 1.5s`; these are mostly attributed to `model_mot_pi3_decode`, `model_build_mot_meta`, `model_point_forward`, or distributed wait points such as `loss_finite_reduce` / `grad_finite_reduce` / `clip_grad_norm`.

Because this run used `MOT_PROFILE_FORWARD_SYNC=0` and `MOT_PROFILE_STEP_SYNC=0`, component timings are asynchronous Python wall-clock timings. A CUDA kernel launched before a timed block can be charged to a later block when the later block touches/synchronizes GPU work. This is why `model_build_mot_meta` can still show ~0.5s spikes even after block-sparse metadata caching: the label is a timing boundary, not necessarily proof that Python metadata construction itself spent 0.5s.

The next optimization target, if needed, is not another cache guess. Run a shorter profile with `MOT_PROFILE_FORWARD_SYNC=1 MOT_PROFILE_STEP_SYNC=1` or Nsight to separate real Pi3/FA4 kernel time from async attribution and collective wait time.

## Summary

This is a debug-only run with `action_chunk_size=16`. Per user instruction, norm/action statistics for this chunk size are treated as placeholders; these results should not be used to judge real training quality or loss correctness.

Result: **passed**. `BATCH_SIZE=2` completed 100/100 steps on 8 GPUs with no OOM, CUDA illegal memory access, or NCCL failure.

Run stamp: `codex_bs2_chunk16_100step_timing_0624_165846`

Command:

```bash
RUN_STAMP=codex_bs2_chunk16_100step_timing_0624_165846 \
BATCH_SIZE=2 NUM_STEPS=100 LOG_INTERVAL=1 SAVE_INTERVAL=999999 EVAL_FREQ=999999 \
DISABLE_WANDB=1 SHOW_ALL_RANK_LOGS=0 \
MOT_PROFILE_TIMING=1 MOT_PROFILE_PHASE_EVENTS=0 MOT_PROFILE_FORWARD_SYNC=0 \
MOT_PROFILE_FORWARD_EVENTS=0 MOT_PROFILE_STEP_SYNC=0 MOT_DEBUG_PHASES=0 \
MASKED_ATTN_BACKEND=fa4 MASTER_PORT=29648 \
bash 1shell/train_mot_mixed_8gpu.sh --action-chunk-size 16
```

Log directory:

```text
train_logs/mix_train_real25_preprocess/codex_bs2_chunk16_100step_timing_0624_165846/torchrun_logs/none_58061j98/attempt_0
```

Timing coverage: `800` `[MOT-TIMING]` entries = `100` steps x `8` ranks.

## Overall Timing

`avg_total_s` is the per-step average across all 8 ranks. It sums measured top-level training fields: data load/barrier, input prep, materialization, transformer forward, loss, backward, grad clipping/reductions, optimizer, scheduler, and zero-grad.

| scope | avg_total_s |
| --- | ---: |
| steps 0-99 | 2.639 |
| steps 1-99 | 1.764 |
| steps 50-99 | 1.649 |
| steps 90-99 | 1.425 |
| median, steps 1-99 | 1.609 |
| p95, steps 1-99 | 2.777 |
| min/max step avg, steps 1-99 | 1.228 / 4.086 |

Step 0 is warmup-heavy: checkpoint/VAE/CUDA kernel warmup makes it `89.323s` average, so steady-state should be read from steps 1-99 or later windows.

## Per-Step Average

| step | avg_total_s | min_rank_s | max_rank_s |
| ---: | ----------: | ---------: | ---------: |
| 0 | 89.323 | 84.130 | 91.525 |
| 1 | 1.657 | 1.654 | 1.671 |
| 2 | 1.522 | 1.462 | 1.807 |
| 3 | 2.768 | 2.673 | 3.034 |
| 4 | 1.672 | 1.665 | 1.695 |
| 5 | 1.627 | 1.614 | 1.652 |
| 6 | 1.533 | 1.510 | 1.563 |
| 7 | 1.738 | 1.716 | 1.763 |
| 8 | 2.520 | 2.367 | 2.789 |
| 9 | 2.293 | 2.286 | 2.299 |
| 10 | 1.572 | 1.568 | 1.581 |
| 11 | 1.574 | 1.568 | 1.582 |
| 12 | 1.700 | 1.682 | 1.726 |
| 13 | 2.202 | 2.123 | 2.682 |
| 14 | 2.919 | 2.798 | 3.221 |
| 15 | 1.432 | 1.425 | 1.452 |
| 16 | 1.543 | 1.527 | 1.602 |
| 17 | 1.681 | 1.669 | 1.693 |
| 18 | 1.763 | 1.730 | 1.808 |
| 19 | 2.529 | 2.526 | 2.536 |
| 20 | 2.549 | 2.426 | 2.930 |
| 21 | 1.516 | 1.510 | 1.530 |
| 22 | 1.626 | 1.609 | 1.655 |
| 23 | 1.687 | 1.676 | 1.713 |
| 24 | 1.647 | 1.643 | 1.659 |
| 25 | 1.527 | 1.505 | 1.557 |
| 26 | 4.086 | 4.082 | 4.093 |
| 27 | 1.924 | 1.922 | 1.926 |
| 28 | 1.519 | 1.510 | 1.550 |
| 29 | 1.600 | 1.590 | 1.627 |
| 30 | 1.609 | 1.598 | 1.632 |
| 31 | 1.700 | 1.675 | 1.724 |
| 32 | 1.831 | 1.563 | 2.102 |
| 33 | 2.587 | 2.504 | 3.038 |
| 34 | 1.839 | 1.830 | 1.869 |
| 35 | 1.519 | 1.509 | 1.555 |
| 36 | 1.568 | 1.560 | 1.580 |
| 37 | 1.634 | 1.628 | 1.660 |
| 38 | 1.539 | 1.531 | 1.569 |
| 39 | 1.595 | 1.522 | 2.057 |
| 40 | 2.226 | 2.083 | 2.677 |
| 41 | 2.047 | 2.042 | 2.053 |
| 42 | 1.531 | 1.514 | 1.572 |
| 43 | 1.563 | 1.553 | 1.586 |
| 44 | 1.592 | 1.576 | 1.618 |
| 45 | 1.587 | 1.575 | 1.643 |
| 46 | 1.495 | 1.491 | 1.510 |
| 47 | 2.431 | 2.191 | 2.850 |
| 48 | 2.290 | 2.133 | 2.743 |
| 49 | 2.073 | 2.061 | 2.121 |
| 50 | 1.628 | 1.614 | 1.651 |
| 51 | 1.538 | 1.529 | 1.559 |
| 52 | 1.635 | 1.619 | 1.662 |
| 53 | 1.628 | 1.601 | 1.668 |
| 54 | 1.603 | 1.590 | 1.660 |
| 55 | 1.636 | 1.551 | 2.182 |
| 56 | 2.457 | 2.364 | 3.053 |
| 57 | 2.859 | 2.770 | 3.392 |
| 58 | 1.373 | 1.370 | 1.378 |
| 59 | 1.493 | 1.484 | 1.516 |
| 60 | 1.803 | 1.796 | 1.832 |
| 61 | 1.676 | 1.662 | 1.731 |
| 62 | 1.540 | 1.520 | 1.594 |
| 63 | 1.686 | 1.668 | 1.706 |
| 64 | 1.741 | 1.560 | 2.264 |
| 65 | 2.168 | 2.162 | 2.182 |
| 66 | 3.055 | 2.887 | 3.550 |
| 67 | 1.414 | 1.412 | 1.418 |
| 68 | 1.423 | 1.408 | 1.460 |
| 69 | 1.435 | 1.430 | 1.438 |
| 70 | 1.260 | 1.255 | 1.263 |
| 71 | 1.244 | 1.239 | 1.249 |
| 72 | 1.246 | 1.243 | 1.247 |
| 73 | 1.248 | 1.241 | 1.251 |
| 74 | 1.349 | 1.267 | 1.899 |
| 75 | 2.604 | 2.600 | 2.608 |
| 76 | 2.557 | 2.554 | 2.560 |
| 77 | 1.928 | 1.923 | 1.933 |
| 78 | 1.310 | 1.306 | 1.315 |
| 79 | 1.267 | 1.265 | 1.271 |
| 80 | 1.261 | 1.259 | 1.262 |
| 81 | 1.260 | 1.251 | 1.267 |
| 82 | 1.253 | 1.250 | 1.255 |
| 83 | 1.246 | 1.240 | 1.255 |
| 84 | 1.261 | 1.257 | 1.268 |
| 85 | 1.258 | 1.255 | 1.265 |
| 86 | 2.775 | 2.681 | 3.410 |
| 87 | 1.991 | 1.986 | 2.008 |
| 88 | 2.799 | 2.797 | 2.806 |
| 89 | 1.280 | 1.275 | 1.288 |
| 90 | 1.228 | 1.227 | 1.231 |
| 91 | 1.237 | 1.235 | 1.239 |
| 92 | 1.236 | 1.231 | 1.240 |
| 93 | 1.241 | 1.240 | 1.244 |
| 94 | 1.247 | 1.243 | 1.251 |
| 95 | 1.244 | 1.242 | 1.246 |
| 96 | 1.242 | 1.239 | 1.245 |
| 97 | 1.438 | 1.237 | 2.040 |
| 98 | 2.047 | 1.940 | 2.761 |
| 99 | 2.087 | 2.084 | 2.089 |

## Top-Level Components

Steady-state averages across steps 1-99 and all 8 ranks.

| component | avg_s | p95_s | max_s |
| --- | ---: | ---: | ---: |
| `data_load` | 0.0013 | 0.0021 | 0.0056 |
| `data_barrier` | 0.0009 | 0.0025 | 0.0046 |
| `prepare_input` | 0.0027 | 0.0038 | 0.0078 |
| `to_device` | 0.0044 | 0.0063 | 0.0300 |
| `materialize_total` | 0.0921 | 0.1255 | 0.6061 |
| `set_gradient_sync` | 0.0026 | 0.0032 | 0.0046 |
| `transformer_forward` | 0.7597 | 1.7819 | 2.7309 |
| `compute_loss` | 0.0009 | 0.0012 | 0.0023 |
| `loss_finite_reduce` | 0.0106 | 0.0131 | 0.5097 |
| `backward` | 0.7560 | 1.1541 | 1.4191 |
| `clip_grad_norm` | 0.0430 | 0.0527 | 0.6917 |
| `grad_finite_reduce` | 0.0090 | 0.0189 | 0.6318 |
| `optimizer_step` | 0.0759 | 0.0777 | 0.8713 |
| `lr_scheduler_step` | 0.0000 | 0.0001 | 0.0002 |
| `zero_grad` | 0.0045 | 0.0146 | 0.0358 |

## Forward Breakdown

Steady-state forward component averages across steps 1-99 and all 8 ranks. Timings are Python wall-clock with `MOT_PROFILE_FORWARD_SYNC=0`, so CUDA async overlap can move time attribution between adjacent components; the step totals are the more reliable throughput number.

| forward component | avg_s | p95_s | max_s |
| --- | ---: | ---: | ---: |
| `model_mot_pi3_decode` | 0.3115 | 0.8795 | 1.5007 |
| `model_mot_joint_block` | 0.1270 | 0.1538 | 0.8531 |
| `model_prepare_train_inputs` | 0.0857 | 0.1352 | 1.0583 |
| `model_front_block` | 0.0838 | 0.1101 | 0.8345 |
| `model_point_forward` | 0.0707 | 0.0887 | 0.6088 |
| `model_build_mot_meta` | 0.0696 | 0.6076 | 0.7531 |
| `model_embed_geometry` | 0.0531 | 0.0610 | 0.8523 |
| `vae_history` | 0.0469 | 0.0666 | 0.5648 |
| `vae_target` | 0.0452 | 0.0709 | 0.5444 |
| `model_build_x_rotary` | 0.0304 | 0.0698 | 0.5248 |
| `model_pi3_point_decoder` | 0.0186 | 0.0187 | 0.5639 |
| `model_pi3_point_conv_head` | 0.0125 | 0.0123 | 0.5206 |
| `model_mot_writeback` | 0.0028 | 0.0033 | 0.0050 |
| `model_mot_pool_registers` | 0.0024 | 0.0030 | 0.0287 |
| `model_build_timestep_emb` | 0.0013 | 0.0018 | 0.0035 |
| `model_final_video` | 0.0008 | 0.0010 | 0.0018 |
| `model_final_action` | 0.0006 | 0.0007 | 0.0018 |
| `model_embed_noisy_video` | 0.0004 | 0.0004 | 0.0261 |
| `model_build_front_meta` | 0.0003 | 0.0004 | 0.0274 |
| `model_pi3_point_output` | 0.0002 | 0.0002 | 0.0024 |

## Slowest Forward Layers

| layer component | avg_s | p95_s | max_s |
| --- | ---: | ---: | ---: |
| `model_mot_pi3_decode.layer_01` | 0.0624 | 0.5421 | 0.7363 |
| `model_mot_pi3_decode.layer_00` | 0.0502 | 0.1318 | 0.6624 |
| `model_front_block.layer_00` | 0.0393 | 0.0395 | 0.7846 |
| `model_mot_joint_block.layer_00` | 0.0388 | 0.0379 | 0.7543 |
| `model_mot_pi3_decode.layer_07` | 0.0226 | 0.0409 | 0.6489 |
| `model_mot_pi3_decode.layer_11` | 0.0172 | 0.0208 | 0.6670 |
| `model_mot_pi3_decode.layer_16` | 0.0171 | 0.0385 | 0.5303 |
| `model_mot_pi3_decode.layer_17` | 0.0165 | 0.0407 | 0.4607 |
| `model_mot_pi3_decode.layer_03` | 0.0121 | 0.0318 | 0.0578 |
| `model_mot_pi3_decode.layer_13` | 0.0116 | 0.0207 | 0.0442 |
| `model_mot_pi3_decode.layer_08` | 0.0114 | 0.0316 | 0.0422 |
| `model_mot_pi3_decode.layer_12` | 0.0114 | 0.0181 | 0.0505 |
| `model_mot_pi3_decode.layer_09` | 0.0114 | 0.0185 | 0.0551 |
| `model_mot_pi3_decode.layer_04` | 0.0104 | 0.0175 | 0.0431 |
| `model_mot_pi3_decode.layer_06` | 0.0103 | 0.0170 | 0.0323 |

## Slowest Steps

| step | avg_total_s | min_rank_s | max_rank_s |
| ---: | ----------: | ---------: | ---------: |
| 26 | 4.086 | 4.082 | 4.093 |
| 66 | 3.055 | 2.887 | 3.550 |
| 14 | 2.919 | 2.798 | 3.221 |
| 57 | 2.859 | 2.770 | 3.392 |
| 88 | 2.799 | 2.797 | 2.806 |
| 86 | 2.775 | 2.681 | 3.410 |
| 3 | 2.768 | 2.673 | 3.034 |
| 75 | 2.604 | 2.600 | 2.608 |
| 33 | 2.587 | 2.504 | 3.038 |
| 76 | 2.557 | 2.554 | 2.560 |
| 20 | 2.549 | 2.426 | 2.930 |
| 19 | 2.529 | 2.526 | 2.536 |

## Slow-Step Cause Analysis

The slow steps are not caused by dataloader stalls. Across steady-state steps, `data_load` is about `0.0013s` avg and remains near that level on the slow steps.

There are two dominant causes:

1. Forward-side spikes around Pi3/FA4 attention and mask setup. Slow steps such as 26, 66, 14, 57, 75, 76, 86, and 87 are dominated by `transformer_forward`, especially `model_mot_pi3_decode`, with occasional time attributed to `model_build_mot_meta`, `model_build_x_rotary`, `model_embed_geometry`, or `model_point_forward`.
2. Distributed synchronization waits. Steps such as 19, 48, 88, and 98 are dominated by `backward`, `loss_finite_reduce`, `grad_finite_reduce`, `clip_grad_norm`, or `optimizer_step`. These are all boundaries where rank skew can surface as wait time.

Important attribution caveat: this run used `MOT_PROFILE_FORWARD_SYNC=0` and `MOT_PROFILE_STEP_SYNC=0`, so forward sub-component timings are Python wall-clock around async CUDA launches. When one rank is slower, other ranks can record the resulting wait under different later components. The rank-level evidence matches that pattern: on several slow steps, all ranks have nearly identical total step time while the apparent slow component differs by rank.

The most likely concrete source of the forward spikes is repeated block-sparse mask construction for token-valid FA4 masks:

- `build_front_x_metadata()` and `build_mot_metadata()` set `cache_key=None` whenever `token_valid_ids` is present.
- `_mot_block_sparse()` can use the global `_MOT_BLOCK_CACHE` only when `cache_key` is present; otherwise the cache is attached to the per-step metadata object and is rebuilt on the next step.
- `_frame_block_sparse()` always calls `_block_sparse_from_flex()` when `FrameCausalMaskSpec.token_valid_ids` is present, so Pi3 frame-causal block sparsity is not reused in that path.

This explains why the slow steps cluster around `model_mot_pi3_decode` / attention metadata work even though the tensor shapes are fixed. Some remaining spikes are ordinary FSDP/optimizer collective skew rather than a single slow Python function.

## Background

Earlier `action_chunk_size=48` tests were blocked by OOM/illegal-memory-access instability, so they did not produce a valid 100-step average. Retesting with `action_chunk_size=16` completed both the non-profile sanity run and this detailed timing run. The chunk16 setting is only for debug memory reduction.

Non-profile sanity run: `codex_bs2_chunk16_100step_nonprofile_dynshape_0624_165146`, completed 100/100 steps.

## Notes

- `action_chunk_size=16` changes V3 dynamic shapes to 4 geometry groups, 10 sampled RGB frames, raw action sequence length 17, and 16 actions per latent frame.
- No residual training processes were left after the run; `nvidia-smi` reported 0 MiB used on all 8 GPUs after completion.
- The largest steady-state forward component by average is `model_mot_pi3_decode` (`0.3115s` avg, `0.8795s` p95). The next largest are `model_mot_joint_block`, `model_prepare_train_inputs`, `model_front_block`, and `model_point_forward`.
