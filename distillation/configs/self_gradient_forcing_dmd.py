"""Self-gradient-forcing DMD configuration."""
import copy

from easydict import EasyDict

from distillation.configs.runtime_dataset import apply_distillation_runtime_overrides
from wan_va.configs import VA_CONFIGS

self_gradient_forcing_dmd_cfg = EasyDict(copy.deepcopy(VA_CONFIGS["wan22_train"]))
apply_distillation_runtime_overrides(self_gradient_forcing_dmd_cfg)
self_gradient_forcing_dmd_cfg.distill = EasyDict(
    method="self_gradient_forcing_dmd",
    model_architecture="autoregressive_va_mot_v1",
    generation_shape={
        "profile_name": "segmented_history_va_v1",
        "order_mode": "segmented",
        "history_frames": 4,
        "chunk_size": 4,
        "window_size": 16,
    },
    max_grad_norm=2.0,
    # optimizer_step 周期：4 次 fake-score，然后 1 次 student，循环。
    fake_score_update_ratio=4,
    # V/A lists are linear denoising progress d, not the timesteps passed to the
    # network. Each scheduler maps d by k=1000-d and t=timesteps[k]. Therefore
    # the same d can produce different video/action t when their shifts differ.
    # SGF then performs velocity -> x0 -> fresh noise -> x_t(next).
    denoisy_step_list=EasyDict(
        video=[1000, 500],
        action=[1000, 500],
    ),
    rollout_horizon_frames=3,
    # Frozen real-score 作为 SGF teacher；CFG 只作用于 video，action 用 conditional。
    teacher_cfg_min=2.0,
    teacher_cfg_max=10.0,
    # Bounds are selected in linear d before warping. With both switches true,
    # DMD uses the strict exit interval [denoisy_to, denoisy_from).
    ts_schedule=True,
    ts_schedule_max=True,
    min_score_timestep=0,
    # DMD maps d to actual network t and applies this clamp after warping.
    dmd_timestep_min=20,
    dmd_timestep_max=980,
    dmd_normalizer_eps=1e-6,
    # fake-score 的 score_input 里 clean 流用 base_input 的 GT clean
    # （默认 False，用生成的 x0 作为 condition）。
    score_input_use_gt_clean=False,
    # student 来自 stage2 EMA export；real-score/teacher 来自 stage1 使用的源
    # teacher checkpoint 并保留原始 wan_va mask；fake-score 也必须是双向模型，
    # 未显式指定时从 real_score_checkpoint 初始化。
    # DCP resume 会恢复 student/fake-score 双 optimizer 的精确状态。
    student_init=None,
    real_score_checkpoint=None,
    fake_score_init=None,
    resume_from=None,
)
