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
    # V/A 使用不同的显式 rollout schedule 和独立 exit_id。SGF 每一步执行
    # velocity -> x0 -> fresh noise -> next x_t；最后 x0 原样写入 history cache。
    denoisy_step_list=EasyDict(
        video=[1000, 833],
        action=[1000, 500],
    ),
    rollout_horizon_frames=3,
    rollout_masked_attn_backend="dense",
    # Frozen real-score 作为 SGF teacher；CFG 只作用于 video，action 用 conditional。
    teacher_cfg_min=2.0,
    teacher_cfg_max=10.0,
    # DMD timestep 由各自 exit 相邻区间动态推导。
    dmd_normalizer_eps=1e-6,
    # student 来自 stage2 EMA export；real-score/teacher 来自 stage1 使用的源
    # teacher checkpoint 并保留原始 wan_va mask；fake-score 也必须是双向模型，
    # 未显式指定时从 real_score_checkpoint 初始化。
    # DCP resume 会恢复 student/fake-score 双 optimizer 的精确状态。
    student_init=None,
    real_score_checkpoint=None,
    fake_score_init=None,
    resume_from=None,
)
