"""Self-gradient-forcing DMD configuration."""
import copy

from easydict import EasyDict

from wan_va.configs import VA_CONFIGS

self_gradient_forcing_dmd_cfg = EasyDict(copy.deepcopy(VA_CONFIGS["umi_3dwam_train"]))
self_gradient_forcing_dmd_cfg.optimization_composition = "va"
self_gradient_forcing_dmd_cfg.distill = EasyDict(
    method="self_gradient_forcing_dmd",
    generation_shape={
        "profile_name": "segmented_history_strict_geometry_v1",
        "order_mode": "segmented",
        "history_frames": 4,
        "chunk_size": 4,
        "window_size": 16,
    },
    max_grad_norm=2.0,
    # optimizer_step 周期：4 次 fake-score，然后 1 次 student，循环。
    fake_score_update_ratio=4,
    # 与 consistency 的 self_rollout 一致：GT history + GT T0，逐帧生成 T1..T3。
    rollout_video_num_steps=2,
    rollout_action_num_steps=2,
    rollout_horizon_frames=3,
    # Frozen real-score 作为 SGF teacher；CFG 只作用于 video，action 用 conditional。
    teacher_cfg_min=2.0,
    teacher_cfg_max=10.0,
    # 重新给 student_x0/generated 加 score noise 时的 nominal timestep 范围。
    # video/action 用各自 scheduler 把同一 t 映射成各自 sigma。
    score_timestep_min=0,
    score_timestep_max=1000,
    # DMD normalizer 和 flow 反解的数值安全下界；不改变 mask 语义。
    dmd_normalizer_eps=1e-6,
    flow_target_eps=1e-6,
    # student 来自 stage2 EMA export；real/fake score 通常都来自 stage1 AR。
    # DCP resume 会恢复 student/fake-score 双 optimizer 的精确状态。
    student_init=None,
    real_score_checkpoint=None,
    fake_score_init=None,
    resume_from=None,
)
