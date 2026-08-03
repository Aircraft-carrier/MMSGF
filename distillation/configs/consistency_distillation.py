"""Consistency distillation configuration."""
import copy

from easydict import EasyDict

from wan_va.configs import VA_CONFIGS

consistency_distillation_cfg = EasyDict(copy.deepcopy(VA_CONFIGS["umi_3dwam_train"]))
consistency_distillation_cfg.optimization_composition = "va"
consistency_distillation_cfg.cfg_prob = 0.0
consistency_distillation_cfg.distill = EasyDict(
    method="consistency_distillation",
    # distillation wrapper 在原生 metadata 上应用两段式 order；不改变物理 packing。
    generation_shape={"order_mode": "segmented", "chunk_size": 4, "window_size": 16},
    max_grad_norm=2.0,
    # 分别决定 scheduler 离散表上 t -> t_next 的跨度；不是每次 forward 数量。
    video_num_steps=2,
    action_num_steps=2,
    # Teacher CFG 只作用于 video flow，action 始终使用 conditional prediction。
    cfg_min=2.0,
    cfg_max=10.0,
    # Video consistency boundary scaling 的数据尺度；必须 > 0。
    sigma_data=0.5,
    # 额外的 action exact-flow regression 权重。
    action_aware_weight=0.01,
    # 仅在成功 student optimizer.step 后更新 EMA。
    ema_decay=0.9999,
    rollout_interval=500,
    rollout_video_num_steps=2,
    rollout_action_num_steps=2,
    rollout_chunk_pairs=1,
    # fresh run: student_init 初始化 raw student 和 EMA；teacher_checkpoint 冻结。
    # resume: DCP 会覆盖 raw student/optimizer/EMA，teacher 仍从来源重建。
    student_init=None,
    teacher_checkpoint=None,
    resume_from=None,
)
