# Self-Gradient-Forcing DMD 实现与数据流程

本文说明 `self_gradient_forcing_dmd` 的当前实现。所有 rollout、record、mask 和
teacher/student 数据组织都位于 `distillation/`，不调用
`inference/mot_inference.py`，也不修改 `wan_va/`。

## 1. 固定窗口、order 与 mask

标准训练窗口为：

```text
frame:       H0 H1 H2 H3 | T0 T1 T2 T3
V/G order:    0  0  0  0 |  2  4  6  8
A order:      1  1  1  1 |  3  5  7  9
loss mask:    0  0  0  0 |  0  1  1  1
```

`H0..H3` 是 GT history，`T0` 是 GT target anchor。SGF 只生成和监督
`T1..T3`，因此和 AR、Consistency 使用同一个
`segmented_history_strict_geometry_v1` profile。

Geometry 继续满足严格历史约束：

```text
G_query(frame=i) 只能读取 frame_id < i 的 committed geometry
```

## 2. 模型所有权

| 模型 | 初始化来源 | 梯度 | 作用 |
| --- | --- | --- | --- |
| student | stage2 consistency EMA export | 可训练 | no-grad rollout、SGF replay、DMD student update |
| real-score / teacher | stage1 AR export | 永久冻结 | GT-clean CFG teacher、DMD real score |
| fake-score | stage1 AR export | 独立 optimizer | DMD fake score、fake-score regression |

Frozen teacher 保持 `eval()` 和 `requires_grad=False`，但所有 full-window joint
forward 都调用：

```python
model(input_dict, mode="train")
```

这里 `mode="train"` 表示走原模型的 `NV + CV + G + NA + CA` joint training
route，不表示解冻 teacher。

## 3. 一次 self_rollout record

`generate_and_record_context()` 在 `torch.no_grad()` 下执行一次：

```text
commit GT H0..H3 V/G/A
commit GT T0 V/G/A

predict T1 video
  -> encode/commit pred T1 geometry
  -> predict/commit T1 action

predict T2 video -> geometry -> action
predict T3 video -> geometry -> action
```

Video 和 action 各随机选择一个 denoise step。每个生成 frame 在 scheduler
`step()` 之前记录：

```text
sample x_t
timestep t
```

记录器只复制 tensor，不改变 scheduler、CFG、transaction cache 或 committed
cache。最终 clean prediction 仍继续完成并写入 rollout result。

## 4. ReplayContext 的三条数据流

`ReplayContext` 显式区分 student、teacher 和 rollout state：

```python
ReplayContext(
    student_batch=...,       # GT history/T0 + pred target V/A/G
    teacher_batch=...,       # 完整 GT V/A/G
    rollout_timesteps=...,   # record 的 video/action [B,F] timestep
    rollout_noisy=...,       # record 的真实 V/A sampler state
    pred_clean=...,          # student clean stream
    teacher_clean=...,       # teacher GT clean stream
    masks=...,               # 仅 T1..T3，并与 valid mask 相交
)
```

Student clean：

```text
video:    GT H0..H3 | GT T0 predV1 predV2 predV3
action:   GT H0..H3 | GT T0 predA1 predA2 predA3
geometry: GT H0..H3 | GT T0 predG1 predG2 predG3
```

Teacher clean：

```text
video/action/geometry: 完整 dataset GT
```

Rollout noisy：

```text
history/T0/padding: 保持对应 branch 的 clean
T1..T3:            使用 rollout recorder 的真实 x_t
```

Video/action timestep 分开保存，因为两个 scheduler 可以使用不同步数和 sigma。

## 5. Student replay 与 GT-clean teacher CFG

Student 输入：

```text
clean    = pred_clean
noisy    = rollout_noisy
geometry = predicted target geometry
timestep = rollout_timesteps
```

Teacher 输入：

```text
clean    = teacher_clean GT
noisy    = 同一份 rollout-recorded target state
geometry = GT geometry
timestep = 同一份 rollout_timesteps
```

Teacher conditional/unconditional 两次 forward 除 text embedding 外完全一致：

```python
teacher_video = uncond_video + cfg_scale * (cond_video - uncond_video)
teacher_action = cond_action
```

CFG 只作用于 video，action 始终使用 conditional flow。Teacher CFG flow 替换旧的
`(rollout_noisy - pred_clean) / sigma` exact self-replay target：

```text
replay_loss = masked_mse(student_flow, teacher_cfg_flow)
```

这样 student 在自己的 rollout 中间状态上，使用预测 clean condition 进行 forward，
但学习 GT-clean frozen teacher 给出的目标。

## 6. Student DMD 分支

Student flow 转为可微 clean estimate：

```text
student_x0 = rollout_noisy - sigma_rollout * student_flow
```

mask 外显式恢复 `pred_clean`，防止 scheduler 数值 timestep 0 对应非零最小 sigma
时改变 history、T0 或 padding。

随后重新采样一份 DMD score noise：

```text
score_noisy = (1 - sigma_score) * student_x0 + sigma_score * Gaussian noise
```

real-score 和 fake-score 在 `torch.no_grad()` 中接收完全相同的：

- `score_noisy`；
- score timestep；
- `student_x0` clean condition；
- predicted geometry；
- mask、order 和 text condition。

因此两者差值仍具有可比性：

```text
real_x0 = score_noisy - sigma_score * real_flow
fake_x0 = score_noisy - sigma_score * fake_flow

dmd_target = student_x0 - (fake_x0 - real_x0).detach() / normalizer
dmd_loss = masked_mse(student_x0, dmd_target)
```

Student update：

```text
student_total = teacher_replay_loss + dmd_loss
```

## 7. Fake-score update

Fake-score update 复用 detached `pred_clean` 和 `student_batch`：

```text
pred_clean
  -> sample score timestep and Gaussian noise
  -> fake_score(..., mode="train")
  -> exact synthetic flow target
  -> fake-score optimizer
```

该分支不运行 teacher CFG，也不运行 DMD real-score forward。只有 fake-score 参数
建立梯度。

## 8. Optimizer 调度

默认：

```text
fake_score, fake_score, fake_score, fake_score, student, repeat
```

gradient accumulation 期间 `optimizer_step` 不变，因此同一 accumulation window
不会切换模型或 optimizer。real-score 永远不进入 optimizer。

## 9. 每次更新的大模型调用

不计 VAE decode：

| 更新类型 | Student | Real-score / teacher | Fake-score |
| --- | ---: | ---: | ---: |
| fake-score | 1 次 no-grad incremental rollout | 0 | 1 次有梯度 full-window forward |
| student | 1 次 no-grad incremental rollout + 1 次有梯度 full-window forward | 2 次 teacher CFG + 1 次 DMD real-score | 1 次 no-grad DMD fake-score |

所有 full-window forward 都使用 `mode="train"` 和同一个 segmented strict-geometry
attention policy。

## 10. 关键配置

| 配置 | 默认值 | 作用 |
| --- | ---: | --- |
| `rollout_video_num_steps` | 2 | self_rollout video denoise 步数 |
| `rollout_action_num_steps` | 2 | self_rollout action denoise 步数 |
| `rollout_horizon_frames` | 3 | 从 GT T0 后生成 T1..T3 |
| `teacher_cfg_min` | 2.0 | SGF teacher video CFG 下界 |
| `teacher_cfg_max` | 10.0 | SGF teacher video CFG 上界 |
| `fake_score_update_ratio` | 4 | 每次 student update 前的 fake-score update 数 |
| `score_timestep_min` | 0 | DMD score timestep 下界，包含 |
| `score_timestep_max` | 1000 | DMD score timestep 上界，不包含 |
| `dmd_normalizer_eps` | `1e-6` | DMD normalizer 下界 |

如果需要固定 teacher CFG，令：

```text
teacher_cfg_min == teacher_cfg_max
```

## 11. 验证边界

实现必须持续满足：

```text
SGF 不调用 inference.mot_inference
SGF 不重新生成 H1..H3
T0 保持 GT anchor
只有 T1..T3 进入 replay/DMD/fake-score mask
student 使用 pred target V/A/G
teacher 使用 GT target V/A/G
teacher noisy target 使用 rollout record
teacher CFG 只作用 video
AR/CM/SGF profile 完全一致
wan_va/ 无代码修改
```
