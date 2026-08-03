# Self-Gradient-Forcing DMD 实现与数据流程

本文说明 `self_gradient_forcing_dmd` 的当前真实实现。它在 consistency distillation 导出的 EMA student 基础上继续训练，并使用 frozen real-score model 与 trainable fake-score model 构造 DMD 更新。

## 1. 训练目标

该方法包含两种互斥 optimizer 更新：

```text
fake-score update:
  detached generated V/A
    -> add score noise
    -> fake-score predicts flow
    -> exact synthetic flow target
    -> fake-score optimizer

student update:
  no-grad context generation
    -> gradient replay
    -> replay target loss
    -> real/fake score comparison
    -> DMD surrogate loss
    -> student optimizer
```

每个 microstep 只允许 student 或 fake-score 其中一组参数获得梯度。real-score 始终冻结。

## 2. 代码边界

| 文件 | 责任 |
| --- | --- |
| `distillation/trainer/self_gradient_forcing_dmd.py` | fake-score optimizer、optimizer schedule、当前 OptimizationTarget |
| `distillation/trainer/base.py` | 公共数据、backward、finite check、clip、日志和 checkpoint 生命周期 |
| `distillation/model/dmd.py` | 持有 real/fake score、DMDUpdateSchedule 和 SGF pipeline |
| `distillation/pipeline/self_gradient_forcing_training.py` | context generation、student replay、DMD score、fake-score regression |
| `distillation/schema.py` | `VAPrediction`、`VAMasks`、`ReplayContext`、`DMDUpdateSchedule` |
| `distillation/model/objectives.py` | replay、DMD surrogate、fake-score flow loss |
| `distillation/pipeline/utils.py` | condition-preserving noise、clean target masking、geometry hiding、V/A stream replacement |
| `distillation/scheduler.py` | `[B,F]` sigma、flow-to-x0 和 broadcast |
| `distillation/model/factory.py` | real/fake model 加载、冻结/参数归属和原生 FSDP shard |
| `distillation/checkpoint.py` | student/fake-score 双 optimizer checkpoint 和 student export |

## 3. 模型所有权

| 模型 | 初始化来源 | requires_grad | 使用位置 | checkpoint |
| --- | --- | --- | --- | --- |
| student/generator | consistency checkpoint 的 `transformer/`，即 EMA student export | V/A 分支可训练 | context generation、replay、DMD student update | DCP + `transformer/` export |
| real-score | autoregressive checkpoint | 全冻结 | 只在 student DMD update 中估计 real distribution | 不重复保存 |
| fake-score | autoregressive checkpoint | V/A 分支可训练 | student DMD target 和 fake-score regression | DCP + 独立 optimizer |

`optimization_composition` 固定为 `va`，因此 student 与 fake-score 都复用 `wan_va.train_mot.apply_mot_parameter_ownership()` 冻结 G-owned 参数。real-score 全模型冻结并处于 eval mode。

fake-score 是额外的 trainable MOT，因此同时复用 `apply_ac_mot()`、`apply_ac_vggto()` 和 `shard_mot_model()`；frozen real-score 不加 activation checkpointing。

## 4. Optimizer 调度

`DMDUpdateSchedule(fake_score_steps=R)` 定义一个长度 `R+1` 的循环：

```text
R = 4:
fake_score, fake_score, fake_score, fake_score, student, repeat
```

选择只依赖已经完成的 `optimizer_step`，所有 rank 得到相同结果。默认 `fake_score_update_ratio=4` 表示每 4 次 fake-score 更新后做 1 次 student DMD 更新，而不是相反。

gradient accumulation 期间 `optimizer_step` 不增加，因此同一个 accumulation window 内不会切换 model/optimizer。

student 更新成功时推进 parent LR scheduler；fake-score 更新不推进 student scheduler。fake-score 当前使用独立 AdamW，但没有单独 scheduler。

## 5. 原始 batch 处理

数据读取、device transfer 和 latent materialization 与 consistency distillation 相同：

```text
MOT dataloader batch
  -> convert_input_format
  -> _materialize_batch_latents
  -> _prepare_joint_input_dict(add_noise=False)
```

蒸馏复用原生 MOT 的：

- view-aware dataloader 和 sampler；
- VAE streaming encode；
- text embedding 与 CFG dropout；
- geometry RGB/point/valid mask 组织；
- stream id；
- chunk/window metadata；
- `validate_mot_batch_for_forward()`；
- `ThreeDVAMOTTransformer3DModel(..., mode="train")`；
- attention metadata 和 FSDP layout。

`add_noise=False` 避免原生 input helper 先生成一套马上会被 SGF pipeline 覆盖的 noise、target 和 timestep。

## 6. V/A schema 与 mask

主要 dataclass：

```python
VAPrediction(
    video: Tensor[B,Cv,F,V,H,W],
    action: Tensor[B,Ca,F,N,1],
)

VAMasks(
    video: Tensor[B,F],
    action: Tensor[B,Ca,F,N,1],
)

ReplayContext(
    batch: dict,                    # 写回预测 V/A/G 的新 batch
    timesteps: VATimesteps,         # video/action 各自的 [B,F]
    noisy: VAPrediction,
    generated: VAPrediction,
    masks: VAMasks,
)
```

`VAMasks.frame_mask()` 使用 video frame mask 与 action frame 内任意有效 token 的并集。只有该并集为 True 的 frame 会采样非零 score timestep。

所有加噪操作都显式执行：

```text
masked target location -> use noisy sample
condition location     -> preserve exact clean sample
```

因此 condition frame 不依赖 scheduler timestep 0 的近似 sigma。

## 7. 两次 trajectory record

`generate_and_record_context()` 整体位于 `torch.no_grad()`。

原始 8-latent window 分成两个 4-latent chunk：

```text
history: H0 H1 H2 H3
target:  T0 T1 T2 T3
```

`H0` 和 `T0` 是 clean anchor；需要 replay 的位置是 `H1..H3` 与 `T1..T3`。

### 7.1 Record 1：history chunk

pipeline 构造一个仅含 history 的 4-frame 局部窗口，并把它解释为一个带首帧 anchor 的生成 chunk：

```text
H0 -> rollout H1,H2,H3 video latent
   -> VAE decode generated RGB
   -> recompute generated geometry
   -> rollout H1,H2,H3 action
```

该 rollout 使用原生 `run_mot_inference()` 的完整 V → G → A 顺序。video 和 action 各随机选择一个 denoising step，并记录该 step 的 noisy sample 与实际 scheduler timestep；rollout 本身继续运行到 final clean。

### 7.2 Record 2：target chunk

Record 1 的 final prediction 会写回完整 batch 的 history：

```text
predicted H0..H3 + clean T0
  -> rollout T1,T2,T3 video latent
  -> VAE decode generated RGB
  -> recompute generated geometry
  -> rollout T1,T2,T3 action
```

两次 record 共享同一个 video record-step 和同一个 action record-step，因此相同 modality 的两个 chunk 使用一致的采样位置。

### 7.3 构造 replay batch 与 mask

两次 record 完成后，pipeline 构造新的 full-window batch：

```text
latents      = [H0, pred H1..H3, T0, pred T1..T3]
actions      = [A0, pred AH1..AH3, A4, pred AT1..AT3]
geometry_rgb = anchor geometry + generated history/target geometry
```

replay mask 为：

```text
frame        H0 H1 H2 H3 T0 T1 T2 T3
video mask    0  1  1  1  0  1  1  1
action mask   0  valid action tokens   0  valid action tokens
```

mask 继续与原始 video/action valid mask 相交，所以 padding 和无效 action token 不参与 replay/DMD loss。

`ReplayContext.noisy` 在六个预测 frame 保存两次 rollout 对应 record-step 的原始 noisy sample；anchor/condition 位置保持 clean。`ReplayContext.generated` 保存 final rollout prediction。video/action timestep 分开保存，因为两个 sampler 可使用不同步数。

## 8. 当前 SGF 的准确语义

当前实现是两次完整 no-grad autoregressive record 加一次有梯度 full-window replay：

```text
record history H0 -> H1..H3 V/G/A
  -> write predicted history into full batch
  -> record target T0 -> T1..T3 V/G/A
  -> assemble predicted full-window batch and replay mask
  -> one gradient student forward(mode="train")
```

它已经做到：

- history 和 target 分别进行一次完整 V → G → A rollout；
- target record 真实依赖 predicted history；
- replay batch 使用 predicted latent/action/geometry，而不是 target GT geometry；
- 两次 record 无梯度，student replay 仅一次有梯度 forward；
- replay 使用 record 阶段保存的 noisy sample 和 modality-specific timestep；
- anchor、padding 和 invalid action token 不进入 loss；
- context 通过明确的 `ReplayContext` 传递。

## 9. Student replay loss

student update 首先把 context 重新装入 joint input：

```text
noisy stream = ReplayContext.noisy
clean stream = ReplayContext.generated
timestep     = ReplayContext.timesteps
geometry     = ReplayContext.batch 中的 generated geometry
```

原生 field builder 完成 shape/mask/geometry 校验后，trainer 把 attention metadata
覆盖为与 stage1/stage2 一致的 `generation_shape=1/16`。这不改变真实 8-frame、
16-action-token/frame 的物理 tensor shape。

student 有梯度地预测 `student_flow`。

replay target 由 noisy、generated clean 和 sigma 精确恢复：

```text
target_flow = (noisy - generated) / sigma
```

sigma 为 0 的 condition 位置 target flow 直接置零。随后使用相同的 V/A mask-aware MSE：

```text
replay_loss = weighted_mse(student_flow, target_flow)
```

该 loss 让 student 从 record-step noisy sample 回归两次完整 rollout 的 final clean prediction。

## 10. Student DMD loss

### 10.1 从 replay flow 得到 differentiable x0

```text
student_x0 = noisy - sigma_context * student_flow
```

该 tensor 在 replay mask 内保留 student graph。mask 外的 H0/T0、padding 和无效
action token 随后显式替换回 `ReplayContext.generated` 的 exact clean value。原因是
训练 scheduler 的 timestep 数值 0 可能对应很小但非零的最低 sigma；若直接套公式，
anchor 会被减去一小段 predicted flow，并作为下一次 score forward 的错误条件。

### 10.2 在 student x0 上重新采样 score noise

pipeline 重新采样 `score_t`，并得到：

```text
score_noisy = (1 - sigma_score) * student_x0 + sigma_score * noise
```

condition frame 仍保持 clean。

### 10.3 Real/fake score 估计

real-score 和 fake-score 都在 `torch.no_grad()` 中接收同一个 `score_input`：

```text
real_x0 = score_noisy - sigma_score * real_flow
fake_x0 = score_noisy - sigma_score * fake_flow
```

这两个模型只提供 stop-gradient target，不在 student update 中积累梯度。

### 10.4 Mask-aware DMD normalizer

video/action 分别计算：

```text
normalizer = mean(abs(student_x0.detach() - real_x0), valid masked elements)
normalizer = clamp_min(normalizer, dmd_normalizer_eps)
```

history、padding 或其他无效位置不参与 normalizer，避免它们稀释 DMD gradient。

DMD target：

```text
dmd_target = student_x0 - (fake_x0 - real_x0).detach() / normalizer
```

surrogate loss：

```text
dmd_loss = weighted_masked_mse(student_x0, dmd_target)
student_total = replay_loss + dmd_loss
```

因为 `dmd_target` 对 score models detach，梯度只回到 student replay graph。

## 11. Fake-score update

fake-score update 复用 no-grad context generation，但不运行 real-score：

1. 取 `ReplayContext.generated`，它已经 detach。
2. 重新采样 score timestep 和 noise。
3. 使用 generated V/A 和 generated geometry 构造 clean stream。
4. fake-score 有梯度地预测 flow。
5. 使用已知 synthetic pair 计算精确目标：

```text
target_flow = (noisy - generated) / sigma
```

6. 计算 V/A weighted masked flow MSE。
7. 只更新 fake-score optimizer。

这里不需要 real-score forward，因为对人工加噪的 generated sample，flow target 已知。删除 real-score forward 减少了一次大模型计算和对应显存占用。

## 12. 每种 optimizer 更新的 forward 数量

不计算 VAE materialization 时：

| 更新类型 | Student | Real-score | Fake-score |
| --- | ---: | ---: | ---: |
| fake-score update | 2 次 no-grad 完整 V/G/A record | 0 | 1 次有梯度 regression |
| student update | 2 次 no-grad 完整 V/G/A record + 1 次有梯度 replay | 1 次 no-grad | 1 次 no-grad |

所有 rank 由同一个 optimizer schedule 决定 forward 路径，避免 FSDP collective 次数不一致。

## 13. Backward、finite check 与梯度所有权

`OptimizationTarget` 明确当前：

```python
OptimizationTarget(
    name="student" | "fake_score",
    optimizer=...,
    model=...,
)
```

公共 trainer 只对 target model：

- 设置 FSDP gradient sync；
- 执行 `clip_grad_norm_(..., 2.0)`；
- 扫描 nonfinite gradient location；
- optimizer step；
- zero grad。

任一 rank 出现非有限 loss/grad 时，所有 rank 都跳过当前 optimizer step。fake-score 失败不会错误地只清 student optimizer；student 失败也不会更新 fake-score。

## 14. Checkpoint 与导出

Self-gradient-forcing DMD checkpoint 包含：

```text
checkpoint_step_N/
├── transformer/                 # student export
├── distributed_state/
│   ├── student model state
│   ├── student optimizer state
│   ├── fake-score model state
│   └── fake-score optimizer state
├── training_state.pt            # LR scheduler、step、RNG、update ratio
├── checkpoint_metadata.json     # MOT 初始化兼容字段 + SGF-DMD method/profile
└── _SUCCESS
```

real-score 不保存，因为它由 autoregressive checkpoint 确定且始终冻结。
metadata 继续包含 `checkpoint_type=mot_training`、VGGTO topology、
`optimization_composition=va` 和 `has_full_state=true`，所以最终 student export 可由
原生 MOT inference/stage loader 直接校验。

resume 恢复：

- student 和 student optimizer；
- fake-score 和 fake-score optimizer；
- student LR scheduler；
- optimizer step 与 fake/student schedule 相位；
- RNG、sampler offset、skip/nonfinite 计数；
- `fake_score_update_ratio`。

## 15. 关键配置

| 配置 | 默认值 | 作用 |
| --- | ---: | --- |
| `fake_score_update_ratio` | 4 | 每次 student update 前的 fake-score update 数 |
| `score_timestep_min` | 0 | score timestep 下界，包含 |
| `score_timestep_max` | 1000 | score timestep 上界，不包含，并裁剪到 scheduler 长度 |
| `dmd_normalizer_eps` | `1e-6` | DMD normalizer 下界 |
| `video_loss_weight` | 继承基础 config | video replay/DMD/fake-score 权重 |
| `action_loss_weight` | 继承基础 config | action replay/DMD/fake-score 权重 |
| `generation_profile` | `1/16` | 继续传递原生 chunk/window metadata |

## 16. 当前风险与后续扩展边界

1. 每个 optimizer microstep 都执行两次完整 V → G → A record，计算成本明显高于旧的 full-window approximation。
2. replay clean target 采用完整 rollout 的 final clean，而 noisy input 来自随机 record-step；这是为保证第二条轨迹使用完整 generated history/geometry 所做的明确选择。
3. generated RGB/G 在 record 与 replay 之间 detach；当前梯度只通过一次 student V/A replay forward 回传。
4. pipeline 每次 `_prepare_input()` 都重新构造 text/geometry input；这是为了保持原生 input contract，后续若优化应提取 `wan_va` 公共 context builder，而不是缓存或复制私有格式。
5. real/fake/student 使用同一 MOT 拓扑和 FSDP shard；checkpoint 来源必须拓扑兼容。
