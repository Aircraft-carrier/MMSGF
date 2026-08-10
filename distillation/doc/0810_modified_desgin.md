# Self-Gradient-Forcing DMD 修改设计（0810）

> 状态：设计稿，不包含代码实现。本文中的 `denoisy` 拼写遵循本次需求，避免在配置、运行时结构和日志中同时出现 `denoise` / `denoising` / `denoisy` 三套名称。

## 1. 设计依据与假设

### 1.1 需求来源与已经确定的决策

本设计以以下参考实现为依据：

- `/zsh/code/Self_Gradient_Forcing/model/base.py`
- `/zsh/code/Self_Gradient_Forcing/utils/wan_wrapper.py`
- `/zsh/code/Self_Gradient_Forcing/pipeline/self_gradient_forcing_training.py`
- `/zsh/code/Self_Gradient_Forcing/trainer/distillation.py`
- `/zsh/code/Self_Gradient_Forcing/model/dmd.py`

并以当前仓库下列真实调用链为修改基础：

- `distillation/configs/self_gradient_forcing_dmd.py`
- `distillation/model/dmd.py::SGFDMDModel`
- `distillation/pipeline/self_gradient_forcing_training.py::SelfGradientForcingTrainingPipeline`
- `distillation/trainer/self_gradient_forcing_dmd.py::SelfGradientForcingDMDTrainer`
- `distillation/model/objectives.py`
- `distillation/schema.py::ReplayContext`
- `distillation/self_rollout/engine.py::self_rollout`
- `distillation/self_rollout/recorder.py::SelfRolloutRecorder`
- `distillation/diffusion_utils.py`
- `wan_va/utils/scheduler.py::FlowMatchScheduler`
- `distillation/model/autoregressive_mot.py::AutoregressiveVAMOTTransformer3DModel`
- `wan_va/modules/model_va_mot.py::VAMOTTransformer3DModel`
- `wan_va/modules/mot_attention.py::_va_visibility`
- `distillation/checkpoint.py::DistillationCheckpointIO`

以下决定已经由需求明确：

1. 历史上下文始终使用每个生成位置完成全部 rollout 后得到的最终 clean `x0`。
2. 不使用随机 exit 的 `x0` 写历史 KV cache。
3. 不对 final-clean history/context 再加噪；clean stream 的 `cond_timesteps` 固定为 0，tensor 本身保持原值。
4. 删除 `replay_target_loss`，student 更新只保留 DMD surrogate loss。
5. video 与 action 不共享 `denoisy_step_list`。
6. video 与 action 分别采样独立的 `exit_id`。
7. `denoisy_step_list` 同时决定 exit 采样空间和 `x0 -> x_t(next)` rollout。
8. 每个模态的 `denoisy_from` / `denoisy_to` 由该模态的 `exit_id` 和列表相邻元素推导，只用于该模态 DMD/fake-score 加噪范围。
9. rollout 必须保留两种显式模式：`inference` 保持当前 `scheduler.step()` 轨迹，可直接用于正式推理；`sgf_renoise` 使用 `velocity -> x0 -> fresh noise -> next x_t`，供 SGF 训练构造上下文与 exit state。
10. 一个 rollout 调用从开始到结束只能使用一种 transition mode，禁止在同一条轨迹中混合两种更新公式。

### 1.2 术语与时间步语义

本文统一采用“实际送入模型并可由 scheduler 查到最近 sigma 的 timestep 值”，不再使用参考实现中 `1000 - scheduler index` 的间接表示：

```text
video_denoisy_step_list = [t_v0, t_v1, ..., t_v(Kv-1)]
action_denoisy_step_list = [t_a0, t_a1, ..., t_a(Ka-1)]

t_0 > t_1 > ... > t_last > 0

video_exit_id  ~ UniformInteger[0, Kv)
action_exit_id ~ UniformInteger[0, Ka)

denoisy_from = step_list[exit_id]
denoisy_to   = step_list[exit_id + 1]，若 exit_id 是最后一项则为 0
```

`denoisy_from` 表示区间的高噪声端，`denoisy_to` 表示低噪声端，必须满足 `0 <= denoisy_to < denoisy_from <= num_train_timesteps`。DMD timestep 在闭区间语义 `[denoisy_to, denoisy_from]` 内按模态独立采样；实现使用整数采样时需保证上界可达或明确采用半开区间等价写法。

### 1.3 已确认的当前实现事实

当前实现并非仅缺一个 wrapper，还存在以下设计差异：

1. `self_rollout` 当前使用 `scheduler.step(velocity, t, x_t)` 做 Euler/ODE 更新。这条路径是正确的正式推理模式，必须保留；缺失的是并列的 SGF re-noise 模式，即 `velocity -> x0` 后用新高斯噪声把 `x0` 加噪到下一 timestep。
2. 当前 `SelfRolloutRecorder` 已能为 video/action 记录不同 step，这一能力应保留；问题是 step 来源是两个 `num_steps`，而不是两个显式、可审计的 `denoisy_step_list`。
3. 当前 student loss 是 `replay_target_loss + dmd_loss`，额外 frozen-teacher flow MSE 改变了参考 SGF 的 generator 优化目标。
4. 当前 `fake_score_step()` 直接用 `context.pred_clean`（final rollout clean）训练 fake score，没有先运行 exit replay；它训练的分布与 student DMD 分支中的 generator sample 不是同一个分布。
5. 当前 real-score 由 `autoregressive=False` 构建，fake-score 却由默认 `autoregressive=True` 构建；二者使用不同模型类和 attention order，`fake_x0 - real_x0` 不是同构 score estimator 的可比差值。
6. 当前 DMD 只采样一份 `[B,F] score_t` 给 V/A，共享 nominal timestep；需求要求 V/A 根据各自 exit interval 分别采样。
7. 当前 score timestep 由全局 `score_timestep_min/max` 决定，与 rollout exit 没有关联；`denoisy_from/to` 没有成为 ReplayContext 的一等数据。
8. 当前 `_flow_targets()` 从 `(x_t - x0) / sigma` 反解 fake-score velocity，需要 `flow_target_eps`；加噪时本来就持有 Gaussian noise，直接使用 `noise - x0` 更精确且不需要除法。
9. 当前 `self_rollout()` 在未注入 generator 时每次都以相同 `config.seed` 新建 generator，因此不同 microstep 会重复同一套 rollout 高斯噪声。训练 RNG 应每步推进并能被现有 per-rank RNG checkpoint 恢复。
10. 当前 fresh fake-score 常从 stage1 AR export 初始化；参考 DMD 的 real/fake score 都是双向模型。fake-score 应从双向 `va_mot_v1` export 初始化，通常与 real-score 的初始 checkpoint 相同。
11. 当前 resume 路径可能把 stage3 `resume_from`（其公开 export 是 AR student）当作 fresh fake-score 初始化来源；正确做法是先用双向初始化 checkpoint 构造 fake-score，再由 DCP 覆盖 fake-score 和 optimizer 状态。
12. `ReplayContext.teacher_batch` / `teacher_clean` 仅服务于即将删除的 replay teacher loss，删除后应移除，避免误以为 DMD real-score 使用 GT future 作为上下文。
13. 原实现的三个 distillation config 索引不存在的 `VA_CONFIGS["umi_3dwam_train"]`。实现阶段统一改用当前 registry 唯一注册的 `wan22_train`，并由 `apply_distillation_runtime_overrides()` 注入 distillation dataset 路径与统计信息。

### 1.4 证据映射

```text
velocity -> x0
  -> 当前：pipeline._predict_x0() 局部实现，仅覆盖 full train forward
  -> 目标：model/wan_wrapper.py::WanDiffusionWrapper，覆盖 AR student 与双向 score

x0 -> next x_t rollout
  -> inference 模式：保留 self_rollout/engine.py 的 scheduler.step()
  -> sgf_renoise 模式：每步 x0 后 fresh Gaussian + next timestep add_noise
  -> mode 在一次 rollout 入口选择，轨迹内部不可切换

final-clean clean context
  -> 当前：engine commit 最终 scheduler sample，但没有明确 x0 契约
  -> 目标：rollout 最后一次 wrapper 输出 x0，原样 commit；禁止 context noise

exit 与 DMD 范围
  -> 当前：recorder step 与 score_timestep_min/max 相互独立
  -> 目标：V/A 独立 exit_id -> V/A 独立 denoisy_from/to -> V/A 独立 DMD t

student loss
  -> 当前：replay_target_loss + dmd_surrogate_loss
  -> 目标：只保留 dmd_surrogate_loss

fake sample
  -> 当前：final rollout clean
  -> 目标：同一 ReplayContext 上的 no-grad exit replay x0

score architecture
  -> 当前：real=bidir，fake=AR
  -> 目标：real/fake 均为 bidir、相同 mask、相同 noisy/context 条件
```

## 2. 核心目标

### 2.1 研究与功能目标

在现有 Video+Action MOT 模型、FSDP trainer、增量 KV cache 和 checkpoint 体系内，实现与本需求一致的 SGF-DMD：

```text
no-grad autoregressive rollout(mode="sgf_renoise")
  -> V/A 分别记录各自随机 exit 的真实 x_t
  -> 所有历史 cache 只提交最终 x0，且不加 context noise
  -> 一次 joint replay 恢复 V/A exit x0（student step 有梯度，fake step 无梯度）
  -> V/A 分别在自己的 [denoisy_to, denoisy_from] 内重新加 score noise
  -> 同构 bidirectional real/fake score 估计 DMD direction
  -> student 只优化 DMD surrogate；fake-score 优化 exact velocity regression
```

### 2.2 可观察行为

实现后应能从日志、测试或 ReplayContext 观察到：

- 每个 microstep 有独立的 `video_exit_id` 和 `action_exit_id`，可相同但不强制相同。
- 两个 exit id 分别受各自列表长度约束。
- `self_rollout` 默认/显式 `inference` 模式继续逐步调用 native `scheduler.step()`，无需 SGF recorder 或 re-noise schedule 即可直接服务推理。
- SGF pipeline 显式传入 `transition_mode="sgf_renoise"`；该模式不会被推理入口误选为默认值。
- 在 `sgf_renoise` mode 中，每个生成 frame 的 video/action history cache 值均等于该模态 rollout 最后一次预测的 `x0`；`inference` mode 则保持提交 native scheduler 的最终 sample。
- exit 仅决定 replay 输入 `x_t/t`，不会导致 rollout 提前退出。
- final-clean context 在进入 replay/score model 前未经过 `add_noise`。
- student step 的总 loss 等于加权 V/A DMD loss，不再出现 replay loss metric。
- fake-score step 的 clean sample 来自 no-grad exit replay x0，而不是 final-clean context。
- real/fake score 模型都是 `VAMOTTransformer3DModel`（双向），且读取相同 `score_noisy`、timestep、clean history、text 和 mask。
- video/action DMD timestep 分别落在各自 `denoisy_to/from` 区间。
- wrapper 对手算例 `x_t=5, sigma=.75, velocity=4` 返回 `x0=2`。

### 2.3 非目标

- 不重新设计 stage1 autoregressive training 或 stage2 consistency distillation loss。
- 不给 clean history 增加 random-noise/context-noise/robust teacher forcing 模式。
- 不保留 `exit` cache mode、`last_step_only` 或共享 V/A exit 的兼容开关。
- 不把两种 rollout transition 做成逐 step 随机策略，也不允许一次 rollout 中途切换 mode。
- 不改变 Video→Action 的生成顺序、segmented generation profile 或现有 KV cache 可见性规则。
- 不引入随机生成长度、随机 rollout horizon 或 GT replacement。
- 不修改 DMD normalizer 的现有“按样本、按模态、只统计 mask”改进。
- 不改变现有 fake-score/student optimizer 轮换和 gradient accumulation 边界。

### 2.4 验收标准

1. 两套不同长度、不同值的 V/A `denoisy_step_list` 能通过配置和 launcher 进入 pipeline。
2. `inference` rollout 与修改前的 scheduler/Euler trajectory 等价，并能在没有 SGF 专用参数时直接运行。
3. `sgf_renoise` rollout 对每个相邻 step 执行 `velocity -> x0 -> fresh-noise add_noise(next_t)`，最后直接返回 x0。
4. 单个 joint replay 支持 V/A 不同 timestep，并分别转换为 x0。
5. `ReplayContext` 显式携带 V/A exit id、from/to、exit noisy state 和 final-clean context。
6. `replay_target_loss` 不再被定义、导出、调用或记录。
7. student step 只有 student 参数获得梯度；real/fake score 无梯度。
8. fake-score step 只有 fake-score 参数获得梯度；student replay 在 `no_grad` 中。
9. real/fake score 模型类与 attention profile 相同。
10. resume 后 V/A schedule、optimizer 轮换、随机采样和 fake-score DCP 状态一致恢复。
11. 单元测试与最小 GPU smoke test loss 有限，完成至少一次 fake-score update 和一次 student update。

## 3. 目标目录结构

```text
MMSGF/
  distillation/
    configs/
      runtime_dataset.py                  [MODIFY][CONFIG] 向 wan22 base 注入 distillation dataset 路径/统计
      self_gradient_forcing_dmd.py       [MODIFY][CONFIG] 显式 V/A denoisy lists，删除旧 score/rollout step 配置
    doc/
      0810_modified_desgin.md             [MODIFY][DOC] 本设计
    model/
      __init__.py                         [MODIFY] 导出 wrapper，移除 replay loss 导出
      dmd.py                              [MODIFY] 构造三类 wrapper、双向 fake-score、保存 schedule contract
      objectives.py                       [MODIFY] 删除 replay_target_loss，保留 DMD/fake-score loss
      wan_wrapper.py                      [ADD] 统一 velocity/x0 的 V/A scheduler-aware adapter
    pipeline/
      self_gradient_forcing_training.py   [MODIFY] final-clean rollout、exit replay、V/A 独立 DMD；移除 replay teacher
    self_rollout/
      __init__.py                         [MODIFY] 导出 SGF schedule/record contract
      engine.py                           [MODIFY] 保留 inference transition，新增并列 SGF re-noise transition
      recorder.py                         [MODIFY] 分别记录 V/A exit_id、from/to 与真实 exit x_t
      transitions.py                      [MODIFY] 定义两种 mode，并构造/校验 SGF rollout schedule
    diffusion_utils.py                    [MODIFY] 增加 x0 re-noise 和 interval timestep 采样纯函数
    schema.py                             [MODIFY] 增加 diffusion output、V/A schedule selection，精简 ReplayContext
    train.py                              [MODIFY][CONFIG] V/A list CLI 解析与传播
    trainer/
      self_gradient_forcing_dmd.py        [MODIFY] 正确选择双向 fake init，保持 optimizer ownership
    workflow.py                           [MODIFY] stage3 fake-score 从双向来源初始化
    tests/
      test_wan_diffusion_wrapper.py       [ADD][TEST] velocity/x0、shape、mask 与 dtype 测试
      test_rollout_transition_modes.py    [ADD][TEST] inference 等价性、SGF re-noise 与 mode 隔离测试
      test_self_gradient_forcing_dmd.py   [ADD][TEST] student/fake 梯度、DMD interval、loss 组成测试
  1shell/
    distill/
      _train_distill_common.sh            [MODIFY][SCRIPT] 透传 V/A denoisy list 环境变量
      train_distill_self_gradient_forcing_dmd_4gpu.sh
                                           [EXISTING CONTEXT] stage3 launcher 与 checkpoint 来源检查
```

不修改 `distillation/checkpoint.py`：现有 DCP 已能保存 raw student、student optimizer、fake-score 和 fake-score optimizer。schedule 兼容信息由 `SGFDMDModel.state_dict()` 写入现有 method state。

## 4. 文件级设计

### 4.1 `distillation/model/wan_wrapper.py` `[ADD]`

**总体职责**：为同一 V/A flow-matching 参数化提供唯一的 `velocity -> x0`、模型调用和输出规范化边界。

**新增原因**：当前 conversion 散落在 pipeline，增量 AR rollout 又直接使用 raw velocity，导致 full replay 与 rollout 容易采用不同公式。

**计划内容**：新增非 `nn.Module` 的 `WanDiffusionWrapper`。它借用已经由 trainer/FSDP 拥有的模型，不注册或复制参数，不改变 checkpoint key。wrapper 持有 video/action scheduler 引用，提供 joint full-forward 适配和单模态 conversion。

**契约**：

- caller：SGF pipeline、self-rollout engine。
- callee：AR student 或 bidirectional score 的 `model(input_dict, mode="train")`，以及 `distillation.diffusion_utils.flow_to_x0()`。
- wrapper 不调用 `.to()`、`.train()`、`.eval()`，不拥有 optimizer/state_dict。
- conversion 在输入 dtype/device 上返回；sigma lookup 允许 `[B,F]`。
- V/A shape 分别为 `[B,Cv,F,V,H,W]` 与 `[B,Ca,F,N,1]`。

**验证**：`distillation/tests/test_wan_diffusion_wrapper.py`。

### 4.2 `distillation/schema.py` `[MODIFY]`

**总体职责**：定义 rollout、replay、DMD 之间不可歧义的数据边界。

**已有锚点**：`VAPrediction`、`VATimesteps`、`VAMasks`、`ReplayContext`。

**计划内容**：

- 新增 `VADiffusionOutput(velocity, x0)`。
- 新增 `DenoisyInterval(exit_id, denoisy_from, denoisy_to)`。
- 新增 `VADenoisySelection(video, action)`。
- 修改 `ReplayContext`，删除 `teacher_batch` 与 `teacher_clean`；将 `pred_clean` 明确重命名为 `final_clean_context`；新增 `denoisy_selection`。
- 保留兼容 property 只限一个迁移版本时，应标明 deprecated；推荐直接更新当前唯一 caller，不继续保留语义含混的 `generated` alias。

**序列化**：ReplayContext 不进入 checkpoint；`VADenoisySelection` 是单 microstep runtime state。

**验证**：pipeline tests 检查 shape、范围和 frozen dataclass 不可变性。

### 4.3 `distillation/diffusion_utils.py` `[MODIFY]`

**总体职责**：继续作为支持 `[B,F]` timestep 的纯 tensor scheduler 层。

**已有锚点**：`sigmas_for_timesteps()`、`broadcast_frame_values()`、`add_noise()`、`flow_to_x0()`。

**计划内容**：

- 新增 `renoise_x0(x0, next_timestep, scheduler, *, generator=None)`，采样 fresh Gaussian 并调用 `add_noise()`。
- 新增 `sample_interval_timesteps(interval, shape, device, mask)`，在 from/to 区间按 frame 采样并对无效位置写 0。
- 不用 native `FlowMatchScheduler.add_noise()`，因为其 `t_dim`/CPU timestep 行为不适合通用 `[B,F]` V/A tensor；继续复用本文件逐 frame sigma lookup。

**验证**：固定 generator 的手算与边界测试；确保相邻两次调用消费不同随机数。

### 4.4 `distillation/self_rollout/transitions.py` `[MODIFY]`

**总体职责**：定义 rollout transition mode，保留 inference 使用的 `RolloutSchedulers`，并增加 SGF 专用显式 schedule contract。

**计划内容**：

- 新增 `RolloutTransitionMode = Literal["inference", "sgf_renoise"]`（或等价 `StrEnum`）。
- `inference` 使用现有 `FlowMatchScheduler.step()`；`sgf_renoise` 使用 wrapper conversion 与 `renoise_x0()`。
- 新增 `SGFRolloutSchedule(video_steps, action_steps, video_scheduler, action_scheduler)`。
- 新增 `build_sgf_rollout_schedule(config, video_scheduler, action_scheduler)`。
- V/A list 分别校验：长度至少 1、严格递减、元素 finite、`0 < t <= num_train_timesteps`。
- 不在 builder 内按 `num_steps` 重新生成 timestep；配置 list 是唯一真值来源。

**兼容性**：`build_rollout_schedulers()` 和 inference 的 native scheduler trajectory 行为不变；新增 mode 不能改变现有推理调用的默认结果。

### 4.5 `distillation/self_rollout/recorder.py` `[MODIFY]`

**总体职责**：记录每个模态的独立 exit input，而不是含糊的 “record step”。

**已有锚点**：`RecordedDenoiseState`、`SelfRolloutRecorder`。

**计划内容**：

- `video_step/action_step` 重命名为 `video_exit_id/action_exit_id`。
- `RecordedDenoiseState` 增加 `exit_id`、`denoisy_from`、`denoisy_to`。
- `observe()` 在命中对应 exit id 时保存进入模型前的真实 `x_t`，不能保存 x0 或 transition 后 sample。
- `validate()` 分别按 V/A list 长度检查。

**验证**：不同长度 lists 下，两个模态记录不同 exit id 和 bounds。

### 4.6 `distillation/self_rollout/engine.py` `[MODIFY]`

**总体职责**：保留现有 history/anchor commit、Video→Action、KV transaction 和异常回滚；在统一入口明确分派 inference 与 SGF 两种单帧 sampler transition。

**已有锚点**：`self_rollout()`、内部 `sample_video()` / `sample_action()`、`commit_video()` / `commit_action()`。

**计划内容**：

- `self_rollout()` 新增显式 `transition_mode: RolloutTransitionMode = "inference"`。
- `inference` mode 沿用当前 Euler/native scheduler path：遍历 `RolloutSchedulers.*.timesteps`，每步调用 `scheduler.step(velocity, timestep, sample)`，最终 sample 可直接作为推理结果和 clean cache commit。
- `sgf_renoise` mode 要求同时提供 `sgf_schedule`、`diffusion_wrapper` 和 recorder；遍历对应显式 list，每步先 recorder.observe，再调用 AR model得到 velocity，再由 wrapper 转成 x0。
- `sgf_renoise` 非最后一步用 fresh Gaussian 把 x0 加噪到对应模态下一 timestep；最后一步直接返回 x0。
- `inference` mode 禁止传入 `sgf_schedule`；`sgf_renoise` 缺少 schedule/wrapper/recorder 时 fail fast，杜绝根据可选参数静默猜 mode。
- 两种 mode 共享 cache commit、Video→Action 顺序和 `RolloutResult`；只替换 `sample_video()` / `sample_action()` 内部 transition。
- SGF mode commit 的只能是最后一步 x0，不允许 commit exit x0 或 re-noised sample；inference mode commit native scheduler 的最终 sample。
- 不允许 SGF rollout 启用 GT replacement；stage3 固定 `ground_truth_provider=None`。
- 修复每个 call 重新使用相同 seed 的问题：SGF 随机数使用 trainer 已管理的 rank-local torch RNG，或者用该 RNG 每次生成新 seed 后建立局部 generator；不能每次 `manual_seed(config.seed)`。

**不改变**：inference 的 scheduler 数值路径；history/anchor 为 GT；V/A cache source/version；rollout horizon；异常恢复。

### 4.7 `distillation/model/objectives.py` `[MODIFY]`

**总体职责**：只保存被实际训练路径消费的纯 loss helper。

**计划内容**：

- 删除 `replay_target_loss()`。
- 保留 `_video_mse()` / `_action_mse()` 的 frame/token 归一方式。
- 保留 `dmd_surrogate_loss()` 当前按模态、按有效 mask 计算 normalizer 的设计；它比参考实现把所有维度统一求均值更适合 V/A 不同尺度。
- 保留 `fake_score_flow_loss()`。

**验证**：DMD direction 手算、全 False mask、normalizer eps、V/A 权重测试。

### 4.8 `distillation/pipeline/self_gradient_forcing_training.py` `[MODIFY]`

**总体职责**：拥有一个 microstep 的 SGF trajectory、exit replay、DMD score 和 fake-score regression 数据流。

**删除**：

- `_teacher_cfg_flow()` 作为 replay target 的用法；保留/重命名为 `_real_score_cfg_velocity()`，仅服务 DMD real-score。
- `replay_target_loss` import/call/metric。
- `teacher_batch`、`teacher_clean` 构造。
- `score_t_min/max`。
- `rollout_video_num_steps/action_num_steps`。
- `_flow_targets()` 的 sigma 除法反解。

**新增或修改**：

- 构造 student/real/fake 三个 `WanDiffusionWrapper`，其中 scheduler 引用相同，模型对象不同。
- 从 config 读取两套 list 并构建 SGF schedule。
- `_sample_exit_ids()` 分别按 V/A list 长度采样。
- `generate_and_record_context()` 完整跑完两个模态 lists，构造 final-clean context、exit noisy state 和 V/A bounds。
- `_replay_student_x0(context, *, requires_grad)` 成为 student/fake 两条路径共享的 generator sample 定义。
- `_sample_dmd_timesteps(selection, masks)` 返回 V/A 独立 timestep。
- `_add_dmd_noise()` 返回 `(noisy, sampled_noise)`，供 fake target 直接使用 `noise - x0`。
- `_predict_score_x0()` 通过 wrapper 统一把 real/fake velocity 转 x0。
- `replay_and_score()` 只返回 DMD loss。
- `fake_score_step()` 先 no-grad replay，再训练 fake-score。

**梯度边界**：

```text
student optimizer step:
  no_grad final-clean rollout
  -> grad-enabled one-pass student replay -> student_x0
  -> no_grad real/fake score x0
  -> DMD surrogate MSE -> student only

fake-score optimizer step:
  no_grad final-clean rollout
  -> no_grad one-pass student replay -> generated_x0.detach()
  -> add noise + exact velocity target
  -> grad-enabled fake-score forward -> fake-score only
```

### 4.9 `distillation/model/dmd.py` `[MODIFY]`

**总体职责**：拥有 stage3 student/real-score/fake-score 对象、wrapper/pipeline 生命周期和 optimizer 路由。

**计划内容**：

- real-score 继续 `autoregressive=False`、frozen、保留源 `wan_va` mask。
- fake-score 改为 `autoregressive=False`，且 `install_distillation_profile=False`、`validate_distillation_profile=False`；其模型类和 mask 必须与 real-score 相同。
- student 仍为 stage2 EMA export 的 AR model。
- wrapper 不取代这些模型的 state/parameter ownership。
- `state_dict()` 增加两套 denoisy lists；resume 时与当前 config 比较，不相同则 fail fast。

**验证**：construction test 断言 real/fake 类型相同、student 类型不同、fake trainable、real frozen。

### 4.10 `distillation/trainer/self_gradient_forcing_dmd.py` `[MODIFY]`

**总体职责**：保持双 optimizer 和 accumulation 选择，修正 fake-score 初始化来源。

**计划内容**：

- fresh run 要求 `student_init`、`real_score_checkpoint`；`fake_score_init` 若为空，默认使用 `real_score_checkpoint`。
- resume 时不能再用 `resume_from` 的 AR export 构造 fake-score；仍以显式 `fake_score_init` 或 `real_score_checkpoint` 构造双向 skeleton，随后 DCP restore。
- fake-score optimizer 继续只遍历 `method_model.fake_score` trainable 参数。
- student/fake update ratio与 `_optimization_target()` 不变。

### 4.11 `distillation/configs/self_gradient_forcing_dmd.py` `[MODIFY][CONFIG]`

**计划内容**：用结构化 list 替代旧字段：

```python
distill.denoisy_step_list = EasyDict(
    # Proposed defaults preserve the approximate two-step schedules implied by
    # the current video/action shifts; final experiment values remain explicit.
    video=[1000, 833],
    action=[1000, 500],
)
```

删除：

- `rollout_video_num_steps`
- `rollout_action_num_steps`
- `score_timestep_min`
- `score_timestep_max`
- `flow_target_eps`

保留：generation profile、horizon、masked attention backend、real CFG 范围、DMD normalizer、optimizer ratio 和 checkpoint paths。

默认值 `[1000,833]` / `[1000,500]` 是根据当前 2-step、`snr_shift=5` / `action_snr_shift=1` 的近似显式化，不是参考仓库 `[1000,750,500,250]` 的机械复制。实际实验若选择其他列表，应直接修改两套显式值，不再通过 `num_steps` 隐式推导。

### 4.11a `distillation/configs/runtime_dataset.py` `[MODIFY][CONFIG]`

`wan22_train` 的 dataset 字段默认留空；`apply_distillation_runtime_overrides()` 从准备后的 `MOT_DATASET_ROOT/meta/mot_config.json` 注入 manifest、empty/text embedding cache、action cache 与 normalization statistics，使三个 distillation method 在同一 base config 上获得真实数据路径。

### 4.12 `distillation/train.py` 与 shell `[MODIFY][CONFIG][SCRIPT]`

**计划内容**：

- 新增 `--video-denoisy-step-list`、`--action-denoisy-step-list`，格式为逗号分隔整数。
- `_train_distill_common.sh` 分别从 `DISTILL_VIDEO_DENOISY_STEP_LIST`、`DISTILL_ACTION_DENOISY_STEP_LIST` 透传。
- parser 层完成空值、整数和重复分隔符检查；严格递减/范围校验仍由 schedule builder 执行。
- 删除 SGF 对 `--rollout-video-num-steps` / `--rollout-action-num-steps` 的消费；这些 flag 仍可供 stage2 consistency 使用。

### 4.13 `distillation/workflow.py` `[MODIFY]`

**计划内容**：stage3 的 `fake_score_init` 不再指向 autoregressive checkpoint，而指向初始双向 teacher checkpoint（与 `real_score_checkpoint` 相同，除非调用方提供另一份同构双向 checkpoint）。stage1/stage2 顺序与 student 初始化不变。

### 4.14 测试文件 `[ADD][TEST]`

- `test_wan_diffusion_wrapper.py`：conversion 数学、V/A shape、不同 scheduler sigma、dtype/device、condition mask。
- `test_rollout_transition_modes.py`：inference mode 与现有 scheduler path 等价；SGF mode 支持显式不同 lists、独立 exit、每步 fresh noise、最后 x0 commit、无 context noise、RNG 不重复；非法 mode/参数组合 fail fast。
- `test_self_gradient_forcing_dmd.py`：ReplayContext、DMD interval、loss 组成、real/fake 同构、student/fake 梯度隔离、resume schedule mismatch。

## 5. 类级设计

### 5.1 `WanDiffusionWrapper` `[ADD]`

新类不继承 `nn.Module`，这是有意设计：底层模型已经由 trainer/FSDP 拥有；再次作为子模块注册会改变 state_dict key、FSDP root 和 optimizer 参数路径。

```python
class WanDiffusionWrapper:
    # Status/Location:
    #   [ADD] distillation/model/wan_wrapper.py
    # Responsibility:
    #   统一调用 Wan V/A diffusion backbone，并把 velocity 输出转为 x0；
    #   不拥有模型参数、optimizer、checkpoint 或 rollout cache。
    # Why A Class:
    #   三个模型共享两套 scheduler 和一致 conversion invariant；这些借用资源
    #   跨多次调用存在，且需要防止 pipeline/rollout 各自实现公式。
    # Construction Inputs:
    #   model: 已配置/FSDP 化的 AR 或 bidirectional model，borrowed。
    #   video_scheduler/action_scheduler: trainer 拥有的 1000-step scheduler，borrowed。
    # Owned State:
    #   仅 Python 引用；无 Tensor parameter/buffer，无 state_dict。
    # Collaborators:
    #   model(..., mode="train")；distillation.diffusion_utils.flow_to_x0。
    # Lifecycle:
    #   SGFDMDModel 构造 -> pipeline 多次调用 -> 随 model/trainer 释放。
    # Invariants:
    #   velocity/noisy/timestep 的 B/F 对齐；V/A 使用各自 scheduler。
    # Public Interface:
    #   predict_joint(...), velocity_to_x0(...), video_x0(...), action_x0(...)
    # Runtime Boundaries:
    #   不主动改变 grad/no_grad/autocast/model mode；完全继承 caller context。
    # Persistence:
    #   无。
    # Tests:
    #   conversion、borrowed ownership、FSDP-compatible call boundary。
    pass
```

### 5.2 `DenoisyInterval` / `VADenoisySelection` `[ADD]`

```python
@dataclass(frozen=True, slots=True)
class DenoisyInterval:
    # Status/Location: [ADD] distillation/schema.py
    # Responsibility: 一个模态一次 microstep 的 exit 与 DMD interval。
    # Owned State:
    #   exit_id: Python int；denoisy_from/to: float 或 int actual timestep。
    # Invariants:
    #   exit_id >= 0；0 <= to < from；from 是 list[exit_id]。
    # Persistence:
    #   runtime only；schedule list 本身进入 method checkpoint state。
    exit_id: int
    denoisy_from: float
    denoisy_to: float


@dataclass(frozen=True, slots=True)
class VADenoisySelection:
    # Responsibility: 显式阻止 V/A exit/bounds 被误合并为一个标量。
    video: DenoisyInterval
    action: DenoisyInterval
```

### 5.3 `ReplayContext` `[MODIFY]`

```python
@dataclass(frozen=True, slots=True)
class ReplayContext:
    # Current:
    #   student_batch, teacher_batch, rollout_timesteps, rollout_noisy,
    #   pred_clean, teacher_clean, masks
    # Proposed:
    #   replay_batch, rollout_timesteps, rollout_noisy,
    #   final_clean_context, masks, denoisy_selection
    # Responsibility:
    #   no-grad rollout 与一次 joint exit replay 之间的完整 detached 边界。
    # Invariants:
    #   rollout_noisy 只在 masks 内为 exit x_t；其余位置等于 final clean；
    #   rollout_timesteps V/A 各自等于对应 exit from timestep；
    #   final_clean_context 全部 detach，且从未被 context-noise 处理；
    #   selection 与 recorder 内容一致。
    # Gradient:
    #   所有字段 detached；student graph 只在 replay 时重建。
    pass
```

### 5.4 `SelfRolloutRecorder` `[MODIFY]`

```python
class SelfRolloutRecorder:
    # Current declaration:
    #   SelfRolloutRecorder(video_step: int, action_step: int)
    # Proposed declaration:
    #   SelfRolloutRecorder(video: DenoisyInterval, action: DenoisyInterval)
    # Responsibility:
    #   在完整 rollout 中捕获 V/A 各自 exit forward 的输入 x_t/t。
    # Owned State:
    #   两个 selection；frame_id -> RecordedDenoiseState dictionaries。
    # Invariants:
    #   每 frame/模态恰好记录一次；从不让 exit 中止后续 rollout。
    # Runtime Boundaries:
    #   observe 中 clone+detach；无梯度、无跨 rank broadcast。
    pass
```

### 5.5 `SelfGradientForcingTrainingPipeline` `[MODIFY]`

```python
class SelfGradientForcingTrainingPipeline:
    # Responsibility:
    #   协调 final-clean rollout、独立 exit replay、DMD 与 fake regression。
    # Owned State:
    #   config/trainer/device；三个 borrowed model；三个 wrappers；
    #   V/A step lists；loss weights；CFG/normalizer；rollout horizon。
    # Collaborators:
    #   self_rollout、WanDiffusionWrapper、objectives、scheduler helpers。
    # Lifecycle:
    #   SGFDMDModel 构造一次 -> 每 microstep compute -> trainer 释放。
    # Invariants:
    #   rollout 固定显式选择 sgf_renoise，不能继承 inference 默认值；
    #   clean context=final x0/no noise；V/A exit 独立；real/fake input 相同；
    #   student total loss 只有 DMD。
    # Runtime Boundaries:
    #   generate always no_grad；replay grad 由 optimizer target 决定；
    #   real-score always frozen/no_grad；fake score 仅 fake step 建 graph。
    # Persistence:
    #   schedule values 由 SGFDMDModel method state 校验，不在 pipeline 单独保存。
    pass
```

### 5.6 `SGFDMDModel` `[MODIFY]`

```python
class SGFDMDModel:
    # Responsibility:
    #   拥有三模型角色、双向 fake-score lifecycle、pipeline 与 update schedule。
    # Owned State:
    #   student: borrowed trainer.transformer；real_score: frozen bidirectional；
    #   fake_score: trainable bidirectional；pipeline；DMDUpdateSchedule。
    # Invariants:
    #   type(real_score) is type(fake_score)；二者 attention profile 相同；
    #   student 是 AR generation profile；real 永不进入 optimizer。
    # Persistence:
    #   DCP 保存 student/fake parameters+optimizers；method state 保存 update ratio
    #   和两套 denoisy lists，用于 resume compatibility check。
    pass
```

## 6. 函数级设计

### 6.1 Wrapper API

```python
def velocity_to_x0(
    self,
    velocity: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    modality: Literal["video", "action"],
) -> torch.Tensor:
    # Status: [ADD] WanDiffusionWrapper
    # Inputs:
    #   velocity/noisy: 同形；video [B,Cv,F,V,H,W] 或 action [B,Ca,F,N,1]。
    #   timesteps: [B,F] 或可规范化为 [B,F]；actual scheduler timestep。
    # Outputs:
    #   x0: 同 shape/device/dtype，x0 = x_t - sigma(t) * velocity。
    # Behavior:
    #   选择模态 scheduler -> per-frame sigma lookup -> broadcast -> conversion。
    # Runtime Boundaries:
    #   保留 autograd；不 detach；sigma 转到 noisy dtype/device。
    # Validation:
    #   B/F、dtype floating、finite timestep、modality。
    pass


def predict_joint(
    self,
    input_dict: dict,
    noisy: VAPrediction,
    timesteps: VATimesteps,
) -> VADiffusionOutput:
    # Status: [ADD]
    # Behavior:
    #   model(input_dict, mode="train") -> V/A velocity -> 各自 velocity_to_x0。
    # Gradient:
    #   完全由 caller 的 grad context 决定。
    pass
```

### 6.2 Schedule helpers

```python
def renoise_x0(
    x0: torch.Tensor,
    next_timesteps: torch.Tensor,
    scheduler: FlowMatchScheduler,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Status: [ADD] distillation/diffusion_utils.py
    # Outputs:
    #   x_next: (1-sigma_next)*x0 + sigma_next*noise。
    #   noise: 本次 fresh Gaussian，供测试/诊断；rollout caller可忽略。
    # Behavior:
    #   每次调用重新 torch.randn；不得跨 denoisy step 复用同一个 noise。
    # Runtime:
    #   caller 的 no_grad；不修改 scheduler table。
    pass


def sample_interval_timesteps(
    interval: DenoisyInterval,
    shape: tuple[int, int],
    device: torch.device,
    mask: torch.Tensor,
) -> torch.Tensor:
    # Status: [ADD]
    # Inputs:
    #   shape=[B,F]；mask=[B,F]。
    # Outputs:
    #   [B,F]，有效位置在 [to,from]，无效位置为 0。
    # Behavior:
    #   按位置独立采样；video/action 分开调用，不共享随机 tensor。
    pass
```

### 6.3 Exit 采样

```python
def _sample_exit_ids(self) -> VADenoisySelection:
    # Status: [ADD] SelfGradientForcingTrainingPipeline
    # Reads:
    #   self.video_denoisy_steps / self.action_denoisy_steps。
    # Behavior:
    #   1. video_exit_id ~ randint(len(video_steps))。
    #   2. action_exit_id ~ 独立 randint(len(action_steps))。
    #   3. 分别由当前/后一项推导 from/to；最后一项 to=0。
    # Distributed:
    #   每 rank 可不同；所有 rank 都完整遍历固定长度列表，因此 collective 图不变。
    # Tests:
    #   monkeypatch randint 强制 V/A 不同；边界 exit；长度不同。
    pass
```

### 6.4 两种 rollout transition 伪代码

公共入口显式分派 mode，不根据 `sgf_schedule is None` 猜测算法：

```python
def self_rollout(
    batch: dict[str, Any],
    *,
    transformer: nn.Module,
    config: Any,
    spec: Any,
    device: torch.device,
    empty_text_emb: torch.Tensor,
    video_num_steps: int | None = None,
    action_num_steps: int | None = None,
    transition_mode: RolloutTransitionMode = "inference",
    schedulers: RolloutSchedulers | None = None,
    sgf_schedule: SGFRolloutSchedule | None = None,
    diffusion_wrapper: WanDiffusionWrapper | None = None,
    recorder: SelfRolloutRecorder | None = None,
    **existing_rollout_options,
) -> RolloutResult:
    # Status: [MODIFY] distillation/self_rollout/engine.py
    # Mode contracts:
    #   inference: video/action num_steps 或 RolloutSchedulers；不得要求 SGF 参数。
    #   sgf_renoise: SGFRolloutSchedule + wrapper + recorder；不读取 num_steps。
    # Outputs:
    #   两种 mode 均返回同一 RolloutResult schema，便于推理和训练复用下游。
    # Validation:
    #   未知 mode、缺失 required 参数、传入另一 mode 专属参数均立即报错。
    # Runtime:
    #   整个调用 no_grad；mode 在入口规范化后不可改变。
    pass
```

```python
def sample_stream(
    sample,
    *,
    transition_mode: Literal["inference", "sgf_renoise"],
    inference_scheduler=None,
    sgf_schedule=None,
    interval=None,
    recorder=None,
):
    if transition_mode == "inference":
        require(inference_scheduler is not None)
        reject(sgf_schedule, interval, recorder)
        return sample_inference_stream(sample, inference_scheduler)

    if transition_mode == "sgf_renoise":
        require(sgf_schedule, interval, recorder, wrapper)
        return sample_sgf_stream(sample, sgf_schedule, interval)

    raise ValueError(f"unsupported rollout transition mode: {transition_mode}")
```

#### 6.4.1 Inference 模式（保留当前路径）

```python
def sample_inference_stream(sample, scheduler, predict_velocity):
    for timestep in scheduler.timesteps:
        velocity = predict_velocity(sample, timestep)
        sample = scheduler.step(velocity, timestep, sample)
    return sample  # 可直接作为正式推理输出并 commit
```

此函数必须与修改前 `sample_video()` / `sample_action()` 的数值顺序、CFG 位置、scheduler 调用次数一致。它不执行 velocity→x0 conversion，也不采样 step 间 fresh noise。

#### 6.4.2 SGF re-noise 模式

```python
def sample_sgf_stream(sample, steps, interval, scheduler, predict_velocity):
    # sample 初始为 N(0,1)，shape 保持一个 logical frame。
    final_x0 = None
    for step_id, timestep in enumerate(steps):
        if step_id == interval.exit_id:
            recorder.observe(
                sample=sample,                 # 进入模型前的真实 x_t
                timestep=timestep,
                exit_id=step_id,
                denoisy_from=interval.denoisy_from,
                denoisy_to=interval.denoisy_to,
            )

        velocity = predict_velocity(sample, timestep)
        x0 = wrapper.velocity_to_x0(
            velocity, sample, timestep, modality=modality
        )

        if step_id + 1 < len(steps):
            next_t = steps[step_id + 1]
            sample, _fresh_noise = renoise_x0(x0, next_t, scheduler)
        else:
            final_x0 = x0

    assert final_x0 is not None
    return final_x0                         # 原样写 clean KV cache
```

关键点：即使 `exit_id=0`，循环仍执行到最后；exit 只负责 recorder，不负责 break。

### 6.5 `generate_and_record_context()`

```python
@torch.no_grad()
def generate_and_record_context(self, batch: dict) -> ReplayContext:
    # Status: [MODIFY]
    # Inputs:
    #   materialized batch；latents [B,Cv,F,V,H,W]；actions [B,Ca,F,N,1]。
    # Outputs:
    #   所有 tensor detached 的 ReplayContext。
    # Behavior:
    #   1. sample V/A independent selections。
    #   2. self_rollout(..., transition_mode="sgf_renoise")，用 SGF schedule
    #      完整 rollout T1..T3；每帧 final x0 commit。
    #   3. recorder 取回各生成 frame 的 V/A exit x_t。
    #   4. final_clean_context = GT H/T0 + predicted final x0 T1..T3。
    #   5. replay_noisy 只在 loss mask 内替换为 recorded exit x_t。
    #   6. replay timesteps 分别写 V/A denoisy_from；condition/padding 写 0。
    #   7. replay_batch clean streams 写 final_clean_context，cond_timesteps=0。
    # Invariants:
    #   不构造 teacher_batch；不调用任何 context add_noise。
    pass
```

### 6.6 共享 student exit replay

```python
def _replay_student_x0(
    self,
    context: ReplayContext,
    *,
    requires_grad: bool,
) -> VAPrediction:
    # Status: [ADD]
    # Behavior:
    #   1. replace_va_streams(replay_batch, rollout_noisy,
    #      final_clean_context, rollout_timesteps)。
    #   2. student_wrapper.predict_joint() 得到 V/A exit x0。
    #   3. mask 外用 final_clean_context 精确覆盖，不能依赖 t=0 的近似 sigma。
    # Gradient:
    #   student step requires_grad=True；fake step在 torch.no_grad() 调用。
    # Outputs:
    #   V/A 同 batch shape；mask 内为 exit replay x0，mask 外为 final clean。
    pass
```

### 6.7 Student DMD step

```python
def replay_and_score(self, context: ReplayContext):
    # Status: [MODIFY]
    # Behavior:
    #   1. student_x0 = grad-enabled _replay_student_x0(context)。
    #   2. dmd_t.video 从 video [to,from] 采样；action 独立采样。
    #   3. 分别采样 V/A Gaussian，把 student_x0 加噪；mask 外保持 clean。
    #   4. real/fake 使用相同 score input；real video 做 CFG，action conditional。
    #   5. 两个 score forward 均 no_grad，经 wrapper 得到 real_x0/fake_x0。
    #   6. total = dmd_surrogate_loss(...)；直接 return。
    # Removed:
    #   GT-clean teacher forward、replay_target_loss、replay metrics、等权相加。
    # Metrics:
    #   DMD V/A/total、V/A exit_id、from/to、DMD t mean、real CFG scale。
    pass
```

### 6.8 Fake-score step

```python
def fake_score_step(self, context: ReplayContext):
    # Status: [MODIFY]
    # Behavior:
    #   1. with no_grad: generated_x0 = _replay_student_x0(context, False)。
    #   2. V/A 各自从 selection interval 采样 dmd_t 和 Gaussian noise。
    #   3. noisy = add_noise(generated_x0, noise, dmd_t)。
    #   4. fake velocity = fake_wrapper model forward（只在这里有梯度）。
    #   5. exact target velocity = noise - generated_x0；无需除 sigma。
    #   6. fake_score_flow_loss()。
    # Invariants:
    #   不调用 real-score；student/final context detached；V/A target 各自 mask。
    pass
```

### 6.9 Score 预测

```python
def _predict_score_x0(
    self,
    wrapper: WanDiffusionWrapper,
    input_dict: dict,
    noisy: VAPrediction,
    timesteps: VATimesteps,
    *,
    real_cfg: bool,
) -> tuple[VAPrediction, float | None]:
    # Status: [MODIFY/RENAME]
    # Behavior:
    #   fake: 一次 conditional joint forward。
    #   real: conditional + empty-text forward；只对 video velocity CFG；action
    #         取 conditional velocity；CFG 后统一 velocity_to_x0。
    # Invariants:
    #   real/fake input_dict、noisy、timestep 和 clean history完全相同。
    pass
```

### 6.10 `SGFDMDModel.__init__()`

```python
def __init__(..., real_score_checkpoint: str, fake_score_init: str):
    # Status: [MODIFY]
    # Behavior:
    #   real_score = build_frozen_transformer(..., autoregressive=False,
    #       install_distillation_profile=False, validate_distillation_profile=False)
    #   fake_score = build_trainable_transformer(..., autoregressive=False,
    #       install_distillation_profile=False, validate_distillation_profile=False)
    #   assert exact model class/profile compatibility
    #   pipeline = SelfGradientForcingTrainingPipeline(...)
    # Runtime:
    #   student/fake parameters由各自 optimizer管理；real requires_grad=False。
    pass
```

## 7. 配置与启动设计

### 7.1 Top-Level Configuration Overview

有效 root 是一个从 `VA_CONFIGS[...]` 深拷贝后增加 `distill` 的 flat `EasyDict`；`distillation.train.run()` 再用 CLI 非空值覆盖。优先级为：

```text
base EasyDict defaults
  -> apply_distillation_runtime_overrides() 的受支持环境覆盖
  -> method config 的 distill defaults
  -> distillation.train CLI
  -> rank/local_rank/world_size/device runtime fields
```

有效 base config 使用当前 registry 注册的 `VA_CONFIGS["wan22_train"]`；distillation runtime override 再从 `MOT_DATASET_ROOT/meta/mot_config.json` 注入数据路径、embedding cache 与 normalization statistics。

```python
top_level_config = {
    # [EXISTING] 路径/模型来源：
    # dataset_path, mot_config_path, mot_manifest_path, empty_emb_path,
    # text_emb_cache_path, action_cache_manifest_path,
    # init_model_from_lingbot, wan22_pretrained_model_name_or_path,
    # wan22_transformer_path, lingbot_transformer_path
    # Owner: MOTTrainer dataset/model loaders。

    # [EXISTING] 数据与 packing：
    # dataset_type, model_type, env_type, obs_cam_keys, action_dim,
    # action_representation, action_chunk_size, action_sequence_length,
    # action_per_frame, action_frames, video_downsample_ratio,
    # vae_temporal_factor, sampled_video_frames_per_action_chunk_per_view,
    # latent_frames_per_action_chunk_per_view, height, width,
    # norm_stat, norm_stats_by_task, action_norm_method
    # Owner: dataset/collate/MOTWindowSpec。

    # [EXISTING] 模型/运行时：
    # param_dtype, patch_size, enable_offload, masked_attn_backend,
    # attn_window, frame_chunk_size, init_noise_seed
    # Owner: model construction, attention, FSDP runtime。

    # [EXISTING] 优化：
    # learning_rate, beta1, beta2, weight_decay, warmup_steps,
    # gradient_accumulation_steps, num_steps, snr_shift, action_snr_shift,
    # cfg_prob, video_noisy_cond_prob, video_loss_weight, action_loss_weight
    # Owner: MOTTrainer optimizer/schedulers/loss aggregation。

    # [EXISTING] data/checkpoint/logging：
    # load_worker, max_views_per_gpu, dataloader_pin_memory,
    # dataloader_prefetch_factor, video_decoder_cache_size, action_cache_size,
    # save_root, save_interval, max_checkpoints, save_full_state, resume_from,
    # initialize_from, enable_wandb, wandb_name, wandb_mode, log_interval,
    # train_seed, sampler_seed, gc_interval, performance_jsonl_enabled,
    # performance_jsonl_interval, performance_jsonl_max_steps,
    # memory_jsonl_enabled, memory_jsonl_interval, memory_smaps_interval,
    # eval_with_cpu, eval_cfg
    # Owner: MOTTrainer runtime。

    # [MODIFY] distill: EasyDict
    # Owner: DistillationTrainerBase / SGFDMDModel / SGF pipeline。

    # [RUNTIME] rank, local_rank, world_size, device
    # Owner: distillation.train.run()。
}
```

### 7.2 `distill` namespace

```python
distill = {
    # [EXISTING] method/model_architecture/generation_shape/max_grad_norm
    # [EXISTING] fake_score_update_ratio
    # [ADD] denoisy_step_list.video: list[int|float], required, strictly descending
    # [ADD] denoisy_step_list.action: list[int|float], required, strictly descending
    # [EXISTING] rollout_horizon_frames/rollout_masked_attn_backend
    # [EXISTING] teacher_cfg_min/teacher_cfg_max：仅 real-score DMD CFG
    # [EXISTING] dmd_normalizer_eps
    # [DELETE] rollout_video_num_steps/rollout_action_num_steps
    # [DELETE] score_timestep_min/score_timestep_max
    # [DELETE] flow_target_eps
    # [EXISTING] student_init/real_score_checkpoint/fake_score_init/resume_from
}
```

参数传播：

```text
DISTILL_VIDEO_DENOISY_STEP_LIST="1000,833"
  -> _train_distill_common.sh --video-denoisy-step-list
    -> train.parse_args(): list[float]
      -> config.distill.denoisy_step_list.video
        -> SelfGradientForcingTrainingPipeline.video_denoisy_steps
          -> exit sampler + SGF video rollout + video DMD interval

DISTILL_ACTION_DENOISY_STEP_LIST="1000,500"
  -> --action-denoisy-step-list
    -> config.distill.denoisy_step_list.action
      -> action exit sampler + action rollout + action DMD interval
```

`denoisy_from/to` 不是用户手工配置项；它们必须由 list 与 exit id 推导，避免配置给出自相矛盾的三份真值。

`transition_mode` 同样不作为 SGF 实验超参暴露：它由调用入口拥有。正式 inference/evaluation 调用 `self_rollout(..., transition_mode="inference")`（也是向后兼容默认值）；`SelfGradientForcingTrainingPipeline` 必须显式调用 `transition_mode="sgf_renoise"`。这样既保留两种公共能力，又避免训练配置意外把 SGF 算法切回 inference trajectory。

### 7.3 兼容与 resume

- 旧 `rollout_*_num_steps` 与 `score_timestep_min/max` 不做兼容回退；检测到旧字段可打印迁移错误，不能静默采用另一算法。
- rollout API 的 `transition_mode` 默认值为 `inference`，用于保持已有推理 caller；SGF caller 禁止依赖默认值，必须显式传 `sgf_renoise`。
- method state 保存 canonicalized tuple lists。resume 时 config list 与 checkpoint 不同应报错。
- `fake_score_init` 必须是 `va_mot_v1` 双向 export；AR export 立即因 architecture 检查失败。
- DCP resume 仍覆盖 fake-score 参数/optimizer；构造 skeleton 的双向 init 来源不影响恢复后的数值。

## 8. 调用链设计

### 8.1 Student optimizer path

```text
[EXISTING] _train_distill_common.sh
  -> [MODIFY] distillation.train.run(config)
    -> [MODIFY] SelfGradientForcingDMDTrainer
      -> [MODIFY] SGFDMDModel.compute_step(batch, "student")
        -> [MODIFY] pipeline.generate_and_record_context(batch) [no_grad]
          -> [MODIFY] self_rollout(transition_mode="sgf_renoise", sgf_schedule=...)
            -> AR predict velocity
            -> [ADD] WanDiffusionWrapper.velocity_to_x0
            -> [ADD] renoise_x0(next_t) for non-final steps
            -> commit final x0 without noise
          <- ReplayContext(V/A exit + final clean + bounds)
        -> [ADD] _replay_student_x0(requires_grad=True)
          -> one joint AR student forward
          -> wrapper V/A velocity -> x0
        -> [ADD] V/A independent DMD t/noise
        -> [MODIFY] bidirectional real/fake score [no_grad]
          -> same noisy/context/t; real video CFG
          -> wrapper V/A velocity -> real_x0/fake_x0
        -> [EXISTING] dmd_surrogate_loss
      <- DMD loss + metrics
    -> [EXISTING] backward/clip/student optimizer.step/lr_scheduler.step
    -> [EXISTING] log/checkpoint
```

### 8.2 Inference rollout path

```text
[EXISTING] inference/evaluation caller
  -> [MODIFY] self_rollout(transition_mode="inference")
    -> [EXISTING] build_rollout_schedulers(video_num_steps, action_num_steps)
    -> [EXISTING] Video→Action incremental predict/cache lifecycle
      -> model velocity
      -> native FlowMatchScheduler.step(velocity, timestep, sample)
    <- [EXISTING CONTRACT] RolloutResult + final native scheduler samples
```

该路径不创建 `DenoisyInterval`、不要求 `WanDiffusionWrapper`、不记录 SGF exit，也不调用 `renoise_x0()`；因此可独立直接用于推理。

### 8.3 Fake-score optimizer path

```text
[EXISTING] trainer optimizer schedule -> target="fake_score"
  -> [MODIFY] SGFDMDModel.compute_step
    -> [MODIFY] final-clean rollout + ReplayContext [no_grad]
    -> [ADD] _replay_student_x0(requires_grad=False) [no_grad]
    -> [ADD] V/A independent interval t + Gaussian
    -> add_noise(generated_x0)
    -> [MODIFY] bidirectional fake_score forward [grad]
    -> exact target velocity = noise - generated_x0
    -> [EXISTING] fake_score_flow_loss
  <- fake loss + exit/t metrics
  -> [EXISTING] backward/clip/fake optimizer.step
  -> student LR scheduler 不推进
```

### 8.4 Clean context 与可见性

```text
GT H0..H3 + GT T0
  -> commit clean/no-noise KV

generate T1 video: read committed history -> final video x0 -> commit clean
generate T1 action: read history + T1 final video x0 -> final action x0 -> commit clean
generate T2/T3: repeat

joint exit replay:
  noisy V/A query at each modality's own exit t
  -> noisy query only reads lower-order clean tokens and same-order noisy tokens
  -> final-clean stream provides historical context, not same-position target leakage
```

## 9. 数据结构设计

### 9.1 Rollout mode contract

```python
rollout_transition = {
    "transition_mode": 'Literal["inference", "sgf_renoise"]',
    "inference": {
        "required": "RolloutSchedulers",
        "forbidden": "SGFRolloutSchedule, DenoisyInterval, SGF recorder",
        "transition": "scheduler.step(velocity, t, sample)",
        "output": "native scheduler final sample",
    },
    "sgf_renoise": {
        "required": "SGFRolloutSchedule, V/A intervals, wrapper, recorder",
        "transition": "velocity -> x0 -> fresh-noise add_noise(next_t)",
        "output": "last-step x0",
    },
}
```

mode 是一次 `self_rollout` 调用级不可变值，不随 frame、模态或 step 改变。V/A 在同一次调用中必须使用同一个 mode，但在 `sgf_renoise` 内仍使用不同 step lists 和 exit ids。

### 9.2 配置 schedule

```python
denoisy_step_list = {
    "video": "list[float], Kv>=1, strictly descending, values in (0,T]",
    "action": "list[float], Ka>=1, strictly descending, values in (0,T]",
}
```

生产者是 config/CLI；消费者是 SGF schedule builder；canonical tuple 进入 method state 用于 resume compatibility。

### 9.3 `ReplayContext`

```python
replay_context = {
    "replay_batch": "dict，text/stream/mask metadata + final clean V/A tensors",
    "rollout_timesteps": {
        "video": "Tensor[B,F]，生成 mask 内等于 video denoisy_from",
        "action": "Tensor[B,F]，生成 mask 内等于 action denoisy_from",
    },
    "rollout_noisy": {
        "video": "Tensor[B,Cv,F,V,H,W]，mask 内 recorded video exit x_t",
        "action": "Tensor[B,Ca,F,N,1]，mask 内 recorded action exit x_t",
    },
    "final_clean_context": {
        "video": "detached Tensor[B,Cv,F,V,H,W]，GT context + rollout final x0",
        "action": "detached Tensor[B,Ca,F,N,1]，GT context + rollout final x0",
    },
    "masks": {
        "video": "BoolTensor[B,F]",
        "action": "BoolTensor[B,Ca,F,N,1]",
    },
    "denoisy_selection": {
        "video": "DenoisyInterval(exit_id, from, to)",
        "action": "DenoisyInterval(exit_id, from, to)",
    },
}
```

所有 tensor 由 no-grad producer detach；consumer 不原地修改。ReplayContext 不序列化。

### 9.4 Wrapper output

```python
VADiffusionOutput(
    velocity=VAPrediction(
        video="Tensor[B,Cv,F,V,H,W]",
        action="Tensor[B,Ca,F,N,1]",
    ),
    x0=VAPrediction(
        video="same shape/device/dtype, graph preserved as caller requests",
        action="same shape/device/dtype, graph preserved as caller requests",
    ),
)
```

### 9.5 DMD noise bundle

pipeline 内部无需新增持久 dataclass，局部结构即可：

```python
dmd_state = {
    "timesteps": "VATimesteps，V/A 分别来自自己的 interval",
    "noise": "VAPrediction，V/A 独立 Gaussian",
    "noisy": "VAPrediction，mask 外精确保留 clean",
}
```

### 9.6 Metrics

```python
metrics = {
    "distill/dmd_video_loss": "scalar detached",
    "distill/dmd_action_loss": "scalar detached",
    "distill/dmd_total_loss": "scalar detached",
    "distill/sgf_total_loss": "与 dmd_total 相同，不含 replay",
    "distill/sgf_video_exit_id": "scalar",
    "distill/sgf_action_exit_id": "scalar",
    "distill/sgf_video_denoisy_from": "scalar",
    "distill/sgf_video_denoisy_to": "scalar",
    "distill/sgf_action_denoisy_from": "scalar",
    "distill/sgf_action_denoisy_to": "scalar",
    "distill/dmd_video_t_mean": "valid video positions mean",
    "distill/dmd_action_t_mean": "valid action frames mean",
    "distill/sgf_real_cfg_scale": "student step only",
}
```

不得再出现 `distill/replay_*`。

## 10. 关键伪代码汇总

```python
def compute_student_step(batch):
    # generate_and_record_context 内部显式调用
    # self_rollout(transition_mode="sgf_renoise")。
    ctx = generate_and_record_context(batch)       # no_grad, full final-clean rollout

    student_x0 = replay_student_x0(ctx, grad=True) # V/A use different exit t

    dmd_t = VATimesteps(
        video=sample(ctx.selection.video),
        action=sample(ctx.selection.action),
    )
    score_noisy, _noise = add_dmd_noise(student_x0, dmd_t, ctx.masks)

    score_input = prepare_score_input(
        ctx.replay_batch,
        noisy=score_noisy,
        clean=student_x0,
        timesteps=dmd_t,
    )
    with torch.no_grad():
        real_x0 = real_score_cfg_wrapper(score_input, score_noisy, dmd_t)
        fake_x0 = fake_wrapper.predict_joint(score_input, score_noisy, dmd_t).x0

    loss, metrics = dmd_surrogate_loss(
        student_x0, fake_x0, real_x0, ctx.masks, loss_weights, eps
    )
    return loss, metrics                         # no replay_target_loss


def compute_fake_step(batch):
    ctx = generate_and_record_context(batch)       # internal mode="sgf_renoise"
    with torch.no_grad():
        generated_x0 = replay_student_x0(ctx, grad=False)

    dmd_t = sample_va_intervals(ctx.selection, ctx.masks)
    noisy, noise = add_dmd_noise(generated_x0, dmd_t, ctx.masks)
    fake_velocity = fake_wrapper.predict_joint_velocity(
        prepare_score_input(ctx.replay_batch, noisy, generated_x0, dmd_t)
    )
    exact_target = VAPrediction(
        noise.video - generated_x0.video,
        noise.action - generated_x0.action,
    )
    return fake_score_flow_loss(fake_velocity, exact_target, ctx.masks)
```

## 11. 验证方案

验证顺序从纯 CPU 到 GPU 集成。

### 11.1 静态与配置

1. 验证 `VA_CONFIGS["wan22_train"]` 与 distillation dataset runtime override 可组合导入。
2. `python -m compileall distillation`。
3. parser tests：
   - 两个 list CLI 正确覆盖默认值；
   - 空 list、非数字、重复逗号失败；
   - 非递减、0/越界值在 schedule builder 失败；
   - 旧 `score_timestep_*` 不再有 consumer。

### 11.2 Wrapper 单元测试

`distillation/tests/test_wan_diffusion_wrapper.py`：

- `test_velocity_to_x0_hand_calculation`：`5-.75*4=2`。
- `test_video_action_use_different_scheduler_sigmas`：同一 nominal t 下 V/A x0 不应错误共享 sigma。
- `test_predict_joint_preserves_gradient`：student x0 可回传到 velocity/model parameter。
- `test_wrapper_does_not_register_model_parameters`：wrapper 无 `state_dict`/parameter ownership。
- `test_condition_locations_are_restored_by_pipeline_mask`。

### 11.3 Rollout 单元测试

`distillation/tests/test_rollout_transition_modes.py` 使用小 tensor/fake model/fake scheduler：

- `test_inference_is_backward_compatible_default_mode`。
- `test_inference_mode_matches_existing_scheduler_step_trajectory`：固定 seed 后逐 step 与修改前路径数值一致。
- `test_inference_mode_requires_no_sgf_schedule_or_wrapper`：证明它可独立直接用于推理。
- `test_rollout_rejects_unknown_or_mixed_transition_mode`。
- `test_video_action_accept_different_step_lists_and_exit_ids`。
- `test_exit_record_does_not_stop_rollout`：exit=0 仍调用全部 steps。
- `test_transition_uses_x0_and_fresh_noise`：每个 next state 匹配公式，且 step 间 noise 不同。
- `test_final_return_is_last_x0_not_renoised_sample`。
- `test_committed_context_is_final_x0_without_noise`。
- `test_rollout_rng_advances_between_microsteps`。
- `test_last_exit_maps_to_denoisy_to_zero`。

### 11.4 Pipeline/loss/梯度测试

`distillation/tests/test_self_gradient_forcing_dmd.py`：

- `test_context_contains_independent_va_intervals`。
- `test_dmd_timesteps_respect_each_modality_interval`。
- `test_student_total_is_dmd_only`：总 loss 与 DMD 相等，无 replay metric。
- `test_student_step_gradients_only_student`。
- `test_fake_step_replays_student_without_grad`。
- `test_fake_step_gradients_only_fake_score`。
- `test_fake_target_is_exact_noise_minus_x0`。
- `test_real_and_fake_receive_identical_score_state`。
- `test_real_fake_models_are_bidirectional_and_same_profile`。
- `test_resume_rejects_changed_denoisy_lists`。
- `test_all_false_mask_returns_finite_zero_contribution`。

建议命令（在 config registry 修复后）：

```bash
pytest -q \
  distillation/tests/test_wan_diffusion_wrapper.py \
  distillation/tests/test_rollout_transition_modes.py \
  distillation/tests/test_self_gradient_forcing_dmd.py
```

回归现有 self-rollout/cache：

```bash
pytest -q \
  distillation/tests/test_self_rollout_indexed_attention.py \
  distillation/tests/test_runtime_helpers.py
```

### 11.5 最小 GPU smoke test

建议使用最小可用 prepared batch、单 rank、`gradient_accumulation_steps=1`、`fake_score_update_ratio=1`、`rollout_horizon_frames=1`，运行至少 2 个 optimizer steps，使 fake-score 与 student 各更新一次。观察：

- 进程成功退出，无 FSDP collective hang；
- 两次 rollout RNG/exit 不被固定 seed 重复；
- 单独运行 inference mode 时不创建 SGF recorder，并能产出合法 `RolloutResult`；
- fake/student total loss、V/A component loss、grad norm 全部 finite；
- fake step student 无 grad，student step fake/real 无 grad；
- 日志含 V/A 独立 exit/from/to；
- checkpoint 包含 student/fake score 双 optimizer DCP 状态；
- resume 一个额外 step 的 optimizer target 和 RNG trajectory 连续。

硬件/数据前提：至少一张能同时容纳 AR student、双向 real-score 与双向 fake-score shard 的 CUDA GPU；若单卡不足，使用仓库现有 4-GPU launcher，不降低模型数量来伪造 smoke pass。

## 12. 风险与开放项

### 12.1 Base config 与 dataset runtime override

实现统一采用已注册的 `wan22_train`，同时从 distillation-owned `MOT_DATASET_ROOT` 覆盖 dataset manifest、text embedding cache、empty embedding、action cache 与 normalization statistics。启动 smoke test 必须确认这些路径均非空且存在。

### 12.2 两套 denoisy list 的最终实验值

本文给出的 `[1000,833]` / `[1000,500]` 只是把当前 2-step + 不同 shift 的隐式 schedule 显式化，以便实现有确定默认。若目标实验已有指定 V/A lists，应在编码前替换；接口、bounds 与测试设计不受数值变化影响。

### 12.3 双向 fake-score checkpoint 来源

当前 workflow 把 stage1 AR export 作为 fake init，这是不兼容的。必须提供 `model_architecture="va_mot_v1"` 的双向 export；最小合理选择是与 real-score 相同的初始 teacher checkpoint。若只有 AR checkpoint，需要另行产出双向 export，不能通过 `strict=False` 混载来掩盖架构差异。

### 12.4 计算与显存

fake-score step 从“final clean 直接训练”变为“完整 rollout + 一次 no-grad exit replay + fake forward”，增加一次 student forward，但这是保证 fake distribution 与 generator DMD sample 对齐所必需。应通过 FSDP、activation checkpoint 和 horizon smoke test验证，不应退回错误 sample 定义来节省计算。

### 12.5 Joint V/A 不同 exit 的 attention 语义

一次 joint replay 中 V/A timestep 不同是原生模型支持的：`replace_va_streams` 已分别携带 `latent_dict.timesteps` 和 `action_dict.timesteps`。测试必须覆盖 video 高噪/action 低噪及反向组合，防止未来代码把两者重新合并成一个 tensor。

### 12.6 两种 transition 的行为漂移

两种 mode 共享 cache orchestration，但不能共享会改变数值语义的 transition 状态。尤其禁止为了“复用”而让 inference mode 先算 x0 再还原 velocity，或让 SGF mode 调一次 `scheduler.step()` 后再 re-noise。`test_inference_mode_matches_existing_scheduler_step_trajectory` 应作为强回归门槛，确保新增 SGF 能力不会改变正式推理结果。

### 12.7 名称兼容

需求使用 `denoisy_*`。实现时应一次性统一配置、dataclass、metric 与 CLI 的拼写；不要同时保留旧 `denoising_step_list` alias，否则 checkpoint/config 审计会出现两份真值。对旧配置采用明确迁移错误即可。

## 13. 实现顺序建议

1. 固化 `wan22_train` base config 与 distillation dataset runtime override，并验证配置可导入。
2. 实现 schema、scheduler pure helpers 和 wrapper，并完成 CPU 数学测试。
3. 先把当前 scheduler path 固化为显式 `inference` mode 并完成数值等价测试；再实现 V/A schedule、recorder 与 `sgf_renoise` transition，完成双 mode fake-model rollout 测试。
4. 精简 ReplayContext 和 pipeline，先跑 student DMD-only path，再接 fake replay path。
5. 把 fake-score 切为双向，修正 fresh/resume/workflow 初始化来源。
6. 删除 replay loss 定义/导出与旧配置字段，增加 CLI/shell 传播。
7. 跑现有 cache 回归、pipeline 梯度测试和最小多 GPU smoke。

该顺序保证每一步都有独立可验证的 tensor 契约，也避免在 wrapper、rollout、loss 和 checkpoint 同时变化时难以定位错误。
