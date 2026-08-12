# 一致性蒸馏职责重构设计（0811）

## 1. 目标、假设与不变量

本次修改只重分代码职责，不修改当前 consistency distillation 的数学与训练语义。以下行为是重构前后的等价性边界：

- video/action 分别使用 `snr_shift` / `action_snr_shift` 的 1000-step `FlowMatchScheduler`；
- `sample_consistency_timesteps()` 的 stride、采样空间和 mask=False 时写 0 的行为不变；
- clean、两份独立 Gaussian noise、`add_noise_to_va()` 的逐模态 mask 行为不变；
- teacher 从同一个 `x_t/t` 做 conditional/unconditional forward，只对 video flow 做 CFG，action 使用 conditional flow；
- `flow_step()` 仍执行 `x_next=x_t+(sigma_next-sigma_t)*flow`，不改成 scheduler Euler step 或 fresh re-noise；
- EMA student 在 `x_next/t_next` 上产生 stop-gradient target；raw student 只在原始 `x_t/t` 上建立梯度图；
- video 仍使用 Flash-WAM consistency boundary scaling，action 仍使用 `flow_to_x0()`；
- 总 loss 仍为 `consistency_loss + action_aware_weight * action_aware_loss`，loss mask 与权重不变；
- EMA 只在成功的 student optimizer step 后更新；teacher/EMA 不获得梯度；
- consistency 的自回归 rollout 不参与上述 loss。把 rollout 预测替换 GT clean 或把 exit state 替换随机 `x_t` 都会改变算法，禁止在本次重构中这样做。

“一致性蒸馏必须通过 pipeline rollout”在这里指：consistency 阶段暴露和执行的无梯度自回归 rollout 统一进入 `SelfGradientForcingTrainingPipeline.generate()`，不再由 trainer/model 私自维护另一套 cache、采样器或 recorder。它与训练 trajectory 是并列入口，不能混成一条数据流。

## 2. 当前实现审查

### 2.1 `distillation/model/consistency.py`

当前文件同时负责：加载三份模型、创建 scheduler、采样 timestep/noise、调用 WAN、flow/x0 转换、teacher trajectory、student/EMA forward、loss、EMA 和方法状态。

正确部分：

- `sample_consistency_timesteps()`、`add_noise_to_va()`、`flow_step()` 的调用顺序清晰；
- teacher/EMA 均在一个 `torch.no_grad()` 区域；
- 只有 student forward 保留 autograd；
- video/action 的 mask、scheduler 和参数化彼此独立。

问题：

- scheduler 和 WAN 调用绕过了已经存在的 `WanDiffusionWrapper`，与 DMD 的分工不一致；
- `_predict_consistency()` 与 student 分支重复拆解 raw WAN output 和 action flow/x0 转换；
- model 没有 consistency 阶段的 pipeline rollout 入口；
- `resume_from` 被 model 保存但 model 不应负责加载训练状态。

### 2.2 `distillation/trainer/consistency.py`

trainer 已负责 FSDP/activation checkpoint 包装、冻结、train/eval 状态、optimizer/LR scheduler 和 optimizer-step 后 EMA hook，方向正确。

问题：

- 它直接面向 `model.student/teacher/ema_student` raw module，model 与 DMD 使用 wrapper 的角色表达不统一；
- 缺少通过 model/pipeline 的 rollout 公共入口；
- 当前通用 trainer checkpoint 只写 `model.pt`，没有跨阶段必须的 `transformer/` export 和 metadata；consistency 的 optimizer/EMA state 也没有写入，resume 不能保持训练语义。

### 2.3 `distillation/model/wan_wrapper.py`

当前 wrapper 已拥有 WAN checkpoint 加载、video/action scheduler、joint forward、flow-to-x0、AR predict/commit，是目标职责的主要基础。

问题：

- 只能从 checkpoint 构造，不能包装测试或 trainer 已持有的 module；
- consistency 没有复用它；
- joint forward 计算的 action x0 正好可替代 consistency 中的重复转换，但 video consistency scaling 仍属于 consistency 算法，不能下沉到通用 wrapper。

### 2.4 `distillation/pipeline/*`

`SelfGradientForcingTrainingPipeline` 已负责：

- `@torch.no_grad()` 的逐 block video→action rollout；
- `BasePipeline` 的 cache reset/history+anchor commit；
- noisy prediction transaction 与 final-clean commit；
- video/action 独立 step list、exit id、exit noisy state 和 interval 记录；
- final clean 与 exit state 的 `RolloutResult`。

因此无需新增 `ConsistencyTrainingPipeline`。新增一份最小 pipeline 仍会复制 cache/commit/recorder，且 consistency rollout 与 SGF rollout 本质使用同一个 AR 模型接口。

需要注意：pipeline rollout 使用 `x0 -> fresh noise -> next x_t`；consistency teacher trajectory 使用 `flow_step()`。二者公式不同且用途不同，复用 pipeline 不代表把 pipeline transition 用到 consistency loss。

### 2.5 schema、objectives 与 utils

- `VAPair`、`VATimesteps`、`VAMasks`、`TrainingStepResult` 已足够表达 consistency loss；不需要新增训练专用 DTO。
- `RolloutResult`/`RecordedDenoiseState` 已足够表达 consistency 的无梯度 rollout；不复制 schema。
- `consistency_loss()` 和 `action_aware_loss()` 已按 mask 归一并返回 detached metrics；不修改公式。
- `replace_va_streams()`、`replace_text_condition()`、`add_noise_to_va()` 是 model 算法编排所需的纯 tensor/input helper，保留。

### 2.6 tests 与 configs

当前测试覆盖 SGF pipeline、DMD gradient isolation、wrapper conversion，但没有直接锁定 consistency 的：

- teacher/student/EMA forward 次数与 CFG 组合；
- student-only gradient；
- timestep/noise/mask 轨迹；
- consistency rollout 是否委托共享 pipeline；
- checkpoint 的 raw student/EMA/optimizer 与跨阶段 export 边界。

`consistency_distillation.py` 只有 loss 配置，缺少共享 pipeline rollout 的最小 schedule/horizon 配置。应增加单独的 `rollout_denoising_step_list`，避免与 `video_num_steps/action_num_steps` 的 consistency stride 混淆。

## 3. 目标职责边界

### 3.1 model

`ConsistencyModel` 负责：

- 持有 student/teacher/EMA 三个 `WanDiffusionWrapper` 角色；
- 从 batch/base input 组织 clean、noise、mask、`t/t_next`；
- teacher CFG、teacher adjacent `flow_step`、EMA target 和 student prediction；
- consistency/action-aware loss 与 metrics；
- optimizer step 后 EMA 更新和少量方法状态；
- 构造共享 rollout pipeline，并把 rollout 请求委托给 pipeline。

model 不负责 FSDP、freeze/train/eval、optimizer、LR、checkpoint 读写或数据加载。

### 3.2 pipeline

`SelfGradientForcingTrainingPipeline` 继续唯一负责：

- 无梯度自回归 rollout；
- KV cache 生命周期、history/anchor commit、预测 transaction 清理；
- exit id/interval/noisy state 记录；
- final clean rollout result。

pipeline 不计算 consistency loss，不调用 teacher CFG，不更新 EMA，不持有 optimizer。

### 3.3 wan_wrapper

`WanDiffusionWrapper` 负责：

- 从 export 加载或包装现有 WAN module；
- video/action scheduler 初始化与 ownership；
- joint WAN forward 输出 `velocity+x0`；
- AR video/action predict/commit；
- scheduler-aware flow/x0 转换。

通用 wrapper 不实现 consistency boundary scaling、teacher CFG 或 loss。

### 3.4 trainer

`ConsistencyTrainer` 负责：

- student/teacher/EMA 的 FSDP/AC、requires_grad、train/eval；
- student optimizer、LR scheduler、gradient accumulation/clip 和训练循环；
- 成功 optimizer step 后调用 EMA hook；
- same-stage resume 的 raw student + optimizer + LR + EMA；
- next-stage export 选择 EMA student；
- batch materialize/input prepare，以及对 model rollout 的薄委托。

## 4. 调用链

### 4.1 consistency 训练 microstep

```text
ConsistencyTrainer._train_step
  -> materialize batch / prepare clean base_input
  -> ConsistencyModel.compute_step
     -> sample t,t_next + independent V/A noise + masked add_noise
     -> no_grad:
        teacher wrapper conditional forward
        teacher wrapper unconditional forward
        video CFG/action conditional flow
        flow_step(x_t -> x_next)
        EMA wrapper forward at x_next/t_next
     -> grad-enabled student wrapper forward at x_t/t
     -> video consistency prediction + action x0
     -> consistency_loss + action-aware loss
  -> backward only into student
  -> student optimizer/LR step
  -> EMA update
```

### 4.2 consistency 自回归 rollout

```text
ConsistencyTrainer.rollout(batch)
  -> materialize batch / prepare effective text condition
  -> ConsistencyModel.rollout
  -> SelfGradientForcingTrainingPipeline.generate (@no_grad)
     -> reset/build history+anchor KV cache
     -> EMA wrapper generate_video -> record exit -> commit final x0
     -> EMA wrapper generate_action -> record exit -> commit final x0
     -> RolloutResult(final clean + exit state/status)
```

rollout 使用 EMA student，因为 stage2 对下游发布的也是 EMA student。训练 loss 不读取 `RolloutResult`。

## 5. pipeline 方案选择

选择：复用 `SelfGradientForcingTraining.py`。

不选择新增专用 pipeline 的原因：

- cache、Video→Action commit、independent scheduler、exit recorder 已完全存在；
- EMA student 与 SGF generator 都是同一 `autoregressive_va_mot_v1` 接口；
- 新 pipeline 只能复制逻辑，没有新的算法边界；
- consistency training trajectory 并不是 AR cache rollout，把它命名为 pipeline 会模糊 `flow_step` 与 SGF re-noise 的区别。

配置使用独立名字：

```text
distill.rollout_denoising_step_list.video/action
distill.rollout_horizon_frames
```

不复用 `video_num_steps/action_num_steps`，后者只决定 consistency 的 scheduler stride。

## 6. 文件级改动计划

- `distillation/model/wan_wrapper.py`
  - 支持包装 caller 提供的 module；现有 checkpoint load 和 conversion 不变。
- `distillation/model/consistency.py`
  - 三个角色改用 wrapper；删除自行创建 scheduler；teacher/student/EMA forward 统一经 wrapper；增加共享 pipeline rollout 委托；保持 loss 逐行等价。
- `distillation/trainer/consistency.py`
  - FSDP 针对 wrapper 内 module；增加薄 `rollout()`；optimizer/EMA hook 不变。
- `distillation/configs/consistency_distillation.py`
  - 增加最小 rollout schedule/horizon/block 配置。
- `distillation/trainer/base.py`
  - 修复 checkpoint：保存 optimizer/LR 与 method extra state，并产生下游可加载的 transformer export/metadata；resume 恢复这些状态。
- `distillation/trainer/self_gradient_forcing_dmd.py`
  - 适配统一 checkpoint extra state，不改 optimizer 轮换。
- `distillation/model/autoregressive_mot.py`
  - 修复 `_prepare_metadata()` 返回 `(metadata, diagnostics)`，确保“替换 mask 的 AR 训练”实际可运行。
- `distillation/workflow.py` / launchers
  - 校正四阶段 checkpoint 来源，尤其 SGF real/fake score 必须来自 bidirectional base export，不能把 AR export 当 bidirectional checkpoint。
- `distillation/tests/`
  - 增加 consistency 算法/梯度/rollout 委托测试、AR segmented mask 回归测试和 checkpoint 链路契约测试。

## 7. 四阶段训练链路审计

### 阶段 0：基础 Video+Action MOT

- 入口：`wan_va/configs/va_wan22_train_cfg.py` + `wan_va/train_mot.py`。
- 模型：`VAMOTTransformer3DModel`（bidirectional VA MoT）；video backbone 从 WAN2.2 初始化，action expert 本地初始化。
- mask：原生 `build_x_metadata()` chunk order 与 `_va_visibility()`；clean→clean 因果、noisy→过去 clean、noisy→同 order noisy。
- 可训练：基础 VA transformer 全部按 parameter ownership 规则训练。
- 梯度：native video/action flow loss → bidirectional VA transformer。
- checkpoint：`model_architecture=va_mot_v1`，是 SGF real-score/fake-score 初始化的合法来源。

### 阶段 1：替换 mask 的自回归训练

- 入口：`AutoregressiveTrainer`，训练循环和数据/noise/loss 复用 `wan_va/train_mot.py`。
- 初始化：阶段 0 的 `va_mot_v1` export；参数兼容地加载为 `AutoregressiveVAMOTTransformer3DModel`。
- mask：history 仍按 chunk；target 从 anchor 起逐帧分配 `video_order, action_order=video_order+1`，形成 segmented AR order。`_va_visibility()` 公式本身不变，只替换 order ids。
- 可训练：AR student；无 teacher/EMA。
- 梯度：native flow loss → AR student。
- checkpoint：`model_architecture=autoregressive_va_mot_v1`，交给 consistency 的 student 与 teacher。

### 阶段 2：一致性蒸馏

- 初始化：raw student、frozen teacher、EMA 均来自阶段 1 AR export；same-stage resume 后 raw student/optimizer/LR/EMA 从 resume state 覆盖。
- mask：沿用阶段 1 segmented AR metadata；loss mask 原样来自 dataset batch。
- 可训练：仅 raw student；teacher 与 EMA frozen/eval。
- 梯度：仅 student 原始 `x_t/t` forward；teacher CFG、teacher transition、EMA target 全部 stop-gradient。
- checkpoint：resume state 保存 raw student+optimizer+LR+EMA；公开 `transformer/` 导出 EMA student，交给 SGF generator。
- rollout：EMA student 经共享 pipeline，无梯度且不进入 loss。

### 阶段 3：SGF-DMD

- generator 初始化：阶段 2 EMA AR export。
- real-score/fake-score 初始化：阶段 0 bidirectional `va_mot_v1` export；二者不能从阶段 1 AR export 初始化。
- mask：generator rollout/replay 使用 segmented AR/cache 顺序；real/fake score 使用原生 bidirectional mask，并消费相同 score-noisy/timestep/clean condition。
- 可训练：generator 与 fake-score 按配置轮换；real-score frozen。
- 梯度：generator step 只回 generator，real/fake score 用于 stop-gradient DMD direction；fake-score step 只回 fake-score，generator replay no-grad。
- checkpoint：公开 export 是 generator；same-stage resume 还必须保存两个 optimizer、fake-score、LR 和 update schedule。

### 当前衔接缺口

- `workflow.py` 目前只编排三个 distillation method，没有执行阶段 0；其 `--student-init` 实际应明确为阶段 0 bidirectional checkpoint。
- workflow 目前把 `real_score_checkpoint`/`fake_score_init` 指向 `args.student_init`，方向正确；SGF launcher 的 `DISTILL_AR_CHECKPOINT -> DISTILL_FAKE_SCORE_INIT` fallback 错误，应删除或改为阶段 0 bidirectional 来源。
- 简化后的 distillation checkpoint 只有 `model.pt`，与 wrapper/launcher 要求的 `transformer/` export 不兼容，必须修复后四阶段才闭合。
- AR `_prepare_metadata()` 当前少返回 diagnostics，训练 forward 会在解包处失败；必须修复并加测试。

## 8. 验证标准

1. 固定 RNG 下，重构前后 consistency 的 `t/t_next`、noise、CFG、teacher next state、student/target output 和 loss 数值一致。
2. backward 后只有 raw student 有梯度；teacher/EMA 无梯度；EMA 仅在 optimizer step hook 更新。
3. consistency `rollout()` 的唯一实现路径进入 `SelfGradientForcingTrainingPipeline.generate()`，cache 无残留 transaction，结果含 exit 状态。
4. AR segmented metadata 对 history chunk 与 target per-frame 产生预期 order ids。
5. 阶段 1/2/3 checkpoint 均含可供下一阶段加载的 `_SUCCESS`、metadata、`transformer/config.json` 和 safetensors；same-stage resume state 完整。
6. 运行 `compileall`、相关单测、`python -m pytest distillation/tests/ -q` 和 `git diff --check` 全部通过。

## 9. 实施结果（完成后回填）

- consistency 的三份 WAN 角色已经统一由 `WanDiffusionWrapper` 调用；scheduler 也由 wrapper 持有。训练中的 timestep/noise/mask、teacher `flow_step`、CFG、EMA target、student graph 和两项 loss 的顺序未改。
- consistency 新增 `ConsistencyModel.rollout()` / `ConsistencyTrainer.rollout()`，EMA AR rollout 唯一委托 `SelfGradientForcingTrainingPipeline.generate()`；训练 loss 不读取 rollout 结果。
- `AutoregressiveVAMOTTransformer3DModel._prepare_metadata()` 已恢复 `(metadata, diagnostics)` 契约，并用测试验证 history chunk + target per-frame segmented order。
- distillation checkpoint 已改为 `distributed_state/` DCP 分片保存 raw trainable model/optimizer 及 EMA 或 fake-score；`training_state.pt` 保存 step、LR、RNG 和 method state；`transformer/` 只聚合下阶段所需 export。consistency export 为 EMA，SGF export 为 generator。
- SGF launcher 不再允许用 AR checkpoint 作为默认 fake-score；workflow 与测试固定为 base bidirectional checkpoint同时初始化 real/fake score。
- 新增测试覆盖 consistency forward 次数、mask、梯度隔离、pipeline 委托、wrapper 借用 module、segmented mask、DCP resume/export 和四阶段 checkpoint 角色衔接。

最终验证使用 `/zsh/miniconda3/envs/linbotva/bin/python`（系统默认 `/zsh/miniconda3/bin/python` 未安装 pytest）：

```text
python -m compileall -q .                                      -> pass
python -m pytest distillation/tests/test_consistency_distillation.py -q
                                                               -> 3 passed
python -m pytest distillation/tests/ -q                        -> 42 passed
git diff --check                                               -> pass
```
