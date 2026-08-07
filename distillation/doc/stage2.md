# Stage 2：一致性蒸馏（`consistency_distillation`）

本文讲解蒸馏流水线的第二阶段 `consistency_distillation`。它把 Stage 1 的 AR
checkpoint 作为 frozen teacher，训练一个 student 学会从同一个加噪点出发、沿着
teacher 的 flow 线推进到相邻低噪点，并让 student 在该相邻点的输出与 frozen EMA
student 一致——这就是 consistency training 的核心。

与 stage1.md 一样，先讲整体架构，再逐层进入代码；mask 构造是重点章节
（第 6 节）。注意力 mask 与 Stage 1 完全相同，Stage 2 新增的是"噪声/loss 侧"的
mask 组合，以及训练期 rollout 的增量缓存 mask。

## 1. 阶段定位与整体架构

### 1.1 在流水线中的位置

```text
autoregressive_training       （Stage 1）
  -> consistency_distillation （Stage 2，本文）
  -> self_gradient_forcing_dmd（Stage 3）
```

Stage 2 输入：Stage 1 的 AR checkpoint（同时作为 `student_init` 与
`teacher_checkpoint`）。输出：EMA student 的 export，Stage 3 把它当作 student
初始化。

### 1.2 一次训练更新的总览

```text
clean V/A
  -> 按 mask 逐帧采样 t 与噪声 -> x_t
  -> frozen teacher 在 x_t 预测 conditional/unconditional flow
  -> video 做 CFG（action 只用 conditional）
  -> flow_step: x_next = x_t + (sigma_next - sigma_t) * flow
  -> frozen EMA student 在 x_next 输出归一化 consistency target

x_t（原始点）
  -> trainable student 输出归一化 consistency prediction
  -> masked V/A consistency loss vs EMA target
  -> 辅助项：student action flow 回归精确 flow（noise - clean）
  -> student optimizer step 成功后更新 EMA
```

只有最后一遍 student forward 保留梯度；teacher 与 EMA 都在 `no_grad` 中。

### 1.3 入口与代码地图

```text
1shell/distill/train_distill_consistency_distillation_4gpu.sh
  -> _train_distill_common.sh（DISTILL_METHOD=consistency_distillation）
  -> python -m distillation.train --method consistency_distillation ...
  -> distillation/train.py: ConsistencyTrainer(config)
  -> DistillationTrainerBase + MOTTrainer.train()
```

| 文件 | 责任 |
| --- | --- |
| `distillation/trainer/consistency.py` | 构造 `ConsistencyModel`；student step 成功后触发 EMA；训练期 rollout |
| `distillation/trainer/base.py` | 数据搬运、latent materialize、梯度累积、finite check、clip、日志与 checkpoint 入口 |
| `distillation/model/consistency.py` | 持有 frozen teacher、frozen EMA student 和 pipeline |
| `distillation/pipeline/consistency_training.py` | 单个 microstep 的完整数据流（teacher -> next -> EMA target -> student） |
| `distillation/model/objectives.py` | V/A mask-aware MSE 与加权总 loss |
| `distillation/pipeline/utils.py` | V/A 分 modality 加噪、condition 保持 clean、替换模型输入流 |
| `distillation/scheduler.py` | `[B,F]` timestep 的 sigma lookup、flow step、flow-to-x0、consistency boundary scaling |
| `distillation/self_rollout/` | 训练期 rollout：增量 KV cache、提交顺序、几何/动作生成 |
| `distillation/model/factory.py` | 从 checkpoint 加载并冻结 teacher/EMA，按原生 FSDP 规则配置 |
| `distillation/checkpoint.py` | raw student + optimizer + EMA 的 DCP 保存/恢复 |
| `distillation/schema.py` | `VAMasks` / `VAPrediction` / `VATimesteps` 等共享结构 |

## 2. 配置要点

`distillation/configs/consistency_distillation.py` 的关键字段：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `optimization_composition` | `"va"` | 只训练 V/A 分支；G 冻结、无 geometry loss |
| `cfg_prob` | `0.0` | 关闭随机文本 dropout，保证 teacher/student 条件一致 |
| `video_num_steps` / `action_num_steps` | `2` / `2` | 决定 scheduler 离散表上 `t -> t_next` 的跨度 |
| `cfg_min` / `cfg_max` | `2.0` / `10.0` | teacher video CFG 的随机区间 |
| `sigma_data` | `0.5` | video consistency boundary scaling 的数据尺度 |
| `action_aware_weight` | `0.01` | action 精确 flow 回归辅助项权重 |
| `ema_decay` | `0.9999` | EMA 衰减 |
| `rollout_interval` | `100` | 每 N 个已完成 optimizer step 跑一次 EMA rollout |
| `rollout_horizon_frames` | `3` | 在 T0 之后逐帧生成 T1..T3 |
| `rollout_gt_mode` | `"none"` | 训练期 rollout 不做 GT 替换 |
| `rollout_masked_attn_backend` | `"dense"` | rollout 时把 EMA 的 mask 后端临时切到 dense |

## 3. 模型所有权

`ConsistencyModel`（`distillation/model/consistency.py`）持有三个角色：

| 模型 | 初始化来源 | requires_grad | 模式 | 保存方式 |
| --- | --- | --- | --- | --- |
| student | `student_init/transformer`（Stage 1 export） | V/A 可训练；G 冻结 | train | DCP raw model |
| teacher | `teacher_checkpoint/transformer` | 全冻结 | eval | 不重复保存，resume 时从来源重建 |
| EMA student | `student_init/transformer`，之后 EMA 更新 | 全冻结 | eval | DCP 保存；同时导出到 `transformer/` |

`build_frozen_transformer()` 会先校验 checkpoint 的 `generation_profile` 与当前
配置一致（`validate_checkpoint_generation_profile`），再加载、冻结并执行原生 FSDP
分片。teacher/EMA 调用 `model(input_dict, mode="train")` 只是选择 joint forward
路由，模型对象本身仍是 `eval()`、`requires_grad=False`。

## 4. 一次更新的数据流（`ConsistencyTrainingPipeline.compute_loss`）

`ConsistencyTrainer.compute_step` 直接委托给
`ConsistencyTrainingPipeline.compute_loss`。下面按代码顺序拆解：

### 步骤 1：复用原生 input builder，但跳过原生加噪

```python
base_input = trainer._prepare_joint_input_dict(batch, add_noise=False)
```

`add_noise=False` 仍然完成：mask 归一化、text/empty text、geometry、stream_ids、
shape 校验（`validate_mot_batch_for_forward`）和 attention-window 参数注入。注意力
metadata 本身不在这一步生成，而是在模型 `mode="train"` forward 的
`_prepare_metadata -> _apply_segmented_order` 里构造（与 Stage 1 完全相同，见
stage1.md 第 5 节）。这里只是不生成原生的随机 t/noise/target，因为下面马上会
构造 consistency 专用的 `(x_t -> x_t_next)` 轨迹。

### 步骤 2：读回规范化后的 mask，采样 timestep 对

```python
masks = VAMasks(
    video=base_input["latent_dict"]["video_latent_loss_mask"].reshape(bsz, frames),
    action=base_input["action_dict"]["action_loss_mask"],
)
timesteps, next_timesteps = self._sample_timesteps(masks)
```

video 是帧级 `[B,F]` mask；action 是元素级 `[B,Ca,F,N,1]` mask。action 帧是否采样
由该帧内任一有效 token 决定（`action_mask.any(dim=(1,3,4))`）。

### 步骤 3：构造 clean 与 noise，按 mask 加噪

```python
clean = VAPrediction(batch["latents"], batch["actions"])
noise = VAPrediction(torch.randn_like(clean.video), torch.randn_like(clean.action))
noisy = add_noise_to_va(clean, noise, timesteps, masks, ...)
```

`add_noise_to_va()` 逐模态调用 `add_noise()` 后，再用 `torch.where(mask, noisy,
clean)` 把 mask=False 的位置（history、T0 anchor、padding、未选中的 action 元素）
**显式恢复为 clean**。

### 步骤 4：frozen teacher 在同一 `x_t` 上做 CFG

```python
teacher_input = replace_va_streams(base_input, noisy, clean, timesteps)
teacher_flow, cfg_scale = self._teacher_cfg_flow(teacher_input, batch)
```

`_teacher_cfg_flow` 做两次 teacher forward：conditional（原 text）与
unconditional（`empty_text_emb`）。CFG 只作用在 video 上：

```text
v_cfg = v_uncond + cfg_scale * (v_cond - v_uncond)
action 直接用 conditional prediction
cfg_scale ~ Uniform(cfg_min, cfg_max)
```

### 步骤 5：沿 flow 线直接推进到相邻状态

```python
next_noisy = VAPrediction(
    flow_step(teacher_flow.video, noisy.video, timesteps.video, next_timesteps.video, ...),
    flow_step(teacher_flow.action, noisy.action, timesteps.action, next_timesteps.action, ...),
)
```

### 步骤 6：EMA student 在 `x_next` 构造 stop-gradient target

```python
target_consistency = self._predict_consistency(self.ema_student, ema_input, next_noisy, next_timesteps)
```

video 用 consistency boundary scaling（`consistency_prediction`），action 用线性
`flow_to_x0`。整个步骤在 `torch.no_grad()` 内。

### 步骤 7：student 在原始 `x_t` 预测

```python
student_input = replace_va_streams(base_input, noisy, clean, timesteps)
student_out = self.student(student_input, mode="train")
student_consistency = 与 EMA 相同的归一化函数(student_flow)
```

这是唯一保留 autograd graph 的 forward。

### 步骤 8：主 loss

```python
consistency, metrics = consistency_loss(student_consistency, target_consistency, masks, weights)
```

比较的是归一化后的 consistency output，在有效 V/A mask 内做 MSE。

### 步骤 9：action-aware 辅助项

```python
action_target = trainer.train_scheduler_action.training_target(clean.action, noise.action, timesteps.action)
aware, aware_metrics = action_aware_loss(student_flow.action, action_target, masks.action)
loss = consistency + action_aware_weight * aware
```

复用同一次 student forward 的 action flow，不加模型调用。默认 0.01 的权重防止
action 的 consistency x0 参数化在高噪声处训练信号过弱。

## 5. timestep 采样与噪声轨迹

### 5.1 `sample_consistency_timesteps`

训练 scheduler 是离散表（1000 个状态，从高噪声排到低噪声）。配置步数 K 决定
`t -> t_next` 的跨度：

```text
stride = clamp(floor(S / K), 1, S - 1)        # S=1000, K=2 -> stride=500
current_id ~ Uniform{0, 1, ..., S - stride - 1}
next_id = current_id + stride
t       = scheduler.timesteps[current_id]       # [B,F]
t_next  = scheduler.timesteps[next_id]
```

采样范围明确排除了不能再向低噪声推进的尾部，因此所有监督位置都满足
`next_sigma < sigma`，不会出现 `t_next == t` 的退化 pair。mask=False 的位置
仍返回数值 0，但随后由调用方用 `torch.where` 保持 clean。

### 5.2 `sigmas_for_timesteps`：`[B,F]` 查找

原生 scheduler 只正确处理一维 timestep；蒸馏需要 `[B,F]`。实现把表扩成
`[S,1,1]` 后逐样本逐帧找最近离散点，返回同形 sigma：

```python
indices = (scheduler_timesteps - timesteps.unsqueeze(0)).abs().argmin(dim=0)
return scheduler.sigmas[indices]
```

注意：数值 `timestep=0` 可能映射到一个小但非零的 sigma（`extra_one_step=True`），
所以"condition 是否保持 clean"必须由 mask 显式保证。

### 5.3 加噪与反解公式

```text
x_t     = (1 - sigma_t) * x0 + sigma_t * noise        # add_noise
flow    = noise - x0                                  # training_target
x0_hat  = x_t - sigma_t * flow                        # flow_to_x0
x_next  = x_t + (sigma_next - sigma_t) * flow         # flow_step
```

`consistency_prediction` 是 Flash-WAM 的方差保持边界缩放（只用于 video）：

```text
d        = sigma^2 + sigma_data^2
c_skip   = sigma_data^2 / d
c_out    = sigma * sigma_data / sqrt(d)
f_v      = c_skip * x_t + c_out * x0_hat
```

`sigma=0` 时 `f_v = x_t`；高噪声处 `c_out` 有界。student 与 EMA 必须调用同一个
函数，否则两侧尺度不一致，MSE 没有可解释性。

手算例子（来自 pipeline 文档字符串）：

```text
frame1: x0=2, noise=6, sigma=0.75
  x_t     = (1-0.75)*2 + 0.75*6 = 5
  flow    = 6 - 2 = 4
  sigma_next=0.25
  x_next  = 5 + (0.25 - 0.75)*4 = 3     # 等价于 (1-0.25)*2 + 0.25*6
```

## 6. mask 构造（重点）

Stage 2 涉及三类 mask，需要分开理解：

1. **数据侧 mask**（与 Stage 1 完全相同）：决定哪些帧/token 有效、哪些被监督；
2. **噪声/timestep mask**：决定哪些位置被加噪、以什么 t 加噪；
3. **注意力 mask**：决定 token 之间能否互相看见（与 Stage 1 同一套 segmented
   metadata，见 stage1.md 第 5 节）。

### 6.1 数据侧 mask

`ConsistencyTrainingPipeline` 从 `base_input` 读回的 mask 就是 Stage 1 文档 2.2 节
介绍的那四个 mask（`video_latent_loss_mask` / `video_latent_valid_mask` /
`action_loss_mask` / `action_valid_mask`），以及 geometry 侧
`geometry_group_valid_mask`。形状与语义完全一致：

```text
frame:      H0 H1 H2 H3 | T0 T1 T2 T3
video loss:  0  0  0  0 |  0  1  1  1
action loss: token 级，只覆盖 T1..T3 的有效 token
```

T0（frame 4）是已知 anchor：不加噪、不算 loss，只作为条件。

### 6.2 `VAMasks` 与 `frame_mask` 组合

`distillation/schema.py` 定义：

```python
@dataclass(frozen=True)
class VAMasks:
    video: torch.Tensor            # [B,F]
    action: torch.Tensor           # [B,Ca,F,N,1]

    def frame_mask(self):
        video = self.video.reshape(self.video.shape[0], -1)
        return video | self.action.any(dim=(1, 3, 4))
```

含义：**只要 video 或 action 在某一帧上有任一监督 token，该帧就采样非零 timestep**。
Stage 2 的 `_sample_timesteps` 内联实现了同一逻辑（video 直接用 `masks.video`，
action 用 `masks.action.any(dim=(1,3,4))`）；`frame_mask()` 这个统一入口被 Stage 3
复用。这样 video/action 的监督范围即使不同，也不会出现"action 要监督但该帧
timestep 全是 0"的情况。

### 6.3 mask 在加噪中的作用

`add_noise_to_va()` 是 mask 语义最关键的一步：

```python
noisy_video = add_noise(clean.video, noise.video, timesteps.video, video_scheduler)
noisy_action = add_noise(clean.action, noise.action, timesteps.action, action_scheduler)
return VAPrediction(
    video=torch.where(masks.video[:, None, :, None, None, None], noisy_video, clean.video),
    action=torch.where(masks.action, noisy_action, clean.action),
)
```

它保证：

- `video_latent_loss_mask=True` 的帧：`x_t = (1-σ)x0 + σ·noise`（σ 来自该帧采样的 t）；
- `video_latent_loss_mask=False` 的帧（history + T0 + padding）：**显式**保持 clean，
  不依赖"timestep=0 恰好 sigma=0"；
- action 未选中的元素（condition 或 padding token）：同样逐元素恢复 clean。

### 6.4 mask 在 loss 中的作用

`distillation/model/objectives.py`：

**`_video_mse`**：对 `[B,Cv,F,V,H,W]` 先做 elementwise FP32 MSE，再按帧把
channel/view/space 求均值（`[B,F]`），用 `video_latent_loss_mask` 把非监督帧清零，
每个样本按有效监督帧数归一，最后 batch 平均。这样 view 数或 latent 分辨率不会
放大 video loss。

**`_action_mse`**：mask 展开到 `[B,Ca,F,N,1]` 全形状；每帧先除以该帧有效 token
数，再只对至少含一个有效 token 的帧求均值。padding 多的样本不会因为分母包含
padding 而得到更小的 loss。

**空 mask 安全性**：全 False 样本通过 `clamp_min(1)` 得到有限的零 loss，不 assert、
不除零。

**`action_aware_loss`**：直接 `F.mse_loss(...)` 后 `torch.where(mask, loss, 0).sum() /
mask.sum().clamp_min(1)`。

### 6.5 注意力 mask：与 Stage 1 完全相同

teacher、EMA、student 三个 forward 都走：

```text
_prepare_joint_input_dict(add_noise=False)
  -> AutoregressiveThreeDVAMOTTransformer3DModel.forward(mode="train")
  -> _prepare_metadata -> _apply_segmented_order
  -> 同一个 segmented_history_strict_geometry_v1 可见性规则
```

因此 Stage 2 的**注意力 mask 结构与 Stage 1 完全一致**（order 表
`V=[0,0,0,0,2,4,6,8]`、`A=[1,1,1,1,3,5,7,9]`、G 只读 committed G、X 读 G 严格过去），
详细构造见 stage1.md 第 5 节。Stage 2 改变的不是"谁能看见谁"，而是：

- 哪些位置被加噪（噪声 mask：`video_latent_loss_mask` / `action_loss_mask`）；
- 输入里的 `noisy_latents` / `latent` / `timesteps` 被 `replace_va_streams` 换成
  consistency 轨迹；
- 哪些位置参与 loss（同一组 mask）。

一个容易混淆的点：`replace_va_streams` 只替换 V/A 轨迹字段，text、geometry、
mask 和 window metadata 都保留自 `base_input`。`action_dict["targets"]` 保留一个
形状载体（模型 reshape 输出需要），但 consistency/replay/DMD loss 从不读取它的值。

### 6.6 训练期 rollout 的增量 mask（`self_rollout`）

Stage 2 每 `rollout_interval` 个成功 optimizer step 用 EMA student 做一次
few-step 生成（默认 2 步 video / 2 步 action，生成 T1..T3）。这个路径的 mask 是
**增量缓存语义**，与训练方阵 mask 不同：

提交顺序（`distillation/self_rollout/engine.py`）：

```text
history:  clean V[0..3] -> G[0..3]（同一个 transaction）-> clean A[0..3]
anchor:   clean V[4]    -> G[4]                          -> clean A[4]
frame f:  predict V[f] -> commit clean V[f]
          derive G[f]  -> commit G[f]
          predict A[f] -> commit clean A[f]
```

每个 token 的可见性由 `build_token_metadata` + `build_cache_visibility` /
`build_cache_selection` 决定：

- 已提交的 K/V 是历史，对后续所有 video/action query 可见；
- 预测（denoise 中）的 noisy K/V 只存在于临时 transaction，`discard_transaction`
  后立即消失，commit 只保留 clean V/A 与 G 的 K/V；
- **G query 只能读 committed G 加上自己当前 transaction**（`STREAM_GEOMETRY`
  过滤），永远不能读 V/A key；history 的 G 全部共享一个 transaction，因此
  history G 之间互见，而 anchor/rollout 每帧一个 transaction，形成严格
  "只看已提交 + 自己" 的阶梯可见性；
- video/action query 可读所有 committed stream + 自己 transaction 的 key；
- `indexed_attention` 先按 `query_valid` / `key_valid` 压缩无效行再执行无 mask
  SDPA，等价于把完整矩形 mask 的可视部分单独计算。

rollout 期间每帧传入的有效性 mask：

```python
_frame_video_valid   -> batch["video_latent_valid_mask"][:, frame_id]   # [B,1]
_frame_action_valid  -> batch["action_valid_mask"][:, :, frame_id]      # [B,Ca,1,N,1]
_frame_geometry_valid-> batch["geometry_group_valid_mask"][:, frame_id] # [B,1,S]
```

它们被写入 token metadata 的 `valid_ids`：无效 token 既不参与 query 也不参与 key，
action 采样时还会用 `valid` 把噪声初始化为 0（`sample * valid`），避免在 padding
位置消耗随机数。

## 7. 训练循环、梯度与 EMA 更新

### 7.1 `DistillationTrainerBase._train_step`

Stage 2 复用 `DistillationTrainerBase._train_step()`（不是 Stage 1 的原生
`_train_step`）：

```text
batch -> convert_input_format -> _materialize_batch_latents
target = _optimization_target()                     # stage2 恒为 student
result = _compute_training_step(batch, target)      # ConsistencyModel.compute_step
loss / gradient_accumulation_steps -> backward
累积边界：clip_grad_norm_(max_grad_norm=2.0)
         全局 finite 检查（任一 rank 非有限则跳过并清梯度）
         通过后 optimizer.step() -> _after_optimizer_step(target)
```

`OptimizationTarget` 同时携带 model 与 optimizer，Stage 3 会用同一个机制切换
student/fake-score；Stage 2 永远返回 student。

### 7.2 EMA 更新时机

```python
def _after_optimizer_step(self, target):
    super()._after_optimizer_step(target)           # student LR scheduler
    self.method_model.after_student_step(self.transformer)

def after_student_step(self, student):
    update_ema(self.ema_student, student, decay=self.ema_decay)
```

EMA 公式（`distillation/model/utils.py`）：

```text
ema <- decay * ema + (1 - decay) * student
```

只在 **student optimizer.step 成功后** 执行：NaN/Inf、被跳过的 step、未到梯度
累积边界都不会更新 EMA。顺序是先更新参数再执行 hook，因此 EMA 读到的是本次
新 student 参数。

### 7.3 计数

父循环 `MOTTrainer.train()` 看到 `optimizer_step_event` 才递增 `optimizer_step`，
并每 `save_interval` 调用 `save_checkpoint()`（Stage 2 走
`DistillationCheckpointIO.save`）。`_maybe_run_training_rollout` 在成功的
optimizer step 后按 `rollout_interval` 触发。

## 8. 训练期 rollout 产物

`ConsistencyTrainer._run_rollout` 把 EMA student 的 `masked_attn_backend` 临时切到
`dense`（`temporary_masked_attention_backend`），然后调用 `self_rollout`。rank 0
在 `save_root/rollouts/step_XXXXXXXX/` 保存：

- `rollout_target_vs_generated.mp4`：所有 view 的 target/generated 并排视频；
- `rollout_actions.png`：动作通道 target/generated 轨迹；
- `rollout.pt`：生成/目标 latent、action、geometry tensor；
- `metadata.json`：来源、版本、替换记录、缓存 token 数等诊断。

`rollout_gt_mode="none"` 时不做 GT 替换；`ground_truth_provider` 接口（offline /
provider 模式）留给需要部分 GT 注入的用法。

## 9. checkpoint 与 resume

`DistillationCheckpointIO.save`（`distillation/checkpoint.py`）：

```text
checkpoint_step_N/
├── transformer/                  # EMA student export（下一阶段消费）
├── distributed_state/            # DCP：raw student + optimizer + EMA student
├── training_state.pt             # step、optimizer_step、LR、RNG、method state
├── checkpoint_metadata.json      # method/architecture/generation_profile 校验
└── _SUCCESS
```

关键点：

- `transformer/` 导出的是 **EMA student**，但 DCP 的 `model` 是 **raw student**
  （另存 `ema_student` 键），这样"交给 Stage 3 的稳定模型"与"Stage 2 精确续训
  状态"可以同时满足；
- resume 时 teacher 不从 checkpoint 读，而是从配置的 `teacher_checkpoint` 来源
  重建（它是冻结的，不会变化）；
- `load()` 会校验 `distill_method`、`model_architecture`、`generation_profile`，
  再用 DCP 恢复 raw student/optimizer/EMA，并恢复 step、LR、每 rank RNG 与
  sampler offset。

## 10. 当前实现的关键注意点

- **两个 mask 层面不要混淆**：注意力 mask（谁能看见谁）三段共用同一套
  `segmented_history_strict_geometry_v1`；噪声/loss mask（谁被加噪、谁算 loss）
  是 Stage 2 自己组合的。
- **clean 保持靠 `torch.where`**：`extra_one_step=True` 时数值 `t=0` 不保证严格
  `sigma=0`，所有 condition/anchor/padding 都必须由 mask 显式恢复 clean。
- **`t_next` 一定更低噪声**：采样范围排除尾部，避免 `t_next == t` 的退化 pair。
- **V/A 分别调度**：video 与 action 使用各自 scheduler、各自 `num_steps`；action
  帧是否采样由该帧任一有效 action token 决定。
- **teacher CFG 只用于 video**：action teacher 保持 conditional prediction；
  `cfg_prob=0` 防止条件被随机替换为空文本。
- **student 只在原始 `x_t` forward 一次**：action-aware 辅助项复用该输出，不增加
  forward；teacher/EMA forward 全部在 `no_grad` 中。
- **EMA 只在成功 step 后更新**：失败、跳过、累积中途都不更新。
- **rollout 是独立增量路径**：不调用 `inference/mot_inference.py`，不修改
  `wan_va/`；预测 transaction 用完即弃，commit 顺序是因果边界。
