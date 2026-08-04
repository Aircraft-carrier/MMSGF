# Consistency Distillation 实现与数据流程

本文说明 `consistency_distillation` 的当前真实实现。它是 workflow 中位于 autoregressive training 之后、self-gradient-forcing DMD 之前的训练方法，但代码和配置不使用数字阶段名。

## 1. 目标与输出

Consistency Distillation 的目标是把 autoregressive checkpoint 中的 teacher 能力压缩到 student，并维护一份 EMA student 作为稳定目标和最终导出模型。

一次训练更新完成以下计算：

```text
clean V/A
  -> sample t and noise
  -> x_t
  -> frozen teacher predicts conditional/unconditional flow at t
  -> video teacher CFG; action keeps conditional flow
  -> move x_t directly to x_t_next
  -> frozen EMA student predicts normalized consistency target at t_next

x_t
  -> trainable student predicts normalized consistency output at t
  -> masked V/A consistency loss against EMA target
  -> student optimizer
  -> EMA update after a successful optimizer step
```

checkpoint 的 `transformer/` 导出 EMA student。DCP 中同时保存 raw student、student optimizer 和 EMA student，以便同一训练方法精确 resume。

## 2. 代码边界

核心文件如下：

| 文件 | 责任 |
| --- | --- |
| `distillation/trainer/consistency.py` | 构造 ConsistencyModel；student 更新成功后触发 EMA |
| `distillation/trainer/base.py` | 数据搬运、latent materialization、梯度累积、finite check、clip、日志契约和 checkpoint 入口 |
| `distillation/model/consistency.py` | 持有 frozen teacher、frozen EMA student 和 consistency pipeline |
| `distillation/pipeline/consistency_training.py` | 执行 teacher → next noisy state → EMA target → student prediction |
| `distillation/model/objectives.py` | V/A mask-aware MSE 和加权总 loss |
| `distillation/pipeline/utils.py` | V/A 分 modality 加噪、条件帧保持 clean、替换模型输入流 |
| `distillation/scheduler.py` | 支持 `[B,F]` timestep 的 sigma lookup、flow step、flow-to-x0 和 Flash-WAM boundary scaling |
| `distillation/rollout.py` | 从 batch 做逐 chunk V→G→A rollout，并保存视频、动作图和 tensor |
| `distillation/model/factory.py` | 从已有 transformer export 加载、冻结并按原生 FSDP 规则配置 teacher/EMA |
| `distillation/checkpoint.py` | raw student、optimizer、EMA、scheduler、RNG 和训练计数的原子保存与恢复 |

pipeline 不复制 `wan_va` 的 dataset、VAE、geometry input、text conditioning、attention metadata 或 FSDP 实现。

## 3. 模型所有权

| 模型 | 初始化来源 | requires_grad | 模式 | 保存方式 |
| --- | --- | --- | --- | --- |
| student | `student_init/transformer` | V/A 分支可训练；G 分支冻结 | train | DCP raw model；不直接作为跨方法 export |
| teacher | `teacher_checkpoint/transformer` | 全冻结 | eval | 不重复保存；resume 时从来源 checkpoint 重建 |
| EMA student | `student_init/transformer`，之后做 EMA | 全冻结 | eval | DCP 保存；同时导出到 `transformer/` |

配置会把 `optimization_composition` 固定为 `va`。geometry/VGGTO 只作为条件流参与 forward，不单独训练 geometry objective，也不会更新 G-owned 参数。

teacher 和 EMA 调用 `model(input_dict, mode="train")` 中的 `mode="train"` 只是选择 MOT 的 joint training forward 路由；模型对象本身仍为 `eval()` 且所有参数均 `requires_grad=False`。

## 4. 入口与 trainer 生命周期

单方法入口为：

```bash
python -m distillation.train \
  --method consistency_distillation \
  --student-init /path/to/autoregressive/checkpoint \
  --teacher-checkpoint /path/to/autoregressive/checkpoint \
  --save-root /path/to/output
```

rollout 参数可以直接从同一入口覆盖：

```bash
  --rollout-interval 500 \
  --rollout-video-num-steps 2 \
  --rollout-action-num-steps 2 \
  --rollout-chunk-pairs 1
```

`ConsistencyTrainer` 继承 `DistillationTrainerBase`，后者继续复用 `MOTTrainer.train()`。初始化顺序是：

1. 如果指定 `resume_from`，先把 checkpoint root 作为 parent transformer 初始化来源，避免 parent 使用原生单模型 resume 路径。
2. `MOTTrainer.__init__()` 创建 dataset、view-aware sampler、student、FSDP student optimizer、LR scheduler、VAE 和 flow schedulers。
3. `ConsistencyTrainer._build_method_model()` 加载 teacher 和 EMA student，并按原生 `shard_mot_model` 边界配置 FSDP。
4. 如果是 resume，再通过 `DistillationCheckpointIO.load()` 覆盖 raw student、optimizer、EMA、scheduler、RNG 和计数状态。

## 5. 原始 batch 到模型输入

### 5.1 数据读取与设备搬运

父训练循环通过 `MOTTrainer._get_next_batch()` 从现有 view-aware dataloader 获取 batch。蒸馏 trainer 随后执行：

```text
batch from dataloader
  -> convert_input_format(batch)
  -> recursively move tensors to current CUDA device
  -> _materialize_batch_latents(batch)
  -> use cached latents, or VAE-encode history/target RGB
```

如果 batch 已有 `latents`，不会重复调用 VAE。如果只有 `vae_rgb_history` 和 `vae_rgb_target`，则复用 `MOTTrainer._encode_vae_rgb()`，逐 batch、逐 view 使用 `WanVAEStreamingWrapper` 编码，并沿 frame 轴拼接 history latent 与 target latent。

### 5.2 主要 tensor 形状

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `latents` | `[B,Cv,F,V,H,W]` | clean video latent；`V` 是原生相机视角数 |
| `actions` | `[B,Ca,F,N,1]` | clean action token；当前模型通常 `Ca=20` |
| `video_latent_loss_mask` | `[B,F]` | 哪些 latent frame 参与 video loss |
| `video_latent_valid_mask` | `[B,F]` | 哪些 latent frame 在数据上有效 |
| `action_loss_mask` | `[B,Ca,F,N,1]` | 哪些 action 元素参与 loss |
| `action_valid_mask` | `[B,Ca,F,N,1]` | 哪些 action 元素有效 |
| `text_emb` | 原生 MOT text embedding shape | conditional 文本条件；stage2 关闭随机 CFG dropout |
| `empty_text_emb` | 与 `text_emb` 单样本 shape 相同 | dataset 注入的 unconditional 文本条件；pipeline 也兼容 trainer fallback |
| `stream_ids` | `[B,V]` | LEFT_WRIST/HEAD/RIGHT_WRIST 视角标识 |
| `geometry_rgb` | `[B,G,S,V,3,Hg,Wg]` | geometry tower 的图像组输入 |
| `geometry_pts3d` | `[B,G,S,V,Hg,Wg,3]` | geometry label；蒸馏不计算 geometry loss |
| `geometry_point_valid_mask` | `[B,G,S,V,Hg,Wg]` | point label 有效像素 |
| `geometry_group_valid_mask` | `[B,G,S]` | geometry slot 有效性 |

精确的 `F`、`N`、`G`、`S` 由现有 `MOTWindowSpec` 和 action sequence 配置决定，不在 distillation 中重新定义。

### 5.3 复用原生 input 构造但跳过冗余噪声

pipeline 调用：

```python
trainer._prepare_joint_input_dict(batch, add_noise=False)
```

该路径仍复用原生实现完成：

- loss/valid mask dtype 和 shape 处理；
- conditional/empty text embedding 整理；
- geometry RGB、point、valid mask 和 slot mask 整理；
- `stream_ids` 注入；
- 原生物理 window shape 校验，然后由 distillation trainer 把 attention metadata
  覆盖为所有阶段一致的 `segmented_history_strict_geometry_v1` profile；
- `validate_mot_batch_for_forward()` 完整 shape 校验。

`add_noise=False` 只跳过原生 autoregressive objective 使用的随机 video/action noise、flow target 和 timestep，因为 consistency pipeline 会立即按 modality 构造自己的 timestep 与噪声。如果先走原生加噪再覆盖，会浪费显存带宽、随机数和临时 tensor。

这里的 `generation_shape` 只描述 attention order/window，不改变真实 tensor 仍是
8 latent frames、每 frame 16 个 action token 的物理打包。stage1、stage2、stage3
训练 forward 都使用同一 segmented profile。history frame 共享 order，target 从 T0
开始按 `V/G=2,4,6,...`、`A=3,5,7,...` 递增；geometry query 只能读取严格更早
的 geometry frame。

## 6. Mask 组合

pipeline 把 V/A mask 组合为：

```python
VAMasks(
    video=video_latent_loss_mask,       # [B,F]
    action=action_loss_mask,            # [B,Ca,F,N,1]
)
```

统一 frame mask 为：

```text
frame_mask = video_loss_frame OR any(action_loss_token in frame)
```

因此，只要 video 或 action 在某一 frame 上需要监督，该 frame 就会采样有效 timestep。完全不参与 V/A loss 的条件 frame 使用 timestep 0，并由 `add_noise_to_va()` 显式保持 clean，而不是依赖 scheduler 的 timestep 0 恰好对应严格零噪声。

## 7. Timestep 与噪声轨迹

设 scheduler 离散表按高噪声到低噪声排列，共 `S` 个 state，配置步数为
`K=consistency_num_steps`：

```text
stride = clamp(floor(S / K), 1, S - 1)
current_id ~ UniformInteger[0, S - stride)
next_id = current_id + stride
t      = scheduler.timesteps[current_id]
t_next = scheduler.timesteps[next_id]
```

这里的 `t` 是 `[B,F]` tensor，每个样本、每个 frame 可以不同。采样范围明确
排除了不能再向低噪声推进的尾部 state，因此所有监督位置都满足
`next_id > current_id`，不会出现 `t_next == t` 的退化 pair。原生
`FlowMatchScheduler.add_noise()` 和 `step()` 不能正确处理 `[B,F]`，因此
distillation 通过 `distillation/scheduler.py` 做逐 frame sigma lookup。

flow matching 使用：

```text
x_t = (1 - sigma_t) * x0 + sigma_t * noise
flow = noise - x0
```

`add_noise_to_va()` 分别使用 video scheduler 和 action scheduler，并通过 mask 保持 condition frame clean。

## 8. Teacher、EMA 与 student forward

### 8.1 Teacher 推进到相邻状态

teacher 在 `x_t,t` 上预测 V/A flow：

```text
teacher_v_cond, teacher_a_cond = teacher(input_cond)
teacher_v_uncond, _              = teacher(input_empty)
cfg_scale ~ Uniform(cfg_min, cfg_max)
teacher_flow_v = teacher_v_uncond + cfg_scale * (teacher_v_cond - teacher_v_uncond)
teacher_flow_a = teacher_a_cond
```

这与 Flash-WAM 一致：CFG 只用于 teacher video trajectory；action guidance scale 保持 1。stage2 配置把原生 `cfg_prob` 设为 0，避免 conditional teacher/student 输入被随机替换为空文本。

然后直接推进到配置的 `t_next`：

```text
x_t_next = x_t + (sigma_t_next - sigma_t) * teacher_flow
```

这一步使用 `flow_step()`，不会调用原 scheduler 对整个 `[B,F]` 只取一个全局 timestep index 的 `.step()`。

### 8.2 EMA 构造 target

EMA student 在 `x_t_next,t_next` 上预测 flow。action 使用线性 clean estimate：

```text
target_action = x_t_next_action - sigma_t_next_action * ema_flow_action
```

video 使用 Flash-WAM 的方差保持 consistency boundary scaling（默认 `sigma_data=0.5`）：

```text
pred_x0 = x_t - sigma * flow
c_skip  = sigma_data^2 / (sigma^2 + sigma_data^2)
c_out   = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
f_v     = c_skip * x_t + c_out * pred_x0
```

因此 `sigma=0` 时 `f_v=x_t`，高噪声区域的 video CM target/prediction 也具有受控尺度。

teacher forward、EMA forward 和 target 构造都在 `torch.no_grad()` 中。target 还会在 objective 内再次 `detach()`，保证 consistency target 没有反向路径。

### 8.3 Student prediction

student 在原始 `x_t,t` 上预测 flow，并使用与 EMA 完全相同的模态对应函数：

```text
student_video  = boundary_scaled_consistency(x_t_video, student_flow_video)
student_action = x_t_action - sigma_t_action * student_flow_action
```

只有该 forward 保留梯度。

`replace_va_streams()` 会同时替换：

- `latent_dict.noisy_latents` / `action_dict.noisy_latents`；
- `latent_dict.latent` / `action_dict.latent`；
- V/A timestep；
- clean-stream `cond_timesteps=0`。

text、geometry、stream id、mask 和 window metadata 继续来自同一个 `base_input`。

## 9. 归一化后的 Consistency loss

loss 比较的是上一节的 consistency function 输出，而不是直接比较未缩放的 video `x0`。video 使用 `c_skip/c_out` 归一化；action 按 Flash-WAM 的低噪声分支保留线性 x0 参数化。

video loss：

1. 先计算 elementwise FP32 MSE。
2. 对每个 `[B,F]` frame，把 channel、view 和空间维求均值。
3. 只保留 `video_latent_loss_mask=True` 的 frame。
4. 每个样本按其监督 frame 数归一，再对 batch 求均值。

action loss：

1. 计算 elementwise FP32 MSE。
2. 使用完整 `[B,Ca,F,N,1]` mask。
3. 每个 frame 按有效 action token 数归一。
4. 最后只对存在有效 token 的 frame 求均值。

主 consistency loss：

```text
loss = video_loss_weight * video_loss
     + action_loss_weight * action_loss
```

此外，stage2 对 student action flow 加一个精确 `noise-clean` 回归辅助项：

```text
action_target = noise_action - clean_action
training_loss = consistency_loss
              + action_aware_weight * masked_mse(student_action_flow, action_target)
```

默认 `action_aware_weight=0.01`。它复用同一次 student forward，不增加模型调用。

全 False mask 会通过 `clamp_min(1)` 形成有限的零 loss，不使用 assert，也不会产生除零。

## 10. 训练期 rollout

每个成功 optimizer step 后，`rollout_interval > 0` 且命中周期时，所有 rank 使用
EMA student 参与 FSDP forward；rank 0 保存产物。rollout 调用 distillation 自有的
`self_rollout`，不进入 `inference/mot_inference.py`：

```text
GT history V -> sequential strict-history G -> history A
  -> commit GT target anchor T0 V/G/A
  -> sample T1 video -> encode/commit T1 geometry -> sample/commit T1 action
  -> continue T2, T3 ... from committed semantic/cache state
```

调用方也可以直接执行 `trainer.rollout(batch)`；该入口会完成 device transfer、缺失 latent 时的 VAE 编码，并返回 `RolloutResult`。

新接口使用 `rollout_horizon_frames` 表示 T0 之后要生成的逻辑 frame 数，默认是 3，
即生成 T1..T3。`rollout_chunk_pairs` 属于旧 fixed-window rollout；配置中不得同时
出现这两个字段。`rollout_gt_mode` 支持 `none`、`offline` 和由调用方显式传入
provider 的 `provider`，GT replacement 后 prediction artifact 保留，后续 continuation
使用替换后的 canonical state。

rank 0 在 `save_root/rollouts/step_XXXXXXXX/` 保存：

- `rollout_target_vs_generated.mp4`：所有 view 的 target/generated 并排视频；
- `rollout_actions.png`：全部动作通道随时间的 target/generated 轨迹和 chunk 边界；
- `rollout.pt`：生成/目标 latent、action、geometry tensor；
- `metadata.json`：chunk 数、chunk frame 数和产物路径。

启用 W&B 时，视频和动作图同时写入当前 training run。

## 11. Backward、梯度累积与 EMA 更新

`DistillationTrainerBase._train_step()` 使用 `OptimizationTarget` 明确当前 model 和 optimizer。ConsistencyTrainer 始终返回 student target。

训练顺序：

1. 计算 loss 和日志 metrics。
2. 所有 rank 汇总 nonfinite loss 状态；任一 rank 非有限则跳过更新并清梯度。
3. `loss / gradient_accumulation_steps` backward。
4. 非累积边界关闭 FSDP gradient sync；累积边界才 clip 和 optimizer step。
5. 对 student 参数执行 `clip_grad_norm_(..., 2.0)`。
6. 所有 rank 汇总 nonfinite grad 状态。
7. finite 时执行 student optimizer 和 LR scheduler。
8. 只有 optimizer 成功后，`ConsistencyTrainer._after_optimizer_step()` 才执行 EMA：

```text
ema = decay * ema + (1 - decay) * student
```

因此，NaN/Inf、被跳过的 step 或未到梯度累积边界时都不会更新 EMA。

## 12. 日志数据契约

因为继续复用 `MOTTrainer.train()`，distillation trainer 会返回父循环需要的最小兼容字段：

- `total_loss_raw`；
- `latent_loss_raw`、`action_loss_raw`；
- V/A weighted loss 和 loss weight；
- local sample 数、native view 数；
- pointcloud/pure sample 数和 dataset skip count；
- V/A supervised/valid numerator 与 denominator；
- optimizer step、grad clip、nonfinite 和 skipped-step 事件。

这些字段只用于复用父训练循环的聚合与 W&B 路径，不重新实现一套 logger。

## 13. Checkpoint 与 resume

Consistency checkpoint 包含：

```text
checkpoint_step_N/
├── transformer/                 # EMA student export
├── distributed_state/           # raw student + optimizer + EMA student
├── training_state.pt            # scheduler、step、RNG、method state
├── checkpoint_metadata.json     # MOT 初始化兼容字段 + distill_method/profile
└── _SUCCESS
```

保存流程先写临时目录，所有 rank 汇总错误后由 rank 0 写 `_SUCCESS` 并原子 rename。只有包含 `_SUCCESS` 的目录会被 workflow 作为成功 checkpoint。metadata 同时保留
`checkpoint_type=mot_training`、VGGTO topology、`optimization_composition=va` 和
`has_full_state=true`，因此 stage3 的基础 transformer loader 可以直接验证并读取
stage2 export。

resume 恢复：

- raw student 和 student optimizer；
- EMA student；
- LR scheduler；
- `step`、`optimizer_step` 和 nonfinite/skip 计数；
- 每 rank RNG；
- dataloader sampler offset；
- `_last_checkpoint_step` 和 performance 起始 step。

## 14. 当前实现边界

- consistency 在 scheduler 离散表上随机选 current state，并用固定 index stride
  推到严格更低噪声的 state；尾部不会截断成同一点。
- V/A 分别保存 `[B,F]` timestep，并使用各自 scheduler/配置步数；action frame
  是否采样由该 frame 内任一有效 action token 决定。
- clean stream 使用 batch 中的 clean V/A；是否能看到某个 token 仍由 `wan_va.modules.mot_attention` 的原生 metadata 决定，distillation 不复制 attention mask。
- geometry 作为条件输入保留，但 G-owned 参数冻结且无 geometry loss。
- teacher CFG 只指导 video trajectory；action teacher 保持 conditional prediction。
- rollout 复用原生固定窗口 V→G→A inference；多 pair 需要 batch 提供连续 target chunks。
