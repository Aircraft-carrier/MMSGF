# Stage 1：自回归训练（`autoregressive_training`）

本文讲解蒸馏流水线的第一阶段 `autoregressive_training`。它会从初始 checkpoint 出发，
用"分段历史 + 严格几何顺序"（`segmented_history_strict_geometry_v1`）的注意力语义训练
一个自回归 MOT 模型，产出后续阶段消费的 AR checkpoint。

全文先讲整体架构，再逐层进入代码：数据 mask → token validity → 注意力 metadata →
segmented order → 可见性规则 → 后端 mask，最后回到训练循环和 checkpoint。

## 1. 阶段定位与整体架构

### 1.1 在蒸馏流水线中的位置

`distillation/workflow.py` 把完整蒸馏定义成三个串行方法：

```text
autoregressive_training       （Stage 1，本文）
  -> consistency_distillation （Stage 2，见 stage2.md）
  -> self_gradient_forcing_dmd（Stage 3）
```

Stage 1 的输入是初始 checkpoint（`distill.student_init`），输出是
`checkpoint_step_N/` 下的 `transformer/` export，称为 AR checkpoint。Stage 2 把它同时
当作 student 初始化与 frozen teacher；Stage 3 把它当作 fake-score 初始化。

### 1.2 入口与启动链

```text
1shell/distill/train_distill_autoregressive_training_4gpu.sh
  -> 1shell/distill/_train_distill_common.sh
     （torchrun，DISTILL_METHOD=autoregressive_training）
  -> python -m distillation.train --method autoregressive_training ...
  -> distillation/train.py: AutoregressiveTrainer(config)
  -> wan_va.train_mot.MOTTrainer.train()
```

`distillation/train.py` 只做三件事：按 `--method` 选择 config、把 CLI 参数写入
`config.distill`、实例化 trainer。Stage 1 没有独立的训练循环。

### 1.3 代码地图

| 文件 | 责任 |
| --- | --- |
| `distillation/trainer/autoregressive.py` | Stage 1 trainer：换模型类、装 generation profile、注入 window 参数、写蒸馏 metadata |
| `distillation/model/autoregressive_mot.py` | AR 模型族：segmented order 覆盖、自回归 forward 边界（含 `mode="self_rollout"`） |
| `distillation/model/autoregressive_vggto.py` | AR VGGTO 塔：segmented inter-frame 关系 mask 与增量缓存路径 |
| `distillation/model/autoregressive_types.py` | `AutoregressiveProfile` 与请求/输出 dataclass |
| `distillation/self_rollout/attention.py` | order 计算（`segmented_orders`）、token metadata、可见性规则 |
| `distillation/mask_profile.py` | generation profile 契约与 checkpoint 校验 |
| `distillation/configs/autoregressive_training.py` | Stage 1 配置 |
| `wan_va/train_mot.py` | 被复用的 dataset/VAE/dataloader/FSDP/optimizer/训练循环/checkpoint |
| `wan_va/modules/mot_attention.py` | 原生注意力 metadata 与 dense/flex/FA4 mask |
| `wan_va/modules/model_3dva_mot.py` | 原生 MOT 模型与 token validity 折叠 |
| `wan_va/dataset/mot_dataset.py` | 数据侧 V/A/G mask 的来源 |

### 1.4 架构总览

一次 Stage 1 训练 microstep 的完整数据流：

```text
dataloader batch（含 4 个 V/A mask + geometry mask）
  -> convert_input_format：递归搬到 GPU
  -> _materialize_batch_latents：有缓存 latent 直接用，否则 VAE 编码
  -> _prepare_joint_input_dict：
       video/action 分别采样 timestep 并加噪（clean condition 保持 clean）
       mask/text/geometry/stream_ids 归一化
  -> model(input_dict, mode="train")：
       V/A/G 三个流 joint forward
       _prepare_metadata 生成 x_meta/mot_meta
       AR 模型把 order_ids 覆盖成 segmented 顺序
       MOT block 按 metadata 执行受限注意力
  -> compute_loss：video/action（可选 geometry）加权 MSE
  -> backward -> clip(2.0) -> optimizer.step -> lr_scheduler.step
```

### 1.5 为什么 Stage 1 几乎不改 `MOTTrainer`

`AutoregressiveTrainer` 继承 `wan_va.train_mot.MOTTrainer`，只重写了四个点：

1. `transformer_model_cls = AutoregressiveThreeDVAMOTTransformer3DModel`：换模型族；
2. `__init__`：把 `distill.resume_from / student_init` 翻译成父类认识的
   `config.resume_from / initialize_from`，并安装 generation profile；
3. `_prepare_joint_input_dict`：在原生输入基础上补 `chunk_size / window_size`
   （这两个值本来也来自原生 spec，这里显式保证来自蒸馏配置）；
4. `_write_checkpoint_metadata`：在原生 MOT metadata 上追加蒸馏字段。

因此 dataset、VAE、view-aware sampler、FSDP、AdamW、LR、训练循环、NaN 处理和
checkpoint 全部复用原生实现。这是"自回归训练"阶段保持简单的原因：它只负责把
**注意力 order 语义**换成蒸馏需要的分段自回归顺序，损失仍是原生 flow-matching MSE。

## 2. 训练窗口与数据侧 mask

### 2.1 一个训练窗口的物理形状

当前 MOT 配置下每个样本是 8 个 latent frame：

```text
frame:      H0 H1 H2 H3 | T0 T1 T2 T3
角色:        history      | anchor 生成目标
video loss:  0  0  0  0  |  0  1  1  1
```

- `latents`：`[B,48,8,V,Hl,Wl]`（48 是 video latent channel，`V` 是原生相机视角数）；
- `actions`：`[B,20,8,16,1]`（20 个 action channel，每 latent frame 打包 16 个 action token）；
- T0 是已知锚点帧（condition，不加噪、不算 loss），真正被监督/生成的是 T1..T3。

这些形状由 `wan_va/mot_spec.py` 的 `MOTWindowSpec` 和 `mot_config.json` 决定，蒸馏代码
不重新定义。

### 2.2 四个 V/A mask 的来源（`wan_va/dataset/mot_dataset.py`）

`MotTrainData._getitem_window()` 在数据加载时就构造好四个 mask：

**`video_latent_valid_mask` `[B,F]`**

把原始 RGB frame 的有效性折叠到 latent frame 级别
（`_latent_valid_mask_from_sampled_frames`）：第一帧直接取原始 mask；后续每 4 个
raw frame 对应 1 个 latent frame，只要这 4 帧里有一个有效，该 latent frame 就有效
（"只有全 padding 才为 False"）。

**`video_latent_loss_mask` `[B,F]`**

```python
video_latent_loss_mask = torch.zeros_like(video_latent_valid_mask)
target_start = self.latent_frames_per_action_chunk_per_view + 1   # 4 + 1 = 5
video_latent_loss_mask[target_start:] = video_latent_valid_mask[target_start:]
```

即：历史帧和锚点帧 T0（frame 4）一律不算 video loss，只有 T1..T3（frame 5..7）
在数据有效时才参与。

**`action_valid_mask` `[B,Ca,F,N,1]`**

来自 `_load_actions()`：`keep_mask = action_loss_mask | condition_mask`，即
history 的 condition action token 和 target 的 supervised action token 都是"有效"的。

**`action_loss_mask` `[B,Ca,F,N,1]`**

只标记 target chunk 中真正被监督的 token。`_load_actions()` 里 action 相对
latent frame 有一个固定的偏移：

```python
latent_offset = 1 + action_offset // tokens_per_frame   # 每个 chunk 的 action 整体后移一帧
```

结果就是 `mot_attention.py` 文档注释描述的结构：

```text
全局 latent 0：空（padding/condition slot）
全局 latent 1～3：history action（condition，不算 loss）
全局 latent 4：空
全局 latent 5～7：target action（supervised）
```

这个"A0 是 padding/condition"的偏移是整个注意力语义（尤其 action 的奇数 order）
成立的前提。

另外还有几何侧 mask：

- `geometry_group_valid_mask` `[B,G,S]`：每个 geometry group（G）里哪些 slot（S）有效；
- `geometry_point_valid_mask` `[B,G,S,V,Hg,Wg]`：point label 有效像素；

`geometry_group_valid_mask` 由 `_group_mot_frames` 的 group mask 与 padding 有效性按位
与得到：`geometry_group_mask & geometry_flat_valid_mask`。Stage 1 默认 `optimization
composition="v"`（见 3.3）时 geometry 只作为条件流参与 forward，不产生 geometry loss。

### 2.3 mask 形状总表

| 字段 | 形状 | 语义 |
| --- | --- | --- |
| `video_latent_valid_mask` | `[B,F]` | latent frame 在数据上是否有效（含 padding 判定） |
| `video_latent_loss_mask` | `[B,F]` | 是否参与 video loss（只覆盖 T1..T3） |
| `action_valid_mask` | `[B,Ca,F,N,1]` | history condition + target supervised 都有效 |
| `action_loss_mask` | `[B,Ca,F,N,1]` | 是否参与 action loss（只覆盖 target token） |
| `geometry_group_valid_mask` | `[B,G,S]` | geometry slot 有效性 |
| `geometry_point_valid_mask` | `[B,G,S,V,Hg,Wg]` | point 像素有效性 |

## 3. 加噪、目标与 loss

Stage 1 使用原生 MOTTrainer 的 flow-matching 加噪与 MSE，不引入一致性或 DMD 目标。

### 3.1 视频加噪（`MOTTrainer._add_video_noise`）

```text
每个 target chunk 采样一个 t（uniform），并 repeat 到该 chunk 的每个 latent frame
timesteps = torch.where(video_latent_loss_mask, timesteps, 0)
sigma     = scheduler.sigmas[nearest(timesteps)]
noisy     = (1 - sigma) * latent + sigma * noise
targets   = noise - latent          # flow target（velocity = noise - x0）
noisy     = torch.where(keep, noisy, latent)      # 非监督位置保持 clean
targets   = torch.where(keep, targets, 0)
```

注意：

- `_sample_mot_chunk_timesteps` 会给**所有** target frame（4..7）采样 t，再用 loss
  mask 把 frame 0..4 清零——所以 T0 实际是 `t=0` 的 clean anchor；
- 可选 `video_noisy_cond_prob`：batch 级随机给 clean condition 加中等噪声
  （`cond_timesteps`），当前蒸馏运行脚本一般保留原生默认；
- `extra_one_step=True` 的 scheduler 使 `t=0` 对应一个很小但不严格为 0 的 sigma，
  因此"保持 clean"最终靠 `torch.where(mask, ...)` 显式保证，而不是依赖 timestep 数值。

### 3.2 动作加噪（`MOTTrainer._add_action_noise`）

与视频同构，但 mask 是逐元素 `[B,Ca,F,N,1]`：

```text
action_frame_loss_mask = action_loss_mask.any(dim=(1, 3, 4))   # 每帧只要有有效 token 就采样 t
timesteps = torch.where(action_frame_loss_mask, timesteps, 0)
noisy     = torch.where(action_loss_mask, noisy, actions)      # 未选中的元素保持 clean
targets   = torch.where(action_loss_mask, targets, 0)
```

### 3.3 loss 与优化组合

- video：`_frame_weighted_mse`（帧级归一：先对每帧 channel/view/space 求均值，再按
  该样本有效监督帧数归一，最后 batch 平均）；
- action：`_action_weighted_mse`（token 级 mask 归一）；
- geometry（若启用）：depth/point loss。

`optimization_composition` 决定哪些参数分支可训练、哪些 loss 生效：

- 启动脚本 `1shell/distill/_distill_paths.sh` 默认导出 `MOT_OPTIMIZATION_COMPOSITION=v`，
  因此 Stage 1 默认只训练 video 分支（`apply_mot_parameter_ownership` 按
  `v/a/g` 前缀把 A、G 参数冻结）；
- 需要 action 监督时可在启动环境里覆盖为 `va` 或 `vag`；
- Stage 2 配置固定为 `va`。

## 4. 模型：`AutoregressiveThreeDVAMOTTransformer3DModel`

### 4.1 与原生模型的关系

AR 模型与 `ThreeDVAMOTTransformer3DModel` 参数兼容（`state_dict` 键完全一致，
见 `distillation/tests/test_autoregressive_model_profile.py`），但：

1. `_build_mot_block` / `_build_vggto_tower` 换成 AR 子类，构造期就安装
   `AutoregressiveProfile`，不需要运行时 monkeypatch；
2. `_prepare_metadata` 在原生 metadata 之上做 `_apply_segmented_order` 覆盖；
3. `forward(mode="self_rollout")` 额外支持 Stage 2/3 rollout 需要的
   `predict_video / commit_video / predict_action / commit_action /
   encode_geometry / encode_geometry_history` 增量操作。

### 4.2 组件结构

```text
AutoregressiveThreeDVAMOTTransformer3DModel
├── vggto: AutoregressiveVGGTOGeometryTower
│     ├── patch_embed（DINO ViT 底座 + register tokens）
│     ├── frame_blocks      （逐图像 block）
│     ├── cross_view_blocks （同 frame 跨视角）
│     └── inter_frame_blocks（帧间关系，AR segmented mask）
├── mot_blocks: AutoregressiveThreeDVAMOTBlock[]
│     ├── video_block（WanTransformerBlock）
│     ├── action_block（ActionTransformerBlock）
│     └── geometry（偶数层才有的 GeometryJointStream register 注意力）
├── condition_embedder / action_condition_embedder（timestep + text）
├── rope / action_rope（RoPE 位置）
└── proj_out / action_proj_out（预测头）
```

## 5. mask 构造（重点）

Stage 1 的 mask 有清晰的层级。理解这个链路是理解蒸馏三个阶段的基础，因为
Stage 2/3 复用同一套注意力 mask 语义，只改变"哪些位置加噪、哪些位置算 loss"。

### 5.1 总览：从数据到 kernel 的 mask 链路

```text
数据侧 mask（2.2 节）
  -> token_valid_ids（按 token 折叠，5.2 节）
  -> MOTMaskMetadata（seq/order/stream/noise/frame/window/token_valid，5.3 节）
  -> _apply_segmented_order（覆盖 order_ids，5.4 节）
  -> 可见性规则（_x_to_x/_g_to_g/_x_to_g，5.5 节）
  -> 后端 mask（dense / FlexAttention / FA4，5.6 节）
```

### 5.2 数据 mask -> token validity

`ThreeDVAMOTTransformer3DModel._mot_token_valid_ids()` 把帧级/元素级 mask 折叠成
每个物理 token 的 `[B, L]` bool：

- video：`video_latent_valid_mask[:, :, None].expand(-1, -1, video_tokens_per_frame)`，
  一帧内 `views * h_tokens * w_tokens` 个 token 共享帧有效性；
- action：`action_valid_mask.any(dim=1)[..., 0]`，把 channel 维折叠，得到
  `[B, F, tokens_per_frame]`；
- 物理顺序是 `NV, CV, NA, CA`，所以 x_valid 直接拼 `[video, video, action, action]`；
- MOT（joint）再插入 geometry token：`[NV, CV, G, NA, CA]`；
- geometry register token 由 `_geometry_token_valid_ids` 从
  `slot_valid_mask [B,G,S]` 展开：`[B,G,S,V,register_tokens] -> [B, L]`。

### 5.3 原生 metadata 构造

`wan_va/modules/mot_attention.py` 的 `build_x_metadata` / `build_mot_metadata` /
`build_geometry_metadata` 生成 `MOTMaskMetadata`，每个 token 携带：

| 字段 | 含义 |
| --- | --- |
| `seq_ids` | 样本隔离（packed batch 内不同样本互不可见） |
| `order_ids` | 块时钟（block clock），可见性比较的主轴 |
| `stream_ids` | `STREAM_VIDEO=0 / STREAM_ACTION=1 / STREAM_GEOMETRY=2 / STREAM_PAD=-1` |
| `noise_ids` | `NOISE_NOISY=0 / NOISE_CLEAN=1 / NOISE_GEOMETRY=2` |
| `frame_ids` | 该 token 的帧号（蒸馏 order 覆盖依赖它） |
| `token_valid_ids` | 5.2 节折叠出的有效性 |
| `window_size` | order 差窗口约束 |

原生顺序（chunk-causal）是 `order = (frame // chunk_size) * 2`（video/G）或
`+1`（action），即每个 chunk 内部所有帧共享同一个 order，chunk 之间严格递增：

```text
原生 chunk order（C=4, F=8）:
V/G:  0 0 0 0 | 2 2 2 2
A:    1 1 1 1 | 3 3 3 3
```

### 5.4 segmented order：蒸馏的核心覆盖

`AutoregressiveThreeDVAMOTTransformer3DModel._apply_segmented_order()` 把
`order_ids` 换成 `distillation/self_rollout/attention.py::segmented_orders()` 的结果：

```python
history_segments = ceil(history_frames / chunk_size)
history_order(f) = floor(f / chunk_size) * 2                      # f < history_frames
target_order(f)  = 2 * history_segments + 2 * (f - history_frames) # f >= history_frames

video_order = segmented_orders(...)
action_order = video_order + 1
```

默认窗口（`history_frames=4, chunk_size=4, window_size=16`）得到：

```text
frame:      H0 H1 H2 H3 | T0 T1 T2 T3
V/G order:   0  0  0  0 |  2  4  6  8
A order:     1  1  1  1 |  3  5  7  9
loss mask:   0  0  0  0 |  0  1  1  1
```

语义：

- **history 是一个完整 chunk**：4 帧共享 order，chunk-causal 规则使它们双向可见
  （`k_order <= q_order`），与原生 chunk 行为一致；
- **target 严格逐帧推进**：T0 之后每帧一个独立 order（`2,4,6,8`），因此每帧只能看
  到更早的帧和同帧自身，实现 frame-level 自回归；
- **action 永远比同帧 video/G 大 1**：奇数 order 让 action 能看到"当前帧"的视觉
  条件（见 5.5），对应数据集 action 后移一帧的 inverse-dynamics 约定；
- `_apply_segmented_order` 同时清掉 `cache_key / structure_cache_key`，避免复用
  原生结构的缓存 metadata。

Profile 本身有版本号与契约校验：

- `distillation/mask_profile.py` 的 `PROFILE_VERSION = 2`（v2 把 G->G 从严格帧历史
  改为 segmented order-causal 可见性）；
- `generation_profile_contract()` 生成可序列化契约；`validate_checkpoint_generation_profile()`
  在跨阶段加载 checkpoint 时校验 `checkpoint_metadata.json` 中的 profile 与当前配置一致，
  防止旧 checkpoint 静默复用新拓扑。

### 5.5 可见性规则

`mot_attention.py` 用 `order_ids` 定义三条核心规则（`_x_to_x` / `_g_to_g` / `_x_to_g`），
配合 `seq_ids`、`window_size`、`token_valid_ids`：

```text
_x_to_x（V/A 之间，LingBot 兼容）：
  clean -> clean: k_order <= q_order
  noisy -> clean: k_order <  q_order
  noisy -> noisy: k_order == q_order

_g_to_g（G 自包含，chunk-causal）：
  k_order <= q_order

_x_to_g（X 读 G，严格过去）：
  k_order < q_order
```

结合奇数/偶数 order，得到的典型 pair（来自 `mot_attention.py` 模块注释）：

| 查询 | 键 | 允许 | 原因 |
| --- | --- | --- | --- |
| noisy video (NV1) | 当前 G1 | 否 | video order 偶数，`k_order < q_order` 不满足当前帧 |
| noisy video (NV1) | 过去 G0 | 是 | `k_order < q_order` |
| noisy action (NA1) | 当前 G1 | 是 | action order 奇数，能读到同帧视觉条件（inverse dynamics） |
| noisy action (NA1) | 下一帧 G2 | 否 | `k_order < q_order` 不满足 |
| noisy action (NA1) | clean video CV1 | 是 | `k_order < q_order` |
| noisy action (NA1) | 自己的 clean action CA1 | 否 | noisy→clean 要求严格小于 |
| G query | 任何 V/A token | 否 | G 只走 `_g_to_g` |

`window_size` 仍是 config/checkpoint 契约的一部分，但注意：Stage 1 训练用的是
**方阵** metadata（当前窗口内所有 token），window 约束在 `_window` 里生效；而
`self_rollout` 增量路径（Stage 2/3）用**提交顺序**当因果边界，不按 window 截断。

### 5.6 后端 mask

同一份 `MOTMaskMetadata` 由 `attention_from_meta()` 派发到三个后端，语义完全一致：

- `dense`：`build_dense_mot_mask(meta)` 显式构造 `[B,Q,K]` bool 方阵；
- `flex`：`create_flex_mot_block_mask(meta)` 用同一个 `mask_mod` 生成 FlexAttention
  `BlockMask`；
- `fa4`：`fa4_attention_from_meta(q,k,v,meta)`，内部用 FlexAttention 的 block mask
  描述 FA4 CuTe kernel 访问的 128x128 块（训练默认 `masked_attn_backend="fa4"`，
  需要 H100/SM90）。

最后 `attention_from_meta` 还会把输出乘上 `token_valid_ids`，保证 padding token
对输出贡献为零。

### 5.7 VGGTO 帧间（inter-frame）mask

VGGTO 的关系注意力是独立的 G-only 注意力，在
`AutoregressiveVGGTOGeometryTower._build_segmented_inter_frame_mask()` 构造：

```python
frame_ids   = arange(groups).repeat_interleave(group_size)        # 每组 slot 共享帧号
image_orders = segmented_orders(frame_ids, history_frames, chunk_size)
token_orders = image_orders.repeat_interleave(tokens_per_image)
mask = token_orders[None, :] <= token_orders[:, None]             # k_order <= q_order
```

然后叠加 `image_valid_mask -> token_valid`，并对无效查询行用单位矩阵兜底（防止
全 False 行导致 softmax NaN）。FA4 路径对应
`_build_segmented_inter_frame_metadata()`：先 `build_geometry_metadata` 再
`_apply_segmented_order`，保证与 dense 路径同一 order 语义。

## 6. 训练循环

### 6.1 复用 `MOTTrainer.train()`

`MOTTrainer.train()` 是外层循环：

```text
while step < num_steps:
    batch = _get_next_batch()          # view-aware sampler + DataLoader
    losses = _train_step(batch, step_in_accumulation)
    if optimizer_step_event: optimizer_step += 1
    if losses["should_log"]: 聚合日志、W&B
    if step 命中 save_interval: save_checkpoint()
    step += 1
```

`_train_step()` 内部：`convert_input_format -> _materialize_batch_latents ->
_prepare_joint_input_dict(add_noise=True) -> model(input, mode="train") ->
compute_loss -> backward -> clip_grad_norm_(2.0) -> optimizer.step() ->
lr_scheduler.step()`。Stage 1 没有 EMA、没有 rollout hook：`AutoregressiveTrainer`
直接继承 `MOTTrainer`（不是 `DistillationTrainerBase`），训练循环里不存在
`_maybe_run_training_rollout` 这类调用。

### 6.2 `AutoregressiveTrainer` 的注入点

```python
def _prepare_joint_input_dict(self, batch_dict):
    input_dict = super()._prepare_joint_input_dict(batch_dict)   # 原生加噪 + 校验
    input_dict["chunk_size"] = self.config.distill.generation_shape["chunk_size"]
    input_dict["window_size"] = self.config.distill.generation_shape["window_size"]
    return input_dict
```

这两个键正是 `_prepare_metadata` / `_embed_geometry` 读取的窗口参数。蒸馏把它们
固定为配置值，保证 Stage 1/2/3 的注意力图完全一致。

## 7. checkpoint 与 resume

Stage 1 使用父类 `MOTTrainer.save_checkpoint()`，目录结构：

```text
checkpoint_step_N/
├── transformer/                  # 下一阶段 from_pretrained 的 export
├── distributed_state/            # DCP sharded model/optimizer
├── training_state.pt             # step、LR、RNG、sampler offset
├── checkpoint_metadata.json      # MOT 字段 + 蒸馏字段
└── _SUCCESS
```

`AutoregressiveTrainer._write_checkpoint_metadata()` 在原生 metadata
（`checkpoint_type=mot_training`、`vggto_attention_topology`、
`optimization_composition`、`has_full_state`）上追加：

```json
{
  "distill_method": "autoregressive_training",
  "exported_model": "student",
  "step": 2000,
  "optimizer_step": 2000,
  "model_architecture": "autoregressive_mot_v1",
  "generation_profile": {
    "profile_name": "segmented_history_strict_geometry_v1",
    "profile_version": 2,
    "order_mode": "segmented",
    "history_frames": 4,
    "chunk_size": 4,
    "window_size": 16,
    "geometry_relation": "segmented_order_causal",
    "x_to_g_relation": "strict_order"
  }
}
```

`transformer/config.json` 里也会写入 `generation_profile`。跨阶段加载时
`validate_checkpoint_generation_profile()` 会逐字段比对，防止 profile 漂移。

resume 与 fresh-init 的语义：

```python
if config.distill.resume_from is not None:
    config.resume_from = str(config.distill.resume_from)      # 原生 DCP resume
elif config.distill.student_init is not None:
    config.initialize_from = str(config.distill.student_init) # 从 export 初始化
```

## 8. 与后续阶段的关系

Stage 1 的产出（`transformer/` export + 带 profile 的 metadata）被后续阶段消费：

| 消费方 | 用途 |
| --- | --- |
| Stage 2 `student_init` | raw student 与 EMA student 的初始化 |
| Stage 2 `teacher_checkpoint` | frozen teacher（CFG + flow 推进） |
| Stage 3 `fake_score_init` | DMD fake-score 模型初始化 |

因此 Stage 1 的 `generation_profile` 必须与 Stage 2/3 完全一致；任何 order 语义的
修改都会在 checkpoint 校验阶段被拒绝。
