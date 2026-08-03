# MOT 推理阶段的 Mask 与 Cache：各模态如何交互

本文围绕两个核心机制说明 MOT 模型在推理阶段如何处理 Video、Action、Geometry 三个模态：

1. Mask：决定哪些 token 有效、哪些 token 可以互相读取、哪些数值可以被 scheduler 更新。
2. Cache：决定 geometry 结果和 attention block mask 哪些内容可以复用。

需要区分三种容易混淆的屏蔽：

| 机制 | 作用 | 是否改变输入数值 |
|---|---|---|
| 输入置零 | 进入模型前把目标区域数值设置为 0 | 是 |
| attention mask | 限制 query 可以读取哪些 key/value | 不直接修改输入值，但改变交互 |
| scheduler 更新范围 | 决定 diffusion 状态实际更新哪些位置 | 是 |

例如视频阶段会同时执行：

    video_geometry_rgb[:, target_slice] = 0
    video_geometry_slot_valid[:, target_slice] = False

第一行是输入数值置零，第二行是 token 有效性屏蔽。它们是两种不同的约束。

---

## 1. 完整推理中的模态数据流

完整推理可以表示为：

    GT 历史 RGB/latent/action
                │
                ▼
    1. 预计算历史 geometry condition
                │
                ├── final_geometry
                └── layer_registers
                │
                ▼
    2. 视频 diffusion
       Video + Action + geometry cache 交互
       只更新目标 Video latent
                │
                ▼
    3. 解码视频为 RGB
                │
                ▼
    4. 使用 GT 历史 RGB + 生成 RGB
       重新计算 geometry condition
                │
                ├── final_geometry
                └── layer_registers
                │
                ▼
    5. 动作 diffusion
       Action + 固定 Video + geometry cache 交互
       只更新目标 Action

geometry 并不是每个 video/action diffusion step 都重新运行完整的 VGGTO geometry tower，而是：

    geometry RGB
        → geometry stream
        → final_geometry / layer_registers
        → 被每个 V/A diffusion step 复用

默认 geometry register attention 层为：

    vggto_register_attention_indices = (0, 2, 4, ..., 28)

奇数层只处理 Video/Action；偶数层加入 Geometry registers，执行三模态 joint attention。

---

## 2. Mask 从 batch 到 attention 的传播

推理阶段主要使用以下 mask：

    batch["video_latent_valid_mask"]
    batch["action_valid_mask"]
    batch["action_loss_mask"]
    batch["geometry_group_valid_mask"]

传播路径是：

    dataset/runtime batch mask
                │
                ▼
    run_video_inference / _sample_actions
    对目标区域进行阶段性修改
                │
                ▼
    _inference_input(...)
                │
                ▼
    _mot_token_valid_ids(...)
                │
                ├── video token valid
                ├── action token valid
                └── geometry token valid
                │
                ▼
    build_x_metadata / build_mot_metadata
                │
                ├── seq_ids
                ├── order_ids
                ├── stream_ids
                ├── noise_ids
                ├── window_size
                └── token_valid_ids
                │
                ▼
    attention_from_meta(...)
                │
                ├── dense attention
                ├── FlexAttention
                └── FlashAttention-4
                │
                ▼
    attention output
                │
                └── 对 invalid query 再次乘 token_valid_ids

因此 mask 有两个层次：

- 模态/时间级 mask：例如目标视频阶段把目标 geometry group 设为无效；
- token 级 mask：把一个时间帧或 geometry slot 扩展到对应的全部 patch、register 或 action token。

---

## 3. 四类 mask 的具体语义

### 3.1 video_latent_valid_mask

典型 shape：

    [B, T]

它表示每个视频 latent frame 是否真实有效。

模型中会将它扩展到该帧的所有视频 token：

    video_tokens = video_valid[:, :, None].expand(
        -1,
        -1,
        video_tokens_per_frame,
    ).reshape(bsz, -1)

如果每帧的视频 token 数为：

    video_tokens_per_frame = views × h_tokens × w_tokens

则：

    [B, T]

变为：

    [B, T × views × h_tokens × w_tokens]

由于模型同时有 noisy video 和 clean video 两份流，最终会复制两次。

video_latent_valid_mask=False 的视频 token：

- 不能作为有效 key/value；
- 不能作为有效 query；
- attention 输出还会被二次乘 mask；
- 不应该将无效 padding 信息传播到其他模态。

重要的是，目标视频在正常生成时通常仍然是 valid。它可能是随机噪声或当前 diffusion 状态，但不是 padding。

所以：

    token valid ≠ token 内容是 GT

目标视频可以 valid，同时内容是生成过程中的 noisy latent。

### 3.2 geometry_group_valid_mask

典型 shape：

    [B, T, 4]

其中 4 是每个 latent frame 的 geometry slot 数。

模型将它扩展为：

    [B, T, geometry_slot, view, register_token]

再 flatten 为：

    [B, T × 4 × views × register_tokens]

一个 geometry slot 为 False 时，受影响的是：

    该时间组
      × 该 slot
      × 所有 view
      × 所有 geometry register token

它在 geometry stream 中控制：

- geometry attention 的 token validity；
- image_valid_mask；
- invalid register 的恢复；
- geometry head 读取的有效特征。

它在 joint MOT attention 中控制：

- 对应 geometry register 是否可以作为 key/value；
- 对应 geometry register 是否可以作为 query；
- 该 slot 是否参与 Video/Action 与 Geometry 交互。

### 3.3 action_valid_mask

典型原始 shape：

    [B, action_dim, T, action_per_frame, 1]

模型执行：

    action_valid = action_valid_mask.to(
        device=device,
        dtype=torch.bool,
    ).any(dim=1)[..., 0]

这一步：

1. 沿 action channel 维做 any；
2. 去掉最后一个大小为 1 的维度；
3. 得到每个时间帧、每个 action token 是否有效。

结果 shape：

    [B, T, action_per_frame]

再 flatten 为：

    [B, T × action_per_frame]

并复制到 noisy action 和 clean action 两份流。

action_valid_mask 是 attention 可见性 mask，决定：

- action token 是否可以作为 query；
- action token 是否可以作为 key/value；
- action token 是否可以和 Video/Geometry 交互。

### 3.4 action_loss_mask

action_loss_mask 与 action_valid_mask 不同。

动作采样初始化时：

    target_action_mask = action_loss_mask[
        :, :, target_slice
    ].to(gt_actions.dtype)

    cur_actions = cur_actions * target_action_mask

每次 scheduler 更新后还会再次执行：

    cur_actions = cur_actions * target_action_mask

因此 action_loss_mask 的直接作用是：

    控制目标 action 哪些数值可以被初始化、更新和保留

而 action_valid_mask 的直接作用是：

    控制 action token 是否参与 attention

两者可以相同，但语义不同：

| mask | 主要控制对象 | 使用位置 |
|---|---|---|
| action_loss_mask | 目标动作数值是否采样/保留 | 初始化和每步 scheduler 后 |
| action_valid_mask | 动作 token 是否参与 attention | _mot_token_valid_ids |

代码允许在缺少 action_valid_mask 时回退：

    action_valid_mask = batch.get(
        "action_valid_mask",
        batch["action_loss_mask"],
    )

但这只是输入兼容策略，不代表两者本来语义相同。

---

## 4. 视频阶段的 mask 修改

默认窗口：

    history_end = 4
    target_slice = slice(5, 8)

视频生成阶段会修改三类输入。

### 4.1 目标 geometry RGB 被清零

    video_geometry_rgb = geometry_rgb_gt.clone()
    video_geometry_rgb[:, target_slice] = 0

结果：

    index 0~4：GT geometry RGB
    index 5~7：0

这防止视频阶段直接读取目标 RGB。

### 4.2 目标 geometry slot 被设为 invalid

    video_geometry_slot_valid = geometry_slot_valid_mask.clone()
    video_geometry_slot_valid[:, target_slice] = False

因此目标 geometry：

- 数值上是 0；
- token validity 上也是 False；
- 不会参与 geometry self-attention；
- 不会作为有效 geometry register 被 Video query 读取。

### 4.3 目标 action 被设为 attention-invalid

    video_action_valid_mask = action_valid_mask.clone()
    video_action_valid_mask[:, :, target_slice] = False

历史 action 仍然可以作为条件，目标 action 不参与视频阶段的有效交互。

注意，视频阶段并没有把 action_loss_mask 改成 False。模型中目标 action 的 attention 可见性使用的是 video_action_valid_mask，而 action_loss_mask 仍然是另一个输入字段。

### 4.4 目标 video latent 通常仍然有效

目标 video latent 从随机噪声开始：

    cur_latents = torch.randn_like(
        gt_latents[:, :, target_slice]
    )

正常情况下 target 的 video_latent_valid_mask 仍然是 True。

因此视频阶段的语义是：

    目标 video token 可以参与 attention，
    但其内容来自当前 diffusion 状态，不是 GT。

---

## 5. 动作阶段的 mask 修改

动作阶段使用：

    geometry_rgb = video.action_geometry_rgb

此时：

    历史 geometry = GT
    目标 geometry = 生成视频解码得到的 RGB

geometry slot validity 通常恢复为 batch 原始有效 mask：

    geometry_slot_valid_mask = batch["geometry_group_valid_mask"][
        :, :frame_count
    ].to(device, dtype=torch.bool)

目标 action 数值从噪声开始：

    pred_actions[:, :, target_slice] = 0
    cur_actions = torch.randn_like(
        gt_actions[:, :, target_slice]
    )
    cur_actions = cur_actions * target_action_mask

动作阶段通常具有以下可见性：

    历史 video：有效
    目标 video：已生成，通常有效
    历史 geometry：有效
    目标 geometry：由生成 RGB 重算，通常有效
    历史 action：有效
    目标 action：有效 token 进入 diffusion，invalid token 被屏蔽

动作阶段不会再修改 video latent，也不会重新更新 geometry registers。

---

## 6. MOT metadata：mask 如何变成 attention 规则

模型通过 MOTMaskMetadata 描述每个 token：

    seq_ids
    order_ids
    stream_ids
    noise_ids
    window_size
    frame_ids
    token_valid_ids
    cache_key
    structure_cache_key

字段语义：

| 字段 | 作用 |
|---|---|
| seq_ids | 隔离 packed batch 中不同样本 |
| order_ids | 实现 chunk-level causal 顺序 |
| stream_ids | 区分 Video、Action、Geometry |
| noise_ids | 区分 noisy、clean、geometry |
| window_size | 限制可见的 chunk 时间范围 |
| frame_ids | 记录 token 对应的帧 |
| token_valid_ids | 屏蔽 invalid token |
| cache_key | 所有 token 有效时复用完整 attention block mask |
| structure_cache_key | token validity 动态变化时复用结构 mask |

最终 attention allow mask 的逻辑是：

    same_batch
      AND within_window
      AND query_valid
      AND key_valid
      AND modality_rule

---

## 7. 物理 token 排列

joint MOT layer 的物理排列是：

    NV, CV, G, NA, CA

含义：

| 缩写 | 含义 |
|---|---|
| NV | noisy video |
| CV | clean video |
| G | geometry registers |
| NA | noisy action |
| CA | clean action |

在 joint layer 中：

    qkv_parts.insert(2, g_qkv)

所以五路 token 一起进行 masked self-attention：

    NV + CV + G + NA + CA

奇数层没有 geometry：

    NV + CV + NA + CA

偶数层才加入 geometry registers，并且使用 mot_meta；奇数层使用 x_meta。

这意味着 geometry 不是一个简单加到 video/action hidden 上的外部条件，而是以独立 G token 流参与 attention。

---

## 8. order_ids：模态间的因果时钟

Video 和 Geometry 使用偶数 order：

    video_order = (frame_id // chunk_size) × 2
    geometry_order = (frame_id // chunk_size) × 2

Action 使用奇数 order：

    action_order = (frame_id // chunk_size) × 2 + 1

默认 chunk_size=4：

    frame 0~3:
        Video/G order = 0
        Action order  = 1

    frame 4~7:
        Video/G order = 2
        Action order  = 3

逻辑时钟为：

    Video/G chunk 0
        → Action chunk 0
        → Video/G chunk 1
        → Action chunk 1

    order 0       → order 1
        → order 2       → order 3

Action 的奇数 order 配合数据中的 action 首个 padding/reference slot，实现原始 LingBot 的 video-then-action MDP 和 inverse-dynamics 形式的可见性。

---

## 9. 核心 attention mask 规则

### 9.1 X 到 X：Video/Action 之间

令 X 表示 Video 或 Action。

规则：

    clean → clean:
        k_order <= q_order

    noisy → clean:
        k_order < q_order

    noisy → noisy:
        k_order == q_order

没有明确允许的组合，例如：

    clean query → noisy key

不会被放行。

语义是：

- clean 状态可以看当前及历史 clean 状态；
- noisy query 看不到同 order 的 clean key，防止目标信息泄漏；
- noisy query 可以看同 order 的 noisy key；
- noisy query 只能读取更早 order 的 clean key。

### 9.2 G 到 G：Geometry 自身

    G query → G key:
        k_order <= q_order

Geometry stream 是 chunk-causal：

- 当前 geometry 可以看当前 chunk；
- 可以看历史 chunk；
- 不能看未来 chunk。

独立 geometry 预计算阶段主要使用这条规则。

### 9.3 X 到 G：Video/Action 读取 Geometry

默认语义是：

    clean X query → G key:
        k_order <= q_order

    noisy X query → G key:
        k_order < q_order

所以 noisy Video/Action query 不能直接读取同 order 的 geometry；clean query 可以读取当前/历史 geometry。

这条规则控制目标 geometry 是否能泄漏给 noisy video/action。

### 9.4 G 到 X

当前 allow 逻辑显式包含：

    X → X
    G → G
    X → G

没有单独加入：

    G → X

joint attention 仍然是一次完整 QKV attention，但允许关系由上述规则定义。当前实现的主要设计是：Video/Action query 读取允许的 Geometry key/value，而 Geometry 自身按 G→G 规则保持独立的 chunk-causal 更新。

---

## 10. 目标帧的具体交互例子

默认：

    frame 0~3:
        Video/G order = 0
        Action order  = 1

    frame 4~7:
        Video/G order = 2
        Action order  = 3

考虑目标 frame 5。

### 10.1 目标 noisy video：NV5

    q_order = 2
    q_noise = NOISE_NOISY

它对 X token 可以读取：

    - noisy video/action 的 order 2
    - clean video/action 的 order < 2

它不能读取：

    - clean token 的 order 2
    - future order > 2

它对 Geometry 可以读取：

    - geometry order < 2，也就是历史 geometry

它不能读取：

    - geometry order 2，也就是目标 chunk geometry

因此视频阶段的目标 noisy video 可以看到：

    历史 clean video
    当前目标 noisy video
    允许的 noisy action
    历史 geometry

不能看到：

    目标 chunk 的 clean video
    目标 chunk 的目标 geometry

这和视频阶段把目标 geometry RGB 清零并将目标 slot mask 置 False 是一致的。

### 10.2 目标 noisy action：NA5

    q_order = 3
    q_noise = NOISE_NOISY

它对 Geometry 可以读取：

    geometry order < 3

目标 geometry 的 order 是 2，所以目标 noisy action 可以读取目标 geometry。

因此动作阶段形成：

    生成视频
        → 生成 RGB
        → 重算目标 geometry
        → 目标 action 读取目标 geometry

这正是动作预测需要的条件链。

### 10.3 目标 clean action：CA5

    q_order = 3
    q_noise = NOISE_CLEAN

它可以读取：

    clean Video/Action order <= 3
    Geometry order <= 3

所以 clean action 的可见性比 noisy action 更宽。

---

## 11. 视频阶段与动作阶段的模态差异

### 视频阶段

    geometry RGB:
        历史 = GT
        目标 = 0

    geometry slot valid:
        历史 = True
        目标 = False

    action valid:
        历史 = True
        目标 = False

    video latent:
        历史 = GT
        目标 = 当前 noisy latent

主要交互：

    目标 noisy video
        ← 历史 clean video
        ← 历史 geometry
        ← 当前 noisy video/action

不会读取目标 geometry。

### 动作阶段

    geometry RGB:
        历史 = GT
        目标 = 生成 RGB

    geometry slot valid:
        使用原始有效 mask

    action valid:
        使用原始 action valid mask

    video latent:
        历史 = GT
        目标 = 已生成视频 latent

主要交互：

    目标 noisy action
        ← 已生成 video latent
        ← 历史和目标 geometry registers
        ← 历史 action
        ← 当前 noisy action

动作阶段不再更新 video latent，也不再更新 geometry cache。

---

## 12. Geometry cache：final_geometry

geometry 预计算入口是：

    transformer(
        _geometry_input(...),
        mode="precompute_geometry",
    )

模型运行完整 geometry stream 后返回：

    final_geometry

它是 geometry stream 的最终状态：

    geometry RGB
        → VGGTO geometry embedding
        → geometry layers
        → final_geometry

后续 inference 使用：

    geometry = geometry_condition["final_geometry"]

它用于：

- 确认 geometry group/view/register 的布局；
- 提供 geometry register rotary embedding；
- 让后续 V/A layer 知道当前 geometry condition 的结构。

---

## 13. Geometry cache：layer_registers

这是 Video/Action 和 Geometry 交互时最关键的 cache。

在 geometry layer 的 joint attention 层：

    g_register = frame_tokens[:, :, :patch_start_idx]

模型会保存：

    layer_registers[layer_id] = g_register.contiguous()

并同时运行 geometry-only register attention：

    register_override = block.geometry(
        g_register,
        geometry.register_rotary,
        meta,
        masked_attn_backend=...,
    )

因此 cache 类似：

    layer_registers = {
        0:  [B, groups, register_tokens, geometry_dim],
        2:  [B, groups, register_tokens, geometry_dim],
        4:  ...,
        ...
    }

每一个 joint layer 都必须有一个 snapshot。模型会检查：

    set(layer_registers) == vggto.register_attention_indices

避免 cache 层号缺失或和模型结构不匹配。

---

## 14. Inference 时如何使用 geometry cache

在 inference 中：

    layer_registers = geometry_condition["layer_registers"]
    geometry = geometry_condition["final_geometry"]

每个 joint layer：

    g_register = layer_registers[layer_id].to(
        device=states.noisy_video.device,
        dtype=states.noisy_video.dtype,
    )

    g_rotary = geometry.register_rotary.to(
        device=states.noisy_video.device
    )

然后将 G registers 插入：

    NV, CV, G, NA, CA

并执行一次 masked self-attention。

关键代码语义是：

    states, _g_updated = block(...)

这里 inference 返回的 _g_updated 被丢弃。

因此推理阶段：

    geometry cache 作为固定 G registers
    参与每个 Video/Action diffusion step，
    但不会被 V/A diffusion 反向更新并回写。

这就是 cache 的核心语义：

    geometry condition 在一个推理阶段内固定；
    Video/Action 可以读取它；
    Video/Action 的每一步不会重新改变它。

---

## 15. 两次 geometry cache 的生命周期

完整 run_mot_inference 中会建立两份不同的 geometry condition。

### 15.1 视频生成前的 cache

输入：

    历史 geometry RGB = GT
    目标 geometry RGB = 0
    目标 geometry slot = invalid

生成：

    video_geometry_condition

用途：

    供所有 inference_video cond/uncond step 复用

生命周期：

    precompute_geometry
        → 多个 video diffusion step
        → 视频阶段结束后丢弃

### 15.2 动作生成前的 cache

输入：

    历史 geometry RGB = GT
    目标 geometry RGB = 生成视频解码得到的 RGB

生成：

    geometry_condition

用途：

    供所有 inference_action step 复用

生命周期：

    precompute_geometry
        → 多个 action diffusion step
        → 完整推理结束后释放

两份 cache 不能混用：

| cache | 历史 geometry | 目标 geometry | 用途 |
|---|---|---|---|
| video_geometry_condition | GT | 置零/invalid | 生成目标视频 |
| geometry_condition | GT | 生成视频 RGB | 生成目标动作 |

---

## 16. Geometry 内部的 cached_outputs

geometry context 还有一类用于 geometry head 的 cache：

    cached_outputs = [None] * vggto.depth

在指定 geometry layer：

    cached_outputs[layer_id] = torch.cat(
        [frame_tokens, inter_tokens],
        dim=-1,
    )

之后 depth/point head 使用：

    vggto.dense_forward(cached_outputs, images)
    vggto.point_forward(cached_outputs, images)

它保存 geometry 中间特征，供 depth/point head 读取。

它和 layer_registers 的区别：

| cache | 用途 | 是否给 V/A inference 复用 |
|---|---|---|
| cached_outputs | depth/point head 的中间特征 | 否 |
| layer_registers | joint MOT 的 G registers | 是 |
| final_geometry | geometry 的最终上下文和 rotary/layout | 是 |
| FA4 block cache | attention mask 的稀疏结构 | 间接使用 |

---

## 17. Attention mask cache：为什么还要缓存 mask

geometry condition 被缓存后，每个 diffusion step 仍然要进行 attention。

但不同 step 之间通常不变的是：

    batch size
    video token 数
    geometry token 数
    action token 数
    frame 数
    chunk size
    window size
    stream 顺序
    noise stream 顺序

变化的主要是：

    latent/action 的数值
    timestep 数值
    某些样本的 token validity

因此 attention mask 分为：

1. 静态结构部分：可缓存；
2. 动态 validity 部分：作为运行时条件传入。

FA4 使用全局缓存：

    _MOT_BLOCK_CACHE: dict[tuple, tuple[Any, Any]]

block size：

    _FA4_BLOCK_SIZE = (128, 128)

MOT mask 与 attention head 无关，因此只构建一个 head 的 block-sparse metadata，再广播给所有 heads：

    _BLOCK_MASK_HEADS = 1

---

## 18. cache_key 与 structure_cache_key

MOTMaskMetadata 中有：

    cache_key
    structure_cache_key

### 18.1 所有 token 都有效

当 token_valid_ids 是 None，或者全部为 True 时，metadata 使用：

    cache_key = structure_cache_key

joint MOT 的结构 key 包含：

    (
        "mot",
        batch_size,
        video_tokens_per_frame,
        geometry_tokens_per_frame,
        action_tokens_per_frame,
        num_frames,
        chunk_size,
        window_size,
    )

FA4 再结合 device、head 数和 block size 形成全局 key。

这种情况缓存的是完整的、语义确定的 block-sparse mask。

### 18.2 存在动态 invalid token

如果 token_valid_ids 不是 None 且不是全 True：

    cache_key = None
    structure_cache_key = ...

FA4 使用：

    mask_only = True

缓存 key 变成：

    (
        "mask_only",
        structure_key,
        device,
        block_heads,
        block_size,
    )

构建 block mask 时暂时移除具体 token_valid_ids，只缓存：

    模态结构
    时间顺序
    causal 规则
    window 结构

具体 token validity 通过 FA4 auxiliary tensors 运行时传入：

    seq_ids
    order_ids
    stream_ids
    noise_ids
    token_valid_ids

因此：

    block 结构可以复用，
    每个样本具体哪些 token 有效仍然动态决定。

---

## 19. FA4 mask cache 的完整路径

实际路径可以简化为：

    MOTMaskMetadata
          │
          ├── seq_ids
          ├── order_ids
          ├── stream_ids
          ├── noise_ids
          └── token_valid_ids
          │
          ▼
    _mot_block_sparse(...)
          │
          ├── 命中 _MOT_BLOCK_CACHE
          └── 未命中则 create_block_mask(...)
          │
          ▼
    block_sparse_fwd / block_sparse_bwd
          │
          ▼
    _mot_aux_tensors(meta)
          │
          ▼
    FA4 custom mask
          │
          ▼
    flash_attn_func(...)

其中：

- block sparse cache 负责粗粒度 block 结构；
- auxiliary tensors 负责 batch、时间、模态、noise、validity 的具体语义；
- custom mask 负责最终判断 query-key 是否允许交互。

---

## 20. 不同 attention backend

代码支持：

    dense
    flex
    fa4

统一入口：

    attention_from_meta(
        q,
        k,
        v,
        meta,
        backend=self.masked_attn_backend,
    )

### dense

构造完整的：

    [B, Q, K]

布尔矩阵，最容易检查，但内存开销最大。

### flex

使用 FlexAttention 的 BlockMask，规则仍来自同一套 metadata。

### fa4

使用 FlashAttention-4 custom mask 和 block-sparse cache。

配置中默认：

    masked_attn_backend = "fa4"

因此生产环境通常走 FA4；dense 更适合语义检查和测试。

### 20.1 dense 与 Flex/FA4 的边界差异

代码中存在一处值得特别注意的实现差异。

dense 参考函数中的 clean X → G 是：

    clean_to_g = (
        q_noise == NOISE_CLEAN
        and k_order < q_order
    )

而 Flex/FA4 inline mask 中是：

    clean_to_g = (
        q_noise == NOISE_CLEAN
        and k_order <= q_order
    )

因此：

    dense 参考：
        clean X query 只能读取严格过去的 G

    Flex/FA4：
        clean X query 可以读取当前及过去的 G

noisy X → G 在各路径中都是严格过去：

    k_order < q_order

如果需要验证不同 backend 完全一致，应重点检查 clean X → G 的边界条件。当前默认 backend 是 FA4，所以实际运行语义应以 FA4 路径为准。

---

## 21. Attention output 的二次 mask

attention backend 完成后，代码还会执行：

    if meta.token_valid_ids is not None:
        out = out * meta.token_valid_ids[:, :, None, None]

这不是 attention score mask，而是对 query output 做后处理：

    invalid query token 的 attention output 直接置零

因此 invalid token 受到两层保护：

1. 作为 key/value 时不能被有效 token 读取；
2. 作为 query 时自己的输出被置零。

这能避免 invalid token 通过 attention residual 或后续层继续传播。

---

## 22. Geometry invalid register 的恢复

geometry stream 有专门的 invalid register 恢复逻辑：

    valid = image_valid_mask
    updated = torch.where(
        valid[:, :, None, None],
        updated,
        original,
    )

如果某个 geometry image/slot invalid：

    updated register 不被采用

而是恢复为：

    original register

这比简单置零更保守，因为它不会把 invalid slot 更新成一个由其他 token 混合得到的新 representation。

该逻辑在 geometry-only layer 中使用，也在 training joint layer 中使用。

---

## 23. 三个模态到底如何交互

### 23.1 Video 读取 Geometry

Video query 通过 X → G 规则读取 geometry registers：

- noisy video 只能读取严格过去的 geometry；
- clean video 读取当前/过去 geometry，具体边界依赖 backend；
- invalid geometry token 不能被读取；
- 视频阶段目标 geometry 被置零且 invalid，所以目标 noisy video 看不到目标 geometry。

### 23.2 Action 读取 Video

Video 和 Action 都属于 X stream，通过 X → X 规则交互。

由于 Action order 比同 chunk 的 Video order 大 1：

    Video/G chunk 0: order 0
    Action chunk 0: order 1
    Video/G chunk 1: order 2
    Action chunk 1: order 3

目标 Action 可以读取当前及过去的 Video token，但仍受 noisy/clean 规则和 window 限制。

这使动作模型可以读取已经生成的目标视频。

### 23.3 Action 读取 Geometry

目标 Action 的 order 是 3，目标 Geometry 的 order 是 2，因此目标 Action 可以读取目标 Geometry。

完整因果链是：

    目标视频 latent
        → VAE 解码目标 RGB
        → 重新计算目标 geometry
        → 目标 action 读取目标 geometry

### 23.4 Geometry 是否被 Video/Action 更新

Geometry 在独立 precompute 阶段更新。

后续 V/A inference 阶段：

    geometry cache 作为固定 G registers 进入 joint attention

但 inference 中：

    states, _g_updated = block(...)

_g_updated 被丢弃，所以 geometry 不会被每个 V/A diffusion step 重新更新。

---

## 24. 一个目标帧的完整交互图

以目标 frame 5 为例。

视频阶段：

    NV5
     │
     ├── 读取历史 CV0~CV3
     ├── 读取当前/同 order noisy video
     ├── 读取允许的 noisy action
     ├── 读取历史 G0
     └── 不能读取目标 G2
         因为目标 geometry slot 被置为 invalid

动作阶段：

    NA5
     │
     ├── 读取已生成 video latent 对应的 Video token
     ├── 读取历史 Action
     ├── 读取当前 noisy Action
     ├── 读取历史 G0
     └── 读取目标 G2
         因为目标 G2 已由生成 RGB 重算并缓存

因此因果关系是：

    history
       → target video
       → target geometry
       → target action

而不是：

    GT target geometry
       → target video

---

## 25. 总结

Mask 的核心作用是控制：

    1. 哪些 token 有效；
    2. query 能看哪些 key/value；
    3. 目标区域是否能读取 GT；
    4. padding slot 是否能传播；
    5. 哪些目标 action 数值能被 scheduler 更新；
    6. invalid query 的 output 是否被置零。

Cache 的核心作用分为三层：

    1. geometry condition cache
       final_geometry + layer_registers
       供每个 V/A diffusion step 复用

    2. geometry head feature cache
       cached_outputs
       供 depth/point head 读取 geometry 中间特征

    3. attention block mask cache
       FA4 的 _MOT_BLOCK_CACHE
       复用固定的模态/时序 block-sparse 结构
       动态 token validity 通过 auxiliary tensors 传入

三个模态的交互方式是：

    Video 与 Action：
        通过 X → X mask 交互

    Video/Action 与 Geometry：
        在偶数 joint layer 中将 Geometry registers
        插入 NV, CV, G, NA, CA 的统一 masked attention

    Geometry：
        先独立计算并缓存
        后续 V/A 阶段作为固定 G registers 提供条件

最终可以把推理阶段理解为：

    Mask 决定谁能看谁、哪些 token 有效、哪些数值能更新；
    Geometry cache 决定几何条件是否需要重复计算；
    Attention mask cache 决定稀疏交互结构是否需要重复构建；
    Joint MOT attention 决定 Video、Action、Geometry
    如何在允许的因果窗口内交换信息。

