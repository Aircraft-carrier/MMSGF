
# 从 t=0 开始的 Video → Geometry → Action 自回归生成与索引对照

> 状态说明：本文保留的是旧 fixed-window `autoregressive_rollout` 的索引分析。
> 该实现及 `distillation/rollout.py` 已删除；当前训练使用
> `distillation/self_rollout/engine.py`，可视化位于
> `distillation/self_rollout/artifacts.py`。

本文模拟从初始状态 t=0 开始，执行：

~~~text
先生成 video
    ↓
根据生成 video 重新计算 geometry
    ↓
生成 action
    ↓
把生成结果作为下一轮条件
    ↓
继续生成下一段 video 和 action
~~~

同时记录 raw frame、latent frame、geometry group、action token 的索引，并与训练阶段进行对比。

核心代码：

~~~text
inference/mot_inference.py
distillation/self_rollout/engine.py
wan_va/dataset/mot_dataset.py
~~~

---

## 1. 先明确一个生成单元是什么

MOT 不是每次只生成一张原始图片。默认配置下：

~~~text
action_chunk_size      = 48
video_downsample_ratio = 4
vae_temporal_factor    = 4
latent_frames_per_chunk = 4
action_per_frame        = 16
~~~

一个 action chunk 的时间组织：

~~~text
48 个 raw action
    ↓
13 个采样 RGB frame
    ↓ Wan VAE
4 个 video latent frame
~~~

13 个 raw frame 的组成是：

~~~text
1 + 4 + 4 + 4
~~~

对应：

~~~text
latent 0：1 个 anchor raw frame
latent 1：4 个 raw frame
latent 2：4 个 raw frame
latent 3：4 个 raw frame
~~~

48 个 raw action 的 packing：

~~~text
action offset 0~15   → latent 1 的 16 个 action token
action offset 16~31  → latent 2 的 16 个 action token
action offset 32~47  → latent 3 的 16 个 action token
~~~

因此，一个 chunk 的真正生成内容是：

~~~text
video：
    3 个非 anchor latent

geometry：
    与 4 个 latent 时间位置对齐的 geometry group

action：
    3 × 16 个 action token
~~~

---

## 2. 完整 MOT window 的索引

单次 run_mot_inference 使用一个固定的 8-latent window：

~~~text
[H0, H1, H2, H3, T0, T1, T2, T3]
~~~

| local latent index | 语义 | 是否本轮生成 |
|---:|---|---|
| 0 | history anchor H0 | 否 |
| 1 | history H1 | 否 |
| 2 | history H2 | 否 |
| 3 | history H3 | 否 |
| 4 | target anchor T0 | 否 |
| 5 | target T1 | 是 |
| 6 | target T2 | 是 |
| 7 | target T3 | 是 |

代码通过：

~~~python
history_end = spec.history_latent_frames
generated_target_start = history_end + 1
target_slice = slice(generated_target_start, frame_count)
~~~

默认：

~~~text
history_end = 4
target_slice = [5, 6, 7]
~~~

所以一次 run_mot_inference 的真实语义是：

~~~text
输入条件：
    GT history H0~H3
    GT target anchor T0

生成：
    target video T1~T3
    target geometry T1~T3
    target action T1~T3
~~~

这里的 T0 是目标 chunk 的第一帧 anchor，不是需要 diffusion 生成的目标。

---

## 3. “从 t=0 开始”应该如何理解

### 3.1 raw episode 的物理时间

如果将 episode 的第一张观测图像记为：

~~~text
image/state[0]
~~~

那么通常：

~~~text
action[0]
    是在 state/image[0] 条件下执行的第一个控制动作
    大致影响 state/image[1]
~~~

即：

~~~text
action[0] ≈ image[0] → image[1]
~~~

它不是单独完成 image[0] → image[4] 的动作。

因为 video downsample_ratio=4 只改变视频采样：

~~~text
image[0], image[4], image[8], ...
~~~

action 仍然按原始控制步保留：

~~~text
action[0], action[1], action[2], action[3], ...
~~~

### 3.2 MOT 的模型时间

当前实现需要一个 history chunk，而不是只输入 image[0]：

~~~text
H0,H1,H2,H3
~~~

因此第一次固定窗口可以理解成：

~~~text
已有：
    H0,H1,H2,H3,T0

待生成：
    T1,T2,T3
~~~

如果希望把 episode raw frame 0 作为时间起点，可以把初始观测/历史帧映射到 H0；但 H0~H3 是模型的 latent 时间轴，不等同于 raw frame 0~3。

---

## 4. 第一阶段：生成 Video

run_mot_inference 首先调用：

~~~python
video = run_video_inference(...)
~~~

### 4.1 video 生成区域

run_video_inference 先取出完整的 GT window：

~~~python
gt_latents = batch["latents"][:, :, :frame_count]
~~~

然后只在 target_slice 初始化噪声：

~~~python
cur_latents = torch.randn_like(
    gt_latents[:, :, target_slice]
)
~~~

shape：

~~~text
cur_latents:
    [B, C, 3, V, H_latent, W_latent]
~~~

索引对应：

~~~text
cur_latents[:, :, 0] ↔ local latent 5 ↔ T1
cur_latents[:, :, 1] ↔ local latent 6 ↔ T2
cur_latents[:, :, 2] ↔ local latent 7 ↔ T3
~~~

H0~H3 和 T0 保持已知：

~~~text
local latent 0,1,2,3,4：
    clean/GT condition

local latent 5,6,7：
    noisy/generated target
~~~

### 4.2 video 阶段使用的 geometry

代码把目标 geometry 清零：

~~~python
video_geometry_rgb = geometry_rgb_gt.clone()
video_geometry_rgb[:, target_slice] = 0

video_geometry_slot_valid = geometry_slot_valid_mask.clone()
video_geometry_slot_valid[:, target_slice] = False
~~~

所以 video 阶段只能看到：

~~~text
geometry group 0~3：
    history geometry

geometry group 4：
    target anchor geometry

geometry group 5~7：
    不可见/invalid
~~~

然后调用：

~~~python
video_geometry_condition = transformer(
    _geometry_input(
        rgb=video_geometry_rgb,
        stream_ids=stream_ids,
        slot_valid_mask=video_geometry_slot_valid,
        return_points=False,
    ),
    mode="precompute_geometry",
)
~~~

这个 precompute_geometry 是给 video 生成建立 history geometry 条件，不是提前预测 target geometry。

### 4.3 video 阶段使用的 action

video 生成时：

~~~python
video_action_noisy = gt_actions.clone()
video_action_noisy[:, :, target_slice] = 0

video_action_valid_mask = action_valid_mask.clone()
video_action_valid_mask[:, :, target_slice] = False
~~~

因此：

~~~text
history action：
    保留，作为 video 条件

target action：
    不作为 video 已知条件
~~~

每一个 video denoise timestep 都只更新 local latent 5~7：

~~~python
cur_latents = video_scheduler.step(
    latent_prediction[:, :, target_slice],
    timestep,
    cur_latents,
)
~~~

最终：

~~~text
pred_latents：
    local 0~3 = GT H0~H3
    local 4   = GT T0
    local 5   = generated T1
    local 6   = generated T2
    local 7   = generated T3
~~~

---

## 5. 第一阶段输出的 RGB 和 Geometry

### 5.1 将 target video latent decode 为 RGB

代码 decode：

~~~python
decoded_target_rgb = decode_latents_to_rgb_views(
    pred_latents[:, :, history_end:frame_count]
)
~~~

默认传入：

~~~text
pred_latents[:, :, 4:8]
    = [T0, generated T1, generated T2, generated T3]
~~~

decode 后是 target chunk 的 13 个 raw RGB frame：

~~~text
decoded_target_rgb:
    [B, 13, V, 3, H, W]
~~~

时间组织：

~~~text
T0：
    1 个 anchor raw frame

T1：
    4 个 raw frame

T2：
    4 个 raw frame

T3：
    4 个 raw frame
~~~

### 5.2 从生成 RGB 构造 geometry RGB

代码将 decode 后的 target RGB 分组：

~~~python
decoded_target_geometry = decoded_rgb_to_geometry_groups(
    decoded_target_rgb,
    latent_frames=int(spec.target_latent_frames),
    vae_temporal_factor=int(config.vae_temporal_factor),
)
~~~

得到：

~~~text
decoded_target_geometry:
    [B, 4, 4, V, 3, H, W]
~~~

第 2 维是 target latent，第 3 维是 geometry slot：

~~~text
decoded_target_geometry[:, 0] ↔ T0
decoded_target_geometry[:, 1] ↔ T1
decoded_target_geometry[:, 2] ↔ T2
decoded_target_geometry[:, 3] ↔ T3
~~~

写回 action geometry：

~~~python
action_geometry_rgb[:, target_slice] = decoded_target_geometry[:, 1:]
~~~

因此：

~~~text
action_geometry_rgb[:, 0:4] = history geometry RGB
action_geometry_rgb[:, 4]   = 已知 T0 anchor geometry RGB
action_geometry_rgb[:, 5]   = 生成 T1 的 geometry RGB
action_geometry_rgb[:, 6]   = 生成 T2 的 geometry RGB
action_geometry_rgb[:, 7]   = 生成 T3 的 geometry RGB
~~~

---

## 6. 第二阶段：根据生成 Video 重新计算 Geometry

run_mot_inference 接下来调用：

~~~python
geometry, geometry_condition = _run_geometry_from_rgb(
    batch,
    frame_count,
    geometry_rgb=video.action_geometry_rgb,
    ...
)
~~~

此时输入 geometry RGB 的来源是：

~~~text
H0~H3：
    dataset/已知 history RGB

T0：
    已知 target anchor RGB

T1~T3：
    由生成 video latent decode 得到的 RGB
~~~

调用：

~~~python
condition = transformer(
    _geometry_input(
        rgb=geometry_rgb,
        stream_ids=batch["stream_ids"],
        slot_valid_mask=batch["geometry_group_valid_mask"][:, :frame_count],
        return_points=True,
    ),
    mode="precompute_geometry",
)
~~~

输入 shape：

~~~text
geometry_rgb:
    [B, 8, 4, V, 3, H, W]

slot_valid_mask:
    [B, 8, 4]
~~~

输出包括：

~~~text
condition["depth"]
condition["depth_conf"]
condition["points"]
以及 geometry branch 的中间条件
~~~

这一步的因果链是：

~~~text
generated video latent
    ↓ decode
generated RGB
    ↓ precompute_geometry
predicted depth/points/features
    ↓
作为 action 生成的 geometry condition
~~~

因此 action 阶段使用的是“由生成 video 推出来的 geometry”，而不是直接使用 target GT geometry。

---

## 7. 第三阶段：根据 Video + Geometry 生成 Action

run_mot_inference 最后调用：

~~~python
pred_actions = _sample_actions(...)
~~~

### 7.1 action 生成区域

action 也只生成 local latent 5~7：

~~~python
pred_actions = gt_actions.clone()
pred_actions[:, :, target_slice] = 0

cur_actions = torch.randn_like(
    gt_actions[:, :, target_slice]
)
~~~

shape：

~~~text
cur_actions:
    [B, 20, 3, 16, 1]
~~~

索引：

~~~text
cur_actions[:, :, 0] ↔ local latent 5 ↔ T1 action
cur_actions[:, :, 1] ↔ local latent 6 ↔ T2 action
cur_actions[:, :, 2] ↔ local latent 7 ↔ T3 action
~~~

### 7.2 raw action index 与 latent/token index

如果当前 target raw action 起点是 0：

~~~text
local latent 5 / T1：
    action[0]  ~ action[15]

local latent 6 / T2：
    action[16] ~ action[31]

local latent 7 / T3：
    action[32] ~ action[47]
~~~

如果当前 target 起点是 raw action 100：

~~~text
local latent 5 / T1：
    action[100] ~ action[115]

local latent 6 / T2：
    action[116] ~ action[131]

local latent 7 / T3：
    action[132] ~ action[147]
~~~

映射公式：

~~~python
latent_offset = 1 + action_offset // 16
token_idx = action_offset % 16
latent_idx = chunk_frame_offset + latent_offset
~~~

对于 target chunk：

~~~text
chunk_frame_offset = 4
~~~

所以 raw action 100 是 target chunk 的第一个 raw action，进入：

~~~text
target chunk 内 latent offset 1
全局 local latent index = 4 + 1 = 5
token index = 0
~~~

也就是：

~~~text
action[100] → local latent 5 的 token 0
~~~

### 7.3 action 与 raw image 的时间含义

通常：

~~~text
action[100]
    是在 state/image[100] 条件下执行的第一个控制步
    大致影响 state/image[101]
~~~

所以：

~~~text
action[100] ≈ image[100] → image[101]
~~~

由于 video stride=4：

~~~text
image[100] → image[104]
~~~

这一段大致由：

~~~text
action[100]
action[101]
action[102]
action[103]
~~~

四个连续控制动作共同完成。

不能把 action[100] 解释成 image[100] → image[104] 的完整动作。

### 7.4 action denoise 的输入

每个 action timestep：

~~~python
action_noisy = pred_actions.clone()
action_noisy[:, :, target_slice] = cur_actions

action_clean = pred_actions.clone()
action_clean[:, :, target_slice] = 0
~~~

transformer 的输入同时包括：

~~~text
pred_latents：
    生成的 T1~T3 video latent

action_noisy：
    target T1~T3 的 action noisy sample

geometry_condition：
    由生成 video 重新计算的 depth/points/features

geometry_rgb：
    生成 video decode 后的 geometry RGB

history action：
    作为条件

text_emb、stream_ids
~~~

调用模式：

~~~python
mode="inference_action"
~~~

最后得到：

~~~text
pred_actions：
    history action 保持条件
    target T1~T3 使用生成 action
~~~

---

## 8. 第一轮从 raw index 0 开始的完整索引

假设第一轮目标 action 从 raw action 0 开始。

### 8.1 latent 索引

~~~text
local latent 0 = H0，初始 history
local latent 1 = H1，初始 history
local latent 2 = H2，初始 history
local latent 3 = H3，初始 history
local latent 4 = T0，target anchor
local latent 5 = T1，本轮生成
local latent 6 = T2，本轮生成
local latent 7 = T3，本轮生成
~~~

### 8.2 geometry group 索引

~~~text
geometry group 0 ↔ H0
geometry group 1 ↔ H1
geometry group 2 ↔ H2
geometry group 3 ↔ H3
geometry group 4 ↔ T0
geometry group 5 ↔ T1
geometry group 6 ↔ T2
geometry group 7 ↔ T3
~~~

每个 group 有 4 个 slot：

~~~text
group 0：
    [H0 frame, H0 frame, H0 frame, H0 frame]
    mask [T,F,F,F]

group 1~3：
    每组 4 个真实 history raw frame

group 4：
    [T0 frame, T0 frame, T0 frame, T0 frame]
    mask [T,F,F,F]

group 5~7：
    每组 4 个 target raw frame
~~~

### 8.3 action 索引

~~~text
local latent 5：
    action[0]  ~ action[15]

local latent 6：
    action[16] ~ action[31]

local latent 7：
    action[32] ~ action[47]
~~~

返回 tensor：

~~~text
actions:
    [20, 8, 16, 1]
~~~

概念索引：

~~~text
actions[:, latent=5, token=0, :] ↔ action[0]
actions[:, latent=5, token=1, :] ↔ action[1]
...
actions[:, latent=6, token=0, :] ↔ action[16]
...
actions[:, latent=7, token=15, :] ↔ action[47]
~~~

实际返回 tensor 的 action feature 维在第 0 维，因此完整数组布局仍是：

~~~text
[20, 8, 16, 1]
~~~

---

## 9. 第二轮：将第一次结果反馈为 history

distillation/rollout.py 中 autoregressive_rollout 会把上一轮结果写入 working：

~~~python
working["latents"][:, :, target_start:target_end] = (
    result.pred_latents[:, :, history:]
)

working["actions"][:, :, target_start:target_end] = (
    result.pred_actions[:, :, history:]
)

working["geometry_rgb"][:, target_start:target_end] = (
    result.action_geometry_rgb[:, history:]
)
~~~

默认：

~~~text
chunk = 4
history = 4
~~~

第一轮结束后：

~~~text
global latent 0~3：
    初始 history H0~H3

global latent 4~7：
    第一轮生成 chunk T0~T3
~~~

第二轮局部窗口重新排列为：

~~~text
local 0 = global T0
local 1 = global T1
local 2 = global T2
local 3 = global T3

local 4 = global U0
local 5 = global U1
local 6 = global U2
local 7 = global U3
~~~

第二轮内部仍然生成 local 5~7，但它们对应全局：

~~~text
local 5 ↔ global U1
local 6 ↔ global U2
local 7 ↔ global U3
~~~

因此第二轮不是再次生成全局 T1~T3，而是生成下一段 U1~U3。

---

## 10. 两轮 rollout 的完整索引表

### 10.1 latent/geometry 索引

| 轮次 | local index | 全局语义 | 状态 |
|---:|---:|---|---|
| 1 | 0 | H0 | 初始 history |
| 1 | 1 | H1 | 初始 history |
| 1 | 2 | H2 | 初始 history |
| 1 | 3 | H3 | 初始 history |
| 1 | 4 | T0 | target anchor |
| 1 | 5 | T1 | 第一次生成 |
| 1 | 6 | T2 | 第一次生成 |
| 1 | 7 | T3 | 第一次生成 |
| 2 | 0 | T0 | 第一轮结果作为新 history |
| 2 | 1 | T1 | 第一轮结果作为新 history |
| 2 | 2 | T2 | 第一轮结果作为新 history |
| 2 | 3 | T3 | 第一轮结果作为新 history |
| 2 | 4 | U0 | 下一段 target anchor |
| 2 | 5 | U1 | 第二次生成 |
| 2 | 6 | U2 | 第二次生成 |
| 2 | 7 | U3 | 第二次生成 |

### 10.2 action raw index

第一轮：

~~~text
T1：
    action[0]  ~ action[15]

T2：
    action[16] ~ action[31]

T3：
    action[32] ~ action[47]
~~~

第二轮：

~~~text
U1：
    action[48] ~ action[63]

U2：
    action[64] ~ action[79]

U3：
    action[80] ~ action[95]
~~~

如果第一轮 target 起点是 action 100，则整体平移：

~~~text
第一轮：
    action[100] ~ action[147]

第二轮：
    action[148] ~ action[195]
~~~

### 10.3 raw video frame index

若每个 raw action step 对应一个 raw video frame，stride=4 时，第一轮 target 的采样 frame 可以写成：

~~~text
T0：
    frame 0

T1：
    frame 4, 8, 12, 16

T2：
    frame 20, 24, 28, 32

T3：
    frame 36, 40, 44, 48
~~~

第二轮：

~~~text
U0：
    frame 48

U1：
    frame 52, 56, 60, 64

U2：
    frame 68, 72, 76, 80

U3：
    frame 84, 88, 92, 96
~~~

这里有三套不同的索引：

~~~text
raw action index：
    每个控制步一个 index

raw video frame index：
    原始视频帧 index，视频输入按 stride=4 采样

latent index：
    VAE 压缩后的模型时间 index
~~~

---

## 11. 训练阶段的固定窗口

训练数据集返回一个固定窗口：

~~~text
[H0,H1,H2,H3,T0,T1,T2,T3]
~~~

### 11.1 video 训练数据

dataset 返回：

~~~text
vae_rgb_history：
    13 个 raw RGB frame

vae_rgb_target：
    13 个 raw RGB frame
~~~

trainer 编码后：

~~~text
latents:
    [B,C,8,V,H_latent,W_latent]
~~~

训练阶段的语义：

~~~text
latent 0~3：
    history GT latent

latent 4：
    target anchor GT latent

latent 5~7：
    target GT latent，加噪后进行 video loss
~~~

### 11.2 action 训练数据

dataset 已经完成 action packing：

~~~text
history：
    latent 1~3 的 action condition

target：
    latent 5~7 的 action target
~~~

如果 target 起点是 raw action 0：

~~~text
latent 5：
    action[0]~action[15]

latent 6：
    action[16]~action[31]

latent 7：
    action[32]~action[47]
~~~

mask：

~~~text
action_loss_mask：
    target latent 5~7 为 True

action_valid_mask：
    history latent 1~3 和 target latent 5~7 为 True
~~~

---

## 12. 训练阶段和 rollout 阶段的差别

| 对比项 | 训练阶段 | rollout/inference 阶段 |
|---|---|---|
| history video | GT | 初始时 GT，后续可为前一轮生成 |
| target anchor T0 | GT | 当前窗口的已知 anchor |
| target video T1~T3 | GT 加噪后预测 | 从随机噪声采样 |
| target geometry | GT geometry/label | 根据生成 video decode 后重新计算 |
| history action | GT condition | 初始 GT 或前一轮生成结果 |
| target action | GT 加噪后预测 | 根据生成 video + geometry 采样 |
| video/action 顺序 | 训练 loss 中联合处理 | 先 video，再 geometry，再 action |
| target 输入 | 有 ground truth | target 区域初始为空/噪声 |
| 下一窗口 | 不反馈模型预测 | 当前结果写入 working |
| 误差累积 | 单窗口内主要是监督误差 | video → geometry → action → 下一轮 |

训练可以看作：

~~~text
GT history + GT target
    ↓ 加噪
模型预测
    ↓
与 GT target 比较
~~~

rollout 是：

~~~text
history + target anchor
    ↓
生成 video
    ↓
decode + geometry
    ↓
生成 action
    ↓
反馈给下一轮
~~~

---

## 13. mask 的训练/rollout 差异

### 13.1 dataset/训练 mask

默认：

~~~text
video_latent_loss_mask：
    [F,F,F,F,F,T,T,T]

action_loss_mask：
    history 1~3 = False
    target 5~7  = True

action_valid_mask：
    history 1~3 = True
    target 5~7  = True
~~~

含义：

~~~text
history：
    只能作为条件，不计算 target loss

T0：
    anchor，不作为主要生成 target

T1~T3：
    作为 video/action supervision target
~~~

### 13.2 rollout mask

autoregressive_rollout 会基于 generated_frames 设置：

~~~python
batch["video_latent_loss_mask"] = (
    video_valid & generated_frames
)

batch["action_loss_mask"] = (
    batch["action_valid_mask"] & generated_frames
)
~~~

第一轮：

~~~text
generated_frames = [F,F,F,F,F,T,T,T]
~~~

第二轮的局部窗口仍然是：

~~~text
local generated_frames = [F,F,F,F,F,T,T,T]
~~~

但全局 working 中，第一轮生成结果已经变成下一轮 history。

---

## 14. teacher forcing 与 feedback

### 14.1 训练：teacher forcing

训练时 target GT 已经存在：

~~~text
H0~H3：GT history
T0：GT anchor
T1~T3：GT target
GT geometry
GT action
~~~

模型做的是：

~~~text
对 GT target 加噪
    ↓
预测去噪结果
    ↓
与 GT 比较
~~~

当前窗口中的下一时间位置不会因为前一位置预测错误而改变输入。

### 14.2 rollout：feedback

rollout 时：

~~~text
第 1 轮：
    初始 history → 生成 T1,T2,T3 和 action[0:48]

第 2 轮：
    T0~T3 作为新的 history
    → 生成 U1,U2,U3 和 action[48:96]

第 3 轮：
    U0~U3 作为新的 history
    → 继续生成下一段
~~~

误差链：

~~~text
video prediction error
    ↓
decoded RGB error
    ↓
geometry prediction error
    ↓
action prediction error
    ↓
next-window history error
~~~

---

## 15. 一个从 t=0 到第二轮的简化时序模拟

### 15.1 第一轮

初始：

~~~text
已知：
    H0,H1,H2,H3,T0
~~~

video：

~~~text
生成：
    T1,T2,T3
~~~

geometry：

~~~text
decode T0~T3
    ↓
生成/计算 geometry T1,T2,T3
~~~

action：

~~~text
action[0]~action[47]
~~~

packing：

~~~text
action[0]~action[15]   → T1
action[16]~action[31]  → T2
action[32]~action[47]  → T3
~~~

写回 working：

~~~text
working latent：
    H0,H1,H2,H3,T0,T1,T2,T3

working action：
    history action + action[0]~action[47]

working geometry：
    history geometry + generated geometry T1~T3
~~~

### 15.2 第二轮

构造局部窗口：

~~~text
history：
    T0,T1,T2,T3

target：
    U0,U1,U2,U3
~~~

video：

~~~text
生成：
    U1,U2,U3
~~~

geometry：

~~~text
decode U0~U3
    ↓
生成/计算 geometry U1,U2,U3
~~~

action：

~~~text
action[48]~action[95]
~~~

packing：

~~~text
action[48]~action[63]  → U1
action[64]~action[79]  → U2
action[80]~action[95]  → U3
~~~

---

## 16. 当前 autoregressive_rollout 的全局索引写回

代码的核心写回：

~~~python
target_start = (pair + 1) * chunk
target_end = target_start + chunk

working["latents"][:, :, target_start:target_end] = (
    result.pred_latents[:, :, history:]
)

working["actions"][:, :, target_start:target_end] = (
    result.pred_actions[:, :, history:]
)

working["geometry_rgb"][:, target_start:target_end] = (
    result.action_geometry_rgb[:, history:]
)
~~~

默认：

~~~text
chunk = 4
history = 4
~~~

第一轮：

~~~text
local result 的 target 区域 result[:, :, 4:8]
    ↓
写入 global working[:, :, 4:8]
~~~

第二轮：

~~~text
local result 的 target 区域 result[:, :, 4:8]
    ↓
写入 global working[:, :, 8:12]
~~~

所以多轮全局 latent 轴：

~~~text
global 0:4   = 初始 history
global 4:8   = 第 1 个生成 chunk
global 8:12  = 第 2 个生成 chunk
global 12:16 = 第 3 个生成 chunk
~~~

每个局部窗口都使用：

~~~text
local 0:4 = history
local 4   = target anchor
local 5:8 = 本轮生成区域
~~~

---

## 17. 三种 index 的区别

### 17.1 raw image/frame index

~~~text
image[100]
~~~

表示 episode 原始视频中的第 100 帧。

如果 FPS=30：

~~~text
timestamp = 100 / 30 ≈ 3.333 秒
~~~

### 17.2 latent frame index

~~~text
latent index 5
~~~

表示当前 8-latent MOT window 中的第 6 个时间位置，通常是 target T1。

它不是 raw image[5]，也不是 action[5]。

### 17.3 raw action index

~~~text
action[100]
~~~

通常表示以 state/image[100] 为参考的第 100 个控制动作，近似影响：

~~~text
image[100] → image[101]
~~~

当 target 起点是 100 时，它会被 pack 到：

~~~text
local latent 5, token 0
~~~

但这只是模型内部的 token 对齐位置，不代表 action[100] 负责整个 latent 时间段。

---

## 18. 最终时序图

~~~text
初始 history
    │
    │ local window: H0 H1 H2 H3 T0 | T1 T2 T3
    ▼
生成 video T1,T2,T3
    │
    ▼
decode target RGB
    │
    ▼
重新计算 geometry T1,T2,T3
    │
    ▼
生成 action[0]~action[47]
    │
    ▼
写回 working：
    H0 H1 H2 H3 T0 T1 T2 T3
    │
    │ 下一轮把 T0~T3 作为 history
    ▼
local window: T0 T1 T2 T3 U0 | U1 U2 U3
    │
    ▼
生成 video U1,U2,U3
    │
    ▼
重新计算 geometry U1,U2,U3
    │
    ▼
生成 action[48]~action[95]
~~~

---

## 19. 总结

从 t=0 开始的真实自回归链路是：

~~~text
初始 history/state
    ↓
run_video_inference
    生成 target video latent T1~T3
    ↓
decode generated video RGB
    ↓
_run_geometry_from_rgb
    重新计算 target geometry condition
    ↓
_sample_actions
    生成 target action T1~T3
    ↓
写入 working
    ↓
下一轮把上一轮 T0~T3 当作新的 history
    ↓
继续生成 U1~U3 video
    ↓
继续生成 U1~U3 geometry/action
~~~

索引规则：

~~~text
单轮 local latent：
    0~3 = history
    4   = target anchor
    5~7 = 本轮新生成

action packing：
    latent 5 → 16 个 raw action
    latent 6 → 16 个 raw action
    latent 7 → 16 个 raw action

第一轮：
    action[0]~action[47]

第二轮：
    action[48]~action[95]

训练：
    固定 H0~H3,T0~T3
    使用 GT target 加噪并计算 loss

rollout：
    video → geometry → action
    将生成结果反馈到下一轮
~~~

因此，训练是单窗口 teacher forcing；推理/rollout 是按 chunk 进行的 feedback 式自回归生成。
