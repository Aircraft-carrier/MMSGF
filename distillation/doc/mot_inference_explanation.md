# MOT 推理流程说明

本文说明 `inference/mot_inference.py` 中 `run_mot_inference` 的执行过程，以及它调用的 `run_video_inference`、`_run_geometry_from_rgb` 和 `_sample_actions`。

整体流程是：

```text
GT 历史视频 latent + GT 历史动作 + 文本
                    │
                    ▼
        run_video_inference
        生成目标视频 latent
                    │
                    ▼
        VAE 解码为目标 RGB
                    │
                    ▼
        _run_geometry_from_rgb
        用 GT 历史 RGB + 生成目标 RGB
        重新计算几何
                    │
                    ▼
        _sample_actions
        固定视频和几何，只生成目标动作
                    │
                    ▼
        MOTInferenceResult
```

核心顺序是：

```text
V → G → A
视频 → 几何 → 动作
```

其中：

- `V` 是视频 latent；
- `G` 是 geometry，包括 depth、depth confidence、3D points；
- `A` 是 action；
- 视频生成使用 classifier-free guidance；
- 动作生成当前要求 `action_guidance_scale == 1`，因此不做 cond/uncond 双分支。

---

## 1. 默认 MOT 窗口和时间索引

默认配置通常为：

```python
action_chunk_size = 48
video_downsample_ratio = 4
vae_temporal_factor = 4
```

由此得到：

```text
每个 action chunk 的原始视频帧数：
48 / 4 + 1 = 13

每个 action chunk 对应的 latent 帧数：
(13 - 1) / 4 + 1 = 4

历史 latent 帧：
4

目标 latent 帧：
4

总 latent 帧：
8
```

时间轴是：

```text
latent index:
0    1    2    3    4    5    6    7
├────历史 chunk────┤    ├────目标 chunk────┤
                  anchor
```

代码通过：

```python
history_end = spec.history_latent_frames
generated_target_start = history_end + 1
target_slice = slice(generated_target_start, frame_count)
```

确定区域。默认值为：

```python
history_end = 4
target_slice = slice(5, 8)
```

因此：

- `0:4` 是历史 latent，使用 GT；
- `4` 是目标 chunk 的 GT anchor；
- `5:8` 是真正生成的目标 latent；
- index 4 不是生成目标，但会作为目标 chunk 的第一帧 anchor。

这是整段代码最重要的索引约定。

---

## 2. `run_mot_inference` 入口

函数定义：

```python
@torch.no_grad()
def run_mot_inference(
    batch: dict[str, Any],
    frame_count: int,
    *,
    transformer,
    config,
    spec,
    device: torch.device,
    empty_text_emb: torch.Tensor,
    decode_latents_to_rgb_views: Callable[..., torch.Tensor],
    scheduler_factory=FlowMatchScheduler,
) -> MOTInferenceResult:
```

它依次调用三个阶段：

```python
video = run_video_inference(...)
```

第一阶段生成视频。

```python
geometry, geometry_condition = _run_geometry_from_rgb(
    batch,
    frame_count,
    geometry_rgb=video.action_geometry_rgb,
    ...
)
```

第二阶段用生成后的视频 RGB 重新计算 geometry。

```python
pred_actions = _sample_actions(
    batch,
    frame_count,
    pred_latents=video.pred_latents,
    geometry_rgb=video.action_geometry_rgb,
    geometry_condition=geometry_condition,
    ...
)
```

第三阶段固定视频和 geometry，生成动作。

最终返回：

```python
MOTInferenceResult(
    pred_latents=video.pred_latents,
    pred_actions=pred_actions,
    pred_rgb=video.pred_rgb,
    pred_depth=geometry.pred_depth,
    pred_depth_conf=geometry.pred_depth_conf,
    pred_points=geometry.pred_points,
    action_geometry_rgb=video.action_geometry_rgb,
)
```

返回字段来源：

| 字段 | 来源 |
|---|---|
| `pred_latents` | `run_video_inference` |
| `pred_rgb` | `run_video_inference` 解码结果 |
| `action_geometry_rgb` | GT 历史 RGB + 生成目标 RGB |
| `pred_depth` | `_run_geometry_from_rgb` |
| `pred_depth_conf` | `_run_geometry_from_rgb` |
| `pred_points` | `_run_geometry_from_rgb` |
| `pred_actions` | `_sample_actions` |

---

## 3. 主要输入张量

### 3.1 视频 latent

```python
batch["latents"]
```

典型 shape：

```text
[B, C, T, V, H, W]
```

例如：

```python
[2, 16, 8, 2, 48, 80]
```

测试中为了简化使用：

```python
[1, 1, 8, 1, 1, 1]
```

### 3.2 Geometry RGB

```python
batch["geometry_rgb"]
```

典型 shape：

```text
[B, T, 4, V, 3, H, W]
```

例如：

```python
[2, 8, 4, 2, 3, 224, 384]
```

这里的 4 是一个 geometry group 内的 slot 数。

每个 action chunk 的 4 个 latent frame 对应：

```text
1 + 4 + 4 + 4 = 13 张原始 RGB
```

第一个 latent frame 在 VAE 中只对应一张原始 RGB，为统一 shape 会复制到四个 slot；后面的 latent frame 分别对应连续四张 RGB。

### 3.3 Geometry mask

```python
batch["geometry_group_valid_mask"]
```

典型 shape：

```text
[B, T, 4]
```

它标记每个 latent frame 的四个 geometry slot 是否有效。

第一个 group 可能是：

```text
[True, False, False, False]
```

完整 group 通常是：

```text
[True, True, True, True]
```

### 3.4 动作

```python
batch["actions"]
```

典型 shape：

```text
[B, action_dim, T, action_per_frame, 1]
```

默认：

```text
action_dim = 20
action_per_frame = 16
T = 8
```

所以：

```text
actions.shape = [B, 20, 8, 16, 1]
```

历史动作作为条件，目标动作由 `action_loss_mask` 控制并进行采样。

### 3.5 文本

```python
batch["text_emb"]
```

通常是：

```text
[B, text_seq_len, text_dim]
```

`empty_text_emb` 用于视频采样的无条件分支。

---

## 4. `run_video_inference`

函数目标是：

```text
只使用可用的 GT 历史信息，生成目标视频 V。
```

### 4.1 检查窗口

首先调用：

```python
_validate_complete_window(batch, frame_count, spec, require_latents=True)
```

默认要求：

```python
frame_count == spec.total_latent_frames == 8
batch["latents"].shape[2] == 8
batch["geometry_rgb"].shape[1] == 8
```

这是固定完整窗口推理，不是任意长度的滑动生成。

### 4.2 创建视频 scheduler

```python
video_scheduler = scheduler_factory(
    shift=config.snr_shift,
    sigma_min=0.0,
    extra_one_step=True,
)
video_scheduler.set_timesteps(int(config.num_inference_steps))
guidance_scale = float(config.guidance_scale)
```

视频 scheduler 使用 `snr_shift` 和 `num_inference_steps`；视频 CFG 使用 `guidance_scale`。

### 4.3 准备 GT 输入

```python
gt_latents = batch["latents"][:, :, :frame_count].to(
    device=device,
    dtype=torch.bfloat16,
)
```

`MOT_INFERENCE_DTYPE` 是 `torch.bfloat16`。

同时读取：

```text
gt_actions
action_loss_mask
action_valid_mask
text_emb
geometry_rgb_gt
geometry_group_valid_mask
stream_ids
video_latent_valid_mask
```

真实文本用于 cond 分支，`empty_text_emb` 会被调整为相同 shape 后用于 uncond 分支。

### 4.4 屏蔽目标 geometry

先复制 GT geometry：

```python
video_geometry_rgb = geometry_rgb_gt.clone()
```

然后清零目标区间：

```python
video_geometry_rgb[:, target_slice] = 0
```

默认结果：

```text
index 0~4：保留 GT geometry
index 5~7：全部为 0
```

同时：

```python
video_geometry_slot_valid = geometry_slot_valid_mask.clone()
video_geometry_slot_valid[:, target_slice] = False
```

这样视频模型生成目标时只能看到历史和 anchor，不能看到目标 GT RGB。

### 4.5 预计算视频阶段的 geometry condition

```python
video_geometry_condition = transformer(
    _geometry_input(
        rgb=video_geometry_rgb,
        stream_ids=stream_ids,
        slot_valid_mask=video_geometry_slot_valid,
        chunk_size=chunk_size,
        window_size=window_size,
        return_points=False,
    ),
    mode="precompute_geometry",
)
```

模型内部会：

1. 对 geometry RGB 做 embedding；
2. 运行 geometry stream；
3. 计算 geometry token；
4. 捕获 joint MOT 层需要的 registers；
5. 返回 `final_geometry` 和 `layer_registers`。

这里 `return_points=False`，因为这个阶段只需要 geometry condition，不需要最终的 depth/points 输出。

该 condition 会在视频的每个 diffusion step 中复用，不会重复计算。

### 4.6 初始化目标视频噪声

动作条件先处理为：

```python
video_action_noisy = gt_actions.clone()
video_action_noisy[:, :, target_slice] = 0

video_action_clean = video_action_noisy.clone()

video_action_valid_mask = action_valid_mask.clone()
video_action_valid_mask[:, :, target_slice] = False
```

目标视频 latent 从随机噪声开始：

```python
cur_latents = torch.randn_like(
    gt_latents[:, :, target_slice]
)
```

如果完整 latent shape 是：

```text
[B, C, 8, V, H, W]
```

则目标噪声 shape 是：

```text
[B, C, 3, V, H, W]
```

### 4.7 每个视频 timestep 的输入

循环：

```python
for timestep in video_scheduler.timesteps.to(device):
```

每轮构造：

```python
latent_noisy = gt_latents.clone()
latent_clean = gt_latents.clone()

latent_noisy[:, :, target_slice] = cur_latents
latent_clean[:, :, target_slice] = 0
```

因此：

```text
latent_noisy:
0~4 = GT latent
5~7 = 当前 cur_latents

latent_clean:
0~4 = GT latent
5~7 = 0
```

时间向量：

```python
latent_timesteps = torch.zeros(
    (gt_latents.shape[0], frame_count),
    device=device,
    dtype=torch.float32,
)
latent_timesteps[:, target_slice] = timestep
```

例如当前 timestep 为 `0.8`：

```text
[0, 0, 0, 0, 0, 0.8, 0.8, 0.8]
```

动作 timestep 全部为 0，因为此阶段不采样动作。

### 4.8 cond/uncond 两次 forward

条件分支：

```python
cond_out = transformer(
    _inference_input(text_emb=text_emb, **common_kwargs),
    mode="inference_video",
)
```

无条件分支：

```python
uncond_out = transformer(
    _inference_input(text_emb=uncond_text_emb, **common_kwargs),
    mode="inference_video",
)
```

两次调用的 geometry condition 相同，主要区别是文本 embedding。

输出：

```python
{"latent_pred": ...}
```

模型内部通过 `_final_video` 将输出投影回视频 latent shape。

### 4.9 classifier-free guidance

```python
latent_prediction = uncond_out["latent_pred"] + guidance_scale * (
    cond_out["latent_pred"] - uncond_out["latent_pred"]
)
```

数学形式：

```text
prediction = uncond + guidance_scale × (cond - uncond)
```

例如：

```text
uncond = 1
cond = 3
guidance_scale = 5

prediction = 1 + 5 × (3 - 1) = 11
```

### 4.10 scheduler 更新目标 latent

```python
cur_latents = video_scheduler.step(
    latent_prediction[:, :, target_slice],
    timestep,
    cur_latents,
)
```

只将 `5:8` 的目标部分交给 scheduler，历史 latent 从不修改。

循环结束后：

```python
pred_latents = gt_latents.clone()
pred_latents[:, :, target_slice] = cur_latents
```

所以：

```text
pred_latents[:, :, 0:5] = GT latent
pred_latents[:, :, 5:8] = 生成 latent
```

### 4.11 解码目标视频

```python
decoded_target_rgb = decode_latents_to_rgb_views(
    pred_latents[:, :, history_end:frame_count]
)
```

传入的是：

```python
pred_latents[:, :, 4:8]
```

即 anchor latent 4 加上生成 latent 5、6、7。

Wan VAE 的时间格式是：

```text
1 个首帧 + 每个后续 latent 对应 4 张 RGB
```

所以 4 个 latent 会解码成：

```text
1 + 4 × (4 - 1) = 13 张 RGB
```

典型 shape：

```text
[B, 13, V, 3, H_rgb, W_rgb]
```

### 4.12 重新分组 geometry RGB

```python
decoded_target_geometry = decoded_rgb_to_geometry_groups(
    decoded_target_rgb,
    latent_frames=int(spec.target_latent_frames),
    vae_temporal_factor=int(config.vae_temporal_factor),
)
```

13 张 RGB 会被分成 4 个 latent group：

```text
group 0 = decoded_rgb[0]，复制为 4 个 slot，代表 anchor
group 1 = decoded_rgb[1:5]
group 2 = decoded_rgb[5:9]
group 3 = decoded_rgb[9:13]
```

输出 shape：

```text
[B, 4, 4, V, 3, H_rgb, W_rgb]
```

### 4.13 生成 `pred_rgb` 和 `action_geometry_rgb`

代表性 RGB：

```python
pred_rgb = geometry_rgb_gt[:, :, 0].clone()
pred_rgb[:, target_slice] = decoded_target_geometry[:, 1:, 0]
```

因此：

```text
pred_rgb[:, 0:5] = GT RGB
pred_rgb[:, 5:8] = 生成 RGB 的每组第 0 个 slot
```

完整 geometry RGB：

```python
action_geometry_rgb = geometry_rgb_gt.clone()
action_geometry_rgb[:, target_slice] = decoded_target_geometry[:, 1:]
```

因此：

```text
action_geometry_rgb[:, 0:5] = GT geometry RGB
action_geometry_rgb[:, 5:8] = 生成 RGB 的完整四-slot group
```

区别是：

- `pred_rgb` 每个 latent frame 只保留一个代表性 RGB；
- `action_geometry_rgb` 保留完整的四个 geometry slot，供后续 geometry 和 action 使用。

---

## 5. `_run_geometry_from_rgb`

它的作用是：

```text
对给定的完整 geometry RGB 运行 geometry stream，
得到 depth、depth confidence、points，
同时得到供动作推理复用的 geometry condition。
```

在完整推理中，输入是：

```python
geometry_rgb=video.action_geometry_rgb
```

也就是 GT 历史 geometry 加上生成目标 geometry。

### 5.1 运行 geometry stream

```python
geometry_rgb = geometry_rgb[:, :frame_count].to(
    device,
    dtype=torch.bfloat16,
)

condition = transformer(
    _geometry_input(
        rgb=geometry_rgb,
        stream_ids=batch["stream_ids"].to(device),
        slot_valid_mask=batch["geometry_group_valid_mask"][
            :, :frame_count
        ].to(device, dtype=torch.bool),
        chunk_size=chunk_size,
        window_size=window_size,
        return_points=True,
    ),
    mode="precompute_geometry",
)
```

这次 `return_points=True`，所以模型会返回：

```python
{
    "final_geometry": ...,
    "layer_registers": ...,
    "diagnostics": ...,
    "depth": ...,
    "depth_conf": ...,
    "points": ...,
    "points_conf": ...,
}
```

其中：

- `final_geometry` 是 geometry stream 的最终 token；
- `layer_registers` 供后续 action inference 使用；
- `depth` 是深度预测；
- `depth_conf` 是深度置信度；
- `points` 是 3D 点预测。

### 5.2 处理 depth 时间维

```python
pred_depth = representative_geometry_frames(
    condition["depth"],
    frame_count,
)
pred_depth_conf = representative_geometry_frames(
    condition["depth_conf"],
    frame_count,
)
```

如果输入是：

```text
[B, 4T, ...]
```

会被 reshape 为：

```text
[B, T, 4, ...]
```

再取每个 group 的第 0 个 slot。因此：

```text
condition["depth"] 可能是 [B, 4T, V, H, W]
pred_depth          变成 [B, T, V, H, W]
```

`points` 不做 representative：

```python
pred_points = condition["points"]
```

因此它通常保留所有 geometry slot，例如：

```text
[B, 4T, V, H, W, 3]
```

### 5.3 返回结果和 condition

```python
result = MOTGeometryInferenceResult(
    pred_depth=...,
    pred_depth_conf=...,
    pred_points=condition["points"],
    geometry_rgb=geometry_rgb,
)
return result, condition
```

第二个返回值 `condition` 会传给 `_sample_actions`，因此动作阶段不会再次计算 geometry。

---

## 6. `_sample_actions`

它的作用是：

```text
固定已经生成的视频 latent 和 geometry RGB，
只对目标动作进行扩散采样。
```

### 6.1 action guidance 限制

```python
if float(config.action_guidance_scale) != 1.0:
    raise ValueError(...)
```

当前必须满足：

```python
config.action_guidance_scale == 1
```

视频采样有 cond/uncond 两次 forward，但动作采样每个 timestep 只有一次 cond forward。

### 6.2 创建 action scheduler

```python
action_scheduler = scheduler_factory(
    shift=config.action_snr_shift,
    sigma_min=0.0,
    extra_one_step=True,
)
action_scheduler.set_timesteps(
    int(config.action_num_inference_steps)
)
```

动作 scheduler 与视频 scheduler 独立，使用：

- `action_snr_shift`；
- `action_num_inference_steps`。

### 6.3 初始化动作

```python
pred_actions = gt_actions.clone()
pred_actions[:, :, target_slice] = 0

cur_actions = torch.randn_like(
    gt_actions[:, :, target_slice]
)

target_action_mask = action_loss_mask[
    :, :, target_slice
].to(gt_actions.dtype)

cur_actions = cur_actions * target_action_mask
```

默认情况下：

```text
pred_actions[:, :, 0:5] = GT history actions
pred_actions[:, :, 5:8] = 0
cur_actions.shape = [B, 20, 3, 16, 1]
```

无效目标 token 会被 mask 为 0。

### 6.4 每轮构造 action 输入

```python
action_noisy = pred_actions.clone()
action_clean = pred_actions.clone()

action_noisy[:, :, target_slice] = cur_actions
action_clean[:, :, target_slice] = 0
```

因此：

```text
action_noisy:
0~4 = GT history actions
5~7 = 当前 cur_actions

action_clean:
0~4 = GT history actions
5~7 = 0
```

动作 timestep：

```python
action_timesteps = torch.zeros(
    (gt_actions.shape[0], frame_count),
    device=device,
    dtype=torch.float32,
)
action_timesteps[:, target_slice] = timestep
```

例如：

```text
[0, 0, 0, 0, 0, 0.8, 0.8, 0.8]
```

### 6.5 视频 latent 和 geometry 在动作阶段固定

调用时：

```python
latent_noisy=pred_latents
latent_clean=pred_latents
geometry_condition=geometry_condition
```

并且：

```python
latent_timesteps = 0
```

也就是说：

- 视频 latent 不会再被更新；
- geometry condition 不会再被更新；
- 动作模型可以读取生成视频和生成几何；
- 只有 `cur_actions` 进入动作 scheduler。

### 6.6 调用 `inference_action`

```python
out = transformer(
    _inference_input(
        latent_noisy=pred_latents,
        latent_clean=pred_latents,
        action_noisy=action_noisy,
        action_clean=action_clean,
        geometry_condition=geometry_condition,
        ...
    ),
    mode="inference_action",
)
```

模型内部会：

1. 使用固定的 `pred_latents`；
2. 使用生成 RGB 得到的 `geometry_condition`；
3. 使用真实文本 `text_emb`；
4. 运行 joint MOT layers；
5. 通过 `_final_action` 输出动作预测。

返回：

```python
{"action_pred": ...}
```

典型 shape：

```text
[B, 20, 8, 16, 1]
```

### 6.7 更新目标动作

```python
cur_actions = action_scheduler.step(
    out["action_pred"][:, :, target_slice],
    timestep,
    cur_actions,
)
cur_actions = cur_actions * target_action_mask
```

只更新目标动作 `5:8`，并在每一步重新应用 action mask。

最后：

```python
pred_actions[:, :, target_slice] = cur_actions
```

所以：

```text
pred_actions[:, :, 0:5] = GT history actions
pred_actions[:, :, 5:8] = sampled target actions
```

---

## 7. Transformer 调用次数

假设：

```python
config.num_inference_steps = 25
config.action_num_inference_steps = 50
```

视频阶段：

```text
1 次 precompute_geometry
25 × (1 次 inference_video cond + 1 次 inference_video uncond)
= 51 次
```

生成后 geometry：

```text
1 次 precompute_geometry
```

动作阶段：

```text
50 次 inference_action
```

完整推理总计：

```text
1 + 25 × 2 + 1 + 50 = 102 次 transformer 调用
```

测试中的 fake scheduler 只有两个 timestep，因此事件序列为：

```python
[
    ("precompute_geometry", None),
    ("inference_video", "cond"),
    ("inference_video", "uncond"),
    ("inference_video", "cond"),
    ("inference_video", "uncond"),
    ("precompute_geometry", None),
    ("inference_action", "cond"),
    ("inference_action", "cond"),
]
```

---

## 8. 模拟输入

为了便于展示，使用：

```text
B = 1
C = 1
T = 8
V = 1
H = W = 1
```

### 8.1 spec

```python
spec = SimpleNamespace(
    total_latent_frames=8,
    history_latent_frames=4,
    target_latent_frames=4,
    latent_frames_per_action_chunk_per_view=4,
    attention_window_size=4,
    action_per_frame=1,
)
```

真实默认配置通常是 `action_per_frame = 16`，这里只是为了简化示例。

### 8.2 config

```python
config = SimpleNamespace(
    snr_shift=5.0,
    action_snr_shift=1.0,
    num_inference_steps=2,
    action_num_inference_steps=2,
    guidance_scale=5.0,
    action_guidance_scale=1.0,
    vae_temporal_factor=4,
)
```

### 8.3 latent

```python
latents = torch.arange(
    8,
    dtype=torch.float32,
).view(1, 1, 8, 1, 1, 1)
```

表示：

```text
latents[0, 0, :, 0, 0, 0] = [0, 1, 2, 3, 4, 5, 6, 7]
```

其中：

```text
0~3 = 历史 latent
4   = GT anchor
5~7 = 待生成 latent
```

### 8.4 geometry RGB

```python
geometry_rgb = torch.full(
    (1, 8, 4, 1, 3, 1, 1),
    7.0,
)
geometry_rgb[:, :4] = 1.0
geometry_rgb[:, 4] = 2.0
```

表示：

```text
geometry_rgb[:, 0:4] = 1
geometry_rgb[:, 4]   = 2
geometry_rgb[:, 5:8] = 7
```

### 8.5 其他 batch 字段

```python
batch = {
    "latents": latents,
    "actions": torch.zeros(1, 1, 8, 1, 1),
    "action_loss_mask": torch.ones(
        1, 1, 8, 1, 1,
        dtype=torch.bool,
    ),
    "action_valid_mask": torch.ones(
        1, 1, 8, 1, 1,
        dtype=torch.bool,
    ),
    "text_emb": torch.ones(1, 1, 1),
    "geometry_rgb": geometry_rgb,
    "geometry_group_valid_mask": torch.ones(
        1, 8, 4,
        dtype=torch.bool,
    ),
    "stream_ids": torch.zeros(
        1, 1,
        dtype=torch.long,
    ),
    "video_latent_valid_mask": torch.ones(
        1, 8,
        dtype=torch.bool,
    ),
}
```

---

## 9. 模拟输出

假设：

```python
torch.randn_like(...) = 4
```

视频 fake transformer 输出：

```text
cond latent_pred = 3
uncond latent_pred = 1
```

在 `guidance_scale = 5` 时：

```text
guided = 1 + 5 × (3 - 1) = 11
```

如果 fake scheduler 的 `step` 原样返回输入 sample，则目标 latent 始终为 4：

```text
pred_latents = [0, 1, 2, 3, 4, 4, 4, 4]
```

解码时传入：

```python
pred_latents[:, :, 4:8]
```

它代表：

```text
[anchor latent 4, target latent 5, target latent 6, target latent 7]
```

假设 decoder 返回 13 张全为 9 的 RGB，重新分组后：

```text
group 0 = [9, 9, 9, 9]  # anchor
group 1 = [9, 9, 9, 9]  # latent 5
group 2 = [9, 9, 9, 9]  # latent 6
group 3 = [9, 9, 9, 9]  # latent 7
```

于是：

```text
pred_rgb = [1, 1, 1, 1, 2, 9, 9, 9]
```

``action_geometry_rgb`` 的 group 为：

```text
latent 0: [1, 1, 1, 1]
latent 1: [1, 1, 1, 1]
latent 2: [1, 1, 1, 1]
latent 3: [1, 1, 1, 1]
latent 4: [2, 2, 2, 2]
latent 5: [9, 9, 9, 9]
latent 6: [9, 9, 9, 9]
latent 7: [9, 9, 9, 9]
```

假设第二次 geometry fake transformer 返回：

```text
depth = 2
depth_conf = 22
points = 12
```

如果 depth 原始 shape 是：

```text
[1, 32, 1, 1, 1]
```

因为 `8 × 4 = 32`，经过 `representative_geometry_frames` 后：

```text
pred_depth.shape = [1, 8, 1, 1, 1]
pred_depth 的值 = 2

pred_depth_conf.shape = [1, 8, 1, 1]
pred_depth_conf 的值 = 22
```

points 不做 representative，因此可能是：

```text
pred_points.shape = [1, 32, 1, 1, 1, 3]
pred_points 的值 = 12
```

动作阶段假设初始随机动作全为 4、action mask 全为 True，fake scheduler 原样返回 sample，则：

```text
pred_actions = [0, 0, 0, 0, 0, 4, 4, 4]
```

最终结果可以概括为：

```python
result.pred_latents
# shape: [1, 1, 8, 1, 1, 1]
# value: [0, 1, 2, 3, 4, 4, 4, 4]

result.pred_rgb
# shape: [1, 8, 1, 3, 1, 1]
# representative RGB: [1, 1, 1, 1, 2, 9, 9, 9]

result.pred_depth
# shape: [1, 8, 1, 1, 1]
# value: 2

result.pred_depth_conf
# shape: [1, 8, 1, 1]
# value: 22

result.pred_points
# shape: [1, 32, 1, 1, 1, 3]
# value: 12

result.pred_actions
# shape: [1, 1, 8, 1, 1]
# value: [0, 0, 0, 0, 0, 4, 4, 4]
```

这些是 fake 数值，只用于展示数据流。真实运行时：

- `pred_latents[:, :, 5:8]` 来自视频 scheduler 的多步更新；
- `pred_rgb[:, 5:8]` 来自 VAE 解码；
- `pred_depth` 和 `pred_points` 来自 geometry 网络；
- `pred_actions[:, :, 5:8]` 来自动作 scheduler 的多步更新。

---

## 10. 一句话总结

### `run_mot_inference`

负责整体编排：

```text
生成视频 → 根据生成视频重新估计几何 → 根据视频和几何生成动作
```

### `run_video_inference`

```text
历史 latent 保持 GT
目标 latent 从噪声开始扩散
历史 geometry 作为条件
使用 classifier-free guidance
最后 VAE 解码成 RGB
```

### `_run_geometry_from_rgb`

```text
RGB → geometry transformer → depth/depth_conf/points + geometry condition
```

### `_sample_actions`

```text
视频 latent 固定
geometry condition 固定
历史 action 固定
目标 action 从噪声开始扩散
每一步只调用一次 inference_action
```

整个 MOT 推理的设计是：

```text
历史信息锁定为 GT，
目标视频先生成，
几何从生成视频重新估计，
动作最后基于生成视频和生成几何预测。
```

