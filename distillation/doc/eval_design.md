# Distilled autoregressive MOT × RoboTwin evaluation design

## 1. 目标与第一版边界

在 MMSGF（policy 环境）实现：

- `distillation/eval/server.py`：加载 distilled autoregressive MOT，提供 HTTP 服务；
- `distillation/eval/infer_pipeline.py`：把在线 RoboTwin 轨迹整理成训练一致的 history/anchor，构建 KV cache，使用 Euler 更新生成 video latent 和 action；
- `GET /healthz`：模型、VAE、text encoder 和数据统计全部就绪后才返回 200；
- `POST /v1/reset`：开始一个新 episode，清空 raw buffer 和 KV cache；
- `POST /v1/actions`：追加本轮执行结果，返回下一段 16 个绝对 EEF action。

第一版固定以下约束：

- `B=1`，一个 server 进程同一时间只处理一个 episode；
- 每次 replan 生成一个 action frame，即 16 个 action；
- video latent 始终先生成，因为模型训练顺序是 Video→Action；
- `return_video=false` 只跳过 VAE decode 和网络返回，不跳过 video latent 生成；
- 每次请求根据 raw ring buffer 重建 history/anchor cache，先保证与 dataset 对齐；
- 不在第一版实现多 chunk、action ensemble、异步 action streaming 和多 session 并发。

直接跳过 video 生成会让 action 缺少训练时存在的前驱 video token，不作为正式评估模式。多 chunk 应在单 chunk 的离线对齐和 RoboTwin smoke test 通过后再增加。

## 2. 现有代码与复用边界

实现应复用而不是复制以下逻辑：

- `wan_va/dataset/mot_dataset.py`
  - `mot_real_window_frame_ids`
  - `_latent_valid_mask_from_sampled_frames`
  - `absolute_actions_to_relative_20d`
  - `relative_20d_to_absolute_actions`
- `wan_va/train_mot.py`
  - streaming VAE 的 `1 + 4 * (F - 1)` 编码规则；应抽成共享 helper，不要 import `MOTTrainer`；
- `distillation/model/autoregressive_mot.py`
  - `AutoregressiveVAMOTTransformer3DModel`
  - `predict_video`、`commit_video`、`predict_action`、`commit_action`；
- `distillation/pipeline/cache.py`
  - `KVCache` transaction/commit 语义；
- `wan_va/utils/scheduler.py`
  - `FlowMatchScheduler.step` 的 Euler 更新。

评估路径不创建 `WanDiffusionWrapper`。`server.py` 直接通过 `AutoregressiveVAMOTTransformer3DModel.from_pretrained(...)` 加载原生 AR MOT；它的 `predict_video()` 和 `predict_action()` 返回模型投影后的 flow，可直接传给 `FlowMatchScheduler.step()`。`infer_pipeline.py` 只负责组织原生模型调用，不增加 model wrapper 或 flow/x0 转换层。

## 3. 两仓职责

MMSGF policy repo 负责模型相关逻辑：

```text
distillation/eval/
├── server.py                 # HTTP、模型生命周期、串行化 inference
├── infer_pipeline.py         # online window、VAE、KV cache、Euler、action decode
├── protocol.py               # 请求/响应 schema 和 image codec
└── tests/
    ├── test_online_window.py
    ├── test_infer_pipeline.py
    └── test_server.py
```

RoboTwin repo 负责环境相关逻辑：

```text
policy/mmsgf/deploy_policy.py  # observation adapter、HTTP client、执行 16 个 ee action
script/eval_policy_client.py   # episode loop、progress/result 文件
```

不要让 policy server import RoboTwin/SAPIEN，也不要让 RoboTwin client 加载 MMSGF checkpoint。两个进程使用各自的 Python 环境。

## 4. Wire protocol

### 4.1 标准 observation

RoboTwin adapter 先把环境 observation 转成稳定 schema，再发送给 server：

```python
OnlineObservation = {
    "step": int,  # episode 内 observation 序号；reset 后首帧为 0
    "images": {
        "cam_high": JPEG_BASE64,
        "cam_left_wrist": JPEG_BASE64,
        "cam_right_wrist": JPEG_BASE64,
    },
    "state": [float] * 16,  # 模型坐标约定，见 4.4
}
```

传输前统一 resize/crop 到训练分辨率 `256x320`，保持 RGB、`uint8`。camera mapping 固定为：

| RoboTwin key | 模型 key | stream id |
|---|---|---:|
| `head_camera.rgb` | `cam_high` | 1 |
| `left_camera.rgb` | `cam_left_wrist` | 0 |
| `right_camera.rgb` | `cam_right_wrist` | 2 |

实现前必须用一个真实 observation 打印并验证这些 key；缺任一 camera 直接失败，不用黑图补齐。

### 4.2 Reset

```http
POST /v1/reset
Content-Type: application/json

{
  "session_id": "adjust_bottle-seed-100000",
  "task_name": "adjust_bottle-demo_clean_collect_200-50",
  "instruction": "adjust the bottle",
  "seed": 100000
}
```

响应：

```json
{
  "session_id": "adjust_bottle-seed-100000",
  "next_observation_step": 0
}
```

`reset` 必须清空 raw observations、executed actions、KV cache、上一次幂等响应和 VAE streaming state。`task_name` 必须存在于 `norm_stats_by_task`。

### 4.3 Infer

第一次调用发送首 observation，没有 executed action：

```json
{
  "session_id": "adjust_bottle-seed-100000",
  "request_id": 0,
  "observations": [{"step": 0, "images": "3 JPEG base64 values", "state": "16 floats"}],
  "executed_actions": [],
  "return_video": false
}
```

执行完 server 返回的 16 个 action 后，第二次调用发送 16 个 post-action observation 和实际执行的 16 个 action：

```json
{
  "session_id": "adjust_bottle-seed-100000",
  "request_id": 1,
  "observations": [
    {"step": 1, "images": "3 JPEG base64 values", "state": "16 floats"},
    "...",
    {"step": 16, "images": "3 JPEG base64 values", "state": "16 floats"}
  ],
  "executed_actions": ["16 absolute EEF actions"],
  "return_video": false
}
```

追加契约：若 server 已保存到 observation `t`，则一个请求中的 `k` 个 executed actions 必须和 `k` 个新 observations 成对，表示 `action[t+i] -> observation[t+i+1]`。第一次请求是唯一允许 `1 observation + 0 action` 的情况。

响应固定返回：

```json
{
  "session_id": "adjust_bottle-seed-100000",
  "request_id": 1,
  "observation_step": 16,
  "actions": [["16 floats"]],
  "action_type": "ee",
  "predicted_video": null,
  "timings_ms": {
    "window": 0.0,
    "vae": 0.0,
    "cache": 0.0,
    "video_euler": 0.0,
    "action_euler": 0.0,
    "total": 0.0
  }
}
```

`actions` shape 必须是 `[16,16]`。仅当 `return_video=true` 时返回 predicted-video artifact；否则为 null。同一 `request_id` 重试时返回缓存响应，不能重复 append buffer 或重复推进随机数状态。

### 4.4 State/action 坐标约定

模型内部必须使用和 prepared parquet 完全相同的 16D 字节语义：

```text
[left xyz, left quaternion q1..q4, left gripper,
 right xyz, right quaternion q1..q4, right gripper]  # 16D
```

当前有一个必须先解决的契约冲突：RoboTwin 的 `endpose` 来自 `transforms3d.mat2quat`，是 `wxyz`；prepared parquet 只把四元数字段命名为 `q1..q4`，而 MMSGF helper 当前按 `xyzw` 解释它。不能仅凭 helper 名称决定在线转换。

现有 clean-50 prepared parquet 保留 RoboTwin `transforms3d` 的 `wxyz` 四个槽位，并满足 `action[t] == state[t+1]`。已有 checkpoint 已经在这些未改写槽位上训练，因此 v1 adapter 原样传输这 16D 布局，不额外 permutation，然后调用：

```python
TASK_ENV.take_action(action, action_type="ee")
```

注意：MMSGF helper 当前名字含 `xyzw`，与 source 的 `wxyz` 命名不一致。这是已有 checkpoint 的数据语义，不能只在 eval 端“修正”；若以后修复数据预处理，必须重新生成 norm stats 并重新训练 checkpoint，同时给新 artifact 写入显式 `quaternion_order`。

## 5. Session 和 ring buffer

v1 server 只允许一个 active session，并用一个 inference lock 串行访问模型。使用标准库 HTTP server 启动：

```bash
python -m distillation.eval.server --checkpoint-root ... --dataset-root ... --model-root ...
```

启动，不能多 worker 重复加载模型。

每个 session 保存：

```python
@dataclass
class EpisodeState:
    session_id: str
    task_name: str
    text_emb: torch.Tensor
    observations: deque[OnlineObservation]  # 最多 50 个，保留 global step
    actions: deque[np.ndarray]               # 最多 48 个 absolute 16D action
    last_observation_step: int
    last_request_id: int
    last_response: dict | None
```

50 observations 和 48 actions 来自训练窗口：history 使用 `current-49 ... current-1` 的图像范围、最近 48 个 action，当前 observation 单独作为 anchor。deque 条目必须保留 global step，不能把 ring-buffer 下标当成 episode step。

## 6. Online window：严格复现 dataset 对齐

令当前 anchor observation 为 `O_t`。训练一致的 history raw image id 为：

```python
# 阶段 1/3：与 mot_real_window_frame_ids 完全一致
history_ids = [t - 49 + 4 * i for i in range(13)]

# 阶段 2/3：episode 左边界 padding 到 O_0，同时保留 raw valid mask
padded_ids = [max(0, index) for index in history_ids]
raw_valid = [index >= 0 for index in history_ids]

# 阶段 3/3：O_t 不是 history 的一部分，它是独立 anchor
anchor = observation[t]
```

history 的 13 张图按 streaming VAE 规则变为 4 个 latent：

```text
latent 0 <- sampled image 0
latent 1 <- sampled images 1..4
latent 2 <- sampled images 5..8
latent 3 <- sampled images 9..12
anchor   <- O_t，以新的 VAE streaming wrapper 单帧编码
```

共享 VAE helper 应显式提供 `encode_history_13_frames()` 和 `encode_anchor_frame()`；不能直接调用当前只接受 13 帧的 `_encode_one_view_latent()` 来编码 anchor。

不要把 `O_t` 同时放进 history 最后一组和 anchor。对于第二次 infer，buffer 有 `O_0..O_16` 时：

```text
history ids before padding = [-33,-29,-25,-21,-17,-13,-9,-5,-1,3,7,11,15]
history ids after padding  = [  0,  0,  0,  0,  0,  0, 0, 0, 0,3,7,11,15]
anchor                     = O_16
```

此时只有 history video latent 3 有真实图像；latent 0..2 根据 dataset mask 视为无效。history-only token compaction 会在 commit 前删除它们。

动作也必须使用 dataset packing：

```python
# current=t；早期 episode 允许 start<0
history_action_start = t - 48
reference_step = max(0, history_action_start)
reference_state = state[reference_step]

# action offset 0..47 映射到 latent frame 1..3；frame 0 始终为空
latent_offset = 1 + action_offset // 16
token_offset = action_offset % 16
```

只填充 episode 内真实存在的 action，其他 slot 保持 0 且 valid=false。每个有效 absolute action 先通过 `absolute_actions_to_relative_20d(reference_state, action)` 转为 20D，再用当前 task 的 `q01/q99` 归一化。

建议从 `MotTrainData._load_actions()` 抽出纯函数 `pack_relative_action_chunk(...)`，dataset 和 online builder 共用；不要在 inference 中维护第二份类似但不同的公式。

### 6.1 前四次 replan 的有效 history

| 当前 step | observations | 有效 video history latent | 有效 action history frame | anchor |
|---:|---|---|---|---|
| 0 | `O0` | 无 | 无 | `O0` |
| 16 | `O0..O16` | frame 3 | frame 3（`a0..a15`） | `O16` |
| 32 | `O0..O32` | frame 2..3 | frame 2..3（`a0..a31`） | `O32` |
| 48 | `O0..O48` | frame 1..3 | frame 1..3（`a0..a47`） | `O48` |

## 7. Cache 构建和生成顺序

每次 `/v1/actions` 重建 cache，顺序固定：

```python
# 阶段 1/3：history，mask 只用于 history token compaction
cache = KVCache()
model.commit_video(history_video_latents, frame_ids=[0, 1, 2, 3],
                   token_valid_mask=video_history_valid, cache=cache, ...)
model.commit_action(history_actions, frame_ids=[0, 1, 2, 3],
                    token_valid_mask=action_history_valid, cache=cache, ...)

# 阶段 2/3：当前 observation 是 anchor；anchor action 保持 dense 空 token
model.commit_video(anchor_video_latent, frame_ids=[4], cache=cache, ...)
model.commit_action(zeros([1, 20, 1, 16, 1]), frame_ids=[4], cache=cache, ...)

# 阶段 3/3：严格 Video -> Action
pred_video = euler_video(model, frame_id=5, cache=cache)
model.commit_video(pred_video, frame_ids=[5], cache=cache, ...)
pred_relative_action = euler_action(model, frame_id=5, cache=cache)
```

这里的一个 predicted video latent 对应训练 target chunk 中 anchor 之后的第一个 temporal latent，覆盖到下一次 replan 边界附近的视觉变化；它不是一张独立 RGB 图片。

第一次 infer 的 history valid mask 全 false，因此 history commit 不产生 K/V；只 commit `O0` anchor 和空 anchor action，再生成 frame 5。

如果未来支持多 chunk，应循环 `video(frame) -> commit video -> action(frame) -> commit action`，不能一次生成所有 video 后再生成所有 action。v1 不开放此参数。

## 8. Euler inference

video 和 action 使用各自 scheduler 与步数：

```python
# 阶段 1/3：x_sigma 从标准高斯开始
sample = torch.randn(target_shape, generator=request_generator)
scheduler.set_timesteps(num_steps)

# 阶段 2/3：直接调用原生 AR MOT；返回 flow=noise-x0，不是 x0
for timestep in scheduler.timesteps:
    flow = model.predict_video(  # action 分支使用 model.predict_action
        sample,
        timestep=timestep,
        frame_ids=[5],
        cache=cache,
        ...,
    )
    sample = scheduler.step(flow, timestep, sample)

# 阶段 3/3：最后一个 Euler step 到 sigma=0，sample 即 clean prediction
return sample
```

每个 denoising step 使用 cache transaction：当前 noisy K/V 临时 append，attention 完成后 discard；只有最终 clean video 被 commit 给 action。随机数由 `seed + request_id` 构造 request-local `torch.Generator`，幂等重试不得重新采样。

默认配置建议从 checkpoint/config 读取，CLI 只允许覆盖：

- `video_num_steps`
- `action_num_steps`
- `device`
- `return_video`

不要在 server 中再实现一份 sigma/timestep 映射。

## 9. Action 后处理

模型输出为 `[1,20,1,16,1]`：

```python
# 阶段 1/3：[1,20,1,16,1] -> [16,20]
normalized = prediction[0, :, 0, :, 0].transpose(0, 1)

# 阶段 2/3：task-specific quantile denormalization
relative = (normalized + 1.0) * 0.5 * (q99 - q01 + 1e-6) + q01

# 阶段 3/3：16 个 action 都以当前 anchor state 为 reference
references = np.broadcast_to(anchor_state, (16, 16))
absolute_model_order = relative_20d_to_absolute_actions(references, relative)
```

client 收到后保持第 4.4 节的 raw RoboTwin 顺序，并逐个以 `action_type="ee"` 执行。server buffer 在下一次请求中记录 client 实际执行的 action，而不是默认认为上一次预测全部执行成功。

## 10. Server 生命周期和错误语义

`server.py` 启动顺序：

1. 读取 checkpoint metadata，要求 `model_architecture=autoregressive_va_mot_v1`；
2. 直接用 `AutoregressiveVAMOTTransformer3DModel.from_pretrained()` 加载 transformer；不创建 `WanDiffusionWrapper`；
3. 独立构造 video/action `FlowMatchScheduler`，分别使用 checkpoint/config 中的 `snr_shift`；
4. 加载 VAE、tokenizer/text encoder、dataset `mot_config.json`；
5. 缓存 task norm stats，模型设为 eval/no-grad；
6. 用固定小输入执行一次 warm-up；
7. 标记 ready。

`GET /healthz`：

- ready 前返回 503；
- ready 后返回 200，并包含 checkpoint、architecture、device、dtype；
- 不创建 session，不修改 RNG/KV cache。

错误码：

- 400：image/state/action shape 或 step 序列错误；
- 404：未知 session；
- 409：request id 乱序、另一个 episode 正在占用单 session server；
- 422：未知 task、缺 camera、instruction 无法编码；
- 503：模型尚未 ready。

推理异常不应自动 append 一半 buffer；先验证完整请求，在临时副本构建窗口，成功后再提交 episode state。

## 11. RoboTwin client 行为

episode 开始：

```python
POST /v1/reset
observation = TASK_ENV.get_obs()
pending_observations = [encode_observation(observation, step=0)]
pending_actions = []
```

每次 replan：

```python
response = POST /v1/actions(
    observations=pending_observations,
    executed_actions=pending_actions,
)
pending_observations = []
pending_actions = []

for action in response["actions"]:
    TASK_ENV.take_action(action, action_type="ee")
    pending_actions.append(action)
    pending_observations.append(
        encode_observation(TASK_ENV.get_obs(), step=next_step)
    )
    if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
        break
```

若 episode 在 16 个 action 中提前结束，不再请求下一 chunk；未上报的尾部轨迹不影响该 episode。网络失败可用相同 `request_id` 重试。

client 每完成一个 episode 写 progress JSON，至少包含 `task_name`、`task_config`、`completed_episodes`、`success_count` 和 `next_seed`；最终结果写到 task output 目录的 `_result_clean.txt` 或 `_result_random.txt`。

## 12. 验证顺序与验收标准

### 12.1 纯数据对齐

从 prepared RoboTwin parquet 选一个 episode，在多个 `current_frame` 上比较：

- online builder 和 `MotTrainData.get_window()` 的 history frame ids；
- raw/latent video valid mask；
- packed relative action、action valid mask、reference state；
- task-specific normalization 后 tensor。

要求逐元素一致；这是实现前最重要的测试。

### 12.2 模型级测试

- empty history 只向 cache 写 anchor video/action；
- `t=16/32/48` 的 committed token 数与第 6.1 节一致；
- 每个 Euler step 后没有残留 transaction；
- video/action 输出 shape 分别保持 `[1,C,1,V,H,W]` 和 `[1,20,1,16,1]`；
- action round-trip `absolute -> relative -> normalized -> denormalized -> absolute` 在容差内一致；
- 重复相同 `request_id` 返回相同 action，buffer 长度不变。

### 12.3 服务 smoke test

1. `GET /healthz` 返回 200；
2. reset 后发送一个真实 RoboTwin observation；
3. 返回恰好 16 个有限的 absolute EEF action；
4. 执行一轮后发送 16 observations/actions；
5. server 日志显示 `current_step=16`，只有最后一个 history latent 有效；
6. episode reset 后 buffer、cache、request id 全部归零。

### 12.4 评估 smoke test

先运行：

```text
1 task × 1 episode × 1 client × 1 server
```

确认 result/progress 文件和重复运行 resume 后，再扩到 clean/random、多任务和多 GPU shard。每个 GPU 启一个独立 server/port，不在单 server 内并发多个 episode。

## 13. 实现顺序

1. 抽取 dataset/streaming VAE 共享 helper，并完成离线逐元素对齐测试；
2. 实现 `OnlineMOTWindowBuilder` 和 action decode；
3. 基于原生 AR MOT `predict_*` 实现 Euler，并测试 KV cache 状态；
4. 实现单 session HTTP server、health/reset/actions；
5. 实现 RoboTwin observation adapter 和 16-action execution loop；
6. 完成单任务单 episode smoke test；
7. 最后再做多 GPU shard、resume 和可选 predicted-video artifact。
