# Cached RoboTwin 推理与训练对齐审计

本文记录 `inference/eval/pipeline.py`、`inference/mot_chunk_infer.py`、MOT 训练数据和
`distillation/eval/robotwin_client.py` 的逐 token 对齐审计。分析针对
`model_architecture=va_mot_v1`、8 latent frame、3 camera、48 action 的当前配置。

## 1. 结论

1. 当前 cache 顺序在 token 可见性上可以严格分解固定窗口 mask，不应仅因为使用
   `AutoregressiveVAMOTTransformer3DModel` 或 `KVCache` 就替换掉 cache 结构。
2. 已确认的严重不匹配是在线文本 embedding 没有复现训练缓存的零 padding。模型的
   cross-attention 没有 text mask，导致所有 video/action token 看见数百个训练时为零、
   在线却非零的 padding token。该错误已在 commit `671a303` 修复。
3. 在线只解码 4 个 target latent，与离线解码完整 8 个 latent 数值上有差异，但用一份
   已知正常的 `pred_latents.pt` 验证后，两种解码都能产生合理图像，因此它不是纯噪声视频的
   主要原因。
4. `robotwin_client.py` 的 camera 顺序、EEF state 顺序以及 `action[t] -> observation[t+1]`
   请求时序与训练数据一致。发现 instruction 选择原来不受 episode seed 控制，已改为确定性
   选择。
5. RoboTwin 原始 quaternion 是 `wxyz`，仓库的 relative-action 函数名称和数学实现写成了
   `xyzw`。当前 checkpoint、norm stats、history 编码和 action 反变换全都沿用了这个约定，
   不能只修改在线 client；正确迁移需要重建数据、统计量并重新训练。

## 2. 训练窗口

训练数据由 `MotTrainData._getitem_window()` 构造。当前 spec 为：

```text
latent frame       0    1    2    3  |  4       5       6       7
video              history           |  anchor  future  future  future
video loss         0    0    0    0  |  0       1       1       1
action valid       0    1    1    1  |  0       1       1       1
action loss        0    0    0    0  |  0       1       1       1
```

原始视频 frame 为：

```text
history = current - 49, current - 45, ..., current - 1
target  = current, current + 4, current + 8, current + 12
```

一个 action latent frame 包含 16 个 action token。48 个 history action 放在 latent frame
1～3，48 个 target action 放在 latent frame 5～7。每个 chunk 内的 action 都相对该 chunk
起点 state 转成 20D relative representation。

在线 `OnlineMOTWindowBuilder` 使用相同的 history frame ID、48-action reference 和
frame/token packing。early episode 的缺失 history 使用最早 observation padding，并通过
`video_valid` 和 `action_valid` 排除。

## 3. 训练 mask

模型为每个样本建立四组 token：

```text
NV = noisy video
CV = clean video
NA = noisy action
CA = clean action
```

当前两个 chunk 的 order 为：

```text
video history   order 0
action history  order 1
video target    order 2
action target   order 3
```

可见规则为：

```text
clean query -> clean key: key_order <= query_order
noisy query -> clean key: key_order <  query_order
noisy query -> noisy key: key_order == query_order
```

使用项目真实的 `build_dense_mot_mask()`，并将每个 frame 简化为一个 token 后，目标 token
的精确可见集合如下。

### 3.1 视频目标

```text
NV5 -> NV4 NV5 NV6 NV7
       CV0 CV1 CV2 CV3
       CA1 CA2 CA3
```

也就是说，生成视频 token 看见整个 noisy target video chunk、clean history video 和
clean history action；看不见 noisy history，也看不见同 order 的 clean target video。

### 3.2 动作目标

```text
NA5 -> CV0 CV1 CV2 CV3 CV4 CV5 CV6 CV7
       NA5 NA6 NA7
       CA1 CA2 CA3
```

也就是说，生成 action token 看见完整 clean video、整个 noisy target action chunk 和
clean history action。

## 4. cache factorization 模拟

当前在线顺序是：

```text
commit_video(history)           -> CV_H
commit_action(history)          -> CA_H
predict_video(frame 4..7)       -> NV_T
commit_video(predicted 4..7)    -> CV_T
predict_action(frame 5..7)      -> NA_T
```

因此在线目标 token 的 key 集合是：

```text
predict video:  NV_T <- CV_H + CA_H + NV_T
predict action: NA_T <- CV_H + CA_H + CV_T + NA_T
```

这与第 3 节的固定窗口目标 token 可见集合一致。invalid token 在固定窗口中由 mask 排除，
在 cache 路径中由 `_compact_stream()` 物理移除，对有效目标 query 等价。

### 4.1 单层数值模拟

新增测试 `inference/tests/test_cached_mask_parity.py`：

1. 创建同一个随机 `AutoregressiveVAMOTBlock`；
2. 一次执行完整 `NV/CV/NA/CA + MOTMaskMetadata`；
3. 再按当前 cache 顺序逐段执行；
4. 比较 `NV4:8` 和 `NA5:8` 输出。

实测：

```text
video target max abs diff  = 2.384185791015625e-07
action target max abs diff = 3.5762786865234375e-07
```

误差为 float32 运算舍入量。测试还验证了 cache 的绝对 frame ID RoPE 与固定窗口
`0..7` video/action position 完全一致。由于每一层的依赖关系相同，这个结果可以按层归纳
到多层网络：只要 commit/predict 顺序不变，cache 不会引入额外 token 可见性。

## 5. 已确认错误：在线文本 padding

训练文本缓存会计算有效长度，然后显式将 512 序列的尾部填零：

```python
seq_len = attention_mask.sum()
valid = encoder_output[:seq_len]
text_emb = concat(valid, zeros(512 - seq_len))
```

旧在线代码直接返回 text encoder 的全部 512 个 hidden state。text encoder 的 attention
mask 只控制 encoder 内部注意力，不保证 padding 位置的最终 hidden state 为零。

模型的 video/action cross-attention 没有再传 text mask，因此每个 video/action token 会
看见全部 512 个 text K/V。对一个 14-token 的 bell prompt，旧在线 embedding 实测：

```text
padding max abs  = 1.015625
padding mean abs = 0.05810546875
```

训练缓存对应位置严格为 0。conditional branch 的错误随后还被 CFG=5 放大。这会同时污染
每一层、每个 video token 和 action token，是旧在线视频变成噪声的首要代码原因。

修复后，在线 `TextEmbedder` 与同 prompt 的训练缓存实测：

```text
shape        = (512, 4096)
max abs diff = 0.0
tail nonzero = 0
```

## 6. VAE 编解码对齐

### 6.1 编码

离线 history 和在线 history 都按每个 view 独立创建 streaming wrapper：先 encode 1 帧，
再依次 encode 3 组 4 帧，得到 4 个 history latent。在线 anchor 单帧 encode 与离线 target
13 帧 encode 的第一个 anchor latent 相同。RGB 都归一化到 `[0,1]`，VAE 输入都转到
`[-1,1]`，latent mean/std 相同。

在线额外把 RoboTwin camera resize 到训练分辨率 `480x640`。camera 顺序和 stream ID 为：

```text
cam_high         -> stream 1
cam_left_wrist   -> stream 0
cam_right_wrist  -> stream 2
```

与训练 manifest 一致。

### 6.2 解码

离线 `video_pred.mp4` 解码全部 8 latent，得到 29 raw frame；在线只解码
`anchor + 3 generated latent`，得到 13 raw frame，并丢弃 anchor raw frame，保存 12 帧。

使用已知合理的离线 `pred_latents.pt` 对比：

```text
full-8 decode target shape = (3 views, 3 channels, 12 frames, 480, 640)
target-4 decode shape      = (3 views, 3 channels, 12 frames, 480, 640)
mean absolute difference   = 0.0929683
```

两者有边界差异，但目视都能保持合理场景和机器人结构。因此 target-only decode 会改变颜色/
亮度和局部细节，但不会单独把正常 latent 变成旧日志中的纯噪声。

## 7. RoboTwin client 审计

### 7.1 正确部分

- camera map 与训练数据一致；
- state 为 `[left xyz+quat, left gripper, right xyz+quat, right gripper]`，共 16D；
- server 返回的是 absolute EEF action，client 使用 `action_type="ee"` 直接执行；
- 第一次请求发送 `observation[0]` 和空 action；
- 执行 `action[t]` 后获取 `observation[t+1]`，下一请求把二者成对发送；
- 中途只执行返回 action 的前 N 个是合法 receding-horizon 行为；
- response retry 使用同一个 request ID，server 侧幂等缓存能够避免重复推进 builder；
- predicted video 的三个 view 按 payload 顺序横向拼接，fps 和 frame 数正确。

训练 parquet 的一个完整 episode 实测 `state[t+1]` 与 `action[t]` 逐元素完全相等：
`mean abs = 0`、`max abs = 0`。这证明 client 的请求时序和 absolute EEF 解释正确。新增
client 测试覆盖 camera/state 排列，以及两次请求之间 action/observation 的对应。

### 7.2 已修复部分

旧 `select_instruction()` 使用全局 `np.random.choice()`，相同 episode seed 在重试或恢复后
可能选择不同 instruction，进而改变 text embedding 和模型结果。现在改用：

```python
np.random.default_rng(seed).choice(choices)
```

### 7.3 quaternion 约定

RoboTwin 使用 transforms3d/SAPIEN 的 `wxyz`。训练 parquet 中的 quaternion 也符合
`wxyz`，例如常见初始姿态约为 `[0.70, 0, 0, 0.714]`。但是仓库函数名和实现为
`quaternion_xyzw_to_matrix()`。

这属于数据合同的历史问题，但当前链路是自洽的：训练 target、norm stats、在线 history 和
在线 inverse 都使用同一转换。单独把 client 或 inverse 改成 `wxyz` 会立即破坏现有 checkpoint
输出语义。应在下一版数据格式中显式增加 quaternion order，并同时重建数据和重训。

## 8. 与示例视频比较时的非等价输入

给出的合理离线视频实际是 `move_playingcard_away`、start frame 66；坏的在线视频是
`click_bell` 的 request 4。两者任务、图像、instruction、history action 和噪声 seed 均不同，
不能作为严格 A/B。离线使用缓存文本和数据集 MP4，在线使用实时 RoboTwin RGB 经 JPEG 90
传输，并默认使用 unseen instruction。

这些域差异会影响质量，但旧文本 padding 是已经量化并确认的代码错误。

## 9. 修改和验证

代码修改：

- `distillation/eval/infer_pipeline.py`：在线文本 embedding 尾部按训练逻辑零填充；
- `distillation/eval/robotwin_client.py`：instruction 选择绑定 episode seed；
- `inference/tests/test_cached_mask_parity.py`：固定窗口/cache 单层数值等价和 RoPE 等价；
- `distillation/tests/test_robotwin_client.py`：camera/state 排列、确定性 instruction、请求时序；
- `inference/tests/test_bidirectional_eval_pipeline.py`：保留并验证整 target chunk cache 调用顺序。

已执行的针对性测试：

```text
15 passed
```

剩余的最终验证是用 commit `671a303` 之后的代码重新运行一个 `click_bell` episode。旧的
`20260817_162817` 视频是在文本 padding 修复前生成，不能用于判断修复后的 cache 推理质量。
