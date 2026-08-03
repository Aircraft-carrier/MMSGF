
# wan_va/dataset/mot_dataset.py：一条 MOT 数据的完整加载链路

本文以当前仓库中的 wan_va/dataset/mot_dataset.py 为准，详细说明 MOT 数据集如何从 JSONL manifest 加载一条样本，并最终组织成训练/推理阶段使用的 sample dictionary。

重点包括：

- manifest 中一行数据如何变成一个可采样的 episode segment；
- 当前窗口如何产生 history/target 视频帧；
- 为什么一条样本同时有 26 个原始 RGB 帧、8 个 latent 时间位置和 32 个 geometry slot；
- RGB、点云、action、文本 embedding 如何分别读取并对齐；
- padding、valid mask、loss mask、condition mask 的区别；
- video decoder、PointStore、action mmap cache 如何参与加载；
- 一条具体样本的完整 shape 和数值模拟。

> 说明：本文中的路径、帧号、图像分辨率、embedding 数值是为了展示流程而构造的示例，不代表某个真实 episode。shape 和函数调用关系按照代码实现说明。

---

## 1. 总体结论

MotTrainData 返回的不是单独的一段视频，而是一个已经把以下信息对齐好的多模态样本：

~~~text
视频 RGB      ─┐
geometry RGB ─┼─> 8 个统一的 latent 时间位置
3D 点云      ─┘
action       ───> 每个 latent 时间位置下的 16 个 action token
text         ───> 条件 embedding
stream_ids   ──> view/相机语义标识
mask         ───> 哪些位置是真数据、做条件还是算 loss
~~~

默认参数为：

~~~text
action_chunk_size       = 48
video_downsample_ratio  = 4
vae_temporal_factor     = 4
geometry_group_size     = 4
history_chunks          = 1
target_chunks           = 1
~~~

由此得到：

~~~text
每个 action chunk 的原始视频采样帧数
    = action_chunk_size / video_downsample_ratio + 1
    = 48 / 4 + 1
    = 13

每个 action chunk 对应的 VAE latent 帧数
    = (13 - 1) / 4 + 1
    = 4

history + target 总 latent 帧数
    = 4 + 4
    = 8

每个 latent frame 的 action token 数
    = video_downsample_ratio * vae_temporal_factor
    = 4 * 4
    = 16

action 维度
    = 双臂 × (位置 3 + 旋转 6D + gripper 1)
    = 2 × 10
    = 20
~~~

一个 joint sample 在未 batch 时通常具有：

~~~text
vae_rgb_history:           [13, V, 3, H, W]
vae_rgb_target:            [13, V, 3, H, W]

geometry_rgb:              [8, 4, V, 3, H, W]
geometry_pts3d:            [8, 4, V, H, W, 3]
geometry_point_valid_mask: [8, 4, V, H, W]
geometry_group_valid_mask: [8, 4]

actions:                   [20, 8, 16, 1]
action_loss_mask:          [20, 8, 16, 1]
action_valid_mask:         [20, 8, 16, 1]
action_reference_states:   [16, 8, 16, 1]

video_latent_valid_mask:   [8]
video_latent_loss_mask:    [8]

stream_ids:                [V]
text_emb:                  [text_seq_len, text_dim]
~~~

DataLoader collate 后通常在最前面增加 batch 维：

~~~text
geometry_rgb: [B, 8, 4, V, 3, H, W]
actions:      [B, 20, 8, 16, 1]
~~~

mot_dataset.py 本身返回的是原始 RGB 帧；Wan VAE latent 通常由后续 trainer 或 forward pipeline 对 vae_rgb_history 和 vae_rgb_target 编码得到。

---

## 2. 数据集初始化：MotTrainData.__init__

### 2.1 选择数据 profile

MotTrainData 支持两个 profile：

~~~python
data_profile = "joint"
data_profile = "geometry"
~~~

| profile | 读取内容 | 典型用途 |
|---|---|---|
| joint | RGB、geometry、action、text | 联合训练/联合推理 |
| geometry | RGB、geometry、点云和 mask | geometry 分支预计算或几何训练 |

如果 profile 不是这两个值，初始化直接报错。

MotGeometryLeRobotData 和 MotPureLeRobotData 是在 MotTrainData 上进一步约束数据类型的包装/子类：

- geometry 数据要求 row 有点云；
- pure LeRobot 数据要求 row 没有点云；
- 即使没有点云，joint schema 仍会返回全 0 的 point tensor 和全 False 的 point mask，保证 batch 字段一致。

### 2.2 创建三个运行时 LRU cache

初始化时会创建有界 LRU cache：

~~~text
_video_decoder_cache
    key: video_path
    value: torchcodec.VideoDecoder

_point_store_cache
    key: preprocessed_pointcloud_dir
    value: PointStore

_action_cache
    key: data_file
    value: (actions, states, indices)
~~~

action cache 只在 joint profile 创建。

cache 的行为：

1. get(key) 命中时增加 hits，并把 key 移到 LRU 队尾；
2. 未命中时增加 misses；
3. put 超过最大容量时淘汰最旧项；
4. PointStore 被淘汰时调用 store.close()；
5. action mmap 被淘汰时关闭 mmap；
6. 每条 sample 将 cache 统计编码到 _runtime_cache_stats，便于监控 DataLoader worker 的命中率。

cache 不改变数据内容，主要是避免每条样本重复打开相同的视频、点云 store 或 action mmap。

### 2.3 读取 JSONL manifest

代码调用：

~~~python
self.rows = load_manifest(self.manifest_path)
~~~

load_manifest 的实现是：

1. 将 manifest 路径转为 Path；
2. 逐行读取 JSONL；
3. 跳过空行；
4. 每一行解析成 Python dict；
5. 所有 row 放入 self.rows。

如果 manifest 为空，初始化报错。

一行 row 至少描述：

~~~json
{
  "task_uid": "task_001",
  "norm_stats_key": "task_001",
  "data_file": "/data/episode_000.parquet",
  "dataset_from_index": 1000,
  "dataset_to_index": 1300,
  "timestamp_policy": "episode_local_frame_over_fps_v1",
  "fps": 30.0,
  "has_pointcloud": true,
  "segment": {
    "start_frame": 0,
    "end_frame": 300,
    "action_text": "pick up the red cup"
  },
  "valid_start_range": [64, 220],
  "views": [
    {
      "video_path": "/data/cam_left.mp4",
      "video_from_timestamp": 10.0,
      "stream_id": 0,
      "preprocessed_pointcloud_dir": "/data/pc_left"
    },
    {
      "video_path": "/data/cam_right.mp4",
      "video_from_timestamp": 10.0,
      "stream_id": 2,
      "preprocessed_pointcloud_dir": "/data/pc_right"
    }
  ]
}
~~~

字段含义：

- data_file：LeRobot/Parquet action-state 数据文件；
- dataset_from_index：segment 在全局数据文件中的绝对起点；
- segment.start_frame/end_frame：segment 内的局部帧范围，end_frame 是开区间；
- dataset_to_index：应满足 dataset_from_index + segment.end_frame；
- fps：episode 的 manifest FPS；
- timestamp_policy：当前实现要求为 episode_local_frame_over_fps_v1；
- views：相机/视频视角列表，支持 2 或 3 个原生 view；
- video_from_timestamp：episode 在实际视频文件中的起始时间；
- stream_id：view 的语义 ID，例如左腕、头部、右腕；
- preprocessed_pointcloud_dir：对应 view 的预处理点云目录。

### 2.4 按原生 view 数分桶

初始化按 len(row["views"]) 建立 view_buckets。

当前支持：

~~~text
V = 2
V = 3
~~~

原因是 view 数会直接影响 RGB、geometry 和 stream_ids 的 shape。将 V=2 和 V=3 混入同一采样路径，容易造成 batch shape 不一致。

同时记录 row 到 bucket 的位置：

~~~python
_row_bucket_positions[row_index] = (view_count, position_in_bucket)
~~~

后续 __getitem__ 根据 requested index 找到同 view 数的 bucket。

### 2.5 校验全局 action index 边界

validate_dataset_index_bounds 检查：

~~~text
dataset_to_index
    ==
dataset_from_index + segment.end_frame
~~~

例如：

~~~text
dataset_from_index = 1000
segment.end_frame   = 300
dataset_to_index    = 1300
~~~

不相等说明 manifest 的 segment 元数据和 Parquet/action 索引的全局范围不一致，初始化直接失败。

### 2.6 推导时间轴

默认推导结果：

~~~text
sampled_video_frames_per_action_chunk_per_view = 13
latent_frames_per_action_chunk_per_view        = 4
action_per_frame                               = 16
total_latent_frames                            = 8
geometry_groups                                = 8
~~~

per view 指的是时间长度，不是 view 数 V。相机维度在 RGB 和 point tensor 中另行保留。

---

## 3. __getitem__：一条样本从哪里开始

入口：

~~~python
sample = dataset[idx]
~~~

实际流程：

~~~text
requested idx
    ↓
映射到固定 view 数 bucket
    ↓
从 bucket 中选择 row
    ↓
根据 profile 调用 _getitem_window 或 _getitem_geometry_window
    ↓
失败时尝试同 view bucket 的下一条 row
    ↓
成功 sample 添加 has_pointcloud、dataset_skip_count、runtime cache stats
    ↓
返回 dict
~~~

### 3.1 同 bucket fallback

__getitem__ 最多尝试：

~~~python
attempts = min(len(bucket), MOT_DATASET_MAX_SAMPLE_ATTEMPTS)
~~~

如果样本出现视频文件不存在、PointStore row 缺失、FPS 不一致、action index 不连续等异常，就打印异常并继续尝试下一条。

fallback 只在相同 view 数 bucket 内进行，所以不会把 V=2 替换成 V=3。

---

## 4. 选择一个窗口：history 和 target raw frame ids

_getitem_window 首先从 row 的 valid_start_range 选择 current_frame：

- random_start=True 时通常随机选择；
- 否则使用确定性的起点策略；
- current_frame 是 segment 内的 local frame id。

然后调用：

~~~python
mot_real_window_frame_ids(
    current_frame=current_frame,
    video_downsample_ratio=4,
    action_chunk_size=48,
)
~~~

核心公式：

~~~python
history_ids = [
    current_frame - action_chunk_size - 1 + idx * stride
    for idx in range(13)
]

target_ids = [
    current_frame + idx * stride
    for idx in range(13)
]
~~~

其中 stride=4。

### 4.1 具体例子：current_frame=100

history 起点：

~~~text
100 - 48 - 1 = 51
~~~

history：

~~~text
[51, 55, 59, 63, 67, 71, 75, 79, 83, 87, 91, 95, 99]
~~~

target：

~~~text
[100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 148]
~~~

总 raw video frame 数：

~~~text
13 + 13 = 26
~~~

这些是 episode 内的 local frame id，不是实际 mp4 的 global frame index。读取视频时还要通过 fps 和 video_from_timestamp 转换。

---

## 5. 26 个 raw frame 如何变成 8 个 geometry group

### 5.1 每个 group 固定 4 个 slot

_group_mot_frames 把每个 action chunk 的 13 个采样帧映射到 4 个 latent 时间位置：

~~~text
13 raw frames
    = 1 + 4 + 4 + 4
    = 4 latent positions
~~~

VAE 的时间结构：

- 第一个 latent 时间位置对应一个单独 anchor frame；
- 后续每个 latent 时间位置对应 4 个 raw frame；
- 第一个 group 为统一 shape，复制第一个 frame 填满 4 个 slot；
- 复制 slot 用 False mask 标识为 padding。

### 5.2 具体 geometry groups

history：

~~~text
group 0 = [51, 51, 51, 51]
mask     = [ True, False, False, False]

group 1 = [55, 59, 63, 67]
mask     = [ True,  True,  True,  True]

group 2 = [71, 75, 79, 83]
mask     = [ True,  True,  True,  True]

group 3 = [87, 91, 95, 99]
mask     = [ True,  True,  True,  True]
~~~

target：

~~~text
group 4 = [100, 100, 100, 100]
mask     = [ True, False, False, False]

group 5 = [104, 108, 112, 116]
mask     = [ True,  True,  True,  True]

group 6 = [120, 124, 128, 132]
mask     = [ True,  True,  True,  True]

group 7 = [136, 140, 144, 148]
mask     = [ True,  True,  True, True]
~~~

最终：

~~~text
geometry_frame_ids.shape = [32]
geometry_group_mask.shape = [8,4]
~~~

geometry group 顺序与视频 latent 时间顺序一致：

~~~text
geometry group 0 ↔ latent 0
geometry group 1 ↔ latent 1
...
geometry group 7 ↔ latent 7
~~~

---

## 6. segment 越界：padded id 与 valid mask

窗口可能靠近 segment 边界。合法范围是：

~~~text
[first_valid, last_valid]
=
[segment.start_frame, segment.end_frame - 1]
~~~

_pad_frame_ids_to_segment 对每个请求帧做两件事。

### 6.1 生成可读取的 padded frame id

~~~text
请求 frame < first_valid
    → 用 first_valid 替换

请求 frame > last_valid
    → 用 last_valid 替换
~~~

### 6.2 保留原始请求是否有效

~~~text
valid = first_valid <= requested_frame <= last_valid
~~~

例如 segment 为 [0,120)：

~~~text
requested = [-5, -1, 3, 7, ..., 119, 123]
padded    = [ 0,  0, 3, 7, ..., 119, 119]
valid     = [ F,  F, T, T, ..., T,   F]
~~~

两套信息的用途：

| 信息 | 用途 |
|---|---|
| padded frame id | 保证 decoder/PointStore 能读取合法位置 |
| valid mask | 防止复制出来的边界数据被当成真实 supervision |

### 6.3 raw mask 到 latent mask

- latent 0 直接看第一个 raw frame；
- latent 1/2/3 各看对应的 4 个 raw frame；
- 4 个 raw frame 只要有一个真实有效，该 latent 就有效；
- 只有整个 group 都是 padding，latent 才是 False。

因此当前代码的语义是“只要该 latent group 中有真实帧就 valid”，不是“group 中所有 raw frame 都必须 valid”。

完整窗口内部时：

~~~text
video_latent_valid_mask = [T,T,T,T,T,T,T,T]
~~~

### 6.4 video loss mask

代码使用：

~~~python
target_start = latent_frames_per_action_chunk_per_view + 1
~~~

默认 target_start=5，因此：

~~~text
latent 0：history anchor，不算 video loss
latent 1：history，不算 video loss
latent 2：history，不算 video loss
latent 3：history，不算 video loss
latent 4：target anchor，不算 video loss
latent 5：target diffusion frame，valid 才算 loss
latent 6：target diffusion frame，valid 才算 loss
latent 7：target diffusion frame，valid 才算 loss
~~~

典型值：

~~~text
video_latent_valid_mask = [T,T,T,T,T,T,T,T]
video_latent_loss_mask  = [F,F,F,F,F,T,T,T]
~~~

---

## 7. RGB 加载：local frame id 到 VideoDecoder

### 7.1 去重后解码，再恢复固定长度

_getitem_window 先执行：

~~~python
unique_padded_frame_ids = list(dict.fromkeys(padded_frame_ids))
~~~

如果边界 padding 复制了同一个首帧或末帧，就只解码一次。

然后构造 frame id 到 decoded position 的映射，再用 index_select 恢复完整序列：

~~~text
实际解码成本：unique frame 数
模型输入长度：固定 26 帧
~~~

### 7.2 timestamp 转换

_load_rgb 对每个 view 执行：

1. 读取 manifest fps；
2. 要求 timestamp_policy 为 episode_local_frame_over_fps_v1；
3. 计算 local timestamp：

~~~text
local_timestamp = local_frame_id / manifest_fps
~~~

4. 加上视频 offset：

~~~text
query_timestamp = video_from_timestamp + local_timestamp
~~~

5. 转换为 decoder 的全局 frame index：

~~~text
global_video_frame_index =
    round(query_timestamp * video_fps)
~~~

6. 调用 VideoDecoder.get_frames_at(global_ids)；
7. 归一化：

~~~python
frames.float() / 255.0
~~~

8. 多 view stack。

单 view：

~~~text
[F, 3, H, W]
~~~

多 view：

~~~text
[F, V, 3, H, W]
~~~

### 7.3 FPS 与 PTS 校验

代码检查：

- manifest FPS 与视频 average_fps 差异不能超过 1e-4；
- decoder 返回的 pts_seconds 与请求 timestamp 误差不能达到 1e-4 秒。

如果 manifest fps、video offset 或视频时间轴错误，样本加载失败，__getitem__ 会尝试同 view bucket 的下一条 row。

### 7.4 数值例子

假设：

~~~text
manifest fps = 30
video_from_timestamp = 10.0
local frame id = 51
video fps = 30
~~~

则：

~~~text
local_timestamp = 51 / 30 = 1.7 秒
query_timestamp = 10.0 + 1.7 = 11.7 秒
global frame id = round(11.7 * 30) = 351
~~~

假设某个 uint8 RGB 像素为：

~~~text
[128, 64, 255]
~~~

归一化后为：

~~~text
[0.5019608, 0.2509804, 1.0]
~~~

若 V=2、H=480、W=640：

~~~text
decoded_rgb                [26, 2, 3, 480, 640]
vae_rgb_history            [13, 2, 3, 480, 640]
vae_rgb_target             [13, 2, 3, 480, 640]
~~~

---

## 8. 点云加载：PointStore 到 geometry label

### 8.1 有点云 row

当 row["has_pointcloud"] 为 True 时，_load_points 对每个 local frame、每个 view：

1. 根据 preprocessed_pointcloud_dir 获取 PointStore；
2. 调用 store.row_for_episode_frame(frame_id) 找到对应行；
3. 读取 store.points[row_idx] 和 store.valid_mask[row_idx]；
4. 转为 float32 point tensor 和 bool mask；
5. 先 stack view，再 stack frame。

结果：

~~~text
points: [F, V, H, W, 3]
mask:   [F, V, H, W]
~~~

PointStore 通过 _point_store_cache 复用；LRU 淘汰时调用 close()。

### 8.2 无点云 row

当 row["has_pointcloud"] 为 False 时，代码不会返回 None，而是构造：

~~~text
points = zeros([F,V,H,W,3])
mask   = zeros([F,V,H,W], dtype=bool)
~~~

这样 pointcloud 和 non-pointcloud 样本仍然拥有相同字段。无点云路径需要 target_h_w，通常使用 RGB 的 H、W。

---

## 9. geometry fields：从 32 个 slot 组成 [8,4,...]

_materialize_geometry_fields 接收：

~~~text
padded_geometry_frame_ids: [32]
geometry_flat_valid_mask:  [32]
geometry_group_mask:       [8,4]
decoded_rgb:               [unique_F,V,3,H,W]
decoded_position:          frame_id → decoded index
~~~

### 9.1 geometry RGB

把 32 个 geometry frame id 映射到已经解码的 RGB：

~~~text
geometry_rgb_flat.shape = [32, V, 3, H, W]
geometry_rgb.shape      = [8, 4, V, 3, H, W]
~~~

### 9.2 geometry point cloud

先得到：

~~~text
geometry_points_flat:
    [32, V, H, W, 3]

geometry_point_mask_flat:
    [32, V, H, W]
~~~

再整理为：

~~~text
geometry_pts3d:
    [8, 4, V, H, W, 3]

geometry_point_valid_mask:
    [8, 4, V, H, W]
~~~

### 9.3 合并 geometry mask

有效性由三部分共同决定：

1. group slot 是否是首帧复制 padding；
2. 请求 frame 是否越过 segment；
3. PointStore 对该像素是否有有效点。

代码逻辑：

~~~python
geometry_group_valid_mask =
    geometry_group_mask & geometry_flat_valid_mask.reshape(8, 4)

geometry_point_valid_mask =
    geometry_point_valid_mask
    & geometry_group_valid_mask[:, :, None, None, None]
~~~

因此 padding slot 即使复用了真实点云，也不会参与 geometry supervision 或 geometry token 交互。

---

## 10. action：mmap/Parquet 到相对 20D

### 10.1 读取 action/state

绝对索引范围：

~~~text
absolute_start =
    dataset_from_index + segment.start_frame

absolute_end =
    dataset_from_index + segment.end_frame
~~~

_load_action_state_index_arrays 的优先级：

~~~text
action cache manifest
    ↓
np.load(..., mmap_mode="r")
    ↓
(actions, states, indices)

没有 cache manifest
    ↓
Parquet columns:
action
observation.state
index
~~~

代码会筛选：

~~~text
absolute_start <= index < absolute_end
~~~

并排序、校验完整覆盖 [absolute_start, absolute_end)。

### 10.2 原始格式

双臂原始 action/state：

~~~text
左臂 = [x,y,z,qx,qy,qz,qw,gripper]  8D
右臂 = [x,y,z,qx,qy,qz,qw,gripper]  8D
总计 = 16D
~~~

### 10.3 相对 20D 转换

对每只手：

~~~text
delta_p = R_state^T × (p_action - p_state)
delta_R = R_state^T × R_action
~~~

将 delta_R 转为 6D rotation columns，并保留 gripper：

~~~text
位置 delta  3D
旋转 6D     6D
gripper     1D
每只手      10D
双臂        20D
~~~

模型接收的是相对于 reference state 的 relative_20d，而不是原始绝对 16D。

### 10.4 action packing

初始布局：

~~~text
aligned:           [8, 16, 20]
action_loss_mask:  [8, 16, 20]
condition_mask:    [8, 16, 20]
reference_states:  [8, 16, 16]
~~~

映射：

~~~python
latent_offset = 1 + action_offset // tokens_per_frame
token_idx = action_offset % tokens_per_frame
latent_idx = frame_offset + latent_offset
~~~

48 个 raw action 的 chunk 内布局：

~~~text
offset 0~15   → latent 1，token 0~15
offset 16~31  → latent 2，token 0~15
offset 32~47  → latent 3，token 0~15
~~~

history chunk 的 frame_offset=0，target chunk 的 frame_offset=4：

~~~text
latent 0：history anchor，空
latent 1：history action 0~15
latent 2：history action 16~31
latent 3：history action 32~47

latent 4：target anchor，空
latent 5：target action 0~15
latent 6：target action 16~31
latent 7：target action 32~47
~~~

### 10.5 history/target mask

history：

~~~python
history_start = local_current - 48
history_ref_idx = max(0, history_start)
~~~

- history action 放入 condition_mask；
- history action 不放入 action_loss_mask；
- history 超出 segment 起点的 action 被跳过。

target：

~~~python
action_start = local_current
ref_idx = local_current
~~~

- target action 放入 action_loss_mask；
- target 越过 segment 尾部时重复最后一个 action；
- 是否写入还受 video_latent_loss_mask 控制。

最终：

~~~python
action_valid_mask = action_loss_mask | condition_mask
~~~

返回布局：

~~~text
actions:                 [20, 8, 16, 1]
action_loss_mask:        [20, 8, 16, 1]
action_valid_mask:       [20, 8, 16, 1]
action_reference_states: [16, 8, 16, 1]
~~~

### 10.6 action normalization

从 norm_stats_by_task[row["norm_stats_key"]] 中取 q01 和 q99：

~~~python
normalized = (aligned - q01) / (q99 - q01 + 1e-6) * 2 - 1
normalized = clip(normalized, -1.5, 1.5)
normalized = normalized * (action_loss_mask | condition_mask)
~~~

无效位置最终为 0。

例如某一维原始归一化前值为 0.35，q01=-0.5、q99=0.5：

~~~text
normalized = (0.35 + 0.5) * 2 - 1
           = 0.7
~~~

action offset=17 时：

~~~text
latent_offset = 1 + 17 // 16 = 2
token_idx      = 17 % 16 = 1
~~~

如果是 history，则写入全局 latent=2；如果是 target，则加上 target frame_offset=4，写入全局 latent=6。

---

## 11. text embedding 和 stream_ids

### 11.1 text embedding

joint profile 初始化时整体加载 text embedding cache：

~~~python
torch.load(text_emb_cache_path, map_location="cpu", weights_only=False)
~~~

每条 row 使用：

~~~python
row["segment"]["action_text"]
~~~

进行字典查找。

例如：

~~~text
action_text = "pick up the red cup"
text_emb.shape = [77, 4096]
~~~

如果 prompt 不存在，代码抛出 KeyError，不会静默生成错误 embedding。可选的 empty_text_emb 也会作为 sample 字段返回。

### 11.2 stream_ids

_stream_id_for_view 的优先级：

~~~text
view["stream_id"]
    ↓
view["hand"] 映射
    ↓
view["video_key"] 映射
~~~

例如：

~~~text
left wrist  → 0
head         → 1
right wrist  → 2
~~~

2-view 样本可能返回：

~~~text
stream_ids = tensor([0, 2])
shape = [2]
~~~

---

## 12. 一条具体样本的完整模拟

### 12.1 manifest row

~~~python
row = {
    "task_uid": "task_001",
    "norm_stats_key": "task_001",
    "data_file": "/data/episode_000.parquet",
    "dataset_from_index": 1000,
    "dataset_to_index": 1300,
    "timestamp_policy": "episode_local_frame_over_fps_v1",
    "segment": {
        "start_frame": 0,
        "end_frame": 300,
        "action_text": "pick up the red cup",
    },
    "valid_start_range": [64, 220],
    "fps": 30.0,
    "has_pointcloud": True,
    "views": [
        {
            "video_path": "/data/cam_left.mp4",
            "video_from_timestamp": 10.0,
            "stream_id": 0,
            "preprocessed_pointcloud_dir": "/data/pc_left",
        },
        {
            "video_path": "/data/cam_right.mp4",
            "video_from_timestamp": 10.0,
            "stream_id": 2,
            "preprocessed_pointcloud_dir": "/data/pc_right",
        }
    ]
}
~~~

### 12.2 选择 current frame

假设随机采样：

~~~text
current_frame = 100
~~~

它处于 valid_start_range=[64,220] 内，且 segment 是 [0,300)，因此本例不需要边界 padding。

### 12.3 raw frame 与 mask

~~~text
history:
[51,55,59,63,67,71,75,79,83,87,91,95,99]

target:
[100,104,108,112,116,120,124,128,132,136,140,144,148]

raw_video_frame_count = 26
raw_video_valid_mask   = [T] * 26
~~~

~~~text
video_latent_valid_mask = [T,T,T,T,T,T,T,T]
video_latent_loss_mask  = [F,F,F,F,F,T,T,T]
~~~

### 12.4 geometry

~~~text
latent 0: [51,51,51,51], mask [T,F,F,F]
latent 1: [55,59,63,67], mask [T,T,T,T]
latent 2: [71,75,79,83], mask [T,T,T,T]
latent 3: [87,91,95,99], mask [T,T,T,T]

latent 4: [100,100,100,100], mask [T,F,F,F]
latent 5: [104,108,112,116], mask [T,T,T,T]
latent 6: [120,124,128,132], mask [T,T,T,T]
latent 7: [136,140,144,148], mask [T,T,T,T]
~~~

### 12.5 RGB

假设：

~~~text
V=2
H=480
W=640
manifest fps=30
video_from_timestamp=10.0
~~~

local frame 51：

~~~text
local_timestamp = 51/30 = 1.7 s
query_timestamp = 10.0 + 1.7 = 11.7 s
global video frame = round(11.7*30) = 351
~~~

local frame 100：

~~~text
local_timestamp = 100/30 = 3.333333... s
query_timestamp = 13.333333... s
global video frame = round(13.333333...*30) = 400
~~~

输出：

~~~text
decoded_rgb:
    [26, 2, 3, 480, 640]

vae_rgb_history:
    [13, 2, 3, 480, 640]

vae_rgb_target:
    [13, 2, 3, 480, 640]

geometry_rgb:
    [8, 4, 2, 3, 480, 640]
~~~

假设 decoder 返回 uint8 像素 [128,64,255]，归一化后是 [0.5019608,0.2509804,1.0]。

### 12.6 点云

假设 PointStore 也对应 480×640：

~~~text
geometry_pts3d:
    [8,4,2,480,640,3]

geometry_point_valid_mask:
    [8,4,2,480,640]
~~~

latent 0 的 slot 1/2/3 虽然复用了 frame 51 的数据，但因为 group mask 是 [T,F,F,F]，最终这些 slot 的 point valid 全部为 False。

### 12.7 action

当前 local frame=100：

~~~text
history_start = 100 - 48 = 52
history_ref_idx = 52

target_action_start = 100
target_ref_idx = 100
~~~

写入位置：

~~~text
latent 1, token 0..15：history action offset 0..15
latent 2, token 0..15：history action offset 16..31
latent 3, token 0..15：history action offset 32..47

latent 5, token 0..15：target action offset 0..15
latent 6, token 0..15：target action offset 16..31
latent 7, token 0..15：target action offset 32..47
~~~

最终：

~~~text
actions.shape = [20,8,16,1]
~~~

示例索引：

~~~text
actions[0, 1, 0, 0]
    = history action offset 0 的第 0 个 relative-20D feature

actions[0, 2, 1, 0]
    = history action offset 17 的第 0 个 feature

actions[0, 5, 0, 0]
    = target action offset 0 的第 0 个 feature
~~~

mask 语义：

~~~text
history latent 1..3:
    action_valid_mask = True
    action_loss_mask  = False

target latent 5..7:
    action_valid_mask = True
    action_loss_mask  = True
~~~

### 12.8 最终 sample dict

~~~python
{
    "vae_rgb_history":       Tensor[13, 2, 3, 480, 640],
    "vae_rgb_target":        Tensor[13, 2, 3, 480, 640],

    "geometry_rgb":          Tensor[8, 4, 2, 3, 480, 640],
    "geometry_pts3d":        Tensor[8, 4, 2, 480, 640, 3],
    "geometry_point_valid_mask":
                             BoolTensor[8, 4, 2, 480, 640],
    "geometry_group_valid_mask":
                             BoolTensor[8, 4],

    "stream_ids":             LongTensor[2],

    "video_latent_valid_mask": BoolTensor[8],
    "video_latent_loss_mask":  BoolTensor[8],

    "actions":                FloatTensor[20, 8, 16, 1],
    "action_loss_mask":       BoolTensor[20, 8, 16, 1],
    "action_valid_mask":       BoolTensor[20, 8, 16, 1],
    "action_reference_states": FloatTensor[16, 8, 16, 1],

    "action_q01":             FloatTensor[20],
    "action_q99":             FloatTensor[20],

    "text_emb":               FloatTensor[text_seq_len, text_dim],
    "empty_text_emb":         FloatTensor[text_seq_len, text_dim],

    "has_pointcloud":         BoolTensor[],
    "dataset_skip_count":     LongTensor[],
    "_runtime_cache_stats":   bytes,
}
~~~

---

## 13. geometry-only profile 的区别

geometry-only 主链路：

~~~text
row
  ↓
选择 current frame
  ↓
构造 history/target raw frame ids
  ↓
构造 8×4 geometry groups
  ↓
segment padding + geometry mask
  ↓
解码需要的 RGB
  ↓
读取 PointStore
  ↓
materialize geometry fields
  ↓
返回 geometry sample
~~~

它不会调用：

- _load_actions；
- _text_emb_for。

geometry-only sample 主要包括：

~~~python
{
    "geometry_rgb": [8,4,V,3,H,W],
    "geometry_pts3d": [8,4,V,H,W,3],
    "geometry_point_valid_mask": [8,4,V,H,W],
    "geometry_group_valid_mask": [8,4],
    "has_pointcloud": True,
    "dataset_skip_count": 0,
    "_runtime_cache_stats": bytes,
}
~~~

---

## 14. 多数据集混合

MotBalancedMixDataset 用于按权重混合多个 MOT dataset：

1. 输入多个子 dataset；
2. 按权重分配采样数量或虚拟长度；
3. 将全局 index 映射到某个子 dataset；
4. 调用对应子 dataset 的 __getitem__。

因此一条样本可能先经过 dataset-level mixing；一旦确定具体子 dataset，后续 window、RGB、point、action、text 流程不变。

---

## 15. 8 个 latent 时间位置的统一对齐

| latent idx | 视频含义 | geometry group | action 含义 | video loss | action loss |
|---:|---|---:|---|---|---|
| 0 | history anchor | 0 | 空 | 否 | 否 |
| 1 | history latent 1 | 1 | history token 0~15 | 否 | 否 |
| 2 | history latent 2 | 2 | history token 16~31 | 否 | 否 |
| 3 | history latent 3 | 3 | history token 32~47 | 否 | 否 |
| 4 | target anchor | 4 | 空 | 否 | 否 |
| 5 | target latent 1 | 5 | target token 0~15 | 是，valid 时 | 是，valid 时 |
| 6 | target latent 2 | 6 | target token 16~31 | 是，valid 时 | 是，valid 时 |
| 7 | target latent 3 | 7 | target token 32~47 | 是，valid 时 | 是，valid 时 |

这张表是理解数据协议的关键：视频、geometry、action 不是各自独立排布，而是共享同一个 8-step latent 时间轴。

---

## 16. 从 dataset 输出到下游模型

dataset 返回：

~~~text
RGB:
    [13,V,3,H,W] history/target raw frames

Geometry:
    [8,4,V,3,H,W]
    [8,4,V,H,W,3]
    masks

Action:
    [20,8,16,1]

Condition:
    text_emb, stream_ids
~~~

后续通常是：

1. trainer 将 history/target RGB 送入 Wan VAE；
2. 得到 history/target video latent；
3. 沿 latent 时间维拼成 8-step 视频 latent；
4. geometry RGB 送入 VGGT/geometry branch 或 geometry precompute；
5. action 和 action mask 送入 action branch；
6. text_emb、stream_ids 作为条件；
7. video_latent_loss_mask、action_loss_mask、geometry_group_valid_mask 控制监督和跨模态 token 交互。

dataset 的核心职责是把多种原始数据变成同一时间坐标系下的标准张量，而不是在这里完成 VAE、VGGT 或 transformer forward。

---

## 17. 调试一条样本时建议检查

~~~text
1. timestamp_policy 是否为 episode_local_frame_over_fps_v1
2. dataset_to_index 是否等于 dataset_from_index + segment.end_frame
3. views 数量是否为 2 或 3
4. video_path 是否存在
5. manifest fps 是否和 video average_fps 一致
6. video_from_timestamp 是否正确
7. decoder pts 是否和请求 timestamp 对齐
8. pointcloud row 是否覆盖所有 geometry frame
9. action cache/Parquet 是否存在连续 index
10. action_text 是否在 text embedding cache 中
11. geometry_group_valid_mask 是否屏蔽 padding slot
12. video_latent_loss_mask 是否只打开 target latent 5~7
13. action shape 是否为 [20,8,16,1]
~~~

建议打印：

~~~python
print("history raw ids:", history_frame_ids)
print("target raw ids:", target_frame_ids)
print("geometry group mask:", geometry_group_mask)
print("video valid:", video_latent_valid_mask)
print("video loss:", video_latent_loss_mask)
print("geometry_rgb:", geometry_rgb.shape)
print("geometry_pts3d:", geometry_pts3d.shape)
print("actions:", actions.shape)
print("action valid count:", action_valid_mask.sum())
print("action loss count:", action_loss_mask.sum())
~~~

---

## 18. 总结

一条 MOT 数据的完整加载链路：

~~~text
JSONL manifest
    ↓
MotTrainData.__getitem__
    ↓
同 view 数 bucket 选择 row
    ↓
valid_start_range 选择 current_frame
    ↓
mot_real_window_frame_ids
    ├── history 13 raw frame ids
    ├── target 13 raw frame ids
    └── 8 个 geometry groups × 4 slots
    ↓
segment padding
    ├── padded ids：保证能读取
    └── raw/geometry valid masks：避免错误监督
    ↓
去重 raw frame ids
    ↓
_load_rgb
    ├── local frame → timestamp
    ├── timestamp + video offset
    ├── timestamp → global video index
    └── VideoDecoder → float RGB
    ↓
_load_points
    └── PointStore → points + point mask
    ↓
_materialize_geometry_fields
    └── [32,...] → [8,4,...]
    ↓
_load_actions
    ├── mmap cache / Parquet
    ├── global index coverage 校验
    ├── absolute 16D → relative 20D
    ├── pack 到 [8,16]
    └── normalization + masks
    ↓
text_emb + stream_ids
    ↓
最终 sample dict
~~~

最重要的固定对齐关系：

~~~text
26 个 raw RGB frame
    → 8 个 video latent 时间位置

8 个 latent 时间位置
    ↔ 8 个 geometry group

每个 latent 时间位置
    ↔ 16 个 action token

history / target
    ↔ condition mask / loss mask
~~~

只要掌握这四个关系，就能理解 mot_dataset.py 为什么要同时生成 RGB、geometry、action、text 和多层 mask，以及它如何为后续 MOT 多模态交互准备输入。

相关源码：

- wan_va/dataset/mot_dataset.py
- wan_va/dataset/action_cache.py
- wan_va/dataset/pointcloud_store.py

