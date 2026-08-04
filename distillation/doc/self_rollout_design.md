# Distillation `self_rollout` 设计方案

## 1. 目标

当前 distillation rollout 通过 `inference/mot_inference.py:508` 的固定窗口推理完成：

```text
完整 history + target window
    -> 一次 video inference
    -> 一次 geometry inference
    -> 一次 action inference
```

这个流程不适合当前的增量 attention mask，也不能在预测 token 和真实执行结果之间做精确替换。

新的 `self_rollout` 必须满足：

```text
history latent    -> committed cache
history geometry  -> committed cache
history action    -> committed cache

target anchor latent    -> committed cache
target anchor geometry  -> committed cache
target anchor action    -> committed cache

predict next latent     -> predicted cache
encode next geometry    -> predicted cache
predict next action     -> predicted cache
execute / sample real result
    -> 删除对应 predicted cache
    -> 写入真实 GT latent / geometry / action
    -> 继续下一个时间点
```

所有新代码必须位于 `distillation/` 下，不修改：

```text
wan_va/
inference/mot_inference.py
```

Flash-WAM 的 KV cache 只作为设计参考，不直接 import Flash-WAM 的模型实现。

## 2. 当前实现约束

当前 `ThreeDVAMOTTransformer3DModel` 在每一层把以下 stream 一起构造并送入 MoT attention：

```text
NV + CV + G + NA + CA
```

当前模型已有的 `cached_outputs` 是 VGGTO dense/point head 的中间结果，不是可直接用于 MoT 增量 attention 的 K/V cache。因此不能只在现有 `forward_inference()` 外面包一层循环来实现真正 cache。

distillation 侧需要提供一个 adapter：

1. 复用当前模型的参数、projection、normalization、FFN 和 geometry 模块。
2. 在 distillation 中重新组织每一层的 Q/K/V 计算。
3. 把已经提交的 K/V 保存到 distillation-owned cache。
4. 对当前 query 根据 cache metadata 生成局部可见性，而不是重新构造完整窗口 mask。
5. 不改变 `wan_va` 的源文件和公开接口。

## 3. 时间、SGF order 与阶段模型

`self_rollout` 不能重新发明一套与训练不同的 frame clock。训练和 rollout
都必须以 `distillation/tests/test_visualize_x_metadata_mask.py` 中两份 metadata
所表达的 segmented order 为语义来源：

```text
8-frame 默认布局： H0 H1 H2 H3 T0 T1 T2 T3

Video / Geometry order: 0 0 0 0 2 4 6 8
Action order:           1 1 1 1 3 5 7 9
```

这里的含义不是“把 `chunk_windows` 改成 1”，而是：

1. history 段仍作为一个 prefill segment，共享 chunk-level order；
2. target 段从 anchor 开始逐 frame 增长 order；
3. action 永远位于同一逻辑 video/geometry state 之后，order 为 video order + 1；
4. `window_size` 继续限制 order 距离，不能因为有 KV cache 就绕过窗口约束。

对超过默认 8-frame 的 rollout，order 必须由固定 history 边界扩展，不能根据
当前 rollout 总长度重新取 `num_frames // 2`，否则 rollout 变长时旧 token 的
order 会发生漂移。稳定定义为：

```text
history_frames = spec.history_latent_frames
history_segments = ceil(history_frames / chunk_size)
target_base_order = 2 * history_segments

frame_id < history_frames:
    video_order(frame_id) = 2 * floor(frame_id / chunk_size)

frame_id >= history_frames:
    target_index = frame_id - history_frames
    video_order(frame_id) = target_base_order + 2 * target_index

geometry_order(frame_id) = video_order(frame_id)
action_order(frame_id) = video_order(frame_id) + 1
```

此外，每个 frame transaction 还有执行阶段：

```text
LATENT   = 0
GEOMETRY = 1
ACTION   = 2
```

`phase` 用于控制“当前 frame 哪些值已经产生并允许成为 key”，`order_id` 用于
复现 SGF mask 的训练语义。这两个概念不能合并：history 中多个 frame 可以具有
相同 order，但仍有不同 `frame_id`；geometry 的严格历史约束必须比较
`frame_id`，不能只比较 order。

同一 target frame 的执行顺序固定为：

```text
latent transaction -> geometry transaction -> action transaction
```

普通 X token 最终是否可见，必须同时满足 segmented order、window、stream、
noise 和事务提交状态；不能只使用 append 顺序形成一个简单下三角 mask。

## 4. Cache entry 设计

每个 stream 的 cache entry 至少包含：

```python
CacheEntry(
    layer_id: int,
    sample_id: int,
    frame_id: int,
    phase: int,
    stream_id: int,
    noise_id: int,
    order_id: int,
    token_start: int,
    token_count: int,
    key: Tensor,
    value: Tensor,
    committed: bool,
    predicted: bool,
    source: Literal["history", "anchor", "predicted", "ground_truth"],
    transaction_id: int,
)
```

实现时不建议为每个 token 建一个 Python object；上面的结构是语义模型。实际
存储应以同一 frame/stream/noise/source 的连续 tensor segment 为单位，K/V
形状统一为：

```text
key/value: [B, token_count, num_heads, head_dim]
metadata:  [B, token_count]
```

query metadata 与 key metadata 分开保存，因为增量 attention 是矩形
`[B, Q_current, K_cache_plus_current]`，不能继续假设 Q/K 来自同一份方阵 metadata。

实际存储应按 layer 分开：

```text
SelfRolloutKVCache
├── layer 0
│   ├── latent K/V
│   ├── geometry K/V
│   └── action K/V
├── layer 1
│   ├── latent K/V
│   ├── geometry K/V
│   └── action K/V
└── ...
```

每个 layer 的 cache 需要支持：

```python
append_committed(...)
append_predicted(...)
clear_predicted(frame_id=None)
promote_predicted(frame_id)
replace_frame_with_ground_truth(frame_id, ...)
truncate_from(frame_id)
snapshot()
restore(snapshot)
```

cache 物理上分成三类，避免删除一种状态时误删另一种：

```text
1. committed K/V
   已经可以作为后续 frame 条件的 canonical K/V。

2. transaction K/V
   当前一次 video/action denoise forward 内的临时 K/V；一个 scheduler step
   结束后立即删除。

3. semantic state
   latent、RGB、geometry frame state、action tensor 及 source/version。它不是
   attention K/V，但 GT replacement 回放时必须依赖它重新构造 K/V。
```

### 4.1 committed cache

`committed cache` 只保存已经确认的内容：

```text
history token
target anchor token
已经完成预测并接受的 token
真实执行/采样获得的 GT token
```

后续 query 可以读取 committed cache，但仍必须经过 attention visibility 检查。

对 video/action，“最终 scheduler sample”不能直接把最后一个 denoise step 的
K/V promote 成 committed K/V。最后一个 denoise K/V 仍携带该 step 的 timestep
conditioning，并且只对应 NV 或 NA query。接受最终 sample 后，必须额外执行一次
canonical commit forward：

```text
accepted video latent -- timestep 0 --> NV0 + CV0 committed K/V
accepted action       -- timestep 0 --> NA0 + CA0 committed K/V
```

这样才与下一次完整窗口 inference 中“过去 frame 同时作为 noisy/clean 的零时刻
condition”一致。`promote_predicted()` 只适用于已经以 canonical 形式重新编码的
segment，不能直接 promote scheduler 中间态。

### 4.2 predicted cache

`predicted cache` 保存当前预测事务中的临时 token：

```text
当前 latent denoise 状态
当前 geometry prediction
当前 action denoise 状态
```

这些 token 不能自动永久写入 committed cache。每次 denoise step 开始前，必须清除上一 step 的 predicted K/V，避免不同 noise level 的 K/V 混在一起。

geometry 没有 scheduler denoise，但仍先进入一个 geometry transaction。只有整帧
的 frame-local state、history-only relation state、joint G registers 和 action 所需
的 geometry condition 全部成功生成后，才原子提交；中途失败则恢复 frame checkpoint。

## 5. Attention visibility 规则

这是本设计中最重要的部分。cache 中存在某个 token，不代表当前 query 可以读取它。

对于 query `q` 和 cache key `k`，先检查：

```text
同一个 sample
token 有效
cache entry 已提交，或被当前事务明确允许
```

然后再按以下规则筛选。

所有规则以 distillation 侧的 `segmented_history_strict_geometry` profile 为准。
`build_x_metadata4sgf()`/`build_mot_metadata4sgf()` 提供 segmented order 的基线，
但 joint mask 中原生 G->G 的 `k_order <= q_order` 仍允许同 order 的当前 geometry。
为了满足本需求，训练 AR、训练 consistency 和 `self_rollout` 三条路径都必须在
distillation 内额外叠加：

```text
G query -> G key: key.frame_id < query.frame_id
```

否则会出现“训练 geometry 可看当前 geometry，rollout 却禁止”的图结构不一致。

### 5.1 Geometry query：只读取历史 geometry

Geometry 的 inter-frame attention 必须严格使用历史 geometry：

```text
q.phase == GEOMETRY
    => k.stream == GEOMETRY
    AND k.frame_time < q.frame_time
    AND k.committed == True
```

也就是说，geometry query 不能读取：

```text
当前 frame 的 geometry
未来 frame 的 geometry
predicted geometry
当前 frame 尚未提交的 geometry
```

这与“geometry token 只能 attend 到历史 geometry token”一致。

该规则同时作用于两处：

```text
1. VGGTO non-joint layer 的 same-view inter-frame attention；
2. even MoT layer 中 GeometryJointStream 的 G-register attention。
```

只约束其中一处是不完整的，因为另一处仍可能把当前/预测 geometry 信息写回
geometry hidden，随后被 action 读取。

### 5.2 Geometry 的局部 frame encoding

“Geometry 只能读取历史 geometry”只约束跨 frame geometry attention，不应阻止当前 RGB 经过当前 frame 的 frame-local encoder。

因此 geometry 处理拆成两部分：

```text
当前 RGB
    -> 当前 frame-local VGGTO block
    -> 当前 frame 的 geometry seed

当前 geometry seed
    -> 只查询历史 geometry cache 的 inter-frame attention
```

如果 geometry cache 为空，当前 frame 仍可以得到 frame-local geometry seed；它不会因为没有历史 geometry 而产生空 attention。

当前一个 logical geometry frame 实际对应 dataset 的一个 group：

```text
[S=4 temporal slots, V synchronized views]
```

`frame_id` 指 group id，不是 group 内 slot id。允许的 frame-local 计算包括每张图
的 frame block 和同 slot 的 synchronized cross-view block；history-only
inter-frame block 的 K/V 只能来自更早 group。第一帧无历史 K/V 时，attention
delta 视为 0，但保留 block residual/MLP 路径，不能向 SDPA 传空 key 后产生 NaN。

### 5.3 普通 latent/action query：按复合时间形成因果关系

对非 geometry query，默认可见性是：

```text
abs(k.order_id - q.order_id) <= window_size
AND SGF noise/order relation is allowed
AND key transaction state is readable
```

具体来说：

```text
latent query:
    可以读取 SGF order/window 允许的历史 latent/geometry/action
    可以读取当前 denoise step 的 NV token，用于当前多 view/patch 自注意力
    不能读取当前 frame 的 action
    不能读取当前 frame 尚未生成的 geometry

geometry query:
    只读取历史 geometry，见 5.1

action query:
    可以读取当前 frame 已 canonical commit 的 latent
    可以读取当前 frame已提交的 geometry
    可以读取 SGF order/window 允许的历史 latent/geometry/action
    可以读取当前 denoise step 的 NA token
    不能读取未来 token
```

### 5.4 NV/CV/NA/CA no-leak 规则

时间因果规则不能替代 noise 规则。仍然需要保留：

```text
clean -> clean:
    key_order <= query_order

noisy -> clean:
    key_order < query_order

noisy -> noisy:
    key_order == query_order
```

因此 visibility 的实际形式是：

```text
same_sample
AND valid
AND abs(key.order_id - query.order_id) <= window_size
AND stream_relation_allowed
AND noise_relation_allowed
AND geometry_history_constraint
AND transaction_visibility_allowed
```

矩形增量 mask 的核心关系如下。`current` 表示当前 transaction 临时 token，
`committed` 表示 canonical cache：

| Query | 可读取 current key | 可读取 committed key | 明确禁止 |
|---|---|---|---|
| current NV | 同 step/current frame 的 NV | SGF 允许的历史 NV/CV/NA/CA/G | current CV/G/NA/CA、未来 token |
| canonical NV0/CV0 commit | 同 commit 内、no-leak 允许的 NV0/CV0 | SGF 允许的历史 cache | current G/A、未来 token |
| current G | 无 inter-frame current G key | strict earlier committed G | current/future/predicted G、所有 X key |
| current NA | 同 step/current frame 的 NA | 当前 committed V/G 与 SGF 允许的历史 cache | current CA、未来 token |
| canonical NA0/CA0 commit | 同 commit 内、no-leak 允许的 NA0/CA0 | 当前 committed V/G 与历史 cache | 未来 token |

注意：history prefill 是唯一一个不是逐 frame 单 query 的阶段。history 的多个 frame
共享 segmented order，必须作为一个完整 prefill segment 使用方阵 mask 计算，才能
保持与训练 metadata 相同的同 segment 可见性。不能把 H0、H1、H2、H3 顺序 append
并声称等价；那会意外把同 order 的双向关系改成 frame 下三角关系。

## 6. Rollout 状态机

`self_rollout()` 使用显式状态机，不再调用 `run_mot_inference()`。

```text
INIT
  ↓
PREFILL_HISTORY_LATENT
  ↓
PREFILL_HISTORY_GEOMETRY
  ↓
PREFILL_HISTORY_ACTION
  ↓
PREFILL_TARGET_ANCHOR
  ↓
PREDICT_LATENT
  ↓
ENCODE_GEOMETRY
  ↓
PREDICT_ACTION
  ↓
EXECUTE_OR_SAMPLE_GROUND_TRUTH
  ↓
REPLACE_PRED_WITH_GT
  ↓
NEXT_FRAME / FINISH
```

## 7. History prefill

### 7.1 History latent

history latent 作为一个 segmented prefill block 进入 latent adapter：

```text
history latent [H0 ... H(H-1)]
    -> 构造与 build_x_metadata4sgf 对应的 history 子 metadata
    -> timestep 0 的 NV0/CV0 full-segment forward
    -> 按 layer 切分并写入 committed latent cache
```

不能逐 frame prefill，原因见 5.4：同一 history segment 的 clean/noisy token 可能
具有相同 order，训练 mask 允许的关系不等价于 append-time 因果关系。

### 7.2 History geometry

history geometry 使用对应的历史 RGB/geometry 输入，但严格 history-only 规则使其
必须按 logical geometry group 顺序处理：

```text
history RGB
    -> frame-local geometry encoding
    -> history-only geometry attention
    -> geometry register/state cache
```

geometry cache 只保存已经处理完成的历史 geometry。

这里与 history latent 不同：latent 必须 block-prefill 以复现同 order 的 X mask；
geometry 因用户要求 `key.frame_id < query.frame_id`，因此必须逐 geometry group
prefill。每组内部只运行 frame-local/cross-view，再查询已经提交的 earlier-G cache。

### 7.3 History action

history action 在全部 history latent 和逐组 history geometry cache 已经存在后，
作为一个 segmented prefill block 处理：

```text
history action
    -> action embedding
    -> 对整个 history action segment 构造 NA0/CA0 query
    -> 读取 SGF mask 允许的 latent/geometry/action key
    -> 写入 committed action cache
```

history action 的 self-relation 同样要保留相同 order 下的 mask 语义，因此不逐 frame
append。geometry keys 仍通过 strict frame rule 过滤；X query 读取 G 则遵循
X-to-G 的 SGF order/no-leak 规则。

## 8. Target anchor

默认设计把 target 的第一个 latent 当作已知 anchor：

```text
target anchor latent
target anchor geometry
target anchor action
```

它们直接进入 committed cache，不参与当前预测 denoise。

“直接进入”仍表示依次执行三次 canonical encoding，而不是把原始 tensor 塞入
cache：

```text
encode T0 latent as NV0/CV0 and commit
encode T0 geometry with G-history-only and commit
encode T0 action as NA0/CA0 and commit
```

这样下一个 frame 的预测可以使用：

```text
history + target anchor
```

如果后续确认 target 第一个 latent 也需要预测，则只需要把 anchor 的 `source` 从 `ground_truth` 改成 `predicted`，状态机不需要重新设计。

## 9. 少步 latent prediction

当前 frame 的 latent prediction 使用临时 predicted transaction：

```text
begin_prediction(frame_id=i, phase=LATENT)
    ↓
初始化当前 latent noisy state
    ↓
for denoise_step in video_scheduler:
    clear_predicted(frame_id=i, phase=LATENT)
    仅构造当前 frame 的 NV query/K/V
    读取 visibility 允许的 committed cache
    把本 step 的 current NV K/V 加入矩形 attention key 集合
    执行 attention 和 output projection
    写入本 denoise step 的 predicted K/V
    scheduler 更新 latent
    ↓
得到 final latent
    ↓
删除最后一个 denoise step 的临时 K/V
    ↓
以 timestep=0 对 final latent 执行 canonical NV0/CV0 commit forward
    ↓
写入 committed latent cache
```

重要规则：

```text
denoise 中间状态只存在 predicted cache
同一 frame 的不同 denoise step 不能同时存在
最终 latent 必须重新 canonical encode，不能 promote 带噪 timestep K/V
```

CFG 若保留，conditional 与 unconditional forward 共享同一份只读 committed cache，
但各自拥有独立 transaction K/V，不能先运行 cond 后把它的临时 K/V 留给 uncond。
两次 forward 完成后组合 model output，再由 scheduler 更新 sample。

## 10. Geometry prediction/cache

latent 生成完成后，使用当前生成 latent 解码或得到对应 RGB，再编码 geometry：

```text
generated latent_i
    -> generated RGB_i
    -> geometry frame-local encoder
    -> history geometry cache
    -> geometry_i state/register
```

geometry 的新状态先写入 transaction geometry cache。只有该 frame 的全部
geometry layer 都成功后，才原子提交为 committed geometry cache。

geometry query 的 cache filter 必须强制：

```text
current geometry_i
    只能读 geometry frame_time < i 的 committed geometry
```

不能因为当前 geometry 处在预测事务中，就让它读取同一事务中刚刚生成的 geometry K/V。

## 11. Action prediction/cache

geometry 生成之后，再预测当前 action：

```text
current action noise
    -> action denoise step 0
    -> clear old predicted action cache
    -> action denoise step 1
    -> ...
    -> final action
    -> 删除最后一个 denoise step 的临时 NA K/V
    -> timestep=0 canonical NA0/CA0 commit forward
    -> 写入 committed action cache
```

action query 可以读取：

```text
历史 committed latent
历史 committed geometry
历史 committed action
当前 frame 的已接受 latent
当前 frame 的已接受 geometry
```

不能读取：

```text
未来 latent
未来 geometry
未来 action
当前 frame 未提交的旧 predicted action
```

action denoise 只需要 current NA query；current CA 代表未知 clean target，不能作为
key。接受最终 action 后的 canonical commit 才同时生成 NA0/CA0。action valid/loss
mask 必须继续作用到 current action slot，padding slot 不得被写成有效 cache token。

## 12. Pred cache 删除与 GT 替换

### 12.1 删除 pred token

每个 predicted token 必须带有：

```text
frame_id
phase
source="predicted"
```

删除接口：

```python
cache.clear_predicted(
    frame_id=frame_id,
    phase=phase,
)
```

删除时必须同时清理：

```text
所有 transformer layer 的 K/V
对应 token metadata
geometry register cache
geometry inter-frame cache
action cache
```

不能只删除最终 hidden，而留下旧 K/V，否则后续 attention 会继续读取已删除的预测。

这里需要区分两个动作：

```text
discard_transaction:
    删除当前 scheduler step/当前 phase 的临时 K/V，不改变已经提交的前序 frame。

truncate_committed_from(frame_id):
    删除 frame_id 及之后的 committed 派生 K/V/geometry state，用于 GT replacement。
```

两者不能共用一个模糊的 `clear()`；否则 denoise step 清理可能误删历史，GT 替换也
可能只删临时状态而留下 stale committed K/V。

### 12.2 使用真实采样结果替换

预测 action 执行后，外部环境或数据 replay 可以返回真实结果：

```python
GroundTruthStep(
    frame_id=...,
    caused_by_action_frame_id=...,
    video_latent=...,
    geometry_rgb=...,
    geometry_state=...,
    action=...,
    video_valid=...,
    action_valid=...,
)
```

`frame_id` 必须表示“这份真实数据要写入哪个 logical cache frame”。
`caused_by_action_frame_id` 只用于 online 环境记录因果来源，不能用它暗中推导
cache frame。这样可以同时支持：

```text
offline teacher forcing: predicted frame i -> batch GT frame i
online environment:      action j -> observation mapped explicitly to frame i
```

第一版不自动猜测 action 与 observation 的 offset，因为当前 action packing 存在
`A_i` 表示物理 transition 的 dataset offset；猜错一位会造成无法从数值测试发现的
训练/执行错位。

替换流程：

```text
停止当前 predicted transaction
    ↓
删除该 frame 及其派生 predicted cache
    ↓
写入真实 GT video latent
    ↓
重新编码真实 GT geometry
    ↓
写入真实 GT geometry cache
    ↓
写入真实 GT action cache
    ↓
标记 source="ground_truth"
    ↓
继续下一个 frame
```

GT replacement 必须是原子事务：

```text
checkpoint = snapshot(before earliest_dirty_frame)
try:
    truncate all derived state from earliest_dirty_frame
    append/re-encode requested GT components in latent -> geometry -> action order
    replay any retained later semantic values that are still valid
except:
    restore(checkpoint)
    raise
```

模型预测结果与 continuation cache 必须分开保存。即使 continuation 已替换成 GT，
`RolloutResult.pred_*` 仍保存原预测，供 target-vs-generated artifact 和训练诊断使用；
不能用 GT 覆盖预测输出后再声称完成了 rollout。

### 12.3 依赖失效规则

替换一个 token 后，所有依赖它的派生 cache 必须失效：

```text
替换 video latent_i:
    删除 geometry_i 及后续派生 geometry
    删除 action_i 及后续 action
    删除所有未来 frame 的 predicted cache

替换 geometry_i:
    保留 video_i
    删除 action_i 及后续依赖 geometry 的 cache

替换 action_i:
    删除所有允许读取 action_i 的未来 predicted cache
```

更精确的 first-version 失效矩阵：

| 被替换组件 | frame i 保留 | frame i 必须重建 | 必须删除/回放 |
|---|---|---|---|
| video_i | 无 | canonical video_i、geometry_i、action_i | 所有 `frame > i` 的派生 cache |
| geometry_i | canonical video_i | geometry_i、action_i | 所有 `frame > i` 的派生 cache |
| action_i | video_i、geometry_i | action_i | 所有 `frame > i` 的派生 cache |

若 provider 没有提供被迫重建的下游 GT 值，策略必须由调用方显式选择：

```text
recompute_predicted: 用新条件重新预测下游组件；
require_ground_truth: 缺少 GT 时直接报错，不继续使用旧预测。
```

默认建议 `require_ground_truth` 用于训练可复现路径，避免 replacement 名义上成功、
实际仍混入旧条件下预测出的 action/geometry。

最安全的实现方式是：

```text
在每个 frame 开始前保存 committed cache checkpoint
发生 GT 替换时恢复最近 checkpoint
从被替换 frame 开始重新 append 真值
```

第一版不应尝试在 cache 中就地修改复杂的派生 K/V；应优先使用 checkpoint + truncate + replay，确保不会出现 stale K/V。

## 13. Offline 与 online 两种 GT 来源

### 13.1 Offline teacher-forcing

直接从当前 batch 获取真实值：

```text
batch["latents"]
batch["geometry_rgb"]
batch["actions"]
```

用途：

```text
单元测试
训练期间固定样本 rollout
验证 cache 替换逻辑
```

### 13.2 Online environment/replay

通过 callback 获取真实执行结果：

```python
ground_truth_provider(
    previous_action=generated_action,
    frame_id=frame_id,
    state=rollout_state,
) -> GroundTruthStep
```

用途：

```text
真实机器人执行
环境交互
SGF trajectory replay
```

`self_rollout` engine 不应该直接依赖具体环境，只依赖这个 provider 接口。

## 14. 与 consistency trainer 的集成

当前：

```python
from distillation.rollout import autoregressive_rollout
```

目标：

```python
from distillation.self_rollout import self_rollout
```

`ConsistencyTrainer._run_rollout()` 继续使用：

```python
self.method_model.ema_student
```

但 rollout 函数改成：

```python
self_rollout(
    batch=batch,
    transformer=self.method_model.ema_student,
    config=self.config,
    spec=mot_spec_from_config(self.config),
    device=self.device,
    empty_text_emb=self._get_empty_text_emb(),
    decode_latents_to_rgb_views=self._decode_rollout_latents,
    video_num_steps=...,
    action_num_steps=...,
    rollout_frames=...,
    ground_truth_provider=...,
    replacement_policy=...,
)
```

返回值继续保持现有 `RolloutResult` 兼容格式，确保以下保存逻辑可以复用：

```python
save_rollout_artifacts(...)
```

`self_rollout` 不得 import：

```text
inference.mot_inference
run_mot_inference
```

建议新增/替换的 distillation config 契约：

```text
distill.generation_shape.order_mode = "segmented"
distill.generation_shape.chunk_size = 4
distill.generation_shape.window_size = 16

distill.rollout_horizon_frames
distill.rollout_video_num_steps
distill.rollout_action_num_steps
distill.rollout_gt_mode = "none" | "offline" | "provider"
distill.rollout_replacement_policy = "require_ground_truth" | "recompute_predicted"
```

`rollout_chunk_pairs` 是旧 fixed-window rollout 的概念，新接口应以 anchor 后需要生成
多少个 logical latent frame 为单位。为避免静默改变旧配置，迁移时应明确拒绝同时
设置 `rollout_chunk_pairs` 与 `rollout_horizon_frames`，而不是猜优先级。

`distillation/rollout.py` 中 artifact 类型/保存函数可以保留，但 consistency trainer
的新执行路径不能经由该文件顶层 import 间接加载 `inference.mot_inference`。可选的
边界处理有两种：

```text
A. 把 RolloutResult/save_rollout_artifacts 移到 distillation-owned result/artifacts；
B. 先删除 distillation/rollout.py 的顶层 inference import，并让旧 SGF recorder
   通过调用方显式注入 run_inference。
```

推荐 A，依赖边界最清楚；但迁移只移动被新路径实际使用的类型和保存函数，不顺便
重构旧 SGF 逻辑。

## 15. 推荐的 distillation 文件结构

```text
distillation/
├── self_rollout/
│   ├── __init__.py
│   ├── cache.py
│   ├── attention.py
│   ├── geometry_cache.py
│   ├── mot_adapter.py
│   ├── scheduler.py
│   ├── provider.py
│   ├── engine.py
│   ├── state.py
│   └── result.py
├── rollout.py                 # 保留 artifact 保存和旧兼容类型，逐步移除旧调用
└── trainer/
    └── consistency.py         # 切换到 self_rollout
```

这里不建议压成一个超大 `self_rollout.py`：geometry cache、MoT layer cache 和
transaction rollback 的生命周期不同，混在一个文件中会使删除/替换规则难以审计。
但每个模块只承担上图列出的单一职责，不再为未来 backend 增加额外抽象。

## 16. 验证计划

### 16.1 Cache 单元测试

验证：

```text
committed append
predicted append
predicted delete
canonical transaction commit
frame truncate
snapshot/restore
```

### 16.2 Attention visibility 测试

至少验证：

```text
geometry_i 不能读取 geometry_i
geometry_i 不能读取 geometry_future
geometry_i 可以读取 geometry_history

latent_i 不能读取未来 latent/action
action_i 可以读取 latent_i
action_i 可以读取 geometry_i
action_i 不能读取未来 token
```

### 16.3 GT replacement 测试

测试步骤：

```text
生成 pred frame 1
确认 pred K/V 存在
删除 pred frame 1
确认所有 layer 的 pred K/V 消失
写入真实 frame 1
确认 source 变为 ground_truth
继续生成 frame 2
确认 frame 2 可以读取真实 frame 1
```

### 16.4 Full-window 对照测试

在 deterministic scheduler 和固定 random seed 下，对比：

```text
旧的完整窗口推理
新的 self_rollout
```

需要分别比较：

```text
latent shape
geometry shape
action shape
mask 可见性
cache frame 顺序
```

由于增量执行的数值路径不同，不强制逐元素完全相等，但必须满足 attention visibility 和 stream dependency 不违反设计。

### 16.5 Import boundary 测试

确保：

```text
distillation.self_rollout 不 import inference.mot_inference
distillation.self_rollout 不修改 wan_va 文件
```

## 17. 实现顺序

建议严格按以下顺序实现：

1. 实现 cache entry、commit/predicted、truncate、snapshot/restore。
2. 实现稳定的 segmented order 生成器，并让 AR/consistency 训练 profile 共用。
3. 实现矩形 `cache_visibility(query_meta, key_meta)`，覆盖 window、no-leak、
   geometry strict history 和 transaction state。
4. 先实现 history X full-segment prefill 和单 frame latent/action adapter。
5. 实现 geometry frame-local、non-joint history K/V 与 joint G-register cache。
6. 实现 video/action denoise transaction 和 canonical timestep-0 commit。
7. 实现 `latent -> geometry -> action` 单 frame transaction。
8. 实现 pred 删除和 GT replace，并加入依赖失效/恢复 checkpoint。
9. 实现完整 `self_rollout` frame loop与 provider 接口。
10. 接入 consistency trainer，并保持 artifact result 兼容。
11. 执行 cache、visibility、prefill 等价性、GT replacement、import boundary 和
    小型 rollout 测试。

## 18. 当前默认假设

本设计暂时采用以下假设：

1. target 的第一个 latent 是已知 clean anchor。
2. latent 仍然需要同时维护 NV/CV 两个 stream。
3. action 仍然需要同时维护 NA/CA 两个 stream。
4. geometry 的跨 frame attention 严格只读历史 committed geometry；当前 frame 的 geometry 只能使用 frame-local encoder 得到 seed。
5. video/action denoise K/V 不直接 promote；最终 sample 必须以 timestep 0 重新编码
   为 canonical NV/CV 或 NA/CA cache。
6. 获得真实 GT 后优先使用 checkpoint + truncate + replay，避免 stale K/V。
7. `self_rollout` 返回现有 `RolloutResult` 兼容结构，不改变现有可视化文件格式。
8. history X stream 作为完整 segment prefill；history geometry 因 strict-frame
   规则逐 group prefill。
9. `window_size` 在 cache 推理中仍生效；cache 中存在但窗口外的 token 不可见。
10. online provider 必须显式给出 GT 对应的 logical `frame_id`，engine 不猜 action/
    observation offset。

## 19. 可实现性结论与必须接受的设计取舍

### 19.1 仅修改 metadata 不足以实现 strict-history geometry

当前 `wan_va.modules.mot_attention` 的 G->G 判断本质是：

```text
k.order_id <= q.order_id
```

即使 distillation 把 order 改成 segmented，当前 geometry 的 query/key 仍具有相同
`frame_id` 和相同 `order_id`，所以 self-attention 一定会被允许。`token_valid_ids`
只能按 token 屏蔽，不能表达 pair-wise 的 `key.frame_id < query.frame_id`。

因此，如果同时坚持以下三个条件：

```text
不修改 wan_va
AR 与 consistency 使用同一 strict geometry mask
self_rollout 也使用该 mask
```

那么 distillation 必须拥有自己的 attention policy/adapter，不能只依赖
`install_order_profile()` 修改 `order_ids`。这是满足需求所需的最小结构性变化，
不是可选优化。

### 19.2 推荐方案：distillation-owned policy + model adapter

推荐在 distillation 中定义唯一可见性函数，并让三条路径共用：

```text
AR training full-sequence mask
Consistency training full-sequence mask
self_rollout rectangular incremental mask
```

三者只在 Q/K 形状上不同，pair predicate 完全相同。训练侧 adapter 仍复用原模型
参数和 block 子模块，但在调用 self-attention 时使用 distillation policy。这样无需
修改 `wan_va` 文件，也不会通过全局 monkeypatch 改变 repository 中其他模型实例。

不推荐全局替换
`wan_va.modules.model_3dva_mot.attention_from_meta`：它会影响同一进程内非
distillation 模型，且难以在多 trainer/测试间恢复。

### 19.3 第一版 backend 取舍

第一版 incremental rollout 推荐使用 PyTorch SDPA + 显式矩形 boolean mask：

```text
Q = current transaction tokens
K/V = visible committed segments + current transaction segments
mask = [B, Q, K]
```

原因是现有 FA4/Flex 路径围绕方阵 `MOTMaskMetadata` 构造，直接扩展到动态矩形
cache 会增加大量 backend 工作。先以 dense reference 证明语义，再在不改变 policy
测试的前提下增加高性能 backend。训练现有 full-sequence backend 是否继续使用
FA4，取决于 distillation policy 能否构造等价 BlockMask；这属于实现阶段的性能
分支，不改变本文的可见性契约。

## 20. 核心数据结构契约

### 20.1 TokenMetadataBatch

每个 query segment 和 key segment 都携带同形状 metadata：

```python
TokenMetadataBatch(
    seq_ids: Tensor,          # [B,L]
    frame_ids: Tensor,        # [B,L], logical latent/geometry group frame
    order_ids: Tensor,        # [B,L], stable segmented SGF order
    stream_ids: Tensor,       # VIDEO / ACTION / GEOMETRY
    noise_ids: Tensor,        # NOISY / CLEAN / GEOMETRY
    valid_ids: Tensor,        # [B,L] bool
    committed_ids: Tensor,    # [B,L] bool
    transaction_ids: Tensor,  # [B,L], -1 for committed
    source_ids: Tensor,       # history/anchor/predicted/ground_truth
    version_ids: Tensor,      # semantic state version used to derive K/V
)
```

`frame_ids` 和 `order_ids` 都必须存在。前者用于 strict geometry 与 truncate，
后者用于 SGF/no-leak/window。

### 20.2 LayerKVCache

每个 MoT/VGGTO attention layer 的 cache 以 segment 列表保存：

```python
LayerKVCache(
    committed_segments: list[KVSegment],
    transaction_segments: dict[int, list[KVSegment]],
)
```

`KVSegment` 必须是不可变 append 单元；GT replacement 不原地改 segment 内容，
而是 truncate 后 append 新 version。物化 attention key 时才按逻辑顺序 concat：

```text
materialized K/V order = committed segment append order + explicitly allowed current segment
```

物理 concat 顺序不决定可见性，mask 才决定可见性。

### 20.3 GeometryStateCache

geometry 不能只保存 joint MoT 的 G-register K/V。至少需要：

```text
per frame/group:
    input RGB or decoded RGB reference
    slot/view validity
    VGGTO layer input/output state version

per non-joint VGGTO layer:
    earlier-group inter-frame K/V

per joint MoT layer:
    earlier-group G-register K/V
    committed register snapshot used by later X/action query

for dense/point heads when artifacts需要:
    cached layer outputs required by dense_forward/point_forward
```

这些 cache 的失效边界相同，但用途不同，不能用一个名为 `geometry_cache` 的裸
tensor 混合保存。

### 20.4 SemanticFrameState

每个 logical frame 保存 continuation 所需的原始语义值：

```python
SemanticFrameState(
    frame_id: int,
    video_latent: Tensor | None,
    geometry_rgb: Tensor | None,
    geometry_state: object | None,
    action: Tensor | None,
    video_source: Source | None,
    geometry_source: Source | None,
    action_source: Source | None,
    video_version: int,
    geometry_version: int,
    action_version: int,
)
```

K/V segment 记录其来源 version。调试断言要求：可见 committed segment 的
`version_id` 必须等于当前 semantic state version，从而尽早发现 stale K/V。

### 20.5 RolloutState 与 PredictionLog

```text
RolloutState
├── semantic_frames       # 后续 continuation 的当前事实，可被 GT 替换
├── mot_layer_caches
├── geometry_layer_caches
├── checkpoints
└── current_transaction

PredictionLog
├── predicted_video[i]    # 永久保留原模型输出
├── predicted_geometry[i]
├── predicted_action[i]
├── scheduler diagnostics
└── replacement events
```

`RolloutState` 可以被 rollback/replace，`PredictionLog` 不随 rollback 覆盖。

## 21. 唯一 attention policy

### 21.1 stable segmented order builder

order builder 的输入必须显式包含：

```text
frame_ids
history_frames
chunk_size
stream_ids
```

不能从当前 tensor 长度重新推导 history/target split。训练 8-frame batch 与长
rollout 对同一个 `frame_id` 必须得到同一个 order。

### 21.2 矩形 visibility 伪代码

```python
def visibility(query, key, *, window_size):
    same_sample = query.seq_id == key.seq_id
    valid = query.valid & key.valid
    in_window = abs(query.order_id - key.order_id) <= window_size

    if query.stream == GEOMETRY:
        relation = (
            key.stream == GEOMETRY
            and key.committed
            and key.frame_id < query.frame_id
        )
    else:
        current_transaction_key = (
            key.transaction_id == query.transaction_id
            and key.transaction_id >= 0
        )
        readable_state = key.committed or current_transaction_key

        if key.stream == GEOMETRY:
            relation = readable_state and x_to_g_sgf_rule(query, key)
        else:
            relation = readable_state and x_to_x_no_leak_rule(query, key)

    return same_sample and valid and in_window and relation
```

对 geometry query，即使 `transaction_id` 相同也不能读取 current G。对 X query，
只有同 transaction 的临时 key 可见，其他 frame/其他 CFG branch/上一个 denoise
step 的 predicted key 一律不可见。

### 21.3 X-to-X no-leak

沿用当前两份 mask 的语义：

```text
clean query -> clean key: key.order <= query.order
noisy query -> clean key: key.order <  query.order
noisy query -> noisy key: key.order == query.order
其他 noise pair: False
```

这意味着 current NV/NA 可以读取同 transaction、同 order 的 noisy token，保留
多 view/patch/action-slot 内部自注意力；不能读取同 order 的 clean target。

### 21.4 X-to-G

统一选择 strict order：

```text
X query -> G key: key.order < query.order
```

因此：

```text
current video order 2n 不能读取 current G order 2n
current action order 2n+1 可以读取 current G order 2n
```

这同时满足执行阶段 `latent -> geometry -> action`。实现前应增加一个 distillation
测试锁定该规则，因为当前 `wan_va` 中 dense helper 与部分 backend inline predicate
存在 `<`/`<=` 表达不完全一致的风险；本任务不能通过修改 `wan_va` 解决，只能让
distillation-owned policy 成为 AR、consistency 和 rollout 的共同真值来源。

### 21.5 无可见 key 的处理

第一帧 geometry 会出现 Q 非空、K 为空。attention adapter 必须直接返回零
attention delta，并继续 residual/FFN，不得构造伪 key，也不得让当前 geometry
临时 key 作为 fallback。伪 key 会违反 strict-history 语义。

## 22. MoT 增量 adapter 的逐层设计

### 22.1 复用范围

distillation adapter 复用模型实例上的参数与无状态 helper：

```text
_embed_video
_embed_action
_time_embed_repeated
_video_grid / _action_positions
ThreeDVAMOTBlock._modulation
ThreeDVAMOTBlock._self_qkv
ThreeDVAMOTBlock._attention_output
ThreeDVAMOTBlock._finish_block
GeometryJointStream.qkv_project / attn_delta / ffn_delta
_final_video / _final_action
```

不调用 `_prepare_va_inputs()`，因为它把 action shape 固定为
`[B,20,8,16,1]`；单 frame adapter 必须直接调用 embedding/time/rotary helper。

由于使用了 private helper，adapter 初始化时必须检查所需属性和模型 topology，
缺少时 fail fast；不能静默退回 `forward_inference()`，因为后者会重新走完整窗口。

### 22.2 单 phase、单 layer 执行

以 current NV 为例：

```text
hidden_l
  -> modulation(timestep)
  -> q_l, k_l, v_l
  -> materialize layer_l committed K/V
  -> concat allowed current k_l/v_l
  -> build rectangular visibility
  -> SDPA
  -> self attention output projection
  -> text cross attention
  -> FFN
  -> hidden_(l+1)
  -> save current k_l/v_l under transaction_id
```

current K/V 必须在本层 attention 中可见，同时只在该 layer 的 transaction cache
存在；不能把 layer 0 K/V 错用于 layer 1。

### 22.3 History X prefill

history latent/action 使用 full-segment Q/K/V：

```text
Q = all history segment tokens
K/V = all history segment tokens + any earlier committed segment
mask = distillation full-sequence policy
```

完成所有 layer 后，按 layer 保存 canonical segment K/V。prefill 不是调用完整
`forward_inference()`；它只是 adapter 的 Q 长度从 one-frame 扩展到 history block。

### 22.4 Canonical video commit

```text
input: accepted latent_i
streams: NV0, CV0
timesteps: both 0
queries: NV0 + CV0 for frame i
keys: committed history + current canonical NV0/CV0
mask: same no-leak policy
output: per-layer canonical K/V and optional final hidden
```

commit 在临时 cache 中完整跑通后一次性移动到 committed segments。任何 layer
失败都不能留下半数 layer 已提交的 frame。

### 22.5 Canonical action commit

同理：

```text
input: accepted action_i
streams: NA0, CA0
timesteps: both 0
keys additionally include committed video_i and geometry_i
```

invalid action slot 的 metadata.valid=False；输出 tensor 可保留 padding 值，但其
K/V 永远不可被后续 query 读取。

## 23. Geometry 增量 adapter

### 23.1 logical frame 与 group

dataset 的 geometry 输入是：

```text
[B, G, S=4, V, 3, 224, 224]
```

`G` 与 latent logical frame 对齐。一次 geometry transaction 处理一个
`[B,1,S,V,3,224,224]` group，所有 cache metadata 的 `frame_id` 使用 group id。

### 23.2 每层顺序

对当前 group 的每个 VGGTO layer：

```text
current tokens
  -> run_frame_block                 # 每图局部
  -> if non-joint layer:
         current synchronized cross-view block
         current Q + earlier-group inter-frame K/V
         history-only relation output
     else joint G layer:
         extract current registers
         current G query + earlier committed G-register K/V
         GeometryJointStream residual/FFN
         register_override 写回 current tokens
  -> 保存本层 current state，进入下一层
```

non-joint inter-frame 与 joint G-register 两套 K/V 都使用
`key.frame_id < query.frame_id`。current group 的 S/V token 不加入 inter-frame K/V
集合；current cross-view 只属于 frame-local stage。

### 23.3 第一帧行为

没有历史 geometry 时：

```text
frame block 正常运行
cross-view 正常运行
inter-frame attention delta = 0
joint G attention delta = 0
对应 block residual/FFN 正常运行
```

这是明确语义，不是异常或需要伪造零 token 的场景。

### 23.4 geometry commit 内容

geometry transaction 成功后原子提交：

```text
每个 non-joint layer 的 current inter-frame K/V
每个 joint layer 的 current G-register K/V
action joint attention 将读取的 current G register snapshot
dense/point head 所需 layer outputs（若当前 rollout 请求这些 artifact）
final geometry state
source/version/valid metadata
```

如果 rollout 仅需要 action conditioning，不需要 depth/point artifact，可以不立即
运行 dense/point head，但不能省略 action 实际读取的 layer register snapshots。

### 23.5 generated RGB 来源

predicted latent_i 必须先通过 trainer 提供的 VAE decode callback 得到多视角 RGB，
再按当前 geometry group 的 S=4 契约构造 geometry 输入。这里有一个需要实现前
验证的数据问题：单个 latent frame 解码后是否稳定对应正好 S=4 个 geometry RGB
slot。adapter 不应插值或复制 slot 来凑 shape；应复用现有 dataset/inference 的
latent-to-RGB 时间对齐规则，并用单元测试锁定索引。

若 online provider 直接给出合法 `[B,1,S,V,3,224,224]` geometry RGB，则跳过
predicted latent decode 作为 GT geometry 输入，但仍必须重新运行 geometry adapter
产生与当前模型权重一致的 cache。

## 24. Scheduler 与 denoise transaction

### 24.1 不依赖 inference/mot_inference.py

`distillation/self_rollout/scheduler.py` 只负责：

```text
构造 wan_va.utils.FlowMatchScheduler
设置 video/action timesteps
初始化 noise
执行 scheduler.step
记录必要诊断
```

它不 import `inference.mot_inference`，也不复制完整 fixed-window batch builder。

### 24.2 Video denoise

```python
sample = randn_like(target_frame_shape)
for step_id, timestep in enumerate(video_timesteps):
    tx = begin_transaction(frame_id, LATENT, step_id)
    cond = adapter.predict_video_nv(sample, timestep, cache, tx, text)
    if cfg_enabled:
        discard_transaction(tx)
        uncond_tx = begin_transaction(frame_id, LATENT, step_id, branch="uncond")
        uncond = adapter.predict_video_nv(sample, timestep, cache, uncond_tx, empty_text)
        prediction = uncond + scale * (cond - uncond)
        discard_transaction(uncond_tx)
    else:
        prediction = cond
        discard_transaction(tx)
    sample = scheduler.step(prediction, timestep, sample)

canonical_commit_video(sample, timestep=0)
```

cond transaction 也必须在 uncond 前删除；二者不能共享临时 K/V。

### 24.3 Action denoise

action 与 video 同样管理 transaction，但 action 默认 conditional-only，保持当前
`action_guidance_scale == 1` 的行为。每次 scheduler step 后重新应用
`action_valid_mask`，防止 padding slot 被噪声更新。

### 24.4 随机性

engine 接受显式 `torch.Generator` 或 seed，video/action 使用可追踪的 generator
状态。checkpoint 必须包含 generator state；否则 GT rollback 后 replay 会改变后续
noise，无法区分“条件替换影响”与“随机序列变化”。

## 25. self_rollout engine 详细状态机

### 25.1 输入验证

启动前验证：

```text
latents/actions/geometry_rgb/action_valid_mask 的 batch、frame 轴一致
history_frames 与 spec/config 一致
至少存在 target anchor frame
stream_ids 与 view 数一致
geometry group S/V/224x224 契约成立
rollout horizon 不超过 offline batch 可用范围（offline mode）
EMA student/model topology 支持 adapter
```

### 25.2 主流程伪代码

```python
def self_rollout(...):
    state = RolloutState(...)
    prediction_log = PredictionLog(...)

    prefill_history_latent_segment(state, batch.history_latents)
    prefill_history_geometry_sequential(state, batch.history_geometry)
    prefill_history_action_segment(state, batch.history_actions)

    anchor = history_frames
    canonical_commit_video(state, anchor, batch.latent[anchor], source=ANCHOR)
    commit_geometry(state, anchor, batch.geometry[anchor], source=ANCHOR)
    canonical_commit_action(state, anchor, batch.action[anchor], source=ANCHOR)

    for frame_id in range(anchor + 1, anchor + 1 + rollout_horizon):
        frame_checkpoint = state.snapshot()
        try:
            pred_video = denoise_video_frame(state, frame_id)
            prediction_log.video[frame_id] = pred_video
            canonical_commit_video(state, frame_id, pred_video, source=PREDICTED)

            pred_geometry = encode_geometry_from_video(state, frame_id, pred_video)
            prediction_log.geometry[frame_id] = pred_geometry.artifact
            commit_geometry(state, frame_id, pred_geometry, source=PREDICTED)

            pred_action = denoise_action_frame(state, frame_id)
            prediction_log.action[frame_id] = pred_action
            canonical_commit_action(state, frame_id, pred_action, source=PREDICTED)

            gt_step = provider.maybe_get(frame_id, pred_action, state.public_view())
            if gt_step is not None:
                replace_with_ground_truth(state, gt_step, policy)
                prediction_log.replacements.append(...)
        except Exception:
            state.restore(frame_checkpoint)
            raise

    return build_rollout_result(state, prediction_log, original_batch)
```

### 25.3 phase barrier

每个 frame 只有满足以下条件才进入下一 phase：

```text
LATENT complete:
    no LATENT transaction K/V remains
    canonical NV0/CV0 committed for every layer

GEOMETRY complete:
    all VGGTO/joint layer states committed at same version
    no geometry transaction remains

ACTION complete:
    no ACTION transaction K/V remains
    canonical NA0/CA0 committed for every layer
```

phase barrier 断言是防止“模型输出 tensor 已生成，但少数 layer cache 未提交”的关键。

## 26. 删除、替换与回放算法

### 26.1 earliest dirty frame

根据替换组件计算最早需要 truncate 的位置：

```text
video_i    -> dirty frame i / phase LATENT
geometry_i -> dirty frame i / phase GEOMETRY
action_i   -> dirty frame i / phase ACTION
```

第一版 cache segment 以 frame 为截断单位，因此即使只替换 geometry/action，也可以
先恢复 frame i 开始前的 checkpoint，再按保留的 canonical video_i 重放后续 phase。
这比 phase 内原地删 layer segment 更慢，但状态最容易证明正确。

### 26.2 replacement replay

```python
def replace_with_ground_truth(state, gt, policy):
    frame_id = gt.frame_id
    checkpoint = state.checkpoint_before(frame_id)
    preserved_predictions = state.prediction_log  # 不回滚
    preserved_semantics = state.semantic_frames.copy_from(frame_id)

    state.restore(checkpoint)

    video = gt.video_latent or preserved_semantics[frame_id].video_latent
    canonical_commit_video(state, frame_id, video, source=selected_source)

    geometry_input = gt.geometry_rgb or derive_from(video)
    commit_geometry(state, frame_id, geometry_input, source=selected_source)

    if gt.action is not None:
        canonical_commit_action(state, frame_id, gt.action, source=GROUND_TRUTH)
    elif policy == "recompute_predicted":
        canonical_commit_action(state, frame_id, denoise_action_frame(state, frame_id))
    else:
        raise MissingGroundTruth("action")

    replay_later_frames_if_requested(...)
```

“替换 geometry 但保留 video”仍需从 frame checkpoint replay video canonical commit，
因为 layer cache 是按完整 frame transaction 原子提交的。第一版优先正确性，后续才
考虑 phase checkpoint 优化。

### 26.3 真正采样 GT 的时序

provider 调用发生在 predicted action 已生成之后。provider 返回的 `frame_id` 可以是：

```text
当前 frame：offline 对齐替换；
下一 frame：online 环境执行 action 后获得下一 observation；
其他显式 frame：replay 系统负责映射。
```

engine 只按返回的 `frame_id` 替换，并校验该 frame 不早于可恢复 checkpoint；不根据
调用发生在哪一帧自动加一或减一。

### 26.4 Cache 删除验收断言

truncate 后必须满足：

```text
所有 MoT layer 不存在 frame >= dirty_frame 的 segment
所有 geometry relation layer 不存在 frame >= dirty_frame 的 segment
所有 semantic derived version 不引用被删除 version
不存在活跃 transaction
materialize 后 key metadata 不含被删 frame/source/version
```

## 27. Training mask 与 rollout mask 的统一

### 27.1 profile 名称

当前 `order_mode="segmented"` 只描述 order，不足以声明 strict geometry。建议设计上
把完整 profile 明确命名为：

```text
segmented_history_strict_geometry_v1
```

checkpoint contract 保存：

```text
profile_name
profile_version
chunk_size
window_size
history_frames
geometry_relation = "strict_frame_history"
x_to_g_relation = "strict_order"
```

这样加载旧 checkpoint 时不会把仅 segmented-order 的模型误认为已经在 strict-G
mask 下训练。

### 27.2 AR 与 consistency

两种 trainer 都在构造模型后安装同一个 distillation policy adapter：

```text
autoregressive trainer -> policy v1
consistency trainer    -> policy v1
EMA student            -> 同一 policy v1
teacher                -> 是否安装由 checkpoint contract 决定
```

teacher/student attention graph 不一致会改变蒸馏目标。若 teacher checkpoint 没有
`strict_geometry_v1` contract，不能静默套用新 policy；实现阶段必须选择并记录：

```text
strict load: 拒绝不匹配 checkpoint；或
explicit migration: 用户明确允许 teacher 在新 policy 下运行。
```

推荐默认 strict load。

### 27.3 可视化测试的角色

`test_visualize_x_metadata_mask.py` 中两份 SGF metadata 继续作为 order/stream/noise
的可视化基线；还需设计一份 strict-G pair mask 可视化或断言，显示：

```text
G_i -> G_i = False
G_i -> G_(i-1) = True（且在 window 内）
NA_i -> G_i = True
NV_i -> G_i = False
```

否则当前 joint 图片只能证明 segmented order，不能证明用户新增的 geometry 限制。

## 28. Result 与 artifact 兼容

现有 `RolloutResult` 字段继续保留：

```text
pred_latents / target_latents
pred_actions / target_actions
pred_geometry_rgb / target_geometry_rgb
action_valid_mask
```

新 engine 内部可以额外返回 diagnostics，但 artifact saver 不依赖它：

```text
continuation_sources per frame/component
replacement_events
cache lengths per layer
order/frame ids
video/action scheduler timesteps
```

`pred_*` 始终来自 `PredictionLog`；`target_*` 始终来自原 batch/provider GT；
用于后续 continuation 的 GT-replaced semantic state 不应覆盖这两者。

## 29. 详细验证矩阵

### 29.1 order/profile

```text
8-frame order 精确等于：
V/G [0,0,0,0,2,4,6,8]
A   [1,1,1,1,3,5,7,9]

长 horizon 中旧 frame order 不随总长度变化
AR/consistency/rollout 使用同一 profile version
```

### 29.2 history prefill

```text
history X block prefill 与 full-sequence policy 的 history 子矩阵相同
逐 frame prefill 的错误实现应有一个反例测试，证明它与同-order双向 mask 不等价
history geometry 每个 group 只读取 earlier group
```

### 29.3 denoise cache

```text
每个 scheduler step 开始时无旧 transaction segment
cond/uncond branch 间无 transaction K/V 泄漏
final denoise K/V 不进入 committed cache
canonical timestep-0 commit 后 NV/CV 或 NA/CA 同时存在
```

### 29.4 visibility

除 16.2 外，增加：

```text
current NV 可以读取 current NV
current NV 不能读取 current CV/G/A
current NA 可以读取 current NA/current committed G
current NA 不能读取 current CA
window 外历史 token 即使物理存在也不可见
其他 transaction_id 的 predicted token 不可见
```

### 29.5 geometry

```text
第一帧空历史不会 NaN，attention delta 为 0
non-joint inter-frame 与 joint G-register 都禁止 current G key
geometry group frame_id 与 S=4 slot id 不混淆
删除 geometry frame 后两套 geometry K/V 都消失
```

### 29.6 replacement

```text
prediction log 在 GT replacement 后保持原值
semantic continuation 改为 GT source/version
替换 video 使同帧 geometry/action 与未来 cache 失效
替换 geometry 使同帧 action 与未来 cache 失效
替换 action 使未来 cache 失效
rollback/replay 后 generator state 可复现
provider frame mapping 不自动 offset
```

### 29.7 import/worktree boundary

```text
AST/rg 证明 distillation.self_rollout 不 import inference.mot_inference
ConsistencyTrainer._run_rollout 不调用 autoregressive_rollout/run_mot_inference
git diff -- wan_va 为空
```

### 29.8 小型端到端

先用 fake adapter/scheduler 验证 engine 状态机，再在可用 GPU 环境做真实模型最小
smoke test：

```text
1 history segment
1 target anchor
1 predicted frame
2-step video
2-step action
optional offline GT replacement
```

CPU 单测不应加载完整 VAE/VGGTO checkpoint；真实模型 smoke test 单独标记，避免普通
test suite 因硬件/权重缺失失败。

## 30. 设计完成后的实现边界

本方案要求后续实现仅修改/新增 `distillation/` 下代码和测试。允许从 `wan_va`
读取已有模型参数、调用已有无状态 embedding/projection/scheduler helper，但不允许：

```text
修改 wan_va 源文件
把 Flash-WAM 加入运行时依赖
调用 inference/mot_inference.py 的 run_mot_inference
用完整窗口 forward 伪装成 incremental cache
把 chunk_size/window/chunk_windows 强制改成 1
```

Flash-WAM 只用于参考 committed/temporary cache 生命周期；本仓库的 segmented order、
NV/CV/NA/CA no-leak、VGGTO geometry 和 GT replay 均由 distillation 自己定义与测试。

## 31. 实现状态

当前方案已在 `distillation/self_rollout/` 落地，训练与推理共用
`segmented_history_strict_geometry_v1` / profile version 1。实现包括：

```text
history block V prefill -> sequential strict-history G -> history A prefill
GT T0 anchor V -> G -> A
T1..Tn incremental video -> geometry -> action
transaction K/V + canonical commit
LATENT / GEOMETRY / ACTION phase checkpoint
predicted cache 删除、GT replacement、version/source 审计
offline/provider GT、valid mask 更新、prediction/continuation 分离
generator rollback/replay 可复现
AR / Consistency / SGF 同 profile checkpoint contract
```

Consistency 和 SGF 的新执行路径均直接调用 `distillation.self_rollout`，不调用
`autoregressive_rollout` 或 `inference.mot_inference.run_mot_inference`。旧
`distillation/rollout.py` 仅保留兼容 artifact saver 和旧接口，不是新训练 rollout
的依赖。

CPU 验证覆盖 cache transaction、history block 反例、visibility、geometry 两套 K/V、
phase 删除、GT continuation、provider frame mapping、stale version、generator 序列、
checkpoint profile、SGF record/replay 和 import boundary。真实模型 smoke 仍按 29.8
的约定只在 CUDA 与对应权重可用时执行，不作为无 GPU 的普通测试套件前置条件。
