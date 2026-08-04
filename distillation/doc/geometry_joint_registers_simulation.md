# `GeometryIncrementalAdapter._run_joint_registers` 工作原理与模拟记录

本文只分析代码，不依赖真实数据、GPU 或模型权重。目标函数位于：

```text
distillation/self_rollout/geometry_cache.py::_run_joint_registers
```

它是 `distillation/self_rollout` 中增量 VGGT geometry 编码的一部分，负责在 VGGT 的 register-attention layer 上，把当前 geometry registers 接入 MOT block 的 geometry stream，并通过 commit-ordered KV cache 查询历史 geometry registers。

一句话概括：

```text
_run_joint_registers 不是重新拼完整 geometry 序列跑 precompute_geometry，
而是把当前 transaction 的 geometry register K/V 临时加入 state.mot_cache，
然后让当前 register query 读取 committed geometry K/V + 当前 transaction K/V，
最后把 attention delta 和 FFN delta 加回 register tokens。
```

---

## 1. 函数签名

```python
def _run_joint_registers(
    self,
    registers: torch.Tensor,
    *,
    layer_id: int,
    frame_ids: torch.Tensor,
    groups: int,
    slots: int,
    views: int,
    slot_valid_mask: torch.Tensor | None,
    transaction_id: int,
    source: CacheSource,
    version_id: int,
    state: RolloutState,
) -> torch.Tensor:
```

---

## 2. 它在完整 geometry encode 中的位置

调用位置在 `_encode_groups_and_commit` 的每层循环中：

```python
for layer_id in range(self.vggto.depth):
    frame_tokens = self.vggto.run_frame_block(
        tokens,
        geometry_state.patch_hw,
        layer_id,
    )
    if layer_id in self.vggto.register_attention_indices:
        registers = frame_tokens[:, :, : self.vggto.patch_start_idx]
        layer_registers[layer_id] = registers.contiguous()
        updated = self._run_joint_registers(...)
        tokens = torch.cat(
            [updated, frame_tokens[:, :, self.vggto.patch_start_idx :]],
            dim=2,
        )
    else:
        ...
```

所以一个 VGGT layer 在 `self_rollout` 里被拆成：

```text
1. self.vggto.run_frame_block(...)
   每张图内部独立处理，得到 frame_tokens

2. 如果当前 layer 是 register_attention_indices：
   取出 frame_tokens 前面的 register tokens
   调用 _run_joint_registers(...)
   用 updated registers 替换原 registers

3. 如果当前 layer 不是 register_attention_indices：
   走 _run_cross_view_block(...)
   再走 _run_relation_attention(...)
```

`_run_joint_registers` 只处理 register-attention layer。普通 same-view inter-frame relation attention 不在这个函数里，而是在 `_run_relation_attention` 里。

---

## 3. 输入张量语义

### 3.1 `registers`

```python
registers: torch.Tensor
```

shape：

```text
[B, images, R, C]
```

其中：

```text
B      = batch size
images = groups * slots * views
R      = register token 数量，也就是 self.vggto.patch_start_idx
C      = hidden dim
```

在真实场景里，如果：

```text
B = 1
groups = 8
slots = 4
views = 2
register_tokens = 16
hidden_dim = 1024
```

那么：

```text
images = 8 * 4 * 2 = 64
registers.shape = [1, 64, 16, 1024]
```

如果是 self rollout 中单帧增量 encode，例如当前只 encode `G5`：

```text
groups = 1
slots = 4
views = 2
register_tokens = 16
hidden_dim = 1024
```

则：

```text
images = 1 * 4 * 2 = 8
registers.shape = [1, 8, 16, 1024]
```

### 3.2 `frame_ids`

```python
frame_ids: torch.Tensor
```

shape：

```text
[groups]
```

语义是当前 transaction 正在 encode 哪些 geometry group。

例如：

```text
history prefill:
  frame_ids = [0, 1, 2, 3]
  groups = 4

anchor:
  frame_ids = [4]
  groups = 1

rollout frame 5:
  frame_ids = [5]
  groups = 1
```

### 3.3 `slots`

每个 geometry group 内的 slot 数量，通常等于 VAE temporal factor。

真实配置里常见：

```text
slots = 4
```

### 3.4 `views`

多视角数量。当前用户要求 view 设置为 2，所以模拟中使用：

```text
views = 2
```

### 3.5 `slot_valid_mask`

```python
slot_valid_mask: torch.Tensor | None
```

如果存在，shape 为：

```text
[B, groups, slots]
```

它表示当前 transaction 中每个 group 的每个 slot 是否有效。

注意它没有 view 维度。函数内部会把它扩展到：

```text
[B, groups, slots, views, registers]
```

也就是说，同一个 group/slot 的所有 view 和所有 register token 共享同一个 valid 状态。

### 3.6 `transaction_id`

当前 encode transaction 的 id。

这个 id 的作用是：

```text
1. 当前 K/V 先 append 到 transaction cache
2. materialize 时拿到 committed K/V + 当前 transaction K/V
3. build_cache_visibility 通过 transaction_id 判断“当前 query 能否读当前 transaction 的 key”
4. encode 成功后，这个 transaction 会被 commit
5. encode 失败时，上层会 restore snapshot
```

### 3.7 `source` 和 `version_id`

用于标记 cache 来源和版本：

```text
source:
  HISTORY / ANCHOR / PREDICTED / GROUND_TRUTH

version_id:
  当前 frame 的 cache 版本
```

它们主要用于：

```text
1. 调试
2. assert cache 是否 stale
3. replacement / rollback 时判断 cache 是否和语义状态一致
```

---

## 4. 函数内部逐行解释

原函数核心逻辑：

```python
batch_size, _images, register_tokens, _channels = registers.shape
geometry_stream = self.model.mot_blocks[layer_id].geometry
```

这里取得当前 layer 对应的 MOT geometry stream。

`layer_id` 必须是 VGGTO register-attention layer。初始化 `GeometryIncrementalAdapter` 时已经检查过：

```python
for layer_id in self.vggto.register_attention_indices:
    if self.model.mot_blocks[int(layer_id)].geometry is None:
        raise TypeError(...)
```

因此这里预期：

```text
self.model.mot_blocks[layer_id].geometry
```

不是 `None`。

---

### 4.1 构造 register rotary

```python
rotary = self._register_rotary(
    batch_size=batch_size,
    frame_ids=frame_ids,
    slots=slots,
    views=views,
    register_tokens=register_tokens,
    device=registers.device,
)
```

`_register_rotary` 的逻辑：

```python
frame_ids = torch.as_tensor(frame_ids, device=device, dtype=torch.float32)
frame = frame_ids[:, None, None, None].expand(
    -1,
    slots,
    views,
    register_tokens,
)
spatial = torch.full_like(frame, -1.0)
grid = torch.stack(
    [frame, spatial, spatial, torch.zeros_like(frame)],
    dim=0,
).reshape(4, -1)
return self.model.rope(grid[None].expand(batch_size, -1, -1))[:, :, None]
```

语义：

```text
register token 没有真实 patch 坐标，所以 spatial 位置用 -1
frame 维度使用真实 frame_id
view 和 register token 被展开成一条 token 序列
最后调用 self.model.rope 得到 rotary embedding
```

如果当前 encode `G5`：

```text
frame_ids = [5]
slots = 4
views = 2
register_tokens = 16
```

则 register rotary 对应的 token 数量是：

```text
1 * 4 * 2 * 16 = 128
```

也就是每个 geometry slot、每个 view、每个 register token 都有一个 rotary 位置。

---

### 4.2 对当前 registers 做 Q/K/V projection

```python
query, current_key, current_value = geometry_stream.qkv_project(
    registers.flatten(1, 2),
    rotary,
)
```

`registers` 原 shape：

```text
[B, images, R, C]
```

其中：

```text
images = groups * slots * views
```

flatten 后：

```text
registers.flatten(1, 2)
=> [B, images * R, C]
=> [B, groups * slots * views * R, C]
```

真实单帧例子：

```text
B = 1
groups = 1
slots = 4
views = 2
R = 16
C = 1024

registers.shape                  = [1, 8, 16, 1024]
registers.flatten(1, 2).shape    = [1, 128, 1024]
query/current_key/current_value  = [1, 128, heads, head_dim]
```

这里的 Q/K/V 是当前 transaction 的 geometry register tokens。

---

### 4.3 构造当前 transaction 的 metadata

```python
metadata = self._joint_metadata(
    batch_size=batch_size,
    frame_ids=frame_ids,
    slots=slots,
    views=views,
    registers=register_tokens,
    slot_valid_mask=slot_valid_mask,
    transaction_id=transaction_id,
    source=source,
    version_id=version_id,
    device=registers.device,
)
```

`_joint_metadata` 会调用 `build_token_metadata`：

```python
return build_token_metadata(
    batch_size=batch_size,
    frame_ids=frame_ids,
    tokens_per_frame=slots * views * registers,
    stream_id=STREAM_GEOMETRY,
    noise_id=NOISE_GEOMETRY,
    history_frames=self.history_frames,
    chunk_size=self.chunk_size,
    device=device,
    valid_ids=valid,
    committed=False,
    transaction_id=transaction_id,
    source_id=int(source),
    version_id=version_id,
)
```

所以当前 joint registers 的 metadata 具有以下属性：

```text
stream_id       = STREAM_GEOMETRY
noise_id        = NOISE_GEOMETRY
committed       = False
transaction_id  = 当前 transaction_id
source_id       = HISTORY / ANCHOR / PREDICTED ...
version_id      = 当前版本
```

`tokens_per_frame` 是：

```text
slots * views * register_tokens
```

如果真实单帧：

```text
4 * 2 * 16 = 128
```

则：

```text
metadata.seq_len = 128
```

如果 history prefill 一次 encode `G0..G3`：

```text
frame_ids = [0, 1, 2, 3]
tokens_per_frame = 128
metadata.seq_len = 4 * 128 = 512
```

---

### 4.4 进入 `_attend_transaction`

```python
attended, visible = self._attend_transaction(
    query=query,
    current_key=current_key,
    current_value=current_value,
    metadata=metadata,
    cache=state.mot_cache,
    layer_id=layer_id,
    transaction_id=transaction_id,
)
```

这是 `_run_joint_registers` 的核心。

注意这里传入的 cache 是：

```python
cache=state.mot_cache
```

不是：

```python
geometry_cache.relation_cache
```

也就是说，joint registers 走的是 MOT block 的统一 cache，因为它对应的是 `self.model.mot_blocks[layer_id].geometry` 这个 geometry stream。

`_attend_transaction` 内部做：

```python
cache.append_transaction(
    layer_id,
    transaction_id,
    KVSegment(current_key, current_value, metadata),
)
key, value, key_meta = cache.materialize(
    layer_id,
    transaction_id=transaction_id,
)
mask = build_cache_visibility(
    metadata,
    key_meta,
    window_size=self.window_size,
)
return incremental_attention(query, key, value, mask), mask.any(dim=-1)
```

分解为三步：

#### 第一步：当前 K/V 加入 transaction cache

```text
state.mot_cache[layer_id].transactions[transaction_id] += current K/V
```

此时 current K/V 还不是 committed。

#### 第二步：materialize 当前可读 K/V

```text
key/value = concat(committed K/V, current transaction K/V)
```

也就是说，如果当前 encode `G6`，而 `G0..G5` 已经 commit，那么：

```text
key/value = [G0 committed, G1 committed, G2 committed, G3 committed,
             G4 committed, G5 committed, G6 current transaction]
```

#### 第三步：构造 rectangular visibility mask

`build_cache_visibility(query_meta, key_meta, ...)` 生成：

```text
[B, Q, K]
```

它不是方阵，因为当前 query 只包含当前 transaction，而 key 包含历史 committed + 当前 transaction。

对 geometry query 来说，可见性规则是：

```python
geometry_relation = (
    (q_stream == STREAM_GEOMETRY)
    & (k_stream == STREAM_GEOMETRY)
    & readable
)
```

其中：

```python
readable = key.committed_ids[:, None, :] | same_transaction
```

所以 geometry register query 可以读：

```text
1. 已 committed 的 geometry key
2. 和自己同一个 transaction 的 geometry key
```

不能读：

```text
1. video key
2. action key
3. 其他尚未 committed 的 transaction key
```

---

### 4.5 把 attention 输出变成 register delta

```python
delta = geometry_stream.attn_delta(attended).reshape_as(registers)
```

`attended` shape 通常是：

```text
[B, groups * slots * views * R, heads, head_dim]
```

`geometry_stream.attn_delta(...)` 把多头 attention 输出投影回 hidden dim：

```text
[B, groups * slots * views * R, C]
```

然后：

```python
reshape_as(registers)
```

恢复成：

```text
[B, groups * slots * views, R, C]
```

也就是和输入 `registers` 相同 shape。

---

### 4.6 对没有可见 key 的 query 清零 delta

```python
visible = visible.reshape(
    batch_size,
    groups * slots * views,
    register_tokens,
)
delta = torch.where(visible[:, :, :, None], delta, torch.zeros_like(delta))
```

`visible` 来自：

```python
mask.any(dim=-1)
```

它表示每个 query token 是否至少有一个可见 key。

正常有效 geometry token 都应该有可见 key；但如果 `slot_valid_mask` 把某些 slot 标成 invalid，这些 query 可能没有可见 key。

这里做的是：

```text
如果某个 register query 没有任何可见 key，
则它对应的 attention delta 置 0。
```

这样 invalid token 不会因为 SDPA fallback 产生伪更新。

---

### 4.7 attention residual + FFN residual

```python
residual = registers + delta
updated = residual + geometry_stream.ffn_delta(residual)
```

这是标准 transformer block 结构的简化形式：

```text
registers
  + attention delta
  + FFN delta
```

输出 `updated` shape 仍然是：

```text
[B, groups * slots * views, R, C]
```

---

### 4.8 invalid slot 恢复原 registers

```python
if slot_valid_mask is not None:
    valid = (
        slot_valid_mask[:, :, :, None]
        .to(device=updated.device, dtype=torch.bool)
        .expand(-1, -1, -1, views)
        .reshape(batch_size, groups * slots * views)
    )
    updated = torch.where(
        valid[:, :, None, None],
        updated,
        registers,
    )
```

这一步是 slot 级保护。

如果某个 geometry slot invalid：

```text
updated = registers
```

也就是说 invalid slot 的 register 不更新，保持进入 `_run_joint_registers` 前的状态。

注意这里 valid 的 shape：

```text
slot_valid_mask: [B, groups, slots]
expand views -> [B, groups, slots, views]
reshape      -> [B, groups * slots * views]
```

再 broadcast 到：

```text
[B, groups * slots * views, R, C]
```

所以同一个 invalid slot 下：

```text
所有 view
所有 register token
```

都会被恢复为原始 `registers`。

---

### 4.9 返回 updated registers

```python
return updated
```

返回 shape：

```text
[B, groups * slots * views, R, C]
```

上层会把它拼回完整 token：

```python
tokens = torch.cat(
    [updated, frame_tokens[:, :, self.vggto.patch_start_idx :]],
    dim=2,
)
```

所以 register-attention layer 的输出是：

```text
updated register tokens + 原 frame_block 后的 patch tokens
```

---

## 5. 和 `_run_relation_attention` 的区别

`_run_joint_registers` 和 `_run_relation_attention` 都是 geometry 增量 attention，但物理 cache 不同。

| 函数 | 处理对象 | 使用的 attention block | 使用的 cache | batch 维语义 |
|---|---|---|---|---|
| `_run_joint_registers` | VGGT register tokens | `mot_blocks[layer_id].geometry` | `state.mot_cache` | `[B, tokens, ...]` |
| `_run_relation_attention` | VGGT full tokens | `vggto.inter_frame_blocks[layer_id]` | `geometry_cache.relation_cache` | `[B * views, tokens, ...]` |

`_run_joint_registers` 是 VGGT 和 MOT joint layer 的连接点：

```text
VGGT register tokens
  -> MOT geometry stream qkv_project
  -> state.mot_cache
  -> build_cache_visibility
  -> incremental_attention
  -> geometry_stream.attn_delta / ffn_delta
  -> updated registers
```

`_run_relation_attention` 是 VGGT 原本 same-view inter-frame attention 的增量 cache 版本：

```text
VGGT full tokens
  -> view-major reshape [B*views, G*S*N, C]
  -> VGGTO relation block qkv
  -> geometry_cache.relation_cache
  -> build_cache_visibility
  -> incremental_attention
  -> updated full tokens
```

---

## 6. 模拟配置

下面用真实 self rollout 语义模拟：

```text
B = 1
history_frames = 4
total groups = 8
slots = 4
views = 2
register_tokens = 16
hidden_dim = 1024
```

阶段：

```text
history:
  G0, G1, G2, G3 一次 transaction

anchor:
  G4 单独 transaction

rollout:
  G5 单独 transaction
  G6 单独 transaction
  G7 单独 transaction
```

每个 group 的 joint register token 数：

```text
tokens_per_group = slots * views * register_tokens
                 = 4 * 2 * 16
                 = 128
```

---

## 7. 模拟记录：history.encode_geometry

### 输入

```text
frame_ids = [0, 1, 2, 3]
groups = 4
slots = 4
views = 2
register_tokens = 16
```

当前 registers：

```text
images = groups * slots * views
       = 4 * 4 * 2
       = 32

registers.shape = [1, 32, 16, 1024]
```

flatten 后：

```text
registers.flatten(1, 2).shape = [1, 512, 1024]
```

当前 transaction metadata：

```text
transaction_id = 0
source = HISTORY
version_id = 1
stream = GEOMETRY
noise = GEOMETRY
committed = False

metadata.seq_len = 4 frames * 128 tokens/frame = 512
```

### cache materialize

history 是第一步，之前没有 committed geometry register K/V。

当前 transaction append 后：

```text
state.mot_cache[layer].committed = []
state.mot_cache[layer].transaction[0] = [G0, G1, G2, G3]
```

materialize：

```text
K/V = [G0(tx0), G1(tx0), G2(tx0), G3(tx0)]
```

mask：

```text
Q = [G0, G1, G2, G3]
K = [G0, G1, G2, G3]
```

因为它们属于同一个 transaction，所以：

```text
same_transaction = True
readable = True
```

可见性：

```text
      K: G0 G1 G2 G3
Q G0     1  1  1  1
Q G1     1  1  1  1
Q G2     1  1  1  1
Q G3     1  1  1  1
```

shape：

```text
query.shape = [1, 512, heads, head_dim]
key.shape   = [1, 512, heads, head_dim]
mask.shape  = [1, 512, 512]
```

### 输出

```text
attended.shape = [1, 512, heads, head_dim]
delta.shape    = [1, 32, 16, 1024]
updated.shape  = [1, 32, 16, 1024]
```

上层 encode 成功后 commit transaction：

```text
state.mot_cache[layer].committed = [G0, G1, G2, G3]
transaction[0] 被清空
```

---

## 8. 模拟记录：anchor.encode_geometry

### 输入

```text
frame_ids = [4]
groups = 1
slots = 4
views = 2
register_tokens = 16
```

当前 registers：

```text
images = 1 * 4 * 2 = 8
registers.shape = [1, 8, 16, 1024]
registers.flatten(1, 2).shape = [1, 128, 1024]
```

metadata：

```text
transaction_id = 1
source = ANCHOR
version_id = 1
metadata.seq_len = 128
```

### cache materialize

进入 `_run_joint_registers` 前，history 已经 committed：

```text
committed = [G0, G1, G2, G3]
current transaction = [G4]
```

materialize：

```text
K/V = [G0(committed), G1(committed), G2(committed), G3(committed), G4(tx1)]
```

可见性：

```text
      K: G0 G1 G2 G3 G4
Q G4     1  1  1  1  1
```

shape：

```text
query.shape = [1, 128, heads, head_dim]
key.shape   = [1, 640, heads, head_dim]
mask.shape  = [1, 128, 640]
```

### 输出

```text
updated.shape = [1, 8, 16, 1024]
```

commit 后：

```text
committed = [G0, G1, G2, G3, G4]
```

---

## 9. 模拟记录：rollout.frame5.encode_geometry

### 输入

```text
frame_ids = [5]
source = PREDICTED
version_id = 1
transaction_id = 2
```

当前 registers：

```text
registers.shape = [1, 8, 16, 1024]
flatten 后      = [1, 128, 1024]
```

### cache materialize

之前 committed：

```text
[G0, G1, G2, G3, G4]
```

当前 transaction：

```text
[G5]
```

materialize：

```text
K/V = [G0(committed), G1(committed), G2(committed), G3(committed),
       G4(committed), G5(tx2)]
```

可见性：

```text
      K: G0 G1 G2 G3 G4 G5
Q G5     1  1  1  1  1  1
```

shape：

```text
query.shape = [1, 128, heads, head_dim]
key.shape   = [1, 768, heads, head_dim]
mask.shape  = [1, 128, 768]
```

commit 后：

```text
committed = [G0, G1, G2, G3, G4, G5]
```

---

## 10. 模拟记录：rollout.frame6.encode_geometry

当前 transaction：

```text
G6(tx3)
```

已有 committed：

```text
G0, G1, G2, G3, G4, G5
```

materialize：

```text
K/V = [G0, G1, G2, G3, G4, G5, G6]
```

可见性：

```text
      K: G0 G1 G2 G3 G4 G5 G6
Q G6     1  1  1  1  1  1  1
```

shape：

```text
query.shape = [1, 128, heads, head_dim]
key.shape   = [1, 896, heads, head_dim]
mask.shape  = [1, 128, 896]
```

commit 后：

```text
committed = [G0, G1, G2, G3, G4, G5, G6]
```

---

## 11. 模拟记录：rollout.frame7.encode_geometry

当前 transaction：

```text
G7(tx4)
```

已有 committed：

```text
G0, G1, G2, G3, G4, G5, G6
```

materialize：

```text
K/V = [G0, G1, G2, G3, G4, G5, G6, G7]
```

可见性：

```text
      K: G0 G1 G2 G3 G4 G5 G6 G7
Q G7     1  1  1  1  1  1  1  1
```

shape：

```text
query.shape = [1, 128, heads, head_dim]
key.shape   = [1, 1024, heads, head_dim]
mask.shape  = [1, 128, 1024]
```

最终 committed：

```text
state.mot_cache[layer] contains joint geometry register K/V for:
G0, G1, G2, G3, G4, G5, G6, G7
```

---

## 12. 视角维度如何处理

`_run_joint_registers` 和 `_run_relation_attention` 的一个重要区别是 view 维度。

### `_run_joint_registers`

输入 registers：

```text
[B, groups * slots * views, R, C]
```

然后 flatten：

```text
[B, groups * slots * views * R, C]
```

所以 view 被放在 token 维里。

这意味着 joint register attention 可以在同一个 batch 内让不同 view 的 register tokens 共同进入 MOT geometry stream。

### `_run_relation_attention`

relation attention 会先做 view-major：

```text
[B, G, S, V, N, C]
-> [B * V, G * S * N, C]
```

所以 relation attention 是 same-view inter-frame，不直接跨 view。

因此：

```text
joint_register:
  view 在 token 维，MOT geometry stream 可以处理所有 view 的 registers

relation:
  view 在 batch 维，每个 view 独立做跨 group/frame attention
```

---

## 13. invalid slot 模拟

假设当前 encode `G5`，slot valid mask 是：

```text
slot_valid_mask for G5 = [1, 1, 0, 1]
```

含义：

```text
slot0 valid
slot1 valid
slot2 invalid
slot3 valid
```

因为 `views = 2`，所以 slot2 对应：

```text
G5 slot2 view0
G5 slot2 view1
```

它们的所有 register tokens 都 invalid。

`_joint_metadata` 会把 valid 展开成：

```text
[B, groups, slots, views, registers]
```

对于 slot2：

```text
valid = False for all views and all register tokens
```

attention 后 `_run_joint_registers` 有两层保护：

### 第一层：没有可见 key 的 query，delta 置 0

```python
delta = torch.where(visible[:, :, :, None], delta, torch.zeros_like(delta))
```

### 第二层：invalid slot 直接恢复原 registers

```python
updated = torch.where(
    valid[:, :, None, None],
    updated,
    registers,
)
```

最终：

```text
valid slot:
  updated = registers + attn_delta + ffn_delta

invalid slot:
  updated = 原 registers
```

---

## 14. 为什么它能实现“target VGGT register 查询 history register”

以 `G6` 为例。

进入 `_run_joint_registers` 时：

```text
query = G6 current registers
current_key/current_value = G6 current registers
```

但 `_attend_transaction` 会：

```text
1. append G6 current K/V 到 transaction cache
2. materialize committed K/V + current transaction K/V
```

此时 key/value 是：

```text
G0 committed
G1 committed
G2 committed
G3 committed
G4 committed
G5 committed
G6 current
```

`build_cache_visibility` 对 geometry query 允许：

```text
committed geometry key
same transaction geometry key
```

所以：

```text
G6 query 可以看 G0..G5 的历史 K/V，也可以看 G6 自己的 K/V。
```

这就是 self_rollout 中 “预测出的 VGGT 能查询历史 VGGT 信息” 在 joint register layer 上的实现。

---

## 15. 与 full `precompute_geometry` 的关系

原版 inference 里常见做法是：

```text
action_geometry_rgb = [GT history, GT anchor, predicted target]
transformer(..., mode="precompute_geometry")
```

也就是完整序列重编码。

`_run_joint_registers` 所在的 self_rollout 不是这样做。

self_rollout 的做法是：

```text
history G0..G3:
  一次性 encode 并 commit K/V

anchor G4:
  当前 G4 query 读取 committed G0..G3 K/V + current G4 K/V
  encode 后 commit G4 K/V

rollout G5:
  当前 G5 query 读取 committed G0..G4 K/V + current G5 K/V
  encode 后 commit G5 K/V

...
```

所以它是增量 KV cache 版本。

attention 层面上，每一步的 K/V 拼接是：

```python
torch.cat([committed_key, current_key], dim=1)
torch.cat([committed_value, current_value], dim=1)
```

但不是 RGB 输入层面的完整拼接重跑。

---

## 16. 完整伪代码

下面是 `_run_joint_registers` 的逻辑等价伪代码：

```python
def _run_joint_registers(registers, layer_id, frame_ids, state, transaction_id):
    # registers: [B, G*S*V, R, C]

    geometry_stream = model.mot_blocks[layer_id].geometry

    # 1. Build rotary for geometry registers.
    rotary = build_register_rotary(
        frame_ids=frame_ids,
        slots=slots,
        views=views,
        register_tokens=R,
    )

    # 2. Current transaction Q/K/V.
    flat_registers = registers.flatten(1, 2)
    query, current_key, current_value = geometry_stream.qkv_project(
        flat_registers,
        rotary,
    )

    # 3. Current transaction metadata.
    metadata = build_geometry_metadata(
        frame_ids=frame_ids,
        tokens_per_frame=slots * views * R,
        transaction_id=transaction_id,
        committed=False,
        source=source,
        version=version_id,
    )

    # 4. Temporarily append current K/V.
    state.mot_cache.append_transaction(
        layer_id,
        transaction_id,
        KVSegment(current_key, current_value, metadata),
    )

    # 5. Read committed K/V plus current transaction K/V.
    key, value, key_meta = state.mot_cache.materialize(
        layer_id,
        transaction_id=transaction_id,
    )

    # 6. Build rectangular visibility.
    mask = build_cache_visibility(metadata, key_meta)

    # 7. Attention.
    attended = incremental_attention(query, key, value, mask)

    # 8. Convert attention output to register delta.
    delta = geometry_stream.attn_delta(attended).reshape_as(registers)

    # 9. Residual + FFN.
    updated = registers + delta
    updated = updated + geometry_stream.ffn_delta(updated)

    # 10. Invalid slots are restored to original registers.
    updated = restore_invalid_slots(updated, registers, slot_valid_mask)

    return updated
```

---

## 17. 关键不变量

`_run_joint_registers` 依赖以下不变量：

```text
1. layer_id 必须是 VGGTO register-attention layer
2. self.model.mot_blocks[layer_id].geometry 不能是 None
3. registers 的 image 维必须等于 groups * slots * views
4. frame_ids.shape 必须等于 [groups]
5. metadata.seq_len 必须等于 groups * slots * views * register_tokens
6. current_key/current_value 的 token 数必须等于 metadata.seq_len
7. state.mot_cache.materialize(...) 返回 committed + current transaction
8. geometry query 只能读 geometry key
9. 当前 transaction 成功后必须被 commit
10. 出错时上层 `_encode_groups_and_commit` 会 restore mot_cache 和 geometry_cache snapshot
```

---

## 18. 最核心结论

`_run_joint_registers` 是 self_rollout 中 VGGT register-attention layer 的增量实现。

它做的事情是：

```text
当前 VGGT geometry registers
  -> flatten 成 token 序列
  -> 通过 MOT geometry stream 生成 Q/K/V
  -> 把当前 K/V 临时加入 state.mot_cache transaction
  -> materialize 出 committed history K/V + current K/V
  -> 用 build_cache_visibility 生成矩形 mask
  -> 做 incremental_attention
  -> 得到 attention delta
  -> residual + FFN
  -> invalid slot 恢复原 registers
  -> 返回 updated registers
```

它实现的因果语义是：

```text
G0..G3:
  history prefill 同 transaction，互相可见

G4:
  查询 committed G0..G3 + current G4

G5:
  查询 committed G0..G4 + current G5

G6:
  查询 committed G0..G5 + current G6

G7:
  查询 committed G0..G6 + current G7
```

因此，在 self_rollout 中，target geometry register 并不是靠完整重编码查询 history，而是靠 `state.mot_cache` 中的 committed geometry register K/V 查询 history。
