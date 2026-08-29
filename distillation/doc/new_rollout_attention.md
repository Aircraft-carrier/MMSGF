# History-only token compaction for rollout attention

## 结论

distillation 的增量 rollout 不再把 history 中 `video_latent_valid_mask=False` 或
`action_valid_mask=False` 的 token 写入 KV cache。实现采用 `B=1` token compaction：
在 embedding 之后、QKV 计算之前物理删除无效 history token，因此后续无 mask
SDPA 根本看不到这些 token。

作用范围刻意限制为 history：

- history video/action：根据 dataset valid mask 压紧后写入 cache；
- anchor video/action：保持原来的完整 dense token；
- target/generated video/action：predict 和 commit 均保持原来的完整 dense token；
- 双向训练路径和 `wan_va/modules/mot_attention.py`：不变。

## 【BasePipeline.build_history_cache：只给 history 传 mask】

```python
# 阶段 1/3：读取 dataset 产生的两个有效性 mask
video_valid = batch.get("video_latent_valid_mask")  # [B,F]
action_valid = batch.get("action_valid_mask")       # [B,Ca,F,N,1]

# 阶段 2/3：只有前 history_frames 带 token_valid_mask
generator.commit_video(
    latents[:, :, :history_frames],
    frame_ids=range(history_frames),
    token_valid_mask=video_valid[:, :history_frames],
)
generator.commit_action(
    actions[:, :, :history_frames],
    frame_ids=range(history_frames),
    token_valid_mask=action_valid[:, :, :history_frames],
)

# 阶段 3/3：anchor 不传 mask，保持原 dense 行为
generator.commit_video(latents[:, :, history_frames : history_frames + 1], ...)
generator.commit_action(actions[:, :, history_frames : history_frames + 1], ...)
```

mask 只沿 history 的两次 `commit_*` 调用传入模型。这样不会改变 anchor/target
的 query 数量、输出 shape、denoising scheduler、recorder 或 loss 对齐关系。

## 【_video_input / _action_input：把 mask 展开到 token 轴】

```python
# 阶段 1/3：video mask 是帧级；每帧展开为 V*H*W 个 patch token
# [1,F] -> [1,F*video_tokens_per_frame]
video_token_valid = (
    video_valid[:, :, None]
    .expand(-1, -1, video_tokens_per_frame)
    .reshape(1, -1)
)

# action 的 Ca=20 是特征轴，不是 token 轴；先归约特征有效性
# [1,Ca,F,N,1] -> [1,F*N]
action_token_valid = action_valid.any(dim=1).reshape(1, -1)

# 阶段 2/3：hidden、conditioning 和 RoPE 使用同一组下标
keep = token_valid[0].nonzero().squeeze(1)
hidden = hidden.index_select(1, keep)
conditioning = conditioning.index_select(1, keep)
rotary = rotary.index_select(1, keep)

# 阶段 3/3：压紧后的序列才进入每一层 _self_qkv 和 cache.append
```

RoPE 先使用原始绝对 `frame_ids` 生成，再执行 `index_select`，所以删除 token
不会把剩余 token 重新编号。若 history mask 全为 true，则直接沿用原 dense
stream；若存在 false，则当前实现明确要求 `B=1`。

## 【AutoregressiveVAMOTBlock：无 mask attention 不变】

```python
# 阶段 1/3：输入已经只包含有效 history token
query, current_key, current_value = self._self_qkv(
    block, hidden, modulation, stream.rotary
)

# 阶段 2/3：cache 中不会出现被删除 token 的 K/V
cache.append(layer_id, current_key, current_value, transaction_id=transaction_id)
key, value = cache.materialize(layer_id, transaction_id=transaction_id)

# 阶段 3/3：仍然使用原来的无 mask SDPA
attention_output = torch.nn.functional.scaled_dot_product_attention(
    query.transpose(1, 2),
    key.transpose(1, 2),
    value.transpose(1, 2),
)
```

这里没有新增 attention mask。causality 仍由 commit 顺序保证；变化仅在于无效
history token 不再生成 Q/K/V，也不再占用永久 KV cache。

## 模拟结果

下面模拟默认配置：`B=1`、8 个 latent frames、3 个相机视角、video latent
空间尺寸 `32x40`、patch size `2x2`，因此每个 video frame 有
`3 * 16 * 20 = 960` 个 token；每个 action frame 有 16 个 token。

模拟 history 为 frame 0..3，假设：

- video history mask 为 `[False, False, True, True]`；
- action history 中 frame 0 无效、frame 1..3 有效；
- anchor frame 4 和 target frame 5..7 按需求不做 compaction。

```text
History video:
  dense tokens   = 4 * 960 = 3840
  retained       = 2 * 960 = 1920
  removed        = 1920

History action:
  dense tokens   = 4 * 16 = 64
  retained       = 3 * 16 = 48
  removed        = 16

History total per transformer layer:
  dense KV       = 3904 tokens
  compact KV     = 1968 tokens
  removed        = 1936 tokens (49.59%)

Anchor and target:
  token count and attention behavior unchanged
```

一个最小索引模拟为：dense token 值 `[0,1,2,3]`，valid mask
`[False,True,False,True]`，则 `keep_indices=[1,3]`，进入 QKV 的内容只有
`[1,3]`。原位置对应的 RoPE 也选择索引 `[1,3]`，而不是重新生成位置
`[0,1]`。

注意：`49.59%` 是这组模拟 mask 下 history KV 长度的减少比例，不是端到端
运行时间或显存下降比例；真实收益取决于每个 sample 的 history valid mask。

## 验证

```text
PYTHONPATH=. pytest -q \
  distillation/tests/test_pipeline_rollout.py \
  distillation/tests/test_wan_diffusion_wrapper.py \
  distillation/tests/test_self_gradient_forcing_dmd.py

35 passed
```

测试覆盖：history/anchor mask 传递边界、`B=1` token 选择、hidden / timestep
conditioning / RoPE 同步压紧，以及原有 rollout 和 wrapper 行为。
