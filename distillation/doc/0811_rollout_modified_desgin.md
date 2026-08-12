# 0811 Rollout 简化设计稿

## 1. 目标

- 把 `distillation/self_rollout` 重命名为 `distillation/pipeline`，只保留 cache / pipeline / recorder 核心文件。
- 去掉本包内的 attention、state、transitions 复杂调度、provider 的 GT 替换、artifacts。
- 新增 `SelfGradientForcingTrainingPipeline`，构造输入对齐参考实现 `Self_Gradient_Forcing/pipeline/self_gradient_forcing_training.py`：
  `denoising_step_list`、`scheduler`、`generator`、`num_frame_per_block`、`per_rank_exit_step`。
- 由 `distillation/model/common/wan_wrapper.py` 承担自回归模型的单独 video / action 生成（直接返回 x0），pipeline 不再调用
  `flow_to_x0(pred, sample, t, scheduler)`，也不再使用 `renoise_x0`，加噪直接用 `scheduler.add_noise`。
- 新增 `BasePipeline`，负责持有 KV cache 并构建历史帧 cache，供 pipeline 及后续新 pipeline 复用。
- 暂时移除一致性蒸馏的 rollout 阶段。
- 不引入 `independent_first_frame`、`denoising_step_list_first_chunk`。

## 2. 新目录

```
distillation/pipeline/
  __init__.py        # 导出 BasePipeline / SelfGradientForcingTrainingPipeline / KVCache / RolloutResult / SelfRolloutRecorder
  base_pipeline.py   # BasePipeline：cache 所有权 + 历史帧提交
  pipeline.py        # SelfGradientForcingTrainingPipeline：逐 block 去噪 rollout + recorder
  cache.py           # 简化后的 append-only + transaction KVCache
  recorder.py        # 基本保留
  result.py          # RolloutResult 保留
```

删除：`attention.py`、`state.py`、`transitions.py`、`provider.py`、`artifacts.py`、`engine.py`。

## 3. 核心设计

### 3.1 因果一致性（为什么可以去掉 attention）

rollout 严格按时间顺序逐 block 提交：先 commit 历史帧，再依次 commit 每个生成 block（block 内 video 先、action 后）。
每次模型调用只对「已提交 cache + 当前 block 自身」做标准 SDPA，未来帧的 K/V 尚未进入 cache，因此因果性自动成立。
去掉 `TokenMetadataBatch` / `indexed_attention` / `build_cache_selection` 后，模型增量 forward 简化为：

```python
# AutoregressiveVAMOTBlock.forward_incremental
cache.append(layer_id, key, value, transaction_id=tx_id)
key_all, value_all = cache.materialize(layer_id, transaction_id=tx_id)
out = F.scaled_dot_product_attention(query, key_all, value_all)
```

### 3.2 cache.py

```python
class KVSegment:            # 不再带 TokenMetadataBatch
    key: Tensor             # [B, L, H, D]
    value: Tensor

class KVCache:
    def new_transaction_id(self) -> int: ...
    def append(self, layer_id, key, value, transaction_id=None): ...
        # transaction_id=None -> 直接进 committed；否则进对应 transaction
    def materialize(self, layer_id, transaction_id=None) -> (K, V):
        # committed[layer] + 可选当前 transaction，按 append 顺序 cat
    def commit(self, transaction_id): ...    # 预测后的 clean KV 转 committed
    def discard(self, transaction_id): ...   # 预测的 noisy KV 丢弃
    def layer_ids(self): ...
    def committed_token_count(self, layer_id): ...
```

去掉 snapshot / restore / truncate_from / assert_committed_frame / 版本校验等 GT 替换专属 API。
transaction 仍保留，用于保证每个去噪步的 noisy KV 不进最终 cache，只有最终 clean 提交。

### 3.3 wan_wrapper.py（新增 video/action 生成入口）

`WanDiffusionWrapper` 增加四个方法，直接封装 `AutoregressiveModelRequest` 与速度转 x0，pipeline 只消费 x0：

```python
def generate_video(self, noisy, timestep, frame_ids, cache, text_emb) -> x0_video:
    pred = self.model(AutoregressiveModelRequest("predict_video", {...}), mode="self_rollout")
    return self._convert_flow_pred_to_x0(self.video_scheduler, pred, noisy, timestep)

def generate_action(self, noisy, timestep, frame_ids, cache, text_emb) -> x0_action:
    pred = self.model(AutoregressiveModelRequest("predict_action", {...}), mode="self_rollout")
    return self._convert_flow_pred_to_x0(self.action_scheduler, pred, noisy, timestep)

def commit_video(self, latents, frame_ids, cache, text_emb): ...   # commit clean K/V
def commit_action(self, actions, frame_ids, cache, text_emb): ...
```

### 3.4 base_pipeline.py

```python
class BasePipeline:
    def __init__(self, generator, num_frame_per_block=3):
        self.generator = generator      # WanDiffusionWrapper（AR 模型 + 双 scheduler）
        self.cache = KVCache()
        self.num_frame_per_block = num_frame_per_block

    def reset_cache(self): self.cache = KVCache()

    def build_history_cache(self, batch, history_frames):
        # video 0..history_frames-1 先整体 commit，再 action 整体 commit
        self.generator.commit_video(batch["latents"][:, :, :history_frames],
                                    frame_ids=range(history_frames), cache=self.cache, ...)
        self.generator.commit_action(batch["actions"][:, :, :history_frames],
                                     frame_ids=range(history_frames), cache=self.cache, ...)
```

### 3.5 pipeline.py — SelfGradientForcingTrainingPipeline

```python
class SelfGradientForcingTrainingPipeline(BasePipeline):
    def __init__(self, denoising_step_list, scheduler, generator,
                 num_frame_per_block=3, per_rank_exit_step=True):
        super().__init__(generator, num_frame_per_block)
        # video/action 各自独立的线性 d 列表（denoising_step_list.video/.action）
        self.video_denoising_step_list = normalize(denoising_step_list.video)
        self.action_denoising_step_list = normalize(denoising_step_list.action)
        self.scheduler = scheduler               # video 侧 FlowMatchScheduler
        self.action_scheduler = generator.action_scheduler  # action 侧独立 warp
        self.per_rank_exit_step = per_rank_exit_step

    def _sample_exit_id(self, num_steps, device):
        if self.per_rank_exit_step:
            return int(torch.randint(num_steps, (), device=device).item())
        # rank0 采样后 dist.broadcast（参考 generate_and_sync_list）

    @torch.no_grad()
    def generate(self, batch, *, rollout_frames, recorder=None,
                 empty_text_emb=None, device=None, **kw):
        self.reset_cache()
        self.build_history_cache(batch, history_frames)
        # 各自 warp、各自采样 exit id
        video_steps = warp_denoisy_progress(self.video_denoising_step_list, self.scheduler)
        action_steps = warp_denoisy_progress(self.action_denoising_step_list, self.action_scheduler)
        video_exit_id = self._sample_exit_id(len(video_steps), device)
        action_exit_id = self._sample_exit_id(len(action_steps), device)
        for start in range(anchor + 1, end, num_frame_per_block):
            frame_ids = range(start, min(start + num_frame_per_block, end))
            video = self._sample_video(frame_ids, video_steps, video_exit_id, recorder)
            self.generator.commit_video(video, frame_ids=frame_ids, cache=self.cache, ...)
            action = self._sample_action(frame_ids, action_steps, action_exit_id, recorder)
            self.generator.commit_action(action, frame_ids=frame_ids, cache=self.cache, ...)
        return RolloutResult(...)

    def _sample_video(self, frame_ids, steps, exit_id, recorder):
        sample = randn(...)
        for i, t in enumerate(steps):
            if i == exit_id and recorder is not None:
                recorder.observe("video", frame_id=f, step_index=i, timestep=t, sample=sample)
            x0 = self.generator.generate_video(sample, t, frame_ids, self.cache, text_emb)
            if i + 1 == len(steps):
                sample = x0
            else:
                sample = self.scheduler.add_noise(
                    x0, torch.randn_like(x0), steps[i + 1]
                )  # 不使用 renoise_x0
        return sample
```

模型 op 的 payload 同步简化：`predict_video/commit_video` 等不再传 `state` / `source` / `version`，只传 `cache` 与必要张量。

## 4. autoregressive_mot.py 修改明细

1. 删除顶部 `from distillation.self_rollout.attention import segmented_orders`，以及所有对 `self_rollout.attention` / `self_rollout.state` 的 lazy import。
2. `AutoregressiveStreamInput` 去掉 `metadata` 字段；`AutoregressiveMOTLayerRequest` 把 `state` 换成 `cache`，删除 `stream_id`（`stream.block_kind` 已够用）。
3. `AutoregressiveVAMOTBlock.forward_incremental`：保留调制/QKV/text cross-attn，把
   `indexed_attention(query, key, value, query_valid, key_valid)` 替换成
   `cache.append(layer_id, key, value, tx)` + `F.scaled_dot_product_attention(query, K, V)`。
4. `_video_input` / `_action_input`：删除 `build_token_metadata` 与 `valid_ids/valid_mask` 参数，只返回 hidden / conditioning / rotary / block_kind。
5. `_run_stream`：删除“兼容路径”分支（`attention_module` 整段），统一走 `forward_incremental`。
6. `_run_transaction`：`commit_source` 删除；结束时 `cache.commit(tx)`（commit op）或 `cache.discard(tx)`（predict op）。
7. `predict_video / predict_action / commit_video / commit_action`：`state` → `cache`；删除 `source`、`version_id`、`noise_id`、
   `valid_frames/valid_mask`；transaction id 由 `cache.new_transaction_id()` 分配。
8. 删除 `_assert_clean_commit` / `assert_video_commit` / `assert_action_commit`（版本校验只服务 GT 替换，不再需要）。
9. `forward_autoregressive` 按新 payload 分发；`mode="self_rollout"` 入口名保留，减少调用方改动。

## 5. 一致性蒸馏改动

- `trainer/consistency.py`：删除 `_run_rollout` / `rollout` / `_maybe_run_training_rollout` / `_decode_rollout_latents`，直接使用 base trainer 的 no-op。
- `configs/consistency_distillation.py`：删除 `rollout_interval`、`rollout_video_num_steps`、`rollout_action_num_steps`、`rollout_gt_mode`、`rollout_replacement_policy`、`rollout_masked_attn_backend`。
- `train.py`：删除 `rollout_gt_mode` / `rollout_replacement_policy` 相关 CLI 处理；SGF 的 `denoisy_step_list` 逻辑保留。

## 6. 可扩展性

- 新 pipeline 继承 `BasePipeline` 即可复用历史 cache 与提交逻辑，只替换去噪策略 / recorder / 输出。
- wrapper 统一承担“模型调用 + 速度转 x0”，未来替换 transition（如标准 ODE 推理）或换 backbone 时，pipeline 与模型接口不变。
