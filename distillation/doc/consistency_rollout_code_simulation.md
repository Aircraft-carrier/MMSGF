# consistency.py::rollout 逐行静态模拟记录

> 本文只分析源码，不运行代码、不加载 checkpoint、不执行 VAE 或真实模型。下面的数值是手工示例，不代表 EMA student 的真实预测值。

## 1. 入口与调用链

ConsistencyTrainer.rollout 本身只做三件事：

    batch = self.convert_input_format(batch)
    batch = self._materialize_batch_latents(batch)
    return self._run_rollout(batch, ground_truth_provider=ground_truth_provider)

调用链：

    caller batch
      -> convert_input_format
      -> _materialize_batch_latents
      -> _run_rollout
           -> resolve_ground_truth_provider
           -> transformer = self.method_model.ema_student
           -> spec = mot_spec_from_config(config)
           -> trainer VAE decoder
           -> distillation.self_rollout.self_rollout
                -> history prefill
                -> anchor prefill
                -> video latent -> geometry -> action
                -> optional GT replacement
                -> RolloutResult

所以 rollout 不是训练 loss，也不是 teacher forward，而是用 EMA student 做增量、自回归推理。

## 2. 输入

进入 self_rollout 后至少要求：

| 字段 | 典型形状 | 含义 |
|---|---:|---|
| latents | [B,C,F,V,H_l,W_l] | 视频 latent |
| actions | [B,C_a,F,N,D] | 动作 token |
| geometry_rgb | [B,F,S,V,3,H,W] | geometry RGB 分组，S 通常为 4 |
| action_valid_mask | [B,C_a,F,N,1] | action 有效 mask |
| stream_ids | [B,V] | view 的 stream id |
| text_emb | [1,T,D] 或 [B,T,D] | 文本条件 |

常见可选字段：

- video_latent_valid_mask：[B,F]；缺少时默认为全 True。
- geometry_group_valid_mask：[B,F,S]；缺少时默认为全 True。
- empty_text_emb：若 batch 自带，则覆盖 trainer 传入的空文本 embedding。

如果 batch 没有 cached latents，_materialize_batch_latents 会从 RGB 字段调用训练 VAE 编码；已有 latents 则直接使用。convert_input_format 会递归搬运 tensor 到当前 device。

_run_rollout 传入的核心对象是：

    transformer = self.method_model.ema_student
    spec = mot_spec_from_config(self.config)
    decode_latents_to_rgb_views = self._decode_rollout_latents
    video_num_steps = config.distill.rollout_video_num_steps
    action_num_steps = config.distill.rollout_action_num_steps
    rollout_frames = config.distill.rollout_horizon_frames
    ground_truth_provider = resolved provider
    replacement_policy = config.distill.rollout_replacement_policy

## 3. 输出

RolloutResult 包含：

| 字段 | 含义 |
|---|---|
| pred_latents | history/anchor 保留输入，rollout frame 写入预测 |
| target_latents | 默认是输入副本；GT replacement 时更新 |
| pred_actions | history/anchor 保留输入，rollout frame 写入预测 |
| target_actions | 默认是输入副本；GT replacement 时更新 |
| pred_geometry_rgb | 预测 latent 经 VAE decode 得到的 geometry |
| target_geometry_rgb | 默认是输入 geometry 副本 |
| action_valid_mask | 工作 mask，provider 可更新 |
| chunk_pairs | 新 self-rollout 固定返回 1 |
| chunk_frames | spec.latent_frames_per_action_chunk_per_view |
| diagnostics | source、version、replacement、cache token 等 |

返回长度是：

    horizon = history_latent_frames + rollout_frames + 1

最后的 1 是 anchor，因此输出是 history + anchor + rollout frames。

## 4. 默认时间轴

配置默认值：

    generation_shape.profile_name = segmented_history_strict_geometry_v1
    generation_shape.order_mode   = segmented
    generation_shape.history_frames = 4
    generation_shape.chunk_size = 4
    generation_shape.window_size = 16
    rollout_video_num_steps = 2
    rollout_action_num_steps = 2
    rollout_horizon_frames = 3
    rollout_gt_mode = none
    rollout_replacement_policy = require_ground_truth

所以：

    history_frames = 4
    anchor = 4
    end_frame = 4 + 3 = 7
    horizon = 7 + 1 = 8

时间轴：

    frame:  0   1   2   3 | 4       | 5       6       7
            <--- history --> anchor  <---- EMA student 生成 ---->

frame 4 是已知 anchor，真正采样的是 frame 5、6、7。

## 5. 最小手工模拟约定

为了展示控制流，缩小为：

    B=1, V=1
    history_frames=2
    anchor=2
    rollout_frames=2
    生成 frame=3、4
    输入 frame=0、1、2、3、4、5

手工设定：

    latents[:, :, f]   = f
    actions[:, :, f]   = f
    geometry_rgb[:, f] = f
    所有 valid mask = True

为了只跟踪 frame，假设 adapter 的输出为：

    predict_video(frame_id)  = frame_id
    predict_action(frame_id) = frame_id + 10

这不是实际网络公式，只是占位预测。

## 6. 初始化和 prefill

engine 先检查 rollout_frames、replacement_policy、必要字段、B/F/V 轴、geometry slot 数、anchor 是否存在、末尾 frame 是否越界，以及 generation profile 是否为 segmented profile。

之后创建：

    RolloutState
    GeometryRolloutCache
    MOTIncrementalAdapter
    GeometryIncrementalAdapter
    video scheduler
    action scheduler

RolloutState 保存：

    semantic_frames[frame_id]
    predictions.video / geometry / action
    每个 frame 开始前的完整 checkpoint
    frame 内 LATENT / GEOMETRY / ACTION checkpoint
    随机 generator state

### 6.1 history

对 frame 0、1，依次提交 video、geometry、action。source 都是 HISTORY，version 都是 1：

    commit_video(latents[:, :, :2], frame_ids=[0,1], HISTORY, 1)
    encode_and_commit_geometry(geometry_rgb[:,0:1], frame_id=0, HISTORY, 1)
    encode_and_commit_geometry(geometry_rgb[:,1:2], frame_id=1, HISTORY, 1)
    commit_action(actions[:, :, :2], frame_ids=[0,1], HISTORY, 1)

### 6.2 anchor

anchor 是 frame 2，仍按 video -> geometry -> action 提交，但 source 改为 ANCHOR：

    commit_video(latents[:, :, 2:3], frame_ids=[2], ANCHOR, 1)
    encode_and_commit_geometry(geometry_rgb[:,2:3], frame_id=2, ANCHOR, 1)
    commit_action(actions[:, :, 2:3], frame_ids=[2], ANCHOR, 1)

此时：

| frame | video | geometry | action |
|---:|---|---|---|
| 0 | HISTORY v1 | HISTORY v1 | HISTORY v1 |
| 1 | HISTORY v1 | HISTORY v1 | HISTORY v1 |
| 2 | ANCHOR v1 | ANCHOR v1 | ANCHOR v1 |

## 7. frame 3：video latent

每个新 frame 开始先执行：

    state.save_checkpoint_before(3)

若任一阶段异常，engine 执行 restore_before(3)，整个 frame 回滚，不留下半个 frame 的 cache。

### 7.1 初始化 video 噪声

    reference = latents[:, :, 3:4]
    sample = torch.randn(reference.shape, generator=state.generator)

输入 frame 3 只提供 shape/device/dtype，sample 内容是随机噪声。

### 7.2 两步 video denoise

每个 video timestep：

    conditional = mot_adapter.predict_video(
        sample,
        timestep=timestep,
        frame_id=3,
        stream_ids=stream_ids,
        text_emb=text_emb,
        state=state,
        valid_frames=video_valid_mask[:, 3:4],
    )
    sample = video_scheduler.step(conditional, timestep, sample)

默认 guidance_scale=1.0，只有 conditional 分支。如果不为 1，还会用 empty_text_emb 做 unconditional，然后使用：

    unconditional + guidance_scale * (conditional - unconditional)

按手工约定，两步后：

    predicted_video[3] = 3

随后 commit：

    commit_video(predicted_video, frame_ids=[3], PREDICTED, version=1)
    state.predictions.video[3] = predicted_video.clone()
    state.save_phase_checkpoint(3, GEOMETRY)

## 8. frame 3：geometry

geometry 依赖 latent。当前 frame 会拼接 anchor 到当前预测：

    sequence = [state.frame(i).video_latent for i in range(anchor, frame_id)]
    target_latents = torch.cat([*sequence, predicted_video], dim=2)
    decoded = decode_latents_to_rgb_views(target_latents)

在 frame 3：

    target_latents = [latent_2, predicted_latent_3]

trainer 的 _decode_rollout_latents：

1. [B,C,F,V,H,W] 改为按 view 展开的 [B*V,C,F,H,W]；
2. latent 反归一化：latent * latents_std + latents_mean；
3. 调用 VAE decode；
4. 从 [-1,1] 转为 [0,1]；
5. 恢复 [B,F_sampled,V,3,H,W]。

随后 decoded_rgb_to_geometry_groups 按 vae_temporal_factor 分组。latent frame 数为 L 时，decoded RGB 需要：

    1 + vae_temporal_factor * (L - 1)

输出 geometry 形状：

    [B, L, vae_temporal_factor, V, 3, H, W]

只取最后一个 group 作为当前 geometry。按手工约定：

    predicted_geometry[3] = 3

然后 encode_and_commit，并写入：

    state.predictions.geometry[3] = predicted_geometry.clone()
    state.save_phase_checkpoint(3, ACTION)

## 9. frame 3：action

初始化并施加 valid mask：

    reference = actions[:, :, 3:4]
    sample = torch.randn(reference.shape, generator=state.generator)
    valid = action_valid_mask[:, :, 3:4]
    sample = sample * valid

当前实现要求 action_guidance_scale == 1，不支持 action CFG。每一步：

    prediction = mot_adapter.predict_action(
        sample,
        timestep=timestep,
        frame_id=3,
        text_emb=text_emb,
        state=state,
        valid_mask=valid,
    )
    sample = action_scheduler.step(prediction, timestep, sample)
    sample = sample * valid

按手工约定：

    predicted_action[3] = 13

commit 后：

| frame | video | geometry | action |
|---:|---|---|---|
| 0 | HISTORY v1 | HISTORY v1 | HISTORY v1 |
| 1 | HISTORY v1 | HISTORY v1 | HISTORY v1 |
| 2 | ANCHOR v1 | ANCHOR v1 | ANCHOR v1 |
| 3 | PREDICTED v1 | PREDICTED v1 | PREDICTED v1 |

## 10. frame 4：自回归依赖

frame 4 重复三阶段，但可以读取 frame 3 已经 commit 的状态：

    frame 4 video    读取 history + anchor + frame 3 video K/V
    frame 4 geometry 读取此前 geometry state/K/V
    frame 4 action    读取此前 action K/V

因此不是每个 frame 独立从原始输入开窗口，而是：

    history -> anchor -> predicted frame 3 -> predicted frame 4

按手工约定：

    predicted_video[4]    = 4
    predicted_geometry[4] = 4
    predicted_action[4]   = 14

## 11. 默认无 GT 时的输出

默认 rollout_gt_mode=none，provider 是 None，不会发生 replacement、回滚、version 递增或 target 更新。

循环结束后先复制输入：

    pred_latents = latents[:, :, :horizon].clone()
    pred_actions = actions[:, :, :horizon].clone()
    pred_geometry = geometry_rgb[:, :horizon].clone()

再用 prediction log 覆盖预测 frame。

手工示例最终表：

| frame | pred latent | pred geometry | pred action | source |
|---:|---:|---:|---:|---|
| 0 | 输入 0 | 输入 0 | 输入 0 | HISTORY |
| 1 | 输入 1 | 输入 1 | 输入 1 | HISTORY |
| 2 | 输入 2 | 输入 2 | 输入 2 | ANCHOR |
| 3 | 3 | 3 | 13 | PREDICTED |
| 4 | 4 | 4 | 14 | PREDICTED |

默认情况下：

    pred_*   = history/anchor 使用输入，rollout frame 使用模型预测
    target_* = 使用输入 batch 对应值

默认配置 history=4、rollout=3 时，返回 frame 0..7，预测 frame 5、6、7。

## 12. GT replacement 分支

provider 在当前 frame 的 video、geometry、action 都预测完后调用：

    gt = ground_truth_provider.maybe_get(
        frame_id=frame_id,
        predicted_action=predicted_action,
        state=state.public_view(),
    )

依赖关系：

    video 被替换 -> geometry 失效 -> action 失效
    geometry 被替换 -> action 失效
    action 单独被替换 -> video/geometry 保持

require_ground_truth 要求：

- video replacement 必须同时提供 geometry 和 action；
- geometry replacement 必须同时提供 action。

recompute_predicted 允许在新的 video/geometry commit 后重新生成下游 component。replacement 会更新 target、有效 mask、semantic source/version 和受影响 cache。代码保存/恢复 generator state，避免 replacement 改变后续随机噪声序列。provider 返回未来 frame 时，会先放入 pending_ground_truth，未来 frame 优先消费。

## 13. diagnostics、recorder 和训练周期性 rollout

diagnostics 主要包含：

    sources[frame_id] = {video, geometry, action}
    versions[frame_id] = {video, geometry, action}
    replacements
    profile = segmented_history_strict_geometry_v1
    profile_version = 1
    rollout_frames
    pending_ground_truth_frames
    mot_cache_tokens
    geometry_cache_tokens

无 GT 时，history source 是 history，anchor source 是 anchor，生成 frame source 是 predicted，replacements 为空。

SelfRolloutRecorder 只记录指定 denoise step 的 scheduler 输入 sample/timestep，不改变预测结果，也不是最终 commit 值。

训练期间 _maybe_run_training_rollout 只在：

    rollout_interval > 0
    且 completed_step % rollout_interval == 0

时调用 _run_rollout；之后 rank 0 可能把视频/action artifact 写到 save_root/rollouts/step_XXXXXXXX。artifact 导出不属于 rollout 状态机本身。

## 14. 一页伪代码

    def rollout(batch, provider=None):
        batch = move_tensors_to_device(batch)
        batch = ensure_latents_with_cached_latents_or_vae(batch)
        provider = resolve_gt_provider(mode, batch, provider)
        return self_rollout(
            batch=batch,
            transformer=ema_student,
            spec=mot_spec_from_config(config),
            decode_latents_to_rgb_views=vae_decode,
            video_num_steps=config.distill.rollout_video_num_steps,
            action_num_steps=config.distill.rollout_action_num_steps,
            rollout_frames=config.distill.rollout_horizon_frames,
            ground_truth_provider=provider,
            replacement_policy=config.distill.rollout_replacement_policy,
        )

    def self_rollout(...):
        validate_inputs()
        state = RolloutState()
        commit_history_video_geometry_action()
        commit_anchor_video_geometry_action()
        for frame_id in frames_after_anchor:
            save_full_frame_checkpoint(frame_id)
            video = denoise_video(frame_id)
            commit_video(video, PREDICTED, 1)
            geometry = decode_latent_to_rgb_and_group(video)
            encode_and_commit_geometry(geometry, PREDICTED, 1)
            action = denoise_action(frame_id)
            commit_action(action, PREDICTED, 1)
            optionally_replace_with_ground_truth()
        return assemble_pred_target_and_diagnostics()

## 15. 最重要理解点

1. consistency.py::rollout 是入口包装器，真正循环在 self_rollout/engine.py。
2. 固定顺序是 video latent -> geometry -> action。
3. history 和 anchor 先进入 cache，不是模型生成。
4. 真正生成的是 anchor 之后的连续 frame。
5. 后续 frame 读取此前已 commit 的状态和 K/V，是增量自回归。
6. geometry 来自预测 latent 的 VAE decode 和 temporal grouping。
7. 默认无 GT replacement，target_* 保留输入 batch。
8. version 和 phase checkpoint 处理 stale cache 与异常回滚。
9. rollout 使用 EMA student，不使用 teacher。
10. 该路径不走旧 fixed-window inference，而由 distillation 自己维护增量 cache、attention visibility 和 diagnostics。

