# 双向 VA-MOT RoboTwin 闭环评测设计

## 1. 目标与范围

本设计为 `model_architecture=va_mot_v1` 的双向固定窗口模型增加 RoboTwin 闭环评测。评测加载训练 checkpoint，在 RoboTwin 环境中循环接收观测、预测 48 个 EEF 动作、执行动作，并最终统计每个任务和全部任务的成功率。

本阶段只设计评测，不实现代码。离线样本可视化仍由 `inference/mot_chunk_infer.py` 负责；`autoregressive_va_mot_v1` 仍由 `distillation/eval` 负责。双向评测不引入 KV cache，也不兼容自回归 checkpoint。

## 2. 已确认的模型合约

当前 RoboTwin 数据配置对应一个固定的 8-latent-frame 窗口：

| latent frame | 视频 | 动作 |
| --- | --- | --- |
| 0 | history 起点 | 空 |
| 1～3 | history | 最近 48 个已执行动作，每帧 16 个 token |
| 4 | target chunk 的当前观测 clean anchor | 空 |
| 5～7 | 同一 target chunk 的 generation | 待预测的 48 个动作 |

关键 shape 如下：

- 视频 latent：`[B, Cv, 8, 3, Hl, Wl]`，3 个视角的 stream id 固定为 `[1, 0, 2]`。
- 动作：`[B, 20, 8, 16, 1]`。
- `video_latent_valid_mask`：`[B, 8]`。
- `action_loss_mask` 和 `action_valid_mask`：`[B, 20, 8, 16, 1]`。
- 每次推理输出 frame 5～7 的动作，共 `3 × 16 = 48` 个 20D relative action，反归一化后转换为 `[48, 16]` 的 absolute EEF action。

frame 4～7 共同组成一个 target chunk：第一个 latent 是 clean anchor，后三个 latent 是 generation。它们必须在一次完整窗口 forward 中一起参与注意力计算，不能拆成四次逐 frame 生成。

双向推理顺序必须与现有 `run_mot_inference` 一致：先根据 history、anchor 和历史动作生成 target video，再以生成的视频为上下文生成 target action。target 区域不能放入环境未来帧或未来动作。每次 video diffusion forward 都必须重新放回原始 clean anchor，并且 scheduler 只能更新 frame 5～7，不能用 transformer 对 frame 4 的输出覆盖 anchor。

## 3. 总体架构

```text
inference/eval/run_robotwin_eval.sh
        |
        | 设置双向 server 入口，复用任务分片、重试、resume 和 summary 调度
        v
distillation/eval/run_robotwin_eval.sh
        |
        +---------------- policy env: linbotva ----------------+
        |                                                       |
        |  inference.eval.server                                |
        |      -> BidirectionalPolicyService                    |
        |      -> BidirectionalMOTInferencePipeline             |
        |      -> VAMOTTransformer3DModel (va_mot_v1)           |
        |                                                       |
        +-------------------------------------------------------+
        |
        +---------------- RoboTwin env: robotwin2 --------------+
        |                                                       |
        |  distillation/eval/robotwin_client.py                 |
        |      -> RoboTwin task environment                     |
        |      -> episode video / progress / result             |
        |                                                       |
        +-------------------------------------------------------+
```

双向实现复用现有 HTTP wire protocol、RoboTwin client 和 shell orchestrator，不复制这些代码。server 是 task-agnostic 的：模型只加载一次，每个 episode 的 `/v1/reset` 更新 task、instruction、随机种子和在线窗口状态。现有 server 的单 session 锁与 orchestrator 的“一台 server 同时服务一个 client”约束保持不变。

## 4. 建议文件

```text
inference/
├── design.md
├── eval/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── server.py
│   └── run_robotwin_eval.sh
└── tests/
    ├── test_bidirectional_eval_pipeline.py
    └── test_bidirectional_eval_server.py
```

### `inference/eval/pipeline.py`

负责双向模型加载和在线推理。可以直接复用 `distillation.eval.infer_pipeline` 中已经验证过的 `OnlineMOTWindowBuilder`、`StreamingVAECodec` 和 `TextEmbedder`，但不能复用 `AutoregressiveMOTInferencePipeline` 或 `KVCache`。

主要接口：

```python
class BidirectionalMOTInferencePipeline:
    def reset(self, *, task_name: str, instruction: str, seed: int) -> None: ...

    def infer(
        self,
        *,
        observations: list[OnlineObservation],
        executed_actions: list[np.ndarray],
        request_id: int,
        return_video: bool,
    ) -> dict: ...


def load_pipeline(
    *, checkpoint_root, dataset_root, model_root, device, dtype,
    video_num_steps, action_num_steps, guidance_scale,
    video_snr_shift, action_snr_shift,
) -> BidirectionalMOTInferencePipeline: ...
```

### `inference/eval/server.py`

复用 `distillation.eval.server` 的 `make_handler`、请求顺序检查和重复请求幂等逻辑。新增薄的 `BidirectionalPolicyService`，仅把 health 中的 architecture 改为 `va_mot_v1`，并加载双向 pipeline。

server CLI 与现有 orchestrator 保持兼容：

```text
--checkpoint-root --dataset-root --model-root --host --port --device --dtype
--video-num-steps --action-num-steps --guidance-scale
--video-snr-shift --action-snr-shift
```

### `inference/eval/run_robotwin_eval.sh`

这是薄包装，不重新实现调度器。它设置：

```bash
POLICY_SERVER_ENTRYPOINT=inference.eval.server
ROBOTWIN_CLIENT_ENTRYPOINT=distillation/eval/robotwin_client.py
```

然后把原始参数转交给 `distillation/eval/run_robotwin_eval.sh`。clean、random、任务选择、端口、重试、超时、断点续评和 `summary.json` 因而与 AR 评测保持同一合约。

## 5. 在线推理流程

### 5.1 Episode reset

1. client 调用 `/v1/reset`，传入 `session_id`、裸 RoboTwin `task_name`、instruction 和 seed。
2. pipeline 将裸任务名映射到 `mot_config.json` 的归一化统计键。当前 50 个任务都能唯一映射到一个 `task_name + "-"` 前缀键；若匹配数不是 1，reset 直接失败。
3. 使用对应的 `q01/q99` 创建 `OnlineMOTWindowBuilder`。
4. 在线计算 instruction text embedding，并缓存到本 episode。

不新增手写任务映射表，因为当前数据已经提供唯一、可验证的前缀关系。

### 5.2 Action request

1. clone 当前 builder，再追加本次 observations 和 executed actions。只有整次推理成功后才提交 clone，失败重试不会污染历史。
2. 从 ring buffer 构建 13 张 history 图像、当前 anchor、最近 48 个动作和有效性 mask。
3. VAE 编码得到 4 个 history latent 和 1 个 anchor latent。
4. 补齐 3 个 generation latent 占位，组装完整 8-frame batch：
   - video frame 0～3 为 history；frame 4～7 是同一个 target chunk，其中 frame 4 为 clean anchor，frame 5～7 为噪声生成区域；
   - action frame 0～3 为历史条件，frame 4 为零，frame 5～7 为动作生成区域；
   - target `action_loss_mask/action_valid_mask` 为真，history 仅按实际已执行动作设置 valid；
   - early episode 的 padding 由 history valid mask 表示。
5. video diffusion 的每次 forward 都从 clean 模板重建完整输入：
   - frame 0～3 重新放入 clean history；
   - frame 4 重新放入本次观测编码得到的 clean anchor；
   - 只有 frame 5～7 放入当前 diffusion sample；
   - transformer 虽然处理完整 8-frame 窗口，但 scheduler 只对 frame 5～7 做 step。
6. action diffusion 的每次 forward 都重新使用固定的视频上下文：frame 0～3 为 clean history，frame 4 为 clean anchor，frame 5～7 为刚生成的 video latent；只有 action frame 5～7 随 diffusion step 更新。
7. 取 `pred_actions[:, :, 5:8]`，转换为 `[48, 20]`，用当前任务统计反归一化。
8. 以 anchor 的 16D EEF state 作为 48 个 relative action 的共同 reference，转换为 `[48, 16]` absolute EEF actions。
9. 返回与现有 client 相同的响应：

```json
{
  "observation_step": 0,
  "actions": [[0.0, 0.0]],
  "action_type": "ee",
  "predicted_video": null,
  "timings_ms": {
    "window": 0.0,
    "vae_encode": 0.0,
    "video_diffusion": 0.0,
    "action_diffusion": 0.0,
    "total": 0.0
  }
}
```

示例中的 action 数值和长度被省略；真实 `actions` 必须是 `[48, 16]`。

client 顺序执行最多 48 个动作，若任务提前成功或达到 step limit 则立即停止。下一次请求把实际执行的动作和对应新观测一并发回 server，从最新环境状态重新规划。

## 6. 需要对公共双向推理做的小改动

为闭环评测复用 `inference/mot_inference.py`，建议只增加两个可选参数，默认值保持现有离线推理行为：

1. `generator: torch.Generator | None = None`：video 和 action 初始噪声使用 episode seed 与 request id 派生的 generator，确保同一请求可复现。
2. `decode_output: bool = True`：RoboTwin client 当前传 `return_video=False`，此时跳过不参与动作生成的 VAE decode；离线 `mot_chunk_infer.py` 继续使用默认值并保存预测视频。

不为双向模型增加 `prediction_chunks`。固定窗口模型天然一次输出完整 48-action target chunk，AR 的逐 frame 预测参数不适用。

## 7. 模型与配置加载

`load_pipeline` 必须检查：

- `checkpoint_metadata.json` 的 `model_architecture` 等于 `va_mot_v1`；
- `transformer/config.json` 和 `transformer/diffusion_pytorch_model.safetensors` 存在；
- 使用 `VAMOTTransformer3DModel.from_pretrained` 加载 checkpoint，而不是 AR 模型类；
- `mot_config.json` 能导出当前固定窗口 spec：history=4、target=4、action_per_frame=16；
- `action_guidance_scale` 固定为 1，与当前 `run_mot_inference` 合约一致。

在线评测的 `model_root` 必须同时提供 `vae/`、`tokenizer/` 和 `text_encoder/`。因此建议传 robbyant LingBot-VA snapshot 根目录；刚补齐 VAE 的 Wan2.2 根目录仍没有当前 `TextEmbedder` 期待的 `tokenizer/` 与 `text_encoder/` 目录，不能直接作为在线评测 model root。

正式评测默认沿用离线配置：video 25 steps、action 50 steps、guidance scale 5。单 episode 冒烟测试可以显式降低到 2/4 steps，但该结果不能与正式成功率直接比较。

## 8. 并行、恢复与输出

第一版保持每个 GPU 一个 policy server，`PARA_NUM_PER_GPU=1`。双向 5B checkpoint 本身约 22 GB，不在同一 GPU 上默认启动多个 server。多 GPU 评测可通过不同 GPU、端口和不重叠任务列表分别启动，不在第一版引入新的多 GPU 调度抽象。

沿用现有输出布局：

```text
<run_output_dir>/<environment>/<run_date>/
├── summary.json
└── <task_name>/
    ├── client.log
    ├── episode0.mp4
    ├── _progress_clean.json 或 _progress_random.json
    ├── _result_clean.txt 或 _result_random.txt
    └── _timeout_or_failed_clean.txt 或 _timeout_or_failed_random.txt
```

client 每完成一个 episode 原子更新 progress；最终 result 最后一个非空行写数值成功率。orchestrator 从磁盘结果重建 `summary.json`，因此中断后可继续未完成任务。

## 9. 验证计划

### 单元测试

1. early episode 和完整历史下的 8-frame batch shape、mask 与 frame 位置正确。
2. target 输入不含未来真值，预测动作只从 frame 5～7 读取。
3. 48 个 normalized relative actions 能恢复成 `[48, 16]` absolute EEF actions。
4. 当前 50 个裸任务名都唯一映射到 50 个 norm stats key。
5. 相同 episode seed/request id 产生相同噪声，不同 request id 产生不同噪声。
6. server health 返回 `va_mot_v1`，reset/actions 和重复 request 幂等行为与现有协议一致。
7. 双向 server 拒绝 `autoregressive_va_mot_v1` checkpoint。

### 冒烟测试

1. `EVAL_NUM_EPISODES=1 PARA_NUM_PER_GPU=1 TASK_MAX_RETRIES=0` 跑一个 clean 任务。
2. 确认 result、progress、episode video 和 summary 都生成。
3. 原命令重跑，确认已完成任务被跳过。
4. 用同样规模跑一个 random 任务，确认 config resolver 与输出后缀正确。
5. 冒烟通过后恢复 25/50 diffusion steps，再扩大 episode 数和任务数。

## 10. 完成标准

- 一个 `va_mot_v1` checkpoint 能通过 `inference/eval/run_robotwin_eval.sh` 在 RoboTwin 中完成闭环 episode。
- client/server 两个 Python 环境边界明确，不要求 RoboTwin 环境安装策略模型依赖。
- clean 和 random 都产生可恢复的 per-task progress、最终成功率与汇总结果。
- AR 评测入口和行为不变，双向实现不使用 AR 模型类、KV cache 或 `prediction_chunks`。
