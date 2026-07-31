# Self Gradient Forcing 三阶段迁移设计

> 状态：仅架构设计，不包含实现。
>
> 约束：不修改 `wan_va/train_mot.py`、`wan_va/modules/*`、`wan_va/dataset/*`、
> `inference/*` 等现有代码。后续实现只新增文件。当前文件名沿用仓库已有的
> `degisn.md`，本设计不顺手重命名。

## 1. 设计依据与假设

### 1.1 设计来源

- 当前编辑文件：`wan_va/distillation/degisn.md`。目标实现时，distillation 代码目录放在
  仓库根目录 `distillation/`，与 `wan_va/` 同级；本文档后续也应迁移到
  `distillation/degisn.md`，本轮只修改现有草案内容。
- 目标仓库：`/zsh/code/myWAM`。
- 算法参考：`/zsh/code/Self_Gradient_Forcing`。
- 用户要求的固定三阶段：
  1. 双向模型初始化的自回归训练（Stage 1 AR）。
  2. 一致性蒸馏训练（Stage 2 Consistency）。
  3. DMD 匹配的 Self Gradient Forcing 训练（Stage 3 SGF-DMD）。

### 1.2 已确认的仓库事实

- `wan_va/train_mot.py:MOTTrainer` 已拥有数据加载、VAE 在线编码、V/A/G 输入构造、
  flow-matching loss、FSDP2、梯度累积、有限值检查、DCP checkpoint、日志和评估队列。
- `wan_va/modules/model_3dva_mot.py:ThreeDVAMOTTransformer3DModel` 已拥有同一模型内的
  noisy/clean V/A 双流，以及独立 G stream。
- `wan_va/modules/mot_attention.py` 已实现当前 MoT 的块内双向、块间因果可见性，但它服务于
  现有大 target chunk。新的 AR 训练需要独立的 13-history/anchor/4-latent causal chunk mask；
  mask 规则必须放在 `distillation/pipeline/causal_chunk_mask.py`，不能分散进 model 或 pipeline。
- `wan_va/dataset/mot_dataset.py:MotTrainData` 已返回三阶段需要的 RGB、action、text
  embedding、geometry label 和有效掩码；无需 prompt/latent 专用数据集。
- 当前仓库的旧窗口常量是 `MOT_HISTORY_CHUNKS=1`、`MOT_TARGET_CHUNKS=1`；用户最新目标把
  训练语义改为 13 个 latent history、1 个 latent anchor，再按因果方式展开 4 个 block，
  每个 block 预测 4 个 latent。为保持“只新增文件”，新 layout 由 `distillation/config.py`
  和 `distillation/pipeline/causal_chunk_mask.py` 定义并由 adapter 转成现有模型输入，不修改
  `wan_va/mot_spec.py`。
- `inference/mot_inference.py` 已实现 V 去噪、由生成 RGB 重算 G、再生成 A 的顺序；
  rollout 应复用其数据契约和 scheduler 语义，不复制评估逻辑。
- 参考仓库的张量是 T2V `[B,F,C,H,W]`，本仓库视频是
  `[B,C,F,V,H,W]`，action 是 `[B,20,F,16,1]`，并多出 G stream；参考代码不能直接导入。

### 1.3 “双向模型”的明确解释

本设计把“双向模型自回归训练”解释为：从现有双向/预训练权重初始化同拓扑 MoT，
但训练 mask 不再让一个 anchor 双向预测完整 12-latent video chunk；改为固定
13 个 latent history 作为 context，1 个 latent anchor 作为当前生成锚点，后续按
`4 blocks * 4 latent` 的 causal block 逐段预测。block 内 4 个 latent 可以使用本 block
内部可见性，block 之间严格因果，未来 block 不可见。

Stage 1 因此需要新增 `distillation/model/ar.py` 与 `distillation/pipeline/ar_training.py`，
仍复用 `MOTTrainer` 的数据、VAE、优化器、checkpoint、日志生命周期，但不能再简单等价为
“直接运行当前 MOTTrainer 的旧 target chunk loss”。

如果“双向模型”实际指“所有时间帧完全非因果的独立 teacher backbone”，则需要额外定义
teacher 的全序列 mask 语义；不能把当前 chunk-causal 模型静默当成 fully-bidirectional
teacher。本设计暂不增加这第四种模型。该决定只影响 teacher checkpoint 的来源，不影响下述
目录和 stage 接口。

### 1.4 对原草案的收敛

原草案同时规划了约 50 个文件、独立 registry、dataset、FSDP、causal attention、KV cache、
trainer、evaluator，以及 `stage3_dmd`/`stage3_sgf_dmd` 两个 Stage 3。它们多数重复现有能力。
本设计删除这些重复边界：不新增 dataset、不新增 inference、不新增 registry，不修改旧 attention。
只新增一个 mask 真源文件 `distillation/pipeline/causal_chunk_mask.py` 和一个轻量 model/input adapter，
用于表达新的 causal chunk 训练布局。Stage 3 只有 SGF-DMD 一个正式训练阶段。

原草案曾同时规划 `train_distill_stage3_dmd_4gpu.sh` 与
`train_distill_stage3_sgf_dmd_4gpu.sh`。目标架构只保留当前工作区中的后者；前者不进入实现范围，
避免出现一个未被三阶段 pipeline 消费的第四训练模式。

### 1.5 参考仓库到目标责任的映射

- `pipeline/bidirectional_training.py:BidirectionalTrainingPipeline`：只参考离散 timestep 的
  backward simulation；目标对应 `distillation/pipeline/ar_training.py`，Stage 1 仍复用现有
  `MOTTrainer` 生命周期，不移植参考类。
- `model/naive_consistency.py:NaiveConsistency.generator_loss`：参考
  `teacher(t) -> x_t_next -> student(t)/EMA(t_next)`；目标由
  `distillation/model/consistency.py` 持有模型状态和 loss，由
  `distillation/pipeline/consistency_training.py` 执行三次 forward，并改为 V/A 双模态 mask reduction。
- `pipeline/self_gradient_forcing_training.py:SelfGradientForcingTrainingPipeline`：参考
  “no-grad AR context construction + 有梯度 parallel replay”两遍语义；目标实现位于
  `distillation/pipeline/self_gradient_forcing_training.py`，使用 full-window replay 取代当前模型
  不存在的 KV cache。
- `model/dmd.py:DMD`：参考 real/fake score 差值、gradient normalization、surrogate loss 和
  fake-score denoising；目标由 `distillation/model/dmd.py` 持有 score 模型与交替更新状态，纯张量
  loss 收敛在 `distillation/model/objectives.py`。
- `trainer/distillation.py:Trainer`：只参考 generator/fake-score 交替更新和多模型 checkpoint
  需求；分布式、dataset、VAE、日志和 checkpoint 发布沿用本仓库语义。

## 2. 核心目标

### 2.1 可观察行为

实现完成后应支持：

```bash
# 单独运行一个阶段
bash 1shell/distill/train_distill_stage1_ar_4gpu.sh
bash 1shell/distill/train_distill_stage2_consistency_4gpu.sh
bash 1shell/distill/train_distill_stage3_sgf_dmd_4gpu.sh

# 从 Stage 1 串行运行完整三阶段，自动传递成功 checkpoint
bash 1shell/distill/train_distill_pipeline_4gpu.sh

# 从中断阶段恢复；精确参数由新增 launcher 实现时固定
PIPELINE_RESUME_STAGE=stage2_consistency \
PIPELINE_ROOT=/path/to/run \
bash 1shell/distill/train_distill_pipeline_4gpu.sh
```

每一阶段均生成带 `_SUCCESS` 的 checkpoint。Stage 1 输出初始化 Stage 2；Stage 2 导出的
EMA student 初始化 Stage 3；Stage 1 同时作为 Stage 3 的 frozen real score 和 fake score
初始化。最终 `checkpoint/transformer/` 与现有 `inference.mot_chunk_infer` 完全兼容。

### 2.2 非目标

- 不把参考仓库的 T2V `WanDiffusionWrapper`、text prompt dataset 或 KV cache 搬进来。
- 不修改 MoT attention、模型 forward、现有 dataset 或现有 inference。
- 不支持 framewise/chunkwise 两套重复实现；distillation 首版只有一种新增训练布局：
  `history_latents=13`、`anchor_latents=1`、`causal_blocks=4`、
  `latents_per_causal_block=4`。
- 不新增 GAN、SiD、ODE regression 或独立 `dmd-only` 训练阶段。
- 不在本次设计中决定实验超参的最终数值；只定义必须存在且有真实消费者的参数。

### 2.3 验收标准

1. Stage 1 的输入窗口严格按 13-history + 1-anchor + 4x4 causal block 组织；video、action、
   VGGT/G token 使用同一组时间 block id 和同一份 causal mask。
2. Stage 2 teacher/EMA 均无梯度，student 有非零有限梯度；同一 `t -> t_next` 使用一致噪声轨迹。
3. Stage 3 pass 1 全程 `no_grad`，pass 2 只对 student 建图；real score 无梯度，fake score 只由
   critic loss 更新。
4. SGF pass 2 的 target-chunk loss 能给历史 clean-stream attention 参数产生非零梯度。
5. Stage 3 Pass 1 除首 observation/action anchor 外不得读取真实 history V/A/G；target 使用的 G
   必须由生成 RGB 重算。
6. V/A loss 使用 adapter 产出的 block-level `video_latent_loss_mask`、`action_loss_mask`；
   无有效元素时返回有限零值。
7. 每阶段可独立 resume；跨阶段只加载模型导出，不错误加载上阶段 optimizer。
8. 完整 pipeline 只消费 `_SUCCESS` checkpoint，失败时停在当前阶段，不启动下游阶段。
9. 最终 student checkpoint 可直接运行现有 MOT inference/evaluation，无转换脚本。

### 2.4 新 AR 时间布局

用户最新要求把原本“13 个 latent 历史 + 1 个 latent anchor + 一次性双向预测 12 个 video latent”
改成小 chunk 的 causal AR 训练。本文档采用如下首版解释：

```python
distill_layout = {
    "history_latents": 13,
    "anchor_latents": 1,
    "causal_blocks": 4,
    "latents_per_causal_block": 4,
    "future_latent_slots": "4 * 4 = 16",
    "total_latent_slots": "13 + 1 + 16 = 30",
    "modalities_using_same_clock": ["video_latent", "action", "vggt_or_geometry_token"],
}
```

含义：

- history 13 个 latent 永远作为过去条件，所有 future block 可见。
- anchor 1 个 latent 是 future rollout 的起点，可被 4 个 causal block 可见。
- 每个 causal block 只预测本 block 的 4 个 latent；不能读后续 block。
- block 之间严格因果：block 2 可读 history/anchor/block 1，不能读 block 3/4。
- action 与 VGGT/G token 不再各自定义时间窗口，必须复用 video latent 的 block 边界。
- 如果最终数据监督仍只提供 12 个 future latent，则通过 `supervised_future_latents=12`
  把最后 4 个 slot 的 loss mask 置 False；mask/layout 仍保持 4x4，避免改模型或 stage 逻辑。

## 3. 目标目录结构

以下只列本设计相关文件。`[EXISTING CONTEXT]` 仅表示复用，不修改。

```text
myWAM/
├── 1shell/
│   ├── train_mot_mixed_4gpu.sh                         [EXISTING CONTEXT] 环境与 torchrun 参数参考
│   └── distill/
│       ├── _train_distill_common.sh                    [ADD][SCRIPT] 三个单阶段 launcher 的公共 torchrun 逻辑
│       ├── train_distill_stage1_ar_4gpu.sh             [ADD][SCRIPT] 启动 Stage 1
│       ├── train_distill_stage2_consistency_4gpu.sh    [ADD][SCRIPT] 启动 Stage 2
│       ├── train_distill_stage3_sgf_dmd_4gpu.sh        [ADD][SCRIPT] 启动 Stage 3
│       ├── train_distill_pipeline_4gpu.sh              [ADD][SCRIPT] 串行启动三阶段并支持阶段级恢复
│       └── eval_distilled_ar_4gpu.sh                    [ADD][SCRIPT] 调用现有 MOT evaluation
├── inference/
│   ├── mot_inference.py                                [EXISTING CONTEXT] V -> G -> A rollout 契约
│   └── mot_chunk_infer.py                              [EXISTING CONTEXT] 最终 checkpoint 评估入口
├── wan_va/
│   ├── train_mot.py                                    [EXISTING CONTEXT] Stage 1 与公共训练生命周期
│   ├── mot_spec.py                                     [EXISTING CONTEXT] 旧窗口/chunk shape 真源，不修改
│   ├── checkpoint_retention.py                         [EXISTING CONTEXT] 成功 checkpoint 保留策略
│   ├── configs/
│   │   └── va_umi_3dwam_train_cfg.py                   [EXISTING CONTEXT] 数据、模型、优化基础配置
│   ├── dataset/
│   │   └── mot_dataset.py                              [EXISTING CONTEXT] 三阶段共享真实数据
│   ├── modules/
│   │   ├── model_3dva_mot.py                           [EXISTING CONTEXT] 共享 student/teacher/score 拓扑
│   │   ├── mot_attention.py                            [EXISTING CONTEXT] 旧 MoT attention，不写新 AR mask
│   │   └── vggto_loss.py                               [EXISTING CONTEXT] Stage 1 geometry supervision
│   └── utils/
│       └── scheduler.py                                [EXISTING CONTEXT] flow scheduler 与 ODE step
└── distillation/                                       [ADD] 与 wan_va 同级的 SGF 迁移代码包
    ├── degisn.md                                       [ADD][DOC] 目标设计文档迁移位置
    ├── __init__.py                                     [ADD] 包声明，不做 eager import
    ├── config.py                                       [ADD][CONFIG] 基础配置克隆、stage overlay、layout 校验
    ├── contracts.py                                    [ADD] 共享 tensor/result/layout 类型契约
    ├── checkpoint.py                                   [ADD] 多模型/多 optimizer 的原子 DCP checkpoint
    ├── trainer.py                                      [ADD] 复用 MOTTrainer 生命周期的三阶段 distillation trainer
    ├── train.py                                        [ADD] 单阶段 CLI 与 stage dispatch
    ├── orchestrator.py                                 [ADD] 三阶段进程编排和 checkpoint handoff
    ├── model/                                          [ADD] 模型所有权、输入适配与无状态 loss；不做 rollout
    │   ├── __init__.py                                 [ADD] 仅导出三个固定 stage model
    │   ├── adapter.py                                  [ADD] 新 layout 与现有 MoT input_dict 的适配层
    │   ├── factory.py                                  [ADD] teacher/EMA/score 加载、FSDP、冻结与 EMA
    │   ├── objectives.py                               [ADD] AR、consistency、DMD surrogate、fake-score 纯 loss
    │   ├── ar.py                                       [ADD] Stage 1 model，持有 layout/mask 与 AR loss 语义
    │   ├── consistency.py                              [ADD] Stage 2 model，持有 frozen teacher/EMA 与一致性 loss
    │   └── dmd.py                                      [ADD] Stage 3 model，持有 real/fake score、交替状态与 DMD loss
    ├── pipeline/                                       [ADD] 因果时间执行与 rollout/replay；不持有 optimizer/checkpoint
    │   ├── __init__.py                                 [ADD] 仅导出三个训练 pipeline
    │   ├── causal_chunk_mask.py                        [ADD] 13-history/anchor/4x4 causal mask 唯一实现
    │   ├── ar_training.py                              [ADD] Stage 1 causal-window 输入与 forward 执行
    │   ├── consistency_training.py                     [ADD] Stage 2 teacher/student/EMA timestep 执行
    │   └── self_gradient_forcing_training.py           [ADD] Stage 3 no-grad AR rollout 与 gradient replay
    └── tests/
        ├── test_config.py                              [ADD][TEST] 配置覆盖、layout 校验与非法组合
        ├── test_causal_chunk_mask.py                   [ADD][TEST] 13-history/anchor/4x4 mask 可见性
        ├── test_model_adapter.py                       [ADD][TEST] video/action/VGGT 同步 chunk clock
        ├── test_objectives.py                          [ADD][TEST] loss 数学、mask、detach
        ├── test_training_pipeline.py                   [ADD][TEST] AR/consistency/SGF replay 与梯度路径
        ├── test_model_state.py                         [ADD][TEST] 冻结、EMA、optimizer ownership
        ├── test_checkpoint.py                          [ADD][TEST] resume 与跨阶段 handoff
        └── test_pipeline_smoke.py                      [ADD][SMOKE] 三阶段各一步的编排 smoke
```

没有 `registry.py`：只有三个固定 stage，`if/elif` dispatch 更清楚。没有 `types.py + base.py`
双重抽象：共享结构集中在 `contracts.py`。没有新 dataset/FSDP/evaluator：直接复用当前实现。

### 3.1 `model/` 与 `pipeline/` 的依赖边界

该分层直接沿用参考仓库 `Self_Gradient_Forcing/model` 与 `Self_Gradient_Forcing/pipeline` 的职责划分，
但不复制其 T2V 实现：

```text
trainer.py / checkpoint.py / train.py
  -> model/{ar,consistency,dmd}.py       # 模型所有权、loss、EMA/fake optimizer 的更新语义
    -> pipeline/*_training.py            # 时间步采样、causal forward、SGF rollout/replay
      -> pipeline/causal_chunk_mask.py   # 唯一的 13 + 1 + 4x4 可见性真源
    -> model/{adapter,objectives,factory}.py
       # 输入字段映射、纯张量 loss、FSDP 加载/冻结/EMA
```

允许 `model` 调用 `pipeline`，因为 stage model 需要一次训练轨迹来计算目标函数；禁止反向依赖：
`pipeline` 不 import `model.ar/consistency/dmd`，也不创建 optimizer、保存 checkpoint 或调用 backward。
这样 Stage 3 的 SGF replay 可独立测试，且 Stage 1/2/3 不会把训练状态藏进 rollout。

## 4. 三阶段算法与产物

### 4.1 Stage 1：`stage1_ar`

输入：现有 LingBot/VGGTO 初始化源，或显式 `student_init` transformer checkpoint。

执行：`train.py` 构造 `DistillationTrainer + ARModel`。父类 `MOTTrainer` 仍负责数据、VAE 编码、
优化器、日志、checkpoint 生命周期；`ARModel` 通过 `ARTrainingPipeline` 把 batch 适配成新的 causal layout：
13 个 latent history 作为 clean context，1 个 latent anchor 作为 rollout 起点，后续 4 个
causal block 每个预测 4 个 latent。video、action、VGGT/G token 使用同一个
`CausalChunkLayout` 和同一份 attention/loss mask。

复用：`MOTTrainer` 的数据读取、VAE latent materialization、optimizer/scheduler、finite check、
FSDP、DCP checkpoint、日志和 evaluation hook。新增 `model/adapter.py` 负责复用
`_prepare_joint_input_dict()` 的输出契约，但替换为小 chunk causal mask 和 block-level loss mask。

产物：

```text
stage1_ar/checkpoints/checkpoint_step_N/
├── transformer/                 # AR student，可直接推理/跨阶段初始化
├── distributed_state/           # 现有 DCP full state
├── training_state.pt
├── checkpoint_metadata.json
└── _SUCCESS
```

Stage 1 是正式的 `ARModel + ARTrainingPipeline` 组合：model 负责 layout/mask 不变量与 loss，
pipeline 负责按同一 causal chunk clock 调用模型。把这部分散落到 trainer 会导致 action/VGGT 与
video 时间边界不一致。

### 4.2 Stage 2：`stage2_consistency`

模型所有权：

- `student`：从 Stage 1 checkpoint 初始化，可训练；即 `DistillationTrainer.transformer`。
- `teacher`：同一个 Stage 1 checkpoint 初始化，冻结且 `eval()`。
- `ema_student`：同一个 Stage 1 checkpoint 初始化，冻结且 `eval()`；每个成功 student step 后 EMA 更新。
- G branch 默认冻结；geometry 仍作为真实条件输入，不在一致性 loss 中重复监督。

一步算法：

1. 从真实 batch 得到 clean V/A、text、G、valid/loss mask。
2. 对 4 个 causal block 分别采样离散相邻时间 `t > t_next` 和同一份 noise，构造 `x_t`。
3. `no_grad` teacher 在 `t` 预测 flow，并通过 `FlowMatchScheduler.step()` 得到 `x_t_next`。
4. student 在 `x_t,t` 上预测 V/A flow，并转成 `x0_student`。
5. `no_grad` EMA student 在 `x_t_next,t_next` 上预测 `x0_target`。
6. 对 V/A 分别做 masked MSE，再按现有 `video_loss_weight`、`action_loss_weight` 聚合。
7. 只 backward/update student；成功更新后执行 `ema = decay*ema + (1-decay)*student`。

这里不照抄参考仓库的 `[B,F,C,H,W]` reduction；视频先按本仓库 frame/mask 规则归一，
action 按有效 action token 归一，最后才做模态加权。

Stage 2 导出 EMA student 到 `transformer/`，因为它是下游推理和 Stage 3 的稳定生成器；raw
student、EMA、optimizer、scheduler 都保存在 DCP 中用于精确 resume。

### 4.3 Stage 3：`stage3_sgf_dmd`

模型所有权：

- `student/generator`：从 Stage 2 的导出 EMA checkpoint 初始化，可训练。
- `real_score`：从 Stage 1 AR checkpoint 初始化，冻结且 `eval()`。
- `fake_score`：从 Stage 1 AR checkpoint 初始化，可训练，拥有独立 optimizer。
- G branch 冻结；Pass 1 使用生成 RGB 重算 G，Pass 2 使用 detach 后的生成上下文重放。

SGF 两遍 replay：

1. Pass 1 保留真实 13-history 与 1-anchor，不读取任何 future block 的真实 V/A/G。
2. Pass 1（`torch.no_grad()`）按 block 1 -> 4 的顺序因果生成，每次只生成 4 个 latent；
   下一个 block 的 context 来自前面已生成 block，记录随机 exit step 的 noisy V/A、
   对应 clean prediction、生成 context 和时间边界。由此每个后续 block 依赖 self-generated past。
3. 每生成一个视频 chunk，按现有 inference 契约通过 VAE decode 得到 RGB，并用 student 的
   G stream 重算该 chunk/下一 chunk 条件；再生成对应 action。所有结果均 detach。
4. 本模型没有训练 KV cache，且固定训练窗口很短，因此不新增参考仓库的 KV cache。Pass 1
   缓存的是窗口级 V/A/G tensor；内存有上界且语义可测试。
5. Pass 2（有梯度）：把 generated future context detach 后作为 clean stream，把 exit noisy tensor
   作为 noisy stream，一次并行 `forward_train()`；loss 只落在当前 block 的 supervised mask。
   `pipeline/causal_chunk_mask.py` 产出的 block mask 恢复“后续 block loss -> 前序 generated context
   clean-stream attention 参数”的梯度路径。

DMD generator 更新：

1. 从 Pass 2 的 differentiable `x0_student` 采样 score timestep 并加同一份 noise。
2. `no_grad` 计算 `fake_score(x_t)` 和 `real_score(x_t)`。
3. `dmd_grad = (fake_x0 - real_x0) / mean(abs(x0_student-real_x0)).clamp_min(eps)`。
4. 使用 stop-gradient target 构造 surrogate MSE，使 student 获得等价 DMD 梯度。
5. V/A 独立计算、按各自 mask 归一并加权；所有 NaN/Inf 用有限值检查阻止 optimizer step。

Fake-score 更新：

1. 对 detach 的 SGF 生成 V/A 采样 timestep/noise。
2. fake score 预测 flow，目标仍是 `noise - generated_sample`。
3. 只更新 fake-score optimizer；real score 和 student 均不接收该 loss 的梯度。
4. 每 `fake_score_update_ratio` 个 fake-score step 做一次 generator DMD step。所有 rank 执行
   相同 forward 次数，避免 FSDP collective 不一致。

Stage 3 只有这一个正式模式。参考仓库的 `self_gradient_forcing_cache_mode=exit` 与
`self_gradient_forcing_match_context=true` 固化为首版语义，不增加暂时没有消费者的模式开关。

## 5. 文件级设计

### `distillation/config.py` `[ADD][CONFIG]`

责任：从 `VA_CONFIGS["umi_3dwam_train"]` 深拷贝基础 `EasyDict`，增加单个 `distill`
namespace，并执行跨字段校验。它不注册到现有 `wan_va.configs`，因此无需修改 registry。

主要符号：

```python
def build_distillation_config(args: argparse.Namespace) -> EasyDict:
    # 输入：CLI 已解析参数；环境变量已经在基础 config import 时生效。
    # 输出：每次调用独立的 EasyDict，不修改 VA_CONFIGS 中的共享对象。
    # 合并优先级：CLI > stage defaults > existing MOT/env config。

def validate_distillation_config(config: EasyDict) -> None:
    # 校验 stage 必需 checkpoint、互斥 resume/init、递减 denoising steps、
    # timestep 范围、EMA decay、update ratio、固定 CausalChunkLayout。
    # 默认 layout：history_latents=13, anchor_latents=1, causal_blocks=4,
    # latents_per_causal_block=4, supervised_future_latents=16。
```

### `distillation/contracts.py` `[ADD]`

责任：只定义跨 `model/pipeline/checkpoint` 共享的数据结构，避免松散 tuple。

```python
@dataclass(frozen=True)
class CausalChunkLayout:
    # history_latents: int = 13。
    # anchor_latents: int = 1。
    # causal_blocks: int = 4。
    # latents_per_causal_block: int = 4。
    # supervised_future_latents: int = 16；如果只监督旧 12 latent，则最后 4 slot loss mask 为 False。
    # total_latents: 只读派生值，13 + 1 + 4*4。
    # block_id_per_latent: LongTensor[total_latents]，history=-1, anchor=0, future blocks=1..4。
    # 所有 modality 必须共享该 layout。

@dataclass(frozen=True)
class CausalMaskBundle:
    # attention_mask: BoolTensor[T,T] 或模型需要的 broadcast 形状；True 表示可见。
    # loss_mask_future: BoolTensor[T]，仅 supervised future slots 为 True。
    # block_id_per_latent: LongTensor[T]，用于 video/action/VGGT 对齐。
    # modality_time_index: dict[str, LongTensor]，adapter 显式记录各模态 token 到 latent clock 的映射。

@dataclass(frozen=True)
class SGFRolloutResult:
    # video_x0: Tensor[B,C,F,V,H,W]，Pass 1 生成结果，detached。
    # action_x0: Tensor[B,20,F,16,1]，Pass 1 生成结果，detached。
    # video_noisy_at_exit/action_noisy_at_exit: 同 shape，Pass 2 noisy stream。
    # video_context/action_context: 同 shape，Pass 2 clean stream，detached。
    # exit_timesteps: Tensor[B,F]，各 chunk 内相同。
    # score_timestep_min/max: int，限制 DMD timestep。
    # masks: 原 batch 的 video/action valid 和 loss mask；只读。

@dataclass
class StageStepResult:
    # loss: differentiable scalar；仅当前 optimizer 对应参数可达。
    # metrics: dict[str, scalar Tensor]，值均 detach，用于现有 logger。
    # optimizer_name: "student" 或 "fake_score"。
    # optimizer_stepped: backward/finite check 后由 trainer 设置。
```

### `distillation/pipeline/causal_chunk_mask.py` `[ADD]`

责任：13-history/1-anchor/4x4 causal chunk mask 的唯一实现。model、adapter、训练 pipeline 都只能从
这里拿 mask，不能自己重建 block 规则。它只有时间可见性职责，不引用模型、optimizer 或 checkpoint。

```python
def build_causal_chunk_mask(layout: CausalChunkLayout, *, device: torch.device) -> CausalMaskBundle:
    # 输入：layout 固定为 13 history + 1 anchor + 4 causal blocks * 4 latent。
    # 输出：CausalMaskBundle；video/action/VGGT 共享同一 latent clock。
    # 行为：
    # 1. history 与 anchor 对所有 future block 可见。
    # 2. future block k 只能看 history、anchor 和 block <= k。
    # 3. future block k 不能看 block > k。
    # 4. loss_mask_future 只打开 supervised_future_latents 覆盖的 future slot。

def expand_mask_to_modality(
    bundle: CausalMaskBundle, modality_time_index: Tensor, *, token_ndim: int,
) -> Tensor:
    # 输入：每个 modality token 对应的 latent time index。
    # 输出：可 broadcast 到 video/action/VGGT token 的 bool mask。
    # 约束：不同 modality token 数量可以不同，但 time index 必须来自同一 CausalChunkLayout。
```

不在 `model/objectives.py` 或 `model/*.py` 里写 mask helper；这能保证 AR、consistency、SGF-DMD
三阶段不会出现不同的时间可见性。

### `distillation/model/adapter.py` `[ADD]`

责任：把当前 `MOTTrainer` 产出的 batch/input_dict 适配到新 causal layout，同时保持
`ThreeDVAMOTTransformer3DModel.forward_train` 的输入字段兼容。

```python
def build_distill_model_input(
    base_input: dict, layout: CausalChunkLayout, mask_bundle: CausalMaskBundle,
) -> dict:
    # 输入：父类准备好的 V/A/G/text tensors。
    # 输出：现有 model.forward_train 可消费的 input_dict。
    # 行为：
    # 1. 切分 13 history、1 anchor、4 个 4-latent causal block。
    # 2. 为 video latent 写入 block-level timesteps、cond_timesteps、valid/loss mask。
    # 3. 为 action 写入同一 block clock 的 timesteps、cond_timesteps、valid/loss mask。
    # 4. 为 VGGT/G token 写入同一 latent clock 的 visibility，不允许使用未来 block 的真实 token。

def select_supervised_block(input_dict: dict, block_index: int, layout: CausalChunkLayout) -> dict:
    # 输入：完整 causal window 和 1-based block_index。
    # 输出：只监督当前 4-latent block 的 input_dict view/copy。
    # 用于 Stage 1 AR 与 Stage 3 Pass 2 的逐 block loss。
```

adapter 不拥有模型和 optimizer，不做 scheduler step；它只处理 shape、mask 和字段映射。
`pipeline/*_training.py` 调用它构造 forward 输入，`model/*.py` 调用它选择 loss 对应 block。

### `distillation/model/factory.py` `[ADD]`

责任：加载同拓扑额外模型，并复用现有 activation-checkpoint/FSDP 规则。它不包装 student；
student 仍由父类 `MOTTrainer` 构造。

```python
def load_distillation_transformer(
    checkpoint_root: Path, *, device: torch.device, dtype: torch.dtype,
    trainable: bool, execution_route: str,
) -> ThreeDVAMOTTransformer3DModel:
    # 从 checkpoint_root/transformer 严格加载。
    # trainable=False：requires_grad_(False)、eval，不加 activation checkpoint。
    # trainable=True：应用 apply_ac_mot/apply_ac_vggto。
    # 两者均复用 shard_mot_model + _configure_model，保持 FSDP2 边界一致。

@torch.no_grad()
def update_ema_(ema_model: nn.Module, student: nn.Module, decay: float) -> None:
    # 要求两模型的 FSDP shard/name/shape 完全一致。
    # 浮点参数原地 EMA；非浮点 buffer 直接 copy；不得创建 full state dict。

def assert_model_ownership(stage_models: Mapping[str, nn.Module]) -> None:
    # Stage 2/3 构造结束时验证 requires_grad 和 train/eval mode，不在 step 中重复检查。
```

### `distillation/model/objectives.py` `[ADD]`

责任：无模型状态的纯张量计算；所有 reduction、mask 和 detach 在这里形成唯一实现。

```python
def flow_to_x0(xt: Tensor, flow: Tensor, sigma: Tensor) -> Tensor:
    # xt=(1-sigma)*x0+sigma*noise，flow=noise-x0，因此 x0=xt-sigma*flow。
    # sigma broadcast 到视频 [B,1,F,1,1,1] 或 action [B,1,F,1,1]。

def masked_consistency_loss(pred_x0: Tensor, target_x0: Tensor, mask: Tensor) -> Tensor:
    # target 必须 detached；逐样本按有效元素归一，再做 batch mean。
    # mask 全 False 时返回与 pred_x0 同 device/dtype、可 backward 的有限零值。

def compute_dmd_gradient(
    generated_x0: Tensor, real_x0: Tensor, fake_x0: Tensor, *, eps: float,
) -> Tensor:
    # 整体 no_grad；按样本、非 batch 轴计算 normalizer；nan_to_num 后返回 detached gradient。

def dmd_surrogate_loss(generated_x0: Tensor, dmd_gradient: Tensor, mask: Tensor) -> Tensor:
    # 0.5*MSE(generated_x0, (generated_x0-dmd_gradient).detach())；按 mask 归一。

def fake_score_flow_loss(
    predicted_flow: Tensor, clean_generated: Tensor, noise: Tensor, mask: Tensor,
) -> Tensor:
    # target=noise-clean_generated，target detached，reduction 与 consistency 相同。
```

### `distillation/pipeline/ar_training.py` `[ADD]`

责任：执行 Stage 1 的 causal-window forward。它从 `causal_chunk_mask.py` 取得唯一 mask，并通过
`model.adapter` 生成与现有 `forward_train()` 兼容的输入；不创建模型、不决定 loss、不持有 optimizer。

```python
class ARTrainingPipeline:
    # Owned state：layout、mask bundle cache；借用 student 和 base batch preparation。
    # Runtime：构造完整 13-history + anchor + 4x4 window，执行一次或按 block 的 forward_train。
    # Return：V/A flow、block-level mask 和 adapter metrics；调用方决定如何归约为 AR loss。

    def forward(self, student: nn.Module, base_input: dict) -> dict[str, Tensor]:
        # 只执行 causal layout/mask/input 适配及 student forward，不调用 backward。
```

### `distillation/pipeline/consistency_training.py` `[ADD]`

责任：执行同一 causal layout 上的 `teacher(t) -> x_t_next -> student(t)/EMA(t_next)` 张量流程。
它借用 teacher/student/EMA 和 scheduler，严格把 teacher、EMA forward 放在 no-grad 中；不保存 EMA、
不更新参数，也不解释 checkpoint 来源。

```python
class ConsistencyTrainingPipeline:
    def forward(
        self, *, student: nn.Module, teacher: nn.Module, ema_student: nn.Module, base_input: dict,
    ) -> dict[str, Tensor]:
        # 返回有梯度的 student x0、detach 的 teacher/EMA target 和共享 V/A loss mask。
```

### `distillation/pipeline/self_gradient_forcing_training.py` `[ADD]`

责任：执行 Stage 3 的两遍 SGF 时序。它持有借用的 student、scheduler、VAE/G 计算器与固定 window
spec，拥有每一步短生命周期 tensor cache；不拥有 real/fake score、optimizer 或 DMD loss。

```python
class SelfGradientForcingTrainingPipeline:
    @torch.no_grad()
    def autoregressive_rollout(self, batch: dict, *, exit_index: int) -> SGFRolloutResult:
        # 依 block 生成，V -> decode RGB -> G -> A；不得读取 future 的真实 V/A/G。

    def replay_with_gradient(self, batch: dict, rollout: SGFRolloutResult) -> dict[str, Tensor]:
        # generated context 全 detach；返回有梯度的 student V/A x0 供 DMD model 计算 loss。
```

首版不实现通用 base rollout、teacher-forcing rollout 或 cache manager；仅保留 SGF 需要的两次调用。

### `distillation/model/ar.py` `[ADD]`

```python
class ARModel:
    # Owned state：layout、ARTrainingPipeline；不拥有 student optimizer。
    # compute_step：调用 pipeline forward，按合法 future mask 归约 V/A flow-matching loss。
```

### `distillation/model/consistency.py` `[ADD]`

```python
class ConsistencyModel:
    # Owned state：frozen teacher、EMA student、ema_decay、ConsistencyTrainingPipeline。
    # Persistence：EMA 放 DCP；teacher 仅记录来源路径/hash；只有 student 可训练。

    def compute_step(self, trainer: "DistillationTrainer", batch: dict) -> StageStepResult:
        # 调用 pipeline，使用 objectives 归约 V/A consistency loss。

    @torch.no_grad()
    def after_student_step(self, student: nn.Module) -> None:
        # 仅在 finite gradient 且 optimizer.step 成功后调用 factory.update_ema_。
```

### `distillation/model/dmd.py` `[ADD]`

```python
class SGFDMDModel:
    # Owned state：real_score(frozen)、fake_score(trainable)、fake_optimizer、update counters、SGF pipeline。
    # Persistence：fake_score/fake_optimizer/counters 放 DCP；real_score 只记录来源。
    # Invariant：一次 microstep 只允许 student 或 fake_score 其中一组参数有梯度。

    def optimizer_for_step(self, optimizer_step: int) -> Literal["student", "fake_score"]:
        # 所有 rank 由同一全局 step 决定，不能使用 rank-local 随机分支。

    def compute_step(self, trainer: "DistillationTrainer", batch: dict) -> StageStepResult:
        # fake_score step：no_grad rollout -> fake_score_flow_loss。
        # student step：SGF Pass 1 -> gradient replay -> real/fake scores no_grad -> DMD surrogate。
```

### `distillation/trainer.py` `[ADD]`

```python
class DistillationTrainer(MOTTrainer):
    # Responsibility：负责 Stage 1/2/3 的公共训练生命周期。
    # Why a class：复用父类 dataset/VAE/FSDP/logging/train loop，同时拥有 stage model、layout 和多模型 checkpoint。
    # Construction：把 resume checkpoint 的 transformer export 作为 parent initialize_from，
    #   暂不让 parent 加载旧单模型 DCP；stage 模型构造后再由 DistillationCheckpointIO 精确恢复。
    # Owned state：student=transformer、parent student optimizer/scheduler、layout、stage_model、checkpoint_io。
    # Inheritance：覆盖 _train_step/save_checkpoint；不覆盖数据加载和现有 train loop。
    # Runtime：每个 rank 具有相同 model-call 次数；沿用父类 finite loss/grad 与累积边界。

    def __init__(self, config: EasyDict) -> None:
        # 先构造父类 student/data runtime，再 build stage，最后可选 restore。

    def _train_step(self, input_dict: dict, *, sync_gradients: bool, loss_scale: float) -> dict:
        # 调用 stage_model.compute_step，backward 当前 optimizer，复用父类 finite/clip/log contract。
        # 返回字段必须兼容 MOTTrainer.train：loss、should_log、data counters、stage metrics。

    def save_checkpoint(self) -> Path:
        # 委托 checkpoint_io 原子保存；成功后仍复用 prune_successful_checkpoints 和 eval queue。
```

禁止通过 monkey-patch 替换 `MOTTrainer._train_step`；显式 subclass 才能让 ownership、resume 和测试
边界可见。父类以下 helper 虽以下划线命名，但在不改旧代码约束下是最小复用点：dataset setup、
`_get_next_batch`、`_materialize_batch_latents`、VAE/text/geometry preparation、finite check、日志。

### `distillation/checkpoint.py` `[ADD]`

```python
class DistillationCheckpointIO:
    # Why a class：一次保存协调多模型、多 optimizer、DCP process group、临时目录和原子发布。
    # Lifecycle：绑定 trainer/stage -> save/load 多次 -> 无外部资源。
    # Compatibility：checkpoint 根目录和 transformer export 延续现有格式。

    def save(self, trainer: DistillationTrainer) -> Path:
        # 临时目录写 DCP + export + metadata；所有 rank 汇总错误；最后写 _SUCCESS 并 rename。

    def load(self, trainer: DistillationTrainer, checkpoint_root: Path) -> None:
        # 必须 stage 相同；恢复所有 trainable/EMA 模型、optimizer、scheduler、step、RNG 和 sampler offset。

def find_latest_successful_checkpoint(stage_root: Path) -> Path:
    # 只接受 checkpoint_step_<N> 且含 _SUCCESS；按数字 N 选择，不按字符串/mtime。
```

Stage-specific DCP state：

```python
stage2_state = {
    "student": "FSDP sharded state",
    "ema_student": "FSDP sharded state",
    "student_optimizer": "optimizer state",
    "lr_scheduler": "scheduler state",
}

stage3_state = {
    "student": "FSDP sharded state",
    "fake_score": "FSDP sharded state",
    "student_optimizer": "optimizer state",
    "fake_score_optimizer": "optimizer state",
    "lr_scheduler": "scheduler state",
}
```

`real_score` 和 Stage 2 `teacher` 都是可由来源 checkpoint 重建的 frozen model，不重复序列化。
metadata 必须记录绝对来源路径、stage、配置快照、global step 和格式版本；来源缺失时 resume 失败。

### `distillation/train.py` `[ADD]`

```python
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    # 必需：--stage、--save-root。
    # 按 stage 必需：--student-init、--teacher-checkpoint、--real-score-checkpoint、--fake-score-init。
    # 可选：--resume-from；与对应 init 参数的语义由 config 校验。

def run(args: argparse.Namespace) -> None:
    # init_distributed -> build config -> seed。
    # stage1_ar/stage2_consistency/stage3_sgf_dmd：DistillationTrainer(config).train()。

def main() -> None:
    # 只做 CLI/error boundary；不包含算法。
```

### `distillation/orchestrator.py` `[ADD]`

责任：非分布式父进程编排三个独立 torchrun job。它不是算法 `pipeline/` 包的一部分；命名为
`orchestrator.py` 是为避免进程级编排与模型内训练 pipeline 混淆。阶段之间必须退出并重启进程，以释放 teacher、
EMA、fake score 和 FSDP process group；不能在同一 Python 进程热切 stage。

```python
def run_pipeline(args: argparse.Namespace) -> None:
    # 读取/创建 pipeline_state.json。
    # 顺序启动 stage launcher；成功后 find_latest_successful_checkpoint。
    # 把 Stage 1 checkpoint 传给 Stage 2 teacher/student 和 Stage 3 real/fake；
    # 把 Stage 2 checkpoint 传给 Stage 3 student。
    # 子进程非零退出或缺少 _SUCCESS 时立即停止，原子记录 failed_stage。
```

`pipeline_state.json` 只由 rank 外父进程写：

```python
{
    "format_version": 1,
    "completed": {
        "stage1_ar": "/abs/.../checkpoint_step_N",
        "stage2_consistency": "/abs/.../checkpoint_step_M"
    },
    "running_stage": "stage3_sgf_dmd",
    "failed_stage": None
}
```

### Shell launchers `[ADD][SCRIPT]`

`_train_distill_common.sh` 复用现有 `train_mot_mixed_4gpu.sh` 的环境检查与 torchrun 约定，接收
一个 stage 名和其 checkpoint 参数；三个单阶段脚本只设置默认 stage/save root 后调用它。
`train_distill_pipeline_4gpu.sh` 启动 `python -m distillation.orchestrator`，不自行用 `ls` 猜
checkpoint。`eval_distilled_ar_4gpu.sh` 调用现有 `python -m inference.mot_chunk_infer`。

## 6. 顶层配置与参数传播

### 6.1 有效根配置

配置机制保持仓库原生方式：`EasyDict`。`build_distillation_config()` 深拷贝
`VA_CONFIGS["umi_3dwam_train"]`，因此所有现有 flat 字段继续生效：模型源、dataset manifest、
MOT shape、优化器、FSDP dtype、日志、checkpoint、evaluation 和 cache 字段均为 `[EXISTING]`。
本设计只新增一个第一层 key：

```python
root_config = {
    # [EXISTING] 当前 va_umi_3dwam_train_cfg 的全部字段。
    # owner/consumer：MOTTrainer、MotTrainData、ThreeDVAMOTTransformer3DModel。
    "...existing_mot_fields...": "unchanged",

    # [ADD] distill: EasyDict
    # source：config.py stage defaults + CLI；consumer：model/pipeline/checkpoint。
    "distill": {
        "stage": "stage1_ar | stage2_consistency | stage3_sgf_dmd",
        "student_init": "Path | None",
        "teacher_checkpoint": "Path | None",
        "real_score_checkpoint": "Path | None",
        "fake_score_init": "Path | None",
        "consistency_num_steps": "int >= 2",
        "consistency_guidance_scale": "float >= 0",
        "ema_decay": "float in [0,1)",
        "denoising_steps": "tuple[int, ...], strictly decreasing",
        "fake_score_update_ratio": "positive int",
        "score_timestep_min": "int in [0,999]",
        "score_timestep_max": "int in [1,1000]",
        "dmd_normalizer_eps": "positive float",
        "layout": {
            "history_latents": 13,
            "anchor_latents": 1,
            "causal_blocks": 4,
            "latents_per_causal_block": 4,
            "supervised_future_latents": 16,
        },
    },
}
```

不加入 YAML/OmegaConf：当前项目没有该配置依赖；为三个固定 stage 引入第二套配置系统没有收益。

### 6.2 关键参数传播

```text
shell env / pipeline checkpoint
  -> train.py argparse
    -> config.build_distillation_config
      -> config.distill.*
        -> DistillationTrainer
          -> ARModel / ConsistencyModel / SGFDMDModel
            -> ARTrainingPipeline / ConsistencyTrainingPipeline / SelfGradientForcingTrainingPipeline
            -> objective 与 optimizer schedule
              -> checkpoint_metadata.json + W&B config
```

- `--resume-from` 恢复同阶段全部 DCP state；优先级最高。
- `--student-init` 只加载 transformer 权重，不加载 optimizer；用于跨阶段 handoff。
- Stage 2 的 `teacher_checkpoint` 默认由 pipeline 显式传 Stage 1；单独启动时必须提供。
- Stage 3 的 `real_score_checkpoint`、`fake_score_init` 必须显式提供；pipeline 均传 Stage 1。
- `distill.layout.*` 是新增 mask/adapter 的真源；CLI 可以覆盖 `supervised_future_latents=12`
  以兼容旧 12-latent 监督，但 `history/anchor/blocks/block_size` 首版固定。
- `optimization_composition` 沿用现有字段：Stage 1/2/3 首版要求 `va`，G/VGGT 作为条件 token
  同步参与 mask，但不单独训练 geometry loss。
- `save_root/resume_from/initialize_from` 继续保留现有根字段，但 distillation CLI 负责归一，禁止
  同时把 root `resume_from` 当作跨阶段 init。

## 7. 调用链设计

### Stage 1

```text
[ADD] stage1 shell
  -> [ADD] distillation.train.run(stage1_ar)
    -> [EXISTING] init_distributed + VA_CONFIGS
    -> [ADD] DistillationTrainer(MOTTrainer).__init__
    -> [ADD] ARModel + ARTrainingPipeline build CausalChunkLayout + CausalMaskBundle
    -> [EXISTING] MOTTrainer.train lifecycle
      -> MotTrainData -> VAE encode -> base input preparation
      -> [ADD] model.adapter builds 13-history + anchor + 4x4 causal input
      -> ThreeDVAMOTTransformer3DModel.forward_train
      -> AR masked V/A flow loss
      -> current optimizer/DCP/log/eval with layout metadata
```

### Stage 2

```text
[ADD] stage2 shell -> [ADD] train.run
  -> [ADD] DistillationTrainer(MOTTrainer) setup student/data
  -> [ADD] ConsistencyModel loads frozen teacher + EMA
  -> existing batch/VAE/G preparation
  -> pipeline.causal_chunk_mask/model.adapter align video/action/VGGT block clock
  -> teacher(x_t,t) [no_grad] -> scheduler.step -> x_t_next
  -> student(x_t,t) [grad] -> x0_student
  -> EMA(x_t_next,t_next) [no_grad] -> x0_target
  -> masked V loss + masked A loss -> student optimizer -> EMA update
  -> [ADD] multi-model DCP; export EMA as transformer/
```

### Stage 3

```text
[ADD] stage3 shell -> [ADD] train.run
  -> DistillationTrainer setup student/data
  -> SGFDMDModel loads frozen real_score + trainable fake_score
  -> rollout Pass 1: 4 causal blocks, each V -> decode/G -> A [no_grad]
  -> replay Pass 2: full-window forward_train [student grad]
  -> real_score/fake_score on noisy generated V/A [no_grad]
  -> DMD surrogate -> student optimizer
  -> alternating step: detached generated V/A -> fake flow loss -> fake optimizer
  -> multi-model DCP; export student as transformer/
```

### Pipeline

```text
[ADD] train_distill_pipeline_4gpu.sh
  -> [ADD] orchestrator.run_pipeline
    -> torchrun Stage 1 -> validate _SUCCESS
    -> torchrun Stage 2(student=stage1, teacher=stage1) -> validate _SUCCESS
    -> torchrun Stage 3(student=stage2, real=stage1, fake=stage1) -> validate _SUCCESS
    -> existing eval launcher (显式请求时才运行)
```

## 8. 关键数据结构与梯度边界

```python
model_input = {
    "latent_dict": {
        "noisy_latents": "Tensor[B,48,F,V,H,W], model dtype/device",
        "latent": "Tensor[B,48,F,V,H,W], clean/replayed context",
        "timesteps": "Tensor[B,F]",
        "cond_timesteps": "Tensor[B,F]",
        "text_emb": "Tensor[B,L,4096]",
        "video_latent_loss_mask": "BoolTensor[B,F]",
        "video_latent_valid_mask": "BoolTensor[B,F]",
    },
    "action_dict": {
        "noisy_latents": "Tensor[B,20,F,16,1]",
        "latent": "Tensor[B,20,F,16,1]",
        "timesteps": "Tensor[B,F]",
        "cond_timesteps": "Tensor[B,F]",
        "action_loss_mask": "BoolTensor[B,20,F,16,1]",
        "action_valid_mask": "BoolTensor[B,20,F,16,1]",
    },
    "geometry_dict": {
        "rgb": "Tensor[B,G,S,V,3,224,224]",
        "slot_valid_mask": "BoolTensor[B,G,S]",
        "stream_ids": "LongTensor[B,V]",
    },
    "chunk_size": "spec.latent_frames_per_action_chunk_per_view",
    "window_size": "spec.attention_window_size",
}
```

梯度规则：

- Stage 2：`student=True`，`teacher=False`，`ema=False`。
- Stage 3 student step：`student=True`，`real=False`，`fake=False`；DMD target detach。
- Stage 3 fake step：`student=False`，`real=False`，`fake=True`；生成 sample detach。
- SGF Pass 1：所有模型调用和 VAE/G 重算均 no-grad。
- SGF Pass 2：输入 context detach，但 student clean-stream/attention 参数参与图；这正是恢复的梯度路径。
- frozen model 使用 `eval()`，student/fake 使用 `train()`；FSDP 包装后不得通过父模块 `.train()`
  意外切回 frozen model。

## 9. 验证计划

1. `distillation/tests/test_config.py`
   - 构造三个最小 CLI namespace。
   - 断言不修改 `VA_CONFIGS` 原对象、覆盖优先级正确、缺 checkpoint/非法 timestep 立即失败。

2. `distillation/tests/test_objectives.py`
   - 小 tensor 手算 `flow_to_x0`、consistency、DMD surrogate、fake loss。
   - 覆盖视频/action broadcast、部分 mask、全 False mask、normalizer 为零和 NaN 输入。

3. `distillation/tests/test_training_pipeline.py`
   - 用记录调用的 tiny fake model/scheduler 验证 chunk 顺序、V -> G -> A、exit tensor。
   - 给真实 history 非 anchor 位置填哨兵值，断言所有 student/G 调用均未读到该哨兵。
   - 断言 Pass 1 无 grad；Pass 2 target loss 对历史 attention 参数梯度非零，对 context tensor 无梯度。

4. `distillation/tests/test_model_state.py`
   - Stage 2 仅 student 可训练，成功 step 后 EMA 数值按公式变化，skipped step 不更新。
   - Stage 3 两种 step 的 optimizer ownership 互斥，所有 rank 的调度由 global step 决定。

5. `distillation/tests/test_checkpoint.py`
   - 临时目录保存/恢复 Stage 2、Stage 3 全状态；校验 step、optimizer、EMA/fake 和 RNG。
   - 跨阶段只读取 `transformer/`；不允许 Stage 2 DCP 直接 resume 到 Stage 3。
   - `find_latest_successful_checkpoint` 忽略临时目录和无 `_SUCCESS` checkpoint。

6. `distillation/tests/test_pipeline_smoke.py`
   - mock subprocess：Stage 1/2/3 checkpoint 传递必须为 `stage1 -> stage2 -> stage3` 指定关系。
   - Stage 2 失败时 Stage 3 不启动；resume 从 Stage 2 时不重跑已完成 Stage 1。

7. 提议的单机 GPU smoke（实现后确认命令）：

```bash
MOT_NUM_STEPS=1 MOT_SAVE_INTERVAL=1 NGPU=1 \
bash 1shell/distill/train_distill_pipeline_4gpu.sh
```

硬件/数据前提与当前训练一致：支持配置 attention backend 的 CUDA GPU、可读 MOT dataset、
LingBot/VGGTO/VGGT 初始化源。通过条件：三个进程均退出 0；V/A component loss 和 total loss
有限；每阶段至少一次 optimizer update；checkpoint 含 `_SUCCESS`；最终 transformer 能完成一次
现有 `mot_chunk_infer`。真实 fixed topology 很大，CPU unit test 只用 fake model，不能替代 GPU smoke。

## 10. 风险与待确认项

1. **术语风险**：如果 Stage 1 的“bidir-to-AR”必须由一个完全非因果 teacher 在线提供 prediction
   matching，而不是“从双向预训练权重初始化后做 AR flow matching”，则 Stage 1 算法要新增 teacher
   forward/objective。需要在编码前确认 teacher checkpoint 和目标公式。

2. **显存风险**：Stage 2 同时持有 student/teacher/EMA，Stage 3 同时持有 student/real/fake，且
   每个都是 30-layer V/A/G 模型。FSDP 分片仍可能超过当前训练显存。首轮 smoke 应先用 `va`
   ownership、batch size 1；若仍 OOM，再单独设计 CPU offload，不能预先加入未经验证的 offload 层。

3. **SGF 与 KV cache 的差异**：参考实现用 causal Wan KV cache，本模型没有等价 cache API。本设计
   使用固定窗口的 no-grad AR rollout + full-window gradient replay，保留 SGF 的关键梯度语义，但吞吐
   不等价。不要为了目录对齐而新增未接入模型的空 KV-cache 模块。

4. **G 条件语义**：生成视频必须 decode 后重算 G，不能继续使用未来真实 RGB 的 G，否则会产生
   条件泄漏。因为现有窗口只有一个 history 和一个 target，Stage 3 还必须从首 anchor 自生成
   history remainder；若继续使用 dataset 的完整真实 history，SGF 仍是 teacher forcing。rollout
   测试必须同时证明真实 history 哨兵不可达、G 来源于 generated RGB。

5. **Stage 3 action DMD**：参考仓库只处理视频。这里把 action 作为独立模态纳入 real/fake score 和
   DMD loss，是机器人 WAM 的必要适配；若实验只希望蒸馏视频，应通过现有
   `optimization_composition=v` 明确关闭 A，而不是引入第二套 Stage 3。

6. **现有私有 helper 依赖**：`DistillationTrainer` 会复用 `MOTTrainer` 的下划线 helper。零修改旧代码
   的代价是这个耦合；若未来允许小范围修改，再把稳定公共 runtime 抽出，但本次不要提前重构。

## 11. 实施顺序

1. 新增 config/contracts、`model/objectives.py`、`pipeline/causal_chunk_mask.py` 及 unit tests；验证纯函数和配置。
2. 新增 `model/adapter.py`、`model/factory.py`、三个 stage model、checkpoint/DistillationTrainer；先完成 Stage 2 一步与 resume。
3. 新增 `pipeline/self_gradient_forcing_training.py` 与 `model/dmd.py`；验证两遍梯度路径和
   fake/student 互斥更新。
4. 新增单阶段 shell、`orchestrator.py` 和 smoke；最后用现有 inference 验证导出格式。

每一步只新增上述目标文件；不得顺手清理或重构现有 `MOTTrainer`、dataset、attention 或 inference。
