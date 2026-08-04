# Distillation linked assets and launch guide

本文记录本仓库为了复用 `/fcy/code/uni3dwam` 中的数据和 checkpoint 所做的软链接，以及 distillation 三阶段的启动方式。

## 已建立的软链接

在本仓库根目录 `/zsh/code/MMSGF` 下新增了这些链接：

| 本仓库路径 | 指向 | 用途 |
| --- | --- | --- |
| `data/unified_mix_subset_0730_train` | `/fcy/code/uni3dwam/data/unified_mix_subset_0730_train` | 源仓库统一训练子集总入口 |
| `data/umi_distill_train` | `/fcy/code/uni3dwam/data/unified_mix_subset_0730_train/umi` | distillation 默认 MOT 数据集根目录 |
| `models/uni3dwam_geometry_only_step60000` | `/fcy/code/uni3dwam/train_logs/umi_subset3k_newmot_geometry_only/0723_001712/checkpoints/checkpoint_step_60000` | 源仓库 geometry-only checkpoint 备选初始化 |
| `models/uni3dwam_video_only_step62000` | `/fcy/code/uni3dwam/train_logs/umi_subset3k_newmot_stage2_video_only/0724_105656/checkpoints/checkpoint_step_62000` | distillation 默认初始 MOT checkpoint |
| `models/uni3dwam_video_from_wan_step20000` | `/fcy/code/uni3dwam/train_logs/umi_subset3k_newmot_stage2_video_from_wan/0727_105900/checkpoints/checkpoint_step_20000` | 源仓库 video-from-wan checkpoint 备选初始化 |
| `train_logs/source_uni3dwam` | `/fcy/code/uni3dwam/train_logs` | 只用于快速查看源仓库历史训练日志和 checkpoint |

默认选择 `models/uni3dwam_video_only_step62000` 作为 distillation pipeline 的 `student_init`，因为它是已完成的视频训练 checkpoint，包含 `_SUCCESS`、`checkpoint_metadata.json`、`training_state.pt` 和 `transformer/` export。

## 修改过的启动脚本

新增公共路径文件：

- `1shell/distill/_distill_paths.sh`

它集中设置默认路径和训练环境变量：

- `MOT_DATASET_ROOT=${REPO_ROOT}/data/umi_distill_train`
- `DISTILL_DEFAULT_STUDENT_INIT=${REPO_ROOT}/models/uni3dwam_video_only_step62000`
- `DISTILL_DEFAULT_GEOMETRY_INIT=${REPO_ROOT}/models/uni3dwam_geometry_only_step60000`
- `DISTILL_DEFAULT_VIDEO_FROM_WAN_INIT=${REPO_ROOT}/models/uni3dwam_video_from_wan_step20000`
- `MOT_POINTCLOUD_SAMPLE_PERIOD=8`
- `MOT_VIDEO_DOWNSAMPLE_RATIO=4`
- `MOT_EVAL_WITH_CPU=0`
- `MOT_OPTIMIZATION_COMPOSITION=v`
- `MOT_MAX_VIEWS_PER_GPU=8`
- `WANDB_MODE=offline`
- `WANDB_PROJECT=umi_distillation`
- `HF_HOME=/zsh/cache/hf_cache`
- `HF_DATASETS_CACHE=/zsh/cache/hf_cache/datasets`
- `HUGGINGFACE_HUB_CACHE=/zsh/cache/hf_cache/hub`
- `TRANSFORMERS_CACHE=/zsh/cache/hf_cache/hub`
- `WAN22_PRETRAINED_MODEL_PATH=/zsh/cache/hf_cache/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c`

如果 `/zsh/cache/hf_cache` 不存在，脚本会 fallback 到 `/fcy/cache/huggingface_cache`。

修改后的脚本：

- `1shell/distill/_train_distill_common.sh`
  - source 公共路径文件。
  - `DISTILL_SAVE_ROOT` 不再必填，默认写到 `train_logs/distill/<method>/<MMDD_HHMMSS>`。
  - 启动前检查数据集 manifest 和传入的 checkpoint export 是否存在。
  - 补充支持这些环境变量透传到 `distillation.train`：
    - `DISTILL_ROLLOUT_HORIZON_FRAMES`
    - `DISTILL_ROLLOUT_GT_MODE`
    - `DISTILL_ROLLOUT_REPLACEMENT_POLICY`
    - `DISTILL_TEACHER_CFG_MIN`
    - `DISTILL_TEACHER_CFG_MAX`
- `1shell/distill/train_distill_autoregressive_training_4gpu.sh`
  - 默认从 `models/uni3dwam_video_only_step62000` 初始化。
- `1shell/distill/train_distill_consistency_distillation_4gpu.sh`
  - 不再错误地默认从源 MOT checkpoint 直接启动。
  - 单独跑时必须设置 `DISTILL_AR_CHECKPOINT`，或显式设置 `DISTILL_STUDENT_INIT` 和 `DISTILL_TEACHER_CHECKPOINT`。
- `1shell/distill/train_distill_self_gradient_forcing_dmd_4gpu.sh`
  - 单独跑时必须设置 `DISTILL_CONSISTENCY_CHECKPOINT` 和 `DISTILL_AR_CHECKPOINT`，或显式设置 `DISTILL_STUDENT_INIT`、`DISTILL_REAL_SCORE_CHECKPOINT`、`DISTILL_FAKE_SCORE_INIT`。
- `1shell/distill/train_distill_pipeline_4gpu.sh`
  - `STUDENT_INIT` 不再必填，默认使用 `models/uni3dwam_video_only_step62000`。
  - 启动前检查默认数据和初始 checkpoint。
- `1shell/distill/eval_distilled_ar_4gpu.sh`
  - 默认数据集改为 `data/umi_distill_train`。
  - `CHECKPOINT` 默认使用 `models/uni3dwam_video_only_step62000`，也可以显式覆盖。

## 推荐启动方式：完整三阶段 pipeline

等 GPU 可用后，从仓库根目录启动：

```bash
cd /zsh/code/MMSGF

NGPU=4 \
MASTER_PORT=29561 \
1shell/distill/train_distill_pipeline_4gpu.sh
```

默认会依次运行：

1. `autoregressive_training`
2. `consistency_distillation`
3. `self_gradient_forcing_dmd`

默认输出目录：

```text
train_logs/distill_pipeline/<MMDD_HHMMSS>/
```

每个阶段完成后，`distillation.workflow` 会读取该阶段 `checkpoints/` 下最新带 `_SUCCESS` 的 checkpoint，并自动传给下一阶段：

- stage1 AR 输出传给 stage2 的 `student_init` 和 `teacher_checkpoint`
- stage2 consistency 输出传给 stage3 的 `student_init`
- stage1 AR 输出传给 stage3 的 `real_score_checkpoint` 和 `fake_score_init`

指定输出目录：

```bash
cd /zsh/code/MMSGF

PIPELINE_ROOT=/zsh/code/MMSGF/train_logs/distill_pipeline/my_run \
NGPU=4 \
MASTER_PORT=29561 \
1shell/distill/train_distill_pipeline_4gpu.sh
```

换初始 checkpoint：

```bash
cd /zsh/code/MMSGF

STUDENT_INIT=/zsh/code/MMSGF/models/uni3dwam_video_from_wan_step20000 \
NGPU=4 \
1shell/distill/train_distill_pipeline_4gpu.sh
```

## 单独启动各阶段

### Stage 1: autoregressive training

```bash
cd /zsh/code/MMSGF

NGPU=4 \
MASTER_PORT=29561 \
DISTILL_SAVE_ROOT=/zsh/code/MMSGF/train_logs/distill/ar_test \
1shell/distill/train_distill_autoregressive_training_4gpu.sh
```

默认 `DISTILL_STUDENT_INIT` 为：

```text
/zsh/code/MMSGF/models/uni3dwam_video_only_step62000
```

如果要换源 checkpoint：

```bash
DISTILL_STUDENT_INIT=/zsh/code/MMSGF/models/uni3dwam_video_from_wan_step20000 \
1shell/distill/train_distill_autoregressive_training_4gpu.sh
```

### Stage 2: consistency distillation

Stage 2 需要 stage1 distillation 产出的 AR checkpoint，不能直接用源 MOT checkpoint。推荐写法：

```bash
cd /zsh/code/MMSGF

DISTILL_AR_CHECKPOINT=/zsh/code/MMSGF/train_logs/distill_pipeline/my_run/autoregressive_training/checkpoints/checkpoint_step_<N> \
DISTILL_SAVE_ROOT=/zsh/code/MMSGF/train_logs/distill/consistency_test \
NGPU=4 \
MASTER_PORT=29561 \
1shell/distill/train_distill_consistency_distillation_4gpu.sh
```

等价显式写法：

```bash
DISTILL_STUDENT_INIT=/path/to/ar/checkpoint_step_<N> \
DISTILL_TEACHER_CHECKPOINT=/path/to/ar/checkpoint_step_<N> \
1shell/distill/train_distill_consistency_distillation_4gpu.sh
```

常用可调项：

```bash
DISTILL_CFG_MIN=2.0
DISTILL_CFG_MAX=10.0
DISTILL_SIGMA_DATA=0.5
DISTILL_ROLLOUT_INTERVAL=100
DISTILL_ROLLOUT_VIDEO_NUM_STEPS=2
DISTILL_ROLLOUT_ACTION_NUM_STEPS=2
DISTILL_ROLLOUT_HORIZON_FRAMES=3
```

### Stage 3: self-gradient-forcing DMD

Stage 3 需要 stage2 consistency checkpoint 作为 student，并需要 stage1 AR checkpoint 作为 real/fake score 初始化：

```bash
cd /zsh/code/MMSGF

DISTILL_CONSISTENCY_CHECKPOINT=/zsh/code/MMSGF/train_logs/distill_pipeline/my_run/consistency_distillation/checkpoints/checkpoint_step_<N> \
DISTILL_AR_CHECKPOINT=/zsh/code/MMSGF/train_logs/distill_pipeline/my_run/autoregressive_training/checkpoints/checkpoint_step_<M> \
DISTILL_SAVE_ROOT=/zsh/code/MMSGF/train_logs/distill/sgf_dmd_test \
NGPU=4 \
MASTER_PORT=29561 \
1shell/distill/train_distill_self_gradient_forcing_dmd_4gpu.sh
```

等价显式写法：

```bash
DISTILL_STUDENT_INIT=/path/to/consistency/checkpoint_step_<N> \
DISTILL_REAL_SCORE_CHECKPOINT=/path/to/ar/checkpoint_step_<M> \
DISTILL_FAKE_SCORE_INIT=/path/to/ar/checkpoint_step_<M> \
1shell/distill/train_distill_self_gradient_forcing_dmd_4gpu.sh
```

常用可调项：

```bash
DISTILL_TEACHER_CFG_MIN=2.0
DISTILL_TEACHER_CFG_MAX=10.0
DISTILL_ROLLOUT_VIDEO_NUM_STEPS=2
DISTILL_ROLLOUT_ACTION_NUM_STEPS=2
DISTILL_ROLLOUT_HORIZON_FRAMES=3
```

## 断点续跑

### Pipeline 断点续跑

`train_distill_pipeline_4gpu.sh` 支持从某个方法的 checkpoint 续跑：

```bash
cd /zsh/code/MMSGF

PIPELINE_ROOT=/zsh/code/MMSGF/train_logs/distill_pipeline/my_run \
PIPELINE_RESUME_METHOD=consistency_distillation \
PIPELINE_RESUME_FROM=/zsh/code/MMSGF/train_logs/distill_pipeline/my_run/consistency_distillation/checkpoints/checkpoint_step_<N> \
NGPU=4 \
MASTER_PORT=29561 \
1shell/distill/train_distill_pipeline_4gpu.sh
```

注意：`PIPELINE_RESUME_METHOD` 和 `PIPELINE_RESUME_FROM` 必须成对设置。

### 单阶段断点续跑

```bash
DISTILL_RESUME_FROM=/path/to/same_method/checkpoint_step_<N> \
DISTILL_SAVE_ROOT=/path/to/original_or_new_save_root \
1shell/distill/train_distill_<method>_4gpu.sh
```

`DISTILL_RESUME_FROM` 必须来自同一个 distillation method。比如 consistency 的 checkpoint 只能用于 consistency 续跑，不能用于 AR 或 SGF DMD。

## 快速检查

检查当前链接是否还有效：

```bash
find data models train_logs -maxdepth 1 -type l -printf '%p -> %l\n'
```

检查 distillation 脚本语法：

```bash
bash -n 1shell/distill/*.sh
```

本机的 `/workspace/cache` 对应 `/fcy/cache`，但 `/zsh/cache` 也有可用的本地 cache。已将 `/fcy/cache/huggingface_cache/hub/models--robbyant--lingbot-va-base` 精确同步到 `/zsh/cache/hf_cache/hub/models--robbyant--lingbot-va-base`，同步后两边都是约 23G。启动脚本现在优先使用 `/zsh/cache/hf_cache`，并把 `WAN22_PRETRAINED_MODEL_PATH` 指到 `/zsh/cache` 下已存在的 lingbot-va-base snapshot；如果 `/zsh/cache/hf_cache` 不存在，再 fallback 到 `/fcy/cache/huggingface_cache`。

由于 distillation 默认从已有 MOT checkpoint 初始化，启动前脚本会先检查数据和 checkpoint export；如果运行期仍需要单独的 Wan2.2 Diffusers 目录或 VGGTO/VGGT 原始权重，请在启动前设置对应环境变量：

```bash
WAN22_DIFFUSERS_MODEL_PATH=/path/to/wan2_2_diffusers \
VGGTO_CHECKPOINT_PATH=/path/to/vggt_omega_1b_512.pt \
VGGT_CHECKPOINT_PATH=/path/to/vggt/model.safetensors \
1shell/distill/train_distill_pipeline_4gpu.sh
```
