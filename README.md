# UMI 3DWA MOT VGGTO Training

## Fixed Memory OOM Issue

Long-running FSDP2 training previously exhausted host memory because training-rank
anonymous memory grew linearly with every step, while oversized caches in persistent
DataLoader workers amplified node-level usage. The residual rank leak matched
[PyTorch #181761](https://github.com/pytorch/pytorch/issues/181761): PyTorch 2.10/2.11
retained `DTensorSpec` objects indefinitely under FSDP2, and `gc.collect()` could not
release them. The growing object graph also made scheduled full-GC pauses progressively
slower.

The production setup now avoids the affected PyTorch path by using PyTorch 2.9.1
(the upstream fix is also available in PyTorch 2.13+), forces AdamW to use
`fused=False`, `foreach=True`, and CPU step state for both fresh and resumed training,
and bounds DataLoader memory with `prefetch_factor=2`, a 256-entry video-decoder cache,
and a 2-entry point-store cache per dataset worker. With these changes, rank memory
returns to the same baseline after scheduled GC, worker memory approaches a bounded
cache plateau, and full-GC time remains stable instead of increasing with training
steps.

This branch is the real-data MOT training path for UMI 3DWA. The dataset and
window protocol still use the MOT fixed-window contract, while the geometry
branch now uses VGGTO: Pi3 model-side geometry is removed and replaced by a
VGGTO geometry tower initialized from a VGGT-Omega checkpoint.

`VGGTO` is the production name used in this repository for the
VGGT-Omega-derived geometry module. The implementation keeps the Pi3-like
geometry idea: no camera token, no camera head/loss, no reference-frame token,
one shared register table for every frame/view, token order
`registers + patches`, and `patch_token_start == num_register_tokens`.

## Current Architecture

Core files:

- `wan_va/modules/model_3dva_mot.py`: LingBot-VA + VGGTO MoT main model.
- `wan_va/modules/vggto_geometry.py`: VGGTO geometry tower, staged layer execution,
  chunk-causal geometry attention, and Omega depth/VGGT DPT point-head entry points.
- `wan_va/modules/vggto_loss.py`: VGGT-style relative depth/point supervision.
- `wan_va/modules/mot_attention.py`: VA/action/G attention visibility rules.
- `wan_va/modules/fa4_attention.py`: FA4 mask helpers for MOT and VGGTO causal
  geometry attention.
- `wan_va/modules/mot_init.py`: LingBot + VGGTO checkpoint initialization.
- `wan_va/train_mot.py`: distributed training, losses, checkpointing, optimizer,
  and FSDP wrapping.
- `wan_va/mot_spec.py`: shared fixed-window shape contract.
- `wan_va/configs/va_umi_3dwam_train_cfg.py`: default UMI 3DWA training config.

The model aligns all 30 LingBot layers one-to-one with a 30-layer VGGTO tower:

- Every layer contains separate video and action experts. Video residuals stay
  3072D; action residuals stay 768D. Both project Q/K/V into a shared
  24-head x 128D carrier for masked MoT attention.
- The 15 even layers `(0, 2, ..., 28)` replace native VGGTO register attention
  with MoT attention over V, A, and unpooled G register tokens.
- The 15 odd layers keep native VGGTO full-token inter-frame attention and an
  independent joint V-A MoT block.
- Action text cross-attention owns its query and output projections while
  sharing the live video-owned text K/V projections and K normalization.

Selected MoT layers consume the native unpooled slot/view/register tokens. VGGTO
patch tokens stay inside the geometry tower and are supervised through the depth
and point losses. The directional dependency is `G_next = f(G)` and
`VA_next = f(VA, G)`: G queries read only G, while V/A queries may read G
under the no-leak mask.

During training, VGGTO inter-frame attention is bidirectional inside each
history/target chunk and causal across chunks. When the causal mask is disabled,
tests cover frame-order permutation equivariance for the VGGTO geometry tower.

## Checkpoints

Initializing this branch from pretrained weights requires these sources:

- `WAN22_PRETRAINED_MODEL_PATH`: LingBot model root, expected to contain `transformer/`.
- `INIT_MODEL_FROM_LINGBOT`: use LingBot for V/A initialization when true
  (default), or original Wan2.2 when false.
- `WAN22_DIFFUSERS_MODEL_PATH`: original Wan2.2 Diffusers model root used when
  `INIT_MODEL_FROM_LINGBOT=false`.
- `VGGTO_CHECKPOINT_PATH`: required VGGTO/VGGT-Omega checkpoint.
- `VGGT_CHECKPOINT_PATH`: required original VGGT checkpoint containing `point_head.*`.

Neither geometry checkpoint has a hardcoded fallback. Config loading fails
immediately when either path is missing.

The video expert loads the selected LingBot or Wan2.2 weights directly. The
768D action backbone and conditioning are initialized from corresponding video
parameters with sequential FP32 linear interpolation and Linear fan-in scaling.
Only the bias-free 20D action encoder and 20D output head are reproducibly random.

Example:

```bash
export WAN22_PRETRAINED_MODEL_PATH=/path/to/lingbot-va-base
export INIT_MODEL_FROM_LINGBOT=false
export WAN22_DIFFUSERS_MODEL_PATH=/workspace/model/wan2_2_diffusers
export VGGTO_CHECKPOINT_PATH=/path/to/vggto-or-vggt-omega.pt
export VGGT_CHECKPOINT_PATH=/path/to/VGGT-1B/model.safetensors
```

VGGTO core geometry keys are loaded strictly. Camera and text-alignment keys in
VGGT-Omega checkpoints are ignored and reported. The first/other register tables
are averaged into the single shared register table used by the current VGGTO model.
The point head is loaded strictly from `point_head.*` in the original VGGT
checkpoint because the VGGT-Omega checkpoint contains only the dense depth head.
VGGT pretraining predicts relative points in a shared first-camera frame, while
the current dataset supervises relative points in each view's local camera frame;
training therefore fine-tunes the loaded head to the local-point contract.

VGGTO-24 is expanded with the explicit target-to-source map below; `None`
denotes a newly inserted identity layer:

```text
(0,1,2,3,None,4,None,5,6,7,None,8,9,10,None,
 11,12,13,14,15,16,17,None,18,None,19,20,21,22,23)
```

Source dense-head cache layers `(4,11,17,23)` move to target native layers
`(5,15,21,29)`. The original five register layers map to `(2,8,12,18,26)`;
the other ten even MOT layers are inserted or converted layers.

MoT joint attention and G-stream geometry updates are active at full strength
from initialization. The attention mask directly controls which V/A queries can
read G keys. Inference runs the independent G stream once and reuses the
pre-update register snapshot aligned with each joint MoT layer.

## Data Preparation

The default real-data root is `/team_data/umi_data`, with support for:

```text
/team_data/umi_data
├── lumos_lerobot/
├── genrobot_lerobot_v1_v2/
├── genrobot_lerobot_v3/
├── genrobot_lerobot_v3_1/
└── genrobot_lerobot_v3_2/
```

Each valid task must be LeRobot v3 dual-arm data and contain:

- `meta/info.json`
- `meta/episodes/chunk-*/file-*.parquet`
- `data/chunk-*/file-*.parquet`
- `videos/chunk-*/observation.images.robot_0/*.mp4`
- `videos/chunk-*/observation.images.robot_1/*.mp4`
- Optional training point labels: per-hand
  `pointcloud/.../<left/right hand>*/preprocessed_pointcloud/_SUCCESS` stores.
  Lumos raw `clips/clip_*` directories are preprocess inputs, not runtime
  dataset inputs.

The raw action/state format is fixed at 16D:

```text
[x,y,z,qx,qy,qz,qw,gripper] * 2
```

Training actions are converted to 20D chunk-reference relative Rot6D:

```text
[dx,dy,dz,rot6d_col0_col1,gripper] * 2
```

### Dataset preparation overview

The real-data preparation path stays manifest/index based. It does not expand
every train window into a separate JSONL row. Instead, the final train manifests
store one row per selected episode/session and keep valid train windows as a
single continuous `valid_start_range`. Runtime sampling later chooses a start
frame from that range.

The pipeline is split by responsibility:

1. `build_base_valid_data`: scan raw LeRobot tasks and write a reusable
   episode-level base snapshot.
2. `build_training_selection`: freeze the exact pointcloud and non-pointcloud
   episodes that will be used for training. It can either build a full
   selection from the base snapshot, or sample a smaller selection from an
   existing full selection without rescanning pointcloud directories.
3. `preprocess_lumos_pointcloud`: convert only the selected Lumos raw
   `res.npz` pointcloud clips into runtime `preprocessed_pointcloud` stores.
   Genrobot stores are not rewritten here.
4. `cal_norm_stats`: compute per-task 20D relative-action stats from
   the frozen selected episodes.
5. `build_training_dataset`: materialize the frozen selection, write the final
   train manifests/config/text cache, and optionally build the action mmap
   cache consumed by training.

The model training path imports `wan_va/dataset/mot_dataset.py`; the other
modules are offline preparation tools.

The descriptions below follow the runtime call logic instead of source line
numbers: each step starts from the module entry point, then lists the helper
sequence, validation gates, files written, and how the output is consumed by
later steps or runtime.

### Step 1: base snapshot

Command:

```bash
python -m wan_va.dataset.build_base_valid_data \
  --umi-data-root /team_data/umi_data \
  --output-root data/umi_mot_full_data_base_0710_final
```

Implementation flow:

- The module entry point calls
  [`build_real_mot_base_cache`](wan_va/dataset/build_base_valid_data.py). The
  function owns the whole base snapshot build: task discovery, task validation,
  episode validation, row construction, reports, and base config creation.
- **STEP1: Filter invaild lerobot format tasks.**
  Task discovery starts in `_step1_iter_task_roots_filter_lerobot`. It walks only the supported
  families: `lumos_lerobot/<date>/<task>`, `genrobot_lerobot_v1_v2/<task>`,
  `genrobot_lerobot_v3/<task>`, `genrobot_lerobot_v3_1/<task>`, and
  `genrobot_lerobot_v3_2/<task>`. A directory becomes a task candidate only
  when `data/`, `videos/`, and `meta/` all exist and `meta/info.json` is a file.
  Incomplete candidate directories are written to
  `reports/step1_iter_roots_invaild_lerobot.jsonl` with `missing_data_dir`,
  `missing_videos_dir`, `missing_meta_dir`, or `missing_meta_info`. The same
  reasons are summarized in `base_config.json` under
  `skipped.step1_invaild_lerobot`.
- Task validation then checks the LeRobot metadata. `build_real_mot_base_cache`
  opens `meta/info.json`, requires `codebase_version == "v3.0"`, and calls
  `_is_dual_arm_info` to require both RGB view features plus 16D `action` and
  16D `observation.state`. A task that fails here is recorded in
  `reports/task_skipped.jsonl`; none of its episodes are considered.
- **STEP2: Filter invaild episodes.**
  Episode metadata is loaded by `_read_episode_records` from
  `meta/episodes/chunk-*/file-*.parquet` and indexed by `episode_index`.
  For each episode, `_episode_text` extracts the first task string,
  `_episode_len` computes `dataset_to_index - dataset_from_index`, and
  `_missing_payload_files_for_episode` verifies that the action/state parquet
  and both RGB videos exist. Episodes that fail these checks are recorded in
  `reports/episode_skipped.jsonl`; the rest of the task can still contribute
  valid episodes.
- Each accepted episode emits one `base_valid_episode_manifest.jsonl` row. The row is still a
  full-episode row, not a window row. It stores task IDs, source IDs, episode
  index, fps, parquet path, dataset index bounds, full-episode segment
  `[start_frame=0, end_frame=episode_len]`, action text, and both RGB view
  descriptors.
- **STEP3: Build base cache json (each row represents one episode)**
  After all tasks are scanned, `build_real_mot_base_cache` writes
  `meta/base_valid_episode_manifest.jsonl`, `reports/task_skipped.jsonl`,
  `reports/episode_skipped.jsonl`,
  `reports/step1_iter_roots_invaild_lerobot.jsonl`, `reports/text_check.json`, and an
  empty `meta/task_norm_stats.json` placeholder. It also writes
  `meta/base_config.json`, which records the base root, source root, manifest
  path, episode/task counts, video keys, and step1/task/episode skip counts.

Input directory layout:

```text
/team_data/umi_data/
├── lumos_lerobot/
│   └── <date>/<task>/
│       ├── meta/info.json
│       ├── meta/episodes/chunk-*/file-*.parquet
│       ├── data/chunk-*/file-*.parquet
│       ├── videos/chunk-*/observation.images.robot_0/*.mp4
│       ├── videos/chunk-*/observation.images.robot_1/*.mp4
│       └── pointcloud/...                         # optional
├── genrobot_lerobot_v1_v2/<task>/...
├── genrobot_lerobot_v3/<task>/...
├── genrobot_lerobot_v3_1/<task>/...
└── genrobot_lerobot_v3_2/<task>/...
```

Output directory layout:

```text
data/umi_mot_full_data_base/
├── meta/
│   ├── base_config.json
│   ├── base_valid_episode_manifest.jsonl
│   └── task_norm_stats.json
├── reports/
│   ├── task_skipped.jsonl
│   ├── episode_skipped.jsonl
│   ├── step1_iter_roots_invaild_lerobot.jsonl
│   └── text_check.json
```

Key files:

- `base_config.json`: records the source root, base manifest path, episode/task counts,
  video keys, and step1/task/episode skip counts. Later steps
  use it to locate `base_valid_episode_manifest.jsonl`.
- `base_valid_episode_manifest.jsonl`: episode-level snapshot. Each row contains fields such as
  `task_uid`, `norm_stats_key`, `source_dataset`, `source_lerobot_task_dir`,
  `data_file`, `dataset_from_index` / `dataset_to_index`, `segment`, and `views`.
- `task_norm_stats.json`: base-stage placeholder. Real per-task stats are written
  back to each raw task's `meta/` directory by the stats step.
- `reports/task_skipped.jsonl`: task-level rejection reasons, such as
  `bad_info:*`, `not_lerobot_v3`, `not_dual_arm`,
  `bad_episode_metadata:*`, or `empty_episode_metadata`.
- `reports/episode_skipped.jsonl`: episode-level rejection reasons within an
  otherwise valid task, such as `empty_episode` or `missing_payload_file`.
- `reports/step1_iter_roots_invaild_lerobot.jsonl`: candidate directories discovered at
  the family/task layer but rejected before task validation because the required
  task root structure is incomplete.

The base step does not build the final training dataset, does not decide the
pointcloud/non-pointcloud sampling ratio, and does not build action cache. The
training action cache is built from the final train manifests in Step 5.

Downstream use:

- Step 2 reads `meta/base_config.json` and `meta/base_valid_episode_manifest.jsonl`
  through `--base-root`. It never rescans the raw LeRobot episode parquet.
- Later steps do not read the base manifest directly for sampling. They read
  the Step 2 frozen selection so pointcloud preprocessing, stats, and final
  materialization all operate on the same episode set.
- The final training root writes its own action cache in Step 5 unless
  `--no-action-cache` is passed there.

### Step 2: frozen training selection

Full-selection command:

```bash
python -m wan_va.dataset.build_training_selection \
  --base-root data/umi_mot_full_data_base_0710_final \
  --output-root data/umi_mot_full_data_selection_0711_final
```

Sample-from-existing-selection command:

```bash
python -m wan_va.dataset.build_training_selection \
  --base-root data/umi_mot_full_data_base_0710_final \
  --output-root data/0715_3k_test_subset \
  --sample-from-existing-selection-path data/umi_mot_full_data_selection_0711_final \
  --max-pointcloud-samples 3000 \
  --max-non-pointcloud-samples 3000
```

Implementation flow:

- The module entry point calls
  [`build_real_mot_training_selection`](wan_va/dataset/build_training_selection.py).
  In full-selection mode, it reads Step 1 `meta/base_config.json`, follows
  `base_valid_episode_manifest_path`, loads the base rows, and validates
  `dataset_to_index == dataset_from_index + segment.end_frame` for every row.
  In sample-from-existing-selection mode, it reads the existing selection root's
  `meta/training_selection_config.json`,
  `meta/selected_pointcloud_episode_manifest.jsonl`, and
  `meta/selected_non_pointcloud_episode_manifest.jsonl` directly. That mode does
  not reload base rows, rescan `pointcloud/`, redo session-to-episode mapping,
  or recheck hand directories.
- Full-selection mode always builds the complete pointcloud and non-pointcloud
  episode sets from the base snapshot. Do not pass
  `max_pointcloud_samples` or `max_non_pointcloud_samples` in this mode; those
  limits require `--sample-from-existing-selection-path`. Selection does not
  take `action_chunk_size` or `video_downsample_ratio`, so the same frozen
  episode set can be reused for different window and stride experiments. Norm
  stats and final dataset build do not reselect.
- In full-selection mode, `select_real_mot_train_rows` first builds planned
  pointcloud rows. Planned pointcloud rows intentionally do not require Lumos
  `preprocessed_pointcloud` stores yet, because Step 3 creates them.
- `_build_planned_pointcloud_rows` indexes base rows by
  `(task_uid, episode_index)`, scans pointcloud session roots for every task,
  resolves each physical session to a LeRobot episode, joins it to the base row,
  validates both hand directories, and emits a deep copy of the base row with
  pointcloud planning fields. It does not compute final `valid_start_range`;
  those depend on the explicit Step 5 window parameters.
- Physical pointcloud layouts differ by family. Lumos scans
  `pointcloud/<multi-session-root>/session_*`; genrobot scans
  `pointcloud/<group>/<session>`.
- Mapping is mandatory for both families. Lumos uses `raw_lerobot_idx.jsonl`,
  keyed by the session path relative to `task_root/pointcloud`. Genrobot uses
  `mcap_lerobot_idx_mapping.jsonl`, keyed by mcap stem or mcap filename. A
  pointcloud session without a mapping is reported and cannot enter the
  pointcloud selection.
- Lumos planned rows require both `left_hand*` and `right_hand*` directories and
  raw `clips/` directories. Genrobot planned rows require existing
  `preprocessed_pointcloud/_SUCCESS`, because genrobot is already preprocessed
  and is not rewritten by Step 3.
- Any mapped pointcloud episode that reaches the planning join is blocked from
  the non-pointcloud pool. This is intentional: pointcloud episodes do not enter
  the non-pointcloud manifest, even when sampling later keeps only a subset of
  pointcloud rows.
- In sample-from-existing-selection mode only, pointcloud rows and
  non-pointcloud rows are limited independently. Each group is first split by a
  Lumos/GenRobot 50/50 source policy, then each source is sampled through a
  deterministic task round-robin. This keeps small test subsets from being
  dominated either by the larger source family or by consecutive episodes from
  one task. The limiter is applied to the already-written full selected
  manifests, so repeated subset builds are cheap and deterministic.
- The selected rows are written once and become the source of truth for every
  later step.

Output layout:

```text
data/umi_mot_real_train_stride4_rel20/
├── meta/
│   ├── training_selection_config.json
│   ├── selected_pointcloud_episode_manifest.jsonl
│   └── selected_non_pointcloud_episode_manifest.jsonl
└── reports/
    ├── selection_pointcloud_skipped.json
    └── selection_pointcloud_skipped.jsonl
```

Downstream use:

- Step 3 reads `selected_pointcloud_episode_manifest.jsonl` and preprocesses
  only rows whose `source_dataset == "lumos_lerobot"`.
- Step 4 reads both selected manifests and computes norm stats only for the
  selected episodes, using the `--action-chunk-size` passed to Step 4.
- Step 5 reads the same selected manifests, materializes the selected
  pointcloud rows with its explicit window parameters, and writes the final
  train manifests.
- If either selection limit changes for a test subset, rerun Step 2 in
  sample-from-existing-selection mode from the full selection root, then rerun
  Steps 3-5 for the new sampled selection. Reusing old Step 3-5 outputs with a
  new selection is invalid.

### Step 3: selected Lumos pointcloud preprocessing

Command:

Single data test: (directly use the key *build_hand_store* function)
```bash
python -m wan_va.dataset.preprocess_lumos_pointcloud build \
  --hand-dir /team_data/umi_data/lumos_lerobot/20260225/task_20260213K043_Store_scattered_tableware/pointcloud/multi_sessions_20260225_093755/session_004/left_hand_250801DR48FP25002672 \
  --report-path data/single_lumus_preprocess_test/reports/lumos_pointcloud_preprocess.json \
  --overwrite
```

Full data preprocessing:
```bash
python -m wan_va.dataset.preprocess_lumos_pointcloud build \
  --selection-root data/0710_final_full_selection \
  --report-path data/0710_final_full_selection/reports/lumos_pointcloud_preprocess.json \
  --num-workers 32 
```

`conf_224` and `rays_224` are not written to `metadata.npz` by default. Add
`--save_conf` or `--save_rays` only when those arrays are needed for debugging.
Important: after a successful build, the raw `clips/*/res.npz` and
`downsample/res.npz` files are deleted by default. Deletion happens only after
`preprocessed_pointcloud/_SUCCESS` is written. Add `--no-del_ori_npz` to keep
the raw `res.npz` files for debugging or later rebuilds.

Implementation flow:

- The `build` subcommand in
  [`preprocess_lumos_pointcloud.py`](wan_va/dataset/preprocess_lumos_pointcloud.py)
  either preprocesses one explicit `--hand-dir` with `build_hand_store`, or
  preprocesses selected Lumos sessions through `build_from_root`.
  `_source_dataset_for_path` rejects non-Lumos paths for this script. Genrobot
  stores are intentionally left in their existing format and are handled later
  by the shared point-store reader.
- Batch `build_from_root` reads Step 2
  `meta/training_selection_config.json` and
  `meta/selected_pointcloud_episode_manifest.jsonl`. It keeps only selected
  rows whose `source_dataset == "lumos_lerobot"`.
- `_selected_lumos_sessions` converts each selected Lumos pointcloud row into a
  session task. It validates that the physical session directory still exists
  and that both hand directories still contain raw `clips/`. Rows that fail this
  check are reported; no replacement session is selected.
- `build_from_root` creates one conversion task per selected hand and sends
  those tasks to `_run_hand_store_tasks`, either serially or with process
  workers.
- Each per-hand task calls `build_hand_store`. If a completed
  `preprocessed_pointcloud/_SUCCESS` store already exists under the Lumos hand
  directory and has the current reports, it is reused unless `--overwrite` is
  passed. Partial or older stores without the reports are rebuilt.
- `build_hand_store` reads raw clips from `clips/clip_*_<start>_<end>/res.npz`.
  `_collect_clip_specs` parses clip directories, and `_clip_episode_frame_bounds`
  extracts the inclusive episode-local frame range from the clip name for
  validation. Each clip must also have `frame_mapping.csv`; preprocessing uses
  its `aligned_frame_index` as the final episode-local frame ID and keeps
  `raw_frame_index` for downsample alignment checks.
- For every clip, `build_hand_store` opens `res.npz`. `local_points`, `conf`,
  `rays`, and `camera_poses` are required; missing keys fail preprocessing
  instead of falling back to synthetic defaults. `local_points` may be
  `[1,F,H,W,3]` or `[F,H,W,3]`, and `_squeeze_frame_axis` normalizes that shape.
  The clip's frame count must match both the frame range encoded in the clip
  directory name and dense `local_frame_index == 0..F-1` rows in
  `frame_mapping.csv`.
- For each local clip row, `_resize_depth_rays_frame` resizes depth and rays,
  then reconstructs xy from the resized rays and depth. Confidence is resized
  alongside the points. The resize path matches the UniK3D guardrails: points
  with depth `z <= 0`, depth `z > 5.0`, or ray `z <= 0.2` are set to zero
  before masks are generated, so those pixels become invalid in `valid_mask_*`.
- The resized local row is then assigned its episode-local frame ID from
  `frame_mapping.csv:aligned_frame_index` and appended to the in-memory rows.
  After all clips are loaded, rows are sorted by that episode frame ID,
  duplicate episode frame IDs are rejected, and the final frame IDs must be
  dense `0..N-1`. This density check is the **preprocessing-time guarantee that
  runtime lookup can use** `pointcloud row i == LeRobot episode-local video frame
  i`.
- `downsample/res.npz` is required and must contain `local_points`, `conf`, and
  `rays`; `downsample/frame_mapping.csv` is also required. `_frame_scales_for_rows`
  joins clip rows to downsample rows by `aligned_frame_index`, requires matching
  `raw_frame_index`, and estimates one median scale per clip from all valid
  matched anchors. The clip-level scale is expanded into per-frame
  `frame_scales` and applied to local points and camera translation. Clips with
  no matched anchor or no valid scale anchor fail preprocessing.
- After scaling, `build_hand_store` writes `local_points_224.npy`,
  `metadata.npz`, `clip_table.jsonl`, `alignment_report.json`,
  `protection_report.json`, and `scale_alignment_report.json`. By default
  `metadata.npz` contains camera poses, episode frame indices, clip ids, and
  frame scales; `conf_224` and `rays_224` are only saved when `--save_conf` or
  `--save_rays` is passed.
- `build_hand_store` then generates masks through the shared
  [`pointcloud_store.py`](wan_va/dataset/pointcloud_store.py) helper. Masks
  require finite positive depth and confidence above threshold. Lumos uses a
  default confidence threshold of `0.03`; genrobot uses `0.0`. During
  preprocess, Lumos masks are generated from the in-memory confidence array even
  when `conf_224` is not saved.
- After masks are generated, `build_hand_store` writes `meta.json` and then
  `_SUCCESS`. `meta.json` marks the output as
  `mot_preprocessed_pointcloud_v2`. Batch mode also writes
  `reports/lumos_pointcloud_protection_events.jsonl` next to the main report
  when `--report-path` is provided.
- Per-hand exceptions raised while running `build_hand_store` are returned in
  the main report's `skipped` list with
  `error_source="_build_hand_store_task"` and
  `error_stage="build_hand_store"`, plus the exception `reason` and `detail`.
- After `_SUCCESS` is written, `build_hand_store` deletes only the original
  `clips/*/res.npz` and `downsample/res.npz` files by default. It leaves
  `frame_mapping.csv`, reports, directories, and all preprocessed outputs in
  place. Use `--no-del_ori_npz` to disable this cleanup.
- Runtime code later opens both old genrobot stores and new Lumos stores through
  `PointStore.open`. `PointStore` requires `_SUCCESS`, mmaps points and masks,
  and maps episode frames by direct row index when the requested frame is inside
  the store length.

Input lumos pointcloud shape:

```text
/team_data/umi_data/lumos_lerobot/<date>/<task>/
├── raw_lerobot_idx.jsonl                         # required session -> episode mapping
└── pointcloud/
    └── <multi-session-root>/
        └── session_<id>/
            ├── left_hand*/
            │   ├── clips/clip_<id>_<start>_<end>/res.npz
            │   ├── clips/clip_<id>_<start>_<end>/frame_mapping.csv
            │   ├── downsample/res.npz
            │   └── downsample/frame_mapping.csv
            └── right_hand*/...
```

Output lumos store:

```text
.../session_<id>/<left_hand|right_hand>*/preprocessed_pointcloud/
├── _SUCCESS
├── local_points_224.npy
├── metadata.npz                                  # includes episode_frame_indices
├── meta.json                                     # format: mot_preprocessed_pointcloud_v2
├── clip_table.jsonl
├── alignment_report.json
├── protection_report.json
├── scale_alignment_report.json
└── masks/
    ├── valid_mask_conf0.npy
    └── valid_mask_conf0p03.npy
```

Genrobot pointcloud stores are not rewritten by this step. Existing genrobot
`preprocessed_pointcloud` directories are consumed as-is.

Downstream/runtime use:

- Step 5 discovers the `preprocessed_pointcloud/` directories written here and
  stores their paths in `mot_final_training_pointcloud_manifest.jsonl` as per-view
  `preprocessed_pointcloud_dir` fields.
- Step 5 requires `_SUCCESS` before a Lumos or genrobot pointcloud store can be
  selected for pointcloud-labeled training rows.
- Step 5 checks that both left/right stores have the same row count as the
  mapped LeRobot episode length. This is the final build-time guard before the
  runtime dataset uses direct row lookup.
- Runtime `MotTrainData` never scans raw `clips/clip_*` directories. It opens
  the already selected `preprocessed_pointcloud_dir` values with `PointStore`.
- Runtime `PointStore` mmaps `local_points_224.npy` as the source for `pts3d`
  and `geometry_pts3d`.
- Runtime `PointStore` reads the active mask under `masks/` as the source for
  `valid_mask` and `geometry_point_valid_mask`.
- Runtime pointcloud lookup is direct:
  `PointStore.row_for_episode_frame(frame_id)` returns the same integer frame ID
  when it is inside the store length. `frame_mapping.csv` and
  `aligned_to_row.npy` are not runtime inputs.
- `metadata.npz` is not used to map video frames to pointcloud rows. Runtime
  uses it only for store-level arrays needed by masks and point-store loading.

### Step 4: selected-episode action norm stats

Command:

```bash
python -m wan_va.dataset.cal_norm_stats \
  --selection-root data/umi_mot_full_data_selection_0711_final \
  --action-chunk-size 48 \
  --overwrite
```

Implementation flow:

- The module entry point calls
  [`cal_real_mot_norm_stats`](wan_va/dataset/cal_norm_stats.py). It reads Step 2
  `meta/training_selection_config.json` and the two selected episode manifests.
  It does not call the sampling code.
- `action_chunk_size` is the only window-shape parameter for this step. the same selected 
  episode set can produce stats for multiple action chunk sizes.
- Selected pointcloud rows and selected non-pointcloud rows are grouped by
  `task_uid`. Norm stats are computed from those selected episodes only.
- Before computing a task, `_valid_existing_stats` checks whether the raw task
  already has `meta/norm_stats_deltarot6d_chunk<action_chunk_size>.json` with
  matching `action_chunk_size` and valid 20D `q01` / `q99` / `mean` / `std`.
  Valid existing files are reused unless `--overwrite` is passed.
- Missing task stats are computed either serially or in a process pool through
  `_compute_one_task`, which wraps `compute_task_norm_stats_payload`.
- `compute_task_norm_stats_payload` iterates that task's base episodes. For each
  episode it computes the full valid start range with `_full_valid_start_range_for_stats`,
  reads `action` and `observation.state` from the raw parquet, optionally reads
  the parquet `index` column, slices the arrays down to the episode, and caches
  repeated parquet reads by file path.
- For the valid start range, `_relative_target_values_for_arrays_vectorized`
  converts target actions to the training action representation:
  chunk-reference relative position plus Rot6D orientation columns plus gripper,
  producing 20D values. Values from all valid chunks in the task are concatenated.
- The task's `q01`, `q99`, `mean`, and `std` are computed per action dimension from the
  concatenated 20D values. The payload also records the task ID, source task
  directory, action parameters, contributing episode indices,
  `num_episodes`, `num_chunks`, and `num_values`.
- The computed payload is written back to the raw task directory as
  `meta/norm_stats_deltarot6d_chunk<action_chunk_size>.json`. Step 5 later
  treats this file as required metadata.
- If `--dataset-root` points at an already-built train root,
  `cal_real_mot_norm_stats` can also reopen that root's `meta/mot_config.json`,
  reload the stats for the rows already present in that train root, and update
  `norm_stat`, `norm_stats_by_task`, and `norm_stat_policy`.

Input:

```text
data/umi_mot_real_train_stride4_rel20/
└── meta/
    ├── training_selection_config.json
    ├── selected_pointcloud_episode_manifest.jsonl
    └── selected_non_pointcloud_episode_manifest.jsonl

/team_data/umi_data/.../<task>/
└── data/chunk-*/file-*.parquet                   # reads action and observation.state
```

Output written back to the raw task:

```text
/team_data/umi_data/.../<task>/
└── meta/
    └── norm_stats_deltarot6d_chunk48.json
```

Stats file structure:

```text
{
  "q01": [20 floats],
  "q99": [20 floats],
  "mean": [20 floats],
  "std": [20 floats],
  "task_uid": "...",
  "action_chunk_size": 48,
  "action_dim": 20,
  "action_representation": "relative_to_chunk_reference_state_rot6d_cols",
  "source_lerobot_task_dir": "...",
  "episode_indices": [...],
  "num_episodes": ...,
  "num_chunks": ...,
  "num_values": ...
}
```

This step does not write the final train root. Its job is to let the training
dataset builder map each manifest row's `norm_stats_key == task_uid` to the
corresponding task's `q01/q99/mean/std`. If the stats `action_chunk_size` and
training-dataset `action_chunk_size` differ, the training build fails on missing
or invalid norm stats.

If a train root already exists,
`python -m wan_va.dataset.cal_norm_stats --selection-root <selection-root> --dataset-root <train-root> --action-chunk-size <N>`
can also update `norm_stat` / `norm_stats_by_task` in the existing
`meta/mot_config.json` to the current stats. Add `--overwrite` when the raw task
stats should be recomputed before updating the train root.

Downstream/runtime use:

- Step 5 treats `meta/norm_stats_deltarot6d_chunk<action_chunk_size>.json` as
  required metadata for every selected task. Missing files, mismatched
  `action_chunk_size`, or invalid stat arrays make the train build fail.
- Step 5 copies the loaded stats into `meta/mot_config.json` as
  `norm_stats_by_task`, and also stores one default `norm_stat` for compatibility
  with older config consumers.
- Runtime `MotTrainData` selects a row's stats by `norm_stats_key`, which is the
  task UID from the base row.
- Runtime action loading converts raw absolute 16D actions to relative 20D
  actions, then normalizes them with that task's `q01/q99`.
- Runtime samples return `action_q01`, `action_q99`, and `action_norm_source`
  for debugging, evaluation, and denormalization checks.

### Step 5: final train dataset

Command:

```bash
python -m wan_va.dataset.build_training_dataset \
  --selection-root data/umi_mot_full_data_selection_0711_final \
  --output-root data/data/umi_mot_full_data_train_0712_final \
  --action-chunk-size 48 \
  --video-downsample-ratio 4
```

Implementation flow:

- The module entry point calls
  [`build_real_mot_train_dataset`](wan_va/dataset/build_training_dataset.py).
  It reads Step 2 `training_selection_config.json`, creates the output root, and
  materializes the selected pointcloud rows using the explicit
  `action_chunk_size` and `video_downsample_ratio` passed to this step.
- `_materialize_selected_pointcloud_rows` processes only the rows in
  `selected_pointcloud_episode_manifest.jsonl`. It does not search for
  replacement rows if one selected row fails.
- `_pointcloud_views_for_row` attaches pointcloud stores to the corresponding
  LeRobot RGB views. For each hand it requires
  `<hand_dir>/preprocessed_pointcloud/_SUCCESS`, copies the base RGB view
  descriptor, and adds `hand`, `view_id`, `hand_dir`, and
  `preprocessed_pointcloud_dir`.
- `_pointcloud_store_lengths` opens both stores with `PointStore`. A session is
  skipped unless both store row counts equal the LeRobot episode length from the
  base row, except GenRobot rows where both hands have exactly one fewer
  pointcloud frame are logically truncated by one tail frame. This is the final
  build-time check that video frames and pointcloud rows are aligned.
- `_pointcloud_valid_start_range` enumerates candidate MOT start frames from
  `_full_valid_start_range`. For each start, it computes the history and target
  sampled frame IDs, clamps padded reads to episode boundaries, and requires
  every padded frame to exist in every point store. The result is a single
  compact `[start, end]` range.
- Each accepted pointcloud row is a deep copy of the base row with
  `has_pointcloud=true`, `source_pointcloud_task_dir`,
  `pointcloud_session_dir`, `pointcloud_mapping_policy`,
  `video_downsample_ratio`, `valid_start_range`, and updated per-view
  pointcloud metadata.
- If a selected pointcloud row cannot be materialized, Step 5 writes
  `reports/pointcloud_skipped.jsonl` and continues with the remaining rows. It
  errors only if no train rows remain.
- Non-pointcloud rows are loaded directly from
  `selected_non_pointcloud_episode_manifest.jsonl`. Step 5 does not resample
  them. It computes their final `valid_start_range` from `action_chunk_size`
  and `MOT_MAX_RIGHT_PADDING_RAW_STEPS`; rows without any valid train window are written to
  `reports/non_pointcloud_skipped.json` and
  `reports/non_pointcloud_skipped.jsonl`.
- After selection, `build_real_mot_train_dataset` calls
  `load_required_task_norm_stats`. This loads the Step 4
  `meta/norm_stats_deltarot6d_chunk<action_chunk_size>.json` file for every
  selected task and rejects missing files, mismatched `action_chunk_size`, or
  non-20D `q01` / `q99` / `mean` / `std`.
- The selected rows are written to two manifests:
  `meta/mot_final_training_pointcloud_manifest.jsonl` for pointcloud-labeled rows and
  `meta/mot_final_training_non_pointcloud_manifest.jsonl` for RGB/action-only rows.
- `_make_text_cache` gathers every unique `segment.action_text` from both
  manifests and writes `text_emb_cache.pt` plus `empty_emb.pt`. If no custom
  embedder is provided, it loads the configured Wan tokenizer and text encoder.
- `build_real_mot_train_dataset` writes `meta/mot_config.json`. This is the
  training source of truth: it records manifest paths, text cache paths, camera
  keys, action dimensions, action/video geometry dimensions, `norm_stat`,
  `norm_stats_by_task`, `source_sample_counts`, selection limit policy, and
  pointcloud snapshot policy.
  `source_sample_counts` is split into `lumos_pointcloud`,
  `genrobot_pointcloud`, `lumos_non_pointcloud`, `genrobot_non_pointcloud`,
  `total_pointcloud`, `total_non_pointcloud`, and `total_train_data_num`.
- Pointcloud skip details are written to `reports/pointcloud_skipped.json` and
  `reports/pointcloud_skipped.jsonl`, preserving why sessions were rejected.
- Unless `--no-action-cache` is passed, the final step calls
  `build_mot_action_cache` on both final manifests. The action-cache builder
  writes mmap action/state arrays for the raw parquets used by the final train
  dataset and updates `mot_config.json` with `action_cache_manifest_path`.

View metadata lifecycle:

- Step 1 writes the RGB-only view descriptors into
  `base_valid_episode_manifest.jsonl`. Each base view contains `view_id`,
  `video_key`, `video_path`, and `video_from_timestamp`. These fields are the
  minimum information required to load RGB frames later: runtime converts local
  episode frame IDs to video frame IDs with `video_from_timestamp * fps`, then
  reads from `video_path`.
- Step 2 copies the matching Step 1 view for each pointcloud hand. It finds the
  base view by `video_key`, using the fixed mapping `left ->
  observation.images.robot_0` and `right -> observation.images.robot_1`, then
  adds `hand`, `hand_dir`, and the fixed `view_id` for that hand. The selected
  pointcloud manifest therefore records both the RGB source and the planned raw
  pointcloud hand directory, but it does not yet record
  `preprocessed_pointcloud_dir` for Lumos rows because Step 3 creates those
  stores after selection.
- Step 5 copies the selected view again and materializes the runtime pointcloud
  store path. `_pointcloud_views_for_row` verifies
  `<hand_dir>/preprocessed_pointcloud/_SUCCESS`, then adds
  `preprocessed_pointcloud_dir`. The final `mot_final_training_pointcloud_manifest.jsonl` view is the
  training-time contract: RGB loading uses `video_path` and
  `video_from_timestamp`, while point loading opens `preprocessed_pointcloud_dir`
  with `PointStore`.

This staged write is intentional. Step 1 can only prove that the LeRobot RGB
episode is valid. Step 2 freezes which pointcloud session/hand directories map
to that episode. Step 5 proves that preprocessing completed and writes the
actual `PointStore` paths consumed during training. Keeping the RGB fields from
the base view and adding pointcloud fields later prevents left/right hand
pointclouds from being mismatched with the wrong camera view.

Input:

```text
data/umi_mot_full_data_base/
└── meta/
    └── base_config.json

data/umi_mot_real_train_stride4_rel20/
└── meta/
    ├── training_selection_config.json
    ├── selected_pointcloud_episode_manifest.jsonl
    └── selected_non_pointcloud_episode_manifest.jsonl

/team_data/umi_data/.../<task>/
├── meta/norm_stats_deltarot6d_chunk48.json       # Step 4 output
├── data/chunk-*/file-*.parquet
├── videos/chunk-*/observation.images.robot_0/*.mp4
├── videos/chunk-*/observation.images.robot_1/*.mp4
└── pointcloud/.../<left/right hand>/preprocessed_pointcloud/
    ├── _SUCCESS                                  # required for pointcloud samples
    ├── local_points_224.npy
    ├── metadata.npz
    ├── meta.json
    └── masks/valid_mask_*.npy                    # if masks have been generated
```

Genrobot mapping uses `mcap_lerobot_idx_mapping.jsonl`. Lumos selection uses
`raw_lerobot_idx.jsonl`, and Lumos preprocessing uses raw pointcloud
`frame_mapping.csv` to build dense stores. Runtime does not use
`frame_mapping.csv` or `aligned_to_row.npy` to map LeRobot video frames to
pointcloud rows.

Output directory layout:

```text
data/umi_mot_real_train_stride4_rel20/
├── meta/
│   ├── training_selection_config.json
│   ├── selected_pointcloud_episode_manifest.jsonl
│   ├── selected_non_pointcloud_episode_manifest.jsonl
│   ├── mot_config.json
│   ├── mot_final_training_pointcloud_manifest.jsonl      # pointcloud-labeled rows
│   └── mot_final_training_non_pointcloud_manifest.jsonl  # pure RGB/action rows
├── empty_emb.pt
├── text_emb_cache.pt
├── reports/
│   ├── pointcloud_skipped.json
│   ├── pointcloud_skipped.jsonl
│   ├── non_pointcloud_skipped.json
│   └── non_pointcloud_skipped.jsonl
└── cache/
    └── actions/                                  # unless --no-action-cache is passed
        ├── action_cache_manifest.jsonl
        ├── cache_info.json
        └── files/<hash-prefix>/<hash>/
            ├── actions.npy
            ├── states.npy
            ├── index.npy
            └── meta.json
```

Key files:

- `meta/mot_config.json`: source of truth for training config. The training config
  reads manifest paths, camera keys, action dimensions, sampling parameters,
  `norm_stats_by_task`, text cache paths, and action cache paths from this file.
- `meta/mot_final_training_pointcloud_manifest.jsonl`: training rows with pointcloud labels. Each row contains
  valid start ranges, pointcloud view/store information, raw video/data paths, and
  task metadata.
- `meta/mot_final_training_non_pointcloud_manifest.jsonl`: RGB/action-only rows without pointcloud labels.
- `empty_emb.pt`: empty text embedding.
- `text_emb_cache.pt`: cache from `action_text` to embedding.
- `reports/pointcloud_skipped.*`: reasons why pointcloud sessions/windows were skipped.
- `reports/non_pointcloud_skipped.*`: selected RGB/action-only rows that had no
  valid train window for the requested chunk/stride parameters.
- `cache/actions/action_cache_manifest.jsonl`: action/state mmap cache index for
  raw parquet files used by the final train dataset.

Runtime use:

- `MOT_DATASET_ROOT` points at this Step 5 output root. The training config reads
  `meta/mot_config.json` from that root.
- `mot_config.json` is the runtime source of truth for manifest paths, camera
  keys, action dimensions, sampling parameters, text cache paths, action cache
  paths, and task normalization stats.
- `mot_final_training_pointcloud_manifest.jsonl` feeds `MotGeometryLeRobotData`. Every row must have
  `has_pointcloud=true`, and every view must contain a
  `preprocessed_pointcloud_dir`.
- `mot_final_training_non_pointcloud_manifest.jsonl` feeds `MotPureLeRobotData`. Rows have
  `has_pointcloud=false`; runtime still returns point tensors, but they are zero
  tensors with all-false masks.
- `text_emb_cache.pt` and `empty_emb.pt` provide text conditioning. Runtime
  looks up each row's `segment.action_text` in `text_emb_cache.pt`.
- The final `cache/actions/action_cache_manifest.jsonl` lets `MotTrainData` mmap
  `actions.npy`, `states.npy`, and optional `index.npy`. If the final action
  cache is absent, the dataset falls back to reading raw parquet columns.
- Pointcloud paths in `mot_final_training_pointcloud_manifest.jsonl` are fixed at build time. Adding or
  changing pointcloud stores requires rerunning Step 5 so the manifest and skip
  reports reflect the new store set.

Training setup:

```bash
export MOT_DATASET_ROOT=data/umi_mot_real_train_stride4_rel20
```

`wan_va/configs/va_umi_3dwam_train_cfg.py` reads
`$MOT_DATASET_ROOT/meta/mot_config.json`, then passes the dataset, normalization,
text cache, action cache, and VGGTO geometry training contract to `wan_va/train_mot.py`.

If any of the following selection limit parameters changes for a subset, rerun
Step 2 from an existing full selection root, then rerun Steps 3-5 from that new
frozen sampled selection:

- `--max-pointcloud-samples`
- `--max-non-pointcloud-samples`

The full selection root should normally be built once with both limits omitted.
After that, smaller verification datasets can pass
`--sample-from-existing-selection-path <full-selection-root>` plus the two limit
flags. This reuses the full selected manifests and skips pointcloud directory
traversal, mapping, and hand-directory checks.

Changing `--action-chunk-size` or `--video-downsample-ratio` does not require a
new Step 2 selection. Reuse the same selection root. Rerun Step 4 only when
`--action-chunk-size` changes or when you want to force updated stats with
`--overwrite`; rerun Step 5 with the new explicit training parameters.

## Dataset Contract

Runtime dataset implementation: `wan_va/dataset/mot_dataset.py`.

### Dataset classes

- `MotTrainData` implements the common sample loading logic for both pointcloud
  and non-pointcloud rows.
- `MotGeometryLeRobotData` is the pointcloud manifest wrapper. It validates that
  all rows have pointcloud labels and that every view points to a
  `preprocessed_pointcloud_dir`.
- `MotPureLeRobotData` is the non-pointcloud manifest wrapper. It validates that
  no row has pointcloud labels.
- `MotBalancedMixDataset` wraps the pointcloud and pure datasets and interleaves
  them for training. The final sample contract is the same regardless of which
  child dataset produced the sample.

### Window geometry

Each sample is one MOT fixed window. With the default
`action_chunk_size=48` and `video_downsample_ratio=4`:

- One history action chunk covers 48 raw action steps before `current_frame`.
- One target action chunk covers 48 raw action steps starting at `current_frame`.
- Each chunk reads `action_chunk_size + 1 = 49` raw observation frames before
  downsampling.
- Downsampling by 4 produces 13 RGB frames per view for history and 13 RGB
  frames per view for target.
- Wan VAE encodes each 13-frame RGB sequence into 4 latent frames.
- The model therefore sees 8 latent frames total:

```text
[H0,H1,H2,H3,T0,T1,T2,T3]
```

`MotTrainData._getitem_window` chooses `current_frame` from the row's
`valid_start_range` unless an explicit start is requested through `get_window`.
It calls `mot_real_window_frame_ids` to produce:

- `history_frame_ids`: 13 raw frame IDs for the history RGB chunk.
- `target_frame_ids`: 13 raw frame IDs for the target RGB chunk.
- `frame_ids`: the concatenation of history and target frame IDs.
- `geometry_frame_ids`: 8 groups of 4 frame IDs, aligned with the 8 latent
  frames.
- `geometry_group_mask`: the structural mask for those 8 groups.

The anchor latent in each chunk uses only one real sampled frame and pads the
other three geometry slots with the same frame ID:

```text
H0/T0 anchor group: [frame0, frame0, frame0, frame0], structural mask [T,F,F,F]
H1..H3/T1..T3:      4 consecutive sampled frames, structural mask [T,T,T,T]
```

### Padding policy

The dataset clamps reads to episode boundaries but keeps separate validity masks
that describe whether the unclamped raw frame/action really existed.

`valid_start_range` is bounded so `MOT_MAX_RIGHT_PADDING_RAW_STEPS` directly
controls the maximum target-frame right padding. The rightmost start frame is:

```text
max_current = segment.end_frame + MOT_MAX_RIGHT_PADDING_RAW_STEPS - action_chunk_size - 1
```

The `-1` accounts for the left-closed/right-open segment convention:
`segment.end_frame - 1` is the last real frame. With the defaults
`action_chunk_size=48` and `MOT_MAX_RIGHT_PADDING_RAW_STEPS=16`, a 300-frame
episode has `max_current=267`, and the final requested target frame is
`267 + 48 = 315`, exactly 16 raw steps after the last real frame `299`.

- `_pad_frame_ids_to_segment` clamps every requested frame ID into
  `[segment.start_frame, segment.end_frame - 1]`.
- RGB loading uses the padded IDs so video decoding never requests an out-of-
  episode frame.
- Point loading uses the same padded IDs. For pointcloud rows, every padded ID
  must resolve through `PointStore`; for non-pointcloud rows, the dataset returns
  zero points and false masks.
- `history_raw_video_valid_mask`, `target_raw_video_valid_mask`, and `raw_video_valid_mask`
  preserve the validity of the original unclamped IDs.
- `video_latent_valid_mask` is derived from sampled-frame validity. Anchor
  latents are valid only if their anchor frame is valid; non-anchor latents are
  valid if any of the four sampled frames in that VAE group is valid.
- `geometry_group_valid_mask` combines the structural geometry group mask with
  the unclamped geometry frame validity.
- `geometry_point_valid_mask` combines point-store validity with
  `geometry_group_valid_mask`, so padded geometry slots never contribute to
  depth supervision.

This policy is important near episode boundaries. A partially valid final
visual group can still train video and produce a G token, but only the real
geometry slots participate in depth loss. A fully padded visual group is masked
out of video/G attention. Real action labels can still be supervised even when
the corresponding visual group is fully padded.

### RGB and point loading

RGB loading:

- `_load_rgb` iterates the row's `views`, converts local episode frame IDs to
  video frame IDs with `video_from_timestamp * fps`, then reads frames with
  `torchcodec.VideoDecoder`.
- RGB is returned as float tensors in `[0,1]`, shape `[F,V,3,H,W]`.
- `vae_rgb_history` and `vae_rgb_target` are loaded separately because the
  trainer encodes them as two Wan VAE chunks.
- `geometry_rgb` is loaded from the grouped geometry frame IDs and reshaped to
  `[8,4,V,3,H,W]`.

Point loading:

- For pointcloud rows, `_load_points` opens each view's `PointStore`, maps each
  local episode frame ID to a store row, reads points and the active valid mask,
  and stacks views into `[F,V,H,W,3]` plus `[F,V,H,W]`.
- For non-pointcloud rows, `_load_points` returns zeros and all-false masks with
  the same shapes. This keeps the sample schema identical for mixed training.
- `pts3d` / `valid_mask` are loaded for the 26 RGB sampled frames.
- `geometry_pts3d` / `geometry_point_valid_mask` are loaded for the 8x4 grouped
  geometry slots and then masked by `geometry_group_valid_mask`.

### Action loading

`_load_actions` reads raw 16D `action` and `observation.state` arrays from the
action cache or raw parquet, slices the current episode, and packs actions into
the model's frame-token layout:

- Raw action/state format is `[x,y,z,qx,qy,qz,qw,gripper] * 2`.
- The reference state for a chunk is `state[current_frame]` for the target chunk
  and the earliest available history reference for the history chunk.
- Absolute actions are converted to 20D chunk-reference relative actions:
  `[dx,dy,dz,rot6d_col0_col1,gripper] * 2`.
- The 48 target raw actions are packed into `T1..T3`, 16 action tokens per
  latent frame. `T0` is the target visual anchor and has no action loss tokens.
- History actions are packed into `H1..H3` as conditioning tokens only.
- Actions are normalized by the row's task stats:
  `(value - q01) / (q99 - q01 + 1e-6) * 2 - 1`, clipped to `[-1.5, 1.5]`,
  and zeroed where neither condition nor loss mask is active.

Action masks:

- `action_loss_mask` marks supervised target action tokens.
- `action_valid_mask` marks both history condition action tokens and supervised
  target action tokens. The model uses this for attention validity.
- `action_reference_states` stores the 16D reference state repeated over the
  corresponding action tokens, which is used for denormalization and evaluation.

### Sample fields

Main tensors returned by one unbatched sample:

- `vae_rgb_history`: `[13,V,3,H,W]`
- `vae_rgb_target`: `[13,V,3,H,W]`
- `rgb` / `vae_rgb`: `[26,V,3,H,W]`
- `geometry_rgb`: `[8,4,V,3,H,W]`
- `geometry_pts3d`: `[8,4,V,H,W,3]`
- `geometry_point_valid_mask`: `[8,4,V,H,W]`
- `geometry_group_valid_mask`: `[8,4]`
- `pts3d` / `valid_mask`: `[26,V,H,W,3]` / `[26,V,H,W]`
- `actions`: `[20,8,16,1]`
- `action_loss_mask`: `[20,8,16,1]`
- `action_valid_mask`: `[20,8,16,1]`
- `action_reference_states`: `[16,8,16,1]`
- `video_latent_loss_mask`: `[8]`
- `video_latent_valid_mask`: `[8]`
- `history_raw_video_valid_mask`: `[13]`
- `target_raw_video_valid_mask`: `[13]`
- `raw_video_valid_mask`: `[26]`
- `text_emb`: `[text_seq_len,text_dim]` padded by the text cache policy
- `action_q01` / `action_q99`: `[20]`

Frame/debug metadata returned by each sample:

- `history_frame_ids`, `target_frame_ids`, `frame_ids`: unclamped requested IDs.
- `padded_history_frame_ids`, `padded_target_frame_ids`, `padded_frame_ids`:
  clamped IDs actually used for RGB/point reads.
- `geometry_frame_ids`: `[8,4]` unclamped grouped geometry IDs.
- `padded_geometry_frame_ids`: `[8,4]` clamped grouped geometry IDs.
- `view_ids`: view IDs from the train manifest, currently left hand `0` and
  right hand `1`.
- `frame_stride`, `action_chunk_size`, `action_sequence_length`,
  `action_per_frame`, `sampled_video_frames_per_action_chunk_per_view`,
  `latent_frames_per_action_chunk_per_view`, and `vae_temporal_factor`.
- `meta`: runtime identifiers for logs and NaN diagnosis, including
  `sample_index`, `task_uid`, `source_lerobot_task_dir`, `data_file`,
  `episode_index`, `start_frame`, `source_dataset`, `has_pointcloud`, and
  `pointcloud_session_dir`.

### Trainer and model consumption

The trainer consumes this contract as follows:

- `_materialize_batch_latents` encodes `vae_rgb_history` and
  `vae_rgb_target` separately with Wan VAE, then concatenates them into
  `[B,C,8,H,W]` latents.
- The dataset returns `video_latent_loss_mask` as the target-frame-only region
  intersected with `video_latent_valid_mask`. History latents and `T0` stay clean
  conditioning; only valid `T1..T3` target latents get video noise/loss.
- `_prepare_joint_input_dict` consumes the dataset-provided video/action masks directly.
  Video masks live in `latent_dict`, while action masks live in `action_dict`.
- `_add_action_noise` uses the resulting `action_loss_mask` so only supervised
  target action tokens get action noise and action loss. `action_valid_mask` is
  passed to the model for attention validity.
- The joint route passes `geometry_rgb`, `geometry_pts3d`,
  `geometry_point_valid_mask`, `geometry_group_valid_mask`, and `stream_ids` as
  `geometry_dict`. The G-only route passes only RGB and slot validity to
  `train_geometry`; geometry targets and masks remain trainer-owned.
- The model flattens `geometry_rgb` from `[B,8,4,V,3,H,W]` into VGGTO image
  inputs, expands `geometry_group_valid_mask` into per-image validity, and packs
  selected-layer G tokens in `frame -> slot -> view -> register` order without pooling.
- VGGTO predicts `depth` / `depth_conf` and `points` / `points_conf`. Both use
  `geometry_pts3d`, masked by `geometry_point_valid_mask`.
- Batched validation expects the same fields with a leading batch dimension, for
  example `geometry_pts3d` becomes `[B,8,4,V,H,W,3]`.

## Loss

Total training loss:

```text
[video_loss_weight * latent_loss]
    + [action_loss_weight * action_loss]
    + [geometry_loss_weight * (
          depth_loss_weight * pooled_depth_loss
        + point_loss_weight * pooled_point_loss
      )]
```

Bracketed terms are present only when V, A, or G belongs to
`optimization_composition`. V is sample-balanced, A normalizes each valid frame
before the outer valid-frame mean, and G retains the VGGT/Pi3 pooled
valid-element reduction over the complete local batch. With world size `W` and
`K` geometry-active ranks, each active rank multiplies its pooled G scalar by
`W / K` before distributed gradient averaging. Inactive ranks contribute a
graph-connected zero; if `K = 0`, the geometry contribution is zero.

MOT batching uses a per-GPU view budget `M=max_views_per_gpu`. A synchronized
native view count `V in {2,3}` is selected for each microstep, and every rank
uses `B=floor(M/V)` samples from that V bucket. This applies to all seven
compositions, including G-active training. The current real-data validation is
V=2 only; V=3 remains structurally supported.

Default depth-loss configuration:

- `video_loss_weight = 1.0`
- `action_loss_weight = 1.0`
- `geometry_loss_weight = 1.0`
- `depth_loss_weight = 1.0`
- `gradient_loss_fn = "grad"`
- `point_loss_weight = 1.0`
- `point_gradient_loss_fn = "normal"`
- `valid_range = 0.98`
- `gamma = 1.0`
- `alpha = 0.2`

`optimization_composition` replaces branch-freeze flags. It controls parameter
ownership and active objectives; only G-only uses a specialized geometry
execution/data route, while every other composition keeps the joint forward.

Depth target:

```text
target_depth = geometry_pts3d[..., 2]
valid_mask   = geometry_point_valid_mask
```

VGGTO outputs `depth` and `depth_conf`. Predicted and target depth are both
scale-normalized per grouped slot by the valid-pixel mean of `||geometry_pts3d||`.
The point head follows the original VGGT DPT layout and semantics: signed XYZ uses `inv_log`,
confidence uses `1 + exp`, and point loss combines relative point distance,
confidence calibration, and multi-scale surface-normal consistency. Point targets
use the same per-slot/view scale as depth. These are local relative point maps,
not metric world coordinates. If a batch has too few valid pixels, each geometry
loss returns a graph-preserving zero.

## Attention Semantics

MOT attention uses explicit token order:

- order `0`: history video/G
- order `1`: history action
- order `2`: target video/G
- order `3`: target action

Key rules:

- Target video cannot read target action.
- Target action can read target video/G and history action.
- Target action cannot read clean target action slots.
- G cannot read video/action/text.
- G-to-G attention is bidirectional inside a chunk and causal across chunks.
- At selected layers, the full MoT G output directly replaces native VGGTO
  register attention; there is no pooling or delta writeback.
- Native and MoT G attention use the same chunk-causal clock during training.

Backends:

- `fa4`: default training backend for H100/SM90 FlashAttention-4 environments.
- `flex`: PyTorch FlexAttention path.
- `dense`: reference/debug path.

## Training

Install or activate the environment first; see `INSTALL.md`.

Set the required paths:

```bash
export MOT_DATASET_ROOT=/path/to/prepared_train_dataset
export WAN22_PRETRAINED_MODEL_PATH=/path/to/lingbot-va-base
export VGGTO_CHECKPOINT_PATH=/path/to/vggto-or-vggt-omega.pt
export VGGT_CHECKPOINT_PATH=/path/to/VGGT-1B/model.safetensors
```

Launch the default multi-GPU training script:

```bash
bash 1shell/train_mot_mixed_8gpu.sh
```

Training hyperparameters, data-loader settings, resume behavior, and checkpoint
settings are defined in `wan_va/configs/va_umi_3dwam_train_cfg.py`.

Run the module entry directly:

```bash
python -m wan_va.train_mot \
  --config-name umi_3dwam_train \
  --save-root train_logs/debug
```

Common environment variables:

- `MOT_DATASET_ROOT`: prepared dataset root; must contain `meta/mot_config.json`.
- `MOT_VIDEO_DOWNSAMPLE_RATIO`: must match the dataset config.
- `MOT_ACTION_CACHE_MANIFEST`: optional action cache manifest override.
- `MOT_EVAL_WITH_CPU=1`: launch one asynchronous CPU evaluation after a
  checkpoint is published. The local 4-GPU script enables this by default.
- `WAN22_PRETRAINED_MODEL_PATH`: LingBot checkpoint root.
- `INIT_MODEL_FROM_LINGBOT`: select LingBot (`true`, default) or Wan2.2
  (`false`) as the video/action initialization source.
- `WAN22_DIFFUSERS_MODEL_PATH`: Wan2.2 Diffusers model root; defaults to
  `/workspace/model/wan2_2_diffusers`.
- `VGGTO_CHECKPOINT_PATH`: required VGGTO checkpoint.
- `VGGT_CHECKPOINT_PATH`: required original VGGT checkpoint used for point-head initialization.
- `WANDB_MODE=offline`: default logging mode.

The optimizer keeps all parameters trainable. VGGTO native parameters use
`vggto_lr_multiplier=0.1`; LingBot, MoT, action heads, and new fusion parameters
use the main learning rate.

## Inference

The same fixed-window entrypoint supports three explicit modes:

```bash
export MOT_DATASET_ROOT=/path/to/prepared_dataset
export WAN22_MODEL_ROOT=/path/to/lingbot-va-base

python inference/mot_chunk_infer.py \
  --checkpoint-root /path/to/checkpoint_step_N \
  --mode video

python inference/mot_chunk_infer.py \
  --checkpoint-root /path/to/checkpoint_step_N \
  --mode geometry

python inference/mot_chunk_infer.py \
  --checkpoint-root /path/to/checkpoint_step_N \
  --mode full
```

`video` generates target video only. `geometry` predicts depth and pointcloud
from dataset GT multi-view RGB without loading the VAE. `full` generates video,
decodes it to RGB, recomputes geometry from the generated views, and then
predicts action from the generated V/G conditions.

Inference settings are grouped in `eval_cfg` in
`wan_va/configs/va_umi_3dwam_train_cfg.py`. Automatic evaluation overrides only
its execution device/backend to `cpu`/`dense`; manual GPU evaluation uses the
same defaults and `--device cuda:0`. DCP conversion runs in the evaluation
process and writes the atomic safetensors export after checkpoint publication.
If the previous CPU evaluation is still running, the next one is skipped rather
than starting an unbounded queue.

## Verification

Focused tests:

```bash
PYTHONPATH=. pytest \
  wan_va/tests/test_checkpoint_saving.py \
  wan_va/tests/test_register_mot_replacement.py \
  wan_va/tests/test_action_expert_init.py -q
```

Dataset and point-label tests:

```bash
PYTHONPATH=. pytest \
  wan_va/tests/test_mot_real_dataset.py \
  wan_va/tests/test_dataset_refactor_boundaries.py \
  wan_va/tests/test_preprocess_lumos_pointcloud.py \
  wan_va/tests/test_pointcloud.py \
  wan_va/tests/test_pointcloud_vis_mapping.py -q
```

Key regression coverage:

- VGGTO has no camera token, and `patch_token_start == num_register_tokens`.
- Shared registers are initialized by averaging first/other checkpoint registers.
- VGGTO frame permutation equivariance holds when the causal mask is disabled.
- VGGTO chunk-causal masks match dense/Flex/FA4 semantics.
- Grouped depth normalization covers valid and zero-valid slots.
- MOT trainer/model tests use VGGTO geometry metadata and relative depth/point losses.

## Notes

- Prepared point-label files still use the historical `local_points` name because
  they are raw 3D labels consumed by the dataset. This does not mean the model
  still has a Pi3 geometry tower.
- `v4_degisn.md` records the design plan used to create this branch.
- `wan_va_pi3_haoranVersion/` is an independent legacy/reference directory, not
  part of the current training path.
