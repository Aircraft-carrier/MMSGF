# Changelog: `umi_3dWAM_V4_vggt` vs `umi_3dWAM_V3`

This changelog describes the current working tree on `umi_3dWAM_V4_vggt` compared with `umi_3dWAM_V3`.

Important baseline note: at the time of review, `HEAD`, `origin/umi_3dWAM_V3`, and `umi_3dWAM_V3` all point to commit `b7106b6` (`Add tracker`). The changes described here are working-tree changes relative to that commit, including untracked files.

## Summary

The main direction is replacing the Pi3 geometry branch with a VGGTO geometry branch:

- Pi3 model-side code and vendored Pi3 modules are removed.
- New VGGTO geometry, VGGTO loss, and vendored VGGT-Omega-derived modules are added.
- The combined model initializer changes from LingBot/Pi3 to LingBot/VGGTO.
- The trainer changes from supervising predicted 3D points with masked L1 to supervising VGGTO depth and confidence.

There are also non-geometry differences:

- Pointcloud preprocessing now detects and rebuilds stale stores.
- The dataset download script points to a newer `DATE_ROOT`.
- Evaluation artifacts change from predicted point PLYs to depth tensors.
- Some training defaults change, especially geometry loss warmup.
- MoT architecture depth split changes from `12 + 18` to `6 + 24`.

## File-Level Change Overview

Modified files:

- `download_modelscope_pointcloud.sh`
- `script/prepare_data/preprocess_pointcloud.py`
- `wan_va/configs/va_umi_3dwam_train_cfg.py`
- `wan_va/modules/fa4_attention.py`
- `wan_va/modules/model_3dva_mot.py`
- `wan_va/modules/mot_attention.py`
- `wan_va/modules/mot_init.py`
- `wan_va/tests/test_mot_v3_attention.py`
- `wan_va/tests/test_mot_v3_trainer.py`
- `wan_va/tests/test_preprocess_pointcloud.py`
- `wan_va/train_mot.py`

Removed files:

- `wan_va/modules/pi3_geometry.py`
- `wan_va/modules/pi3_vendored/**`

Added untracked files:

- `v4_degisn.md`
- `wan_va/modules/vggto_geometry.py`
- `wan_va/modules/vggto_loss.py`
- `wan_va/modules/vggto_vendored/**`
- `wan_va/tests/test_vggto_geometry.py`

No diff was found in:

- `wan_va/dataset/**`
- `inference/**`

## Configuration Changes

File: `wan_va/configs/va_umi_3dwam_train_cfg.py`

### Checkpoint Configuration

V3:

- Uses `PI3_CHECKPOINT_PATH`.
- Provides a default Pi3X checkpoint path under the Hugging Face cache.

V4:

- Removes `pi3_checkpoint_path`.
- Adds `vggto_checkpoint_path`.
- Requires `VGGTO_CHECKPOINT_PATH` to be set in the environment.
- Raises immediately if `VGGTO_CHECKPOINT_PATH` is missing.

Impact:

- V4 no longer has a default geometry checkpoint.
- Importing the config now requires the VGGTO checkpoint environment variable unless tests monkeypatch it.
- This is stricter than V3 and can break tools that import the config without setting geometry checkpoint state.

### Geometry Loss Configuration

V3:

- `point_loss_weight = 1.0`
- `point_loss_warmup_steps = 1000`
- `pi3_lr_multiplier = 0.1`

V4:

- `depth_loss_weight = 1.0`
- `depth_loss_warmup_steps = 0`
- `gradient_loss_fn = "grad"`
- `valid_range = 0.98`
- `gamma = 1.0`
- `alpha = 0.2`
- `vggto_lr_multiplier = 0.1`

Impact:

- Geometry supervision is full-strength from step 0 in V4.
- V3 warmed up point supervision over 1000 steps.
- The new loss has extra hyperparameters for confidence, quantile filtering, and gradient regularization.

## Data Preparation Changes

File: `script/prepare_data/preprocess_pointcloud.py`

### Stale Store Detection

V3 reused an existing preprocessed pointcloud store whenever `_SUCCESS` existed and `overwrite=False`.

V4 adds `_is_current_pointcloud_store(store_dir)` and only reuses a store when all of the following are true:

- `_SUCCESS` exists.
- `local_points_224.npy` loads.
- `aligned_to_row.npy` loads.
- `metadata.npz` exists and includes `source_aligned_frame_indices`.
- `aligned_to_row` is exactly `0..N-1`.
- `metadata["aligned_frame_indices"]` is exactly `0..N-1`.
- The lookup length, aligned length, and points row count match.

If any check fails, V4 deletes and rebuilds the store.

Impact:

- Existing legacy stores with raw aligned-frame lookup tables are rebuilt instead of silently reused.
- This is a dataset preprocessing behavior change, not part of the model geometry replacement itself.
- It should improve correctness for the current local-frame-index format, but it can increase preprocessing time when old stores are present.

## Download Script Change

File: `download_modelscope_pointcloud.sh`

V3:

- `DATE_ROOT="20260409"`

V4:

- `DATE_ROOT="20260413"`

Impact:

- The pointcloud download target changes to a newer data root.
- This is outside the model implementation and can change the dataset contents used by downstream preprocessing.

## Model Architecture Changes

File: `wan_va/modules/model_3dva_mot.py`

### Geometry Tower Replacement

V3:

- Uses `Pi3GeometryTower`.
- Initializes from LingBot and Pi3 through `from_lingbot_and_pi3`.
- Maintains Pi3 encoder, local/global decoder blocks, point decoder, and point head.
- Returns `local_points` for geometry supervision and eval.

V4:

- Uses `VGGTOGeometryTower`.
- Initializes from LingBot and VGGTO through `from_lingbot_and_vggto`.
- Adds VGGTO frame blocks, inter-frame blocks, shared register tokens, and dense depth head.
- Returns `depth` and `depth_conf` instead of `local_points`.

Impact:

- The geometry output contract changes from 3D point prediction to depth plus confidence prediction.
- All downstream trainer and eval code must consume `depth` / `depth_conf`.

### MoT Layer Split

V3:

- `mot_num_front_layers = 12`
- `mot_num_layers = 18`
- First 12 LingBot blocks are front-only.
- Final 18 blocks run MoT with geometry.

V4:

- `mot_num_front_layers = 6`
- `mot_num_layers = 24`
- First 6 LingBot blocks are front-only.
- Final 24 blocks run MoT with VGGTO geometry.

Impact:

- This is a real architecture behavior change beyond swapping Pi3 for VGGTO.
- Twelve more LingBot blocks participate in MoT joint attention than V3.
- VGGTO has 24 layers, and each VGGTO layer already contains one frame/local block plus one inter-frame/global block, so V4 aligns one VGGTO layer to one MoT block.
- If the intended invariant is "only geometry model changes", this layer split should be explicitly reviewed.

### Geometry Token Flow

V3:

- Pi3 RGB encoding returns hidden geometry state, Pi3 positional encodings, image shape, G rotary embeddings, and optional slot masks.
- Each MoT layer runs one Pi3 decode step, including local and global/register updates.
- The model pools Pi3 global register states into compact G tokens.
- MoT register deltas are written back into Pi3 hidden state.
- Point head predicts `local_points`.

V4:

- VGGTO encodes grouped RGB into `VGGTOGeometryState`.
- RGB is flattened into `[B, G*S*V, 3, H, W]` for VGGTO.
- The model tracks:
  - `groups`
  - `group_size`
  - `views`
  - `slot_valid_mask`
  - `image_valid_mask`
  - `frame_group_ids`
  - `cached_outputs`
- Each MoT layer runs one VGGTO layer through `_run_vggto_group`.
- VGGTO group registers are pooled into G tokens.
- MoT register deltas are written back into VGGTO register tokens.
- Dense head uses cached intermediate outputs to predict depth and confidence.

Impact:

- Geometry state management is substantially different.
- Slot validity now affects VGGTO image masks and register pooling/writeback.
- Dense prediction depends on cached VGGTO layer outputs rather than a Pi3 point head.

### Standalone Geometry Forward

V3:

- `forward_geometry` runs Pi3 geometry and can return `local_points`.

V4:

- `forward_geometry` runs VGGTO grouped geometry and can return `depth` and `depth_conf`.
- Diagnostics now report VGGTO register token count and register-attention layer indices.

Impact:

- Any caller expecting `local_points` must be updated.
- Geometry eval output semantics are depth-based.

## New VGGTO Geometry Module

File: `wan_va/modules/vggto_geometry.py`

### Core Components

The new module introduces:

- `VGGTOGeometryState`
- `VGGTOGroupOutput`
- `VGGTOGeometryTower`
- `build_frame_causal_visibility_mask`
- `build_group_causal_visibility_mask`
- `init_shared_register_from_first_other_`

### VGGTO Tower Structure

`VGGTOGeometryTower` includes:

- DINOv3/VGGT-style patch embedding path, with a simple fallback patch embed.
- ResNet-style image normalization buffers.
- `frame_blocks`: per-frame transformer blocks.
- `inter_frame_blocks`: inter-frame transformer blocks.
- Shared register tokens.
- Dense depth head.
- Cached layer output collection for dense prediction.

Defaults:

- `patch_size = 16`
- `embed_dim = 1024`
- `depth = 24`
- `num_heads = 16`
- `num_register_tokens = 16`
- `register_attention_block_indices = (2, 6, 9, 14, 20)`
- `cached_layer_indices = (4, 11, 17, 23)`

### Causal Attention Semantics

VGGTO uses group-causal visibility:

- Keys are visible when `key_group_id <= query_group_id`.
- Optional image validity masks remove invalid keys.
- The diagonal is preserved so invalid slots can still self-attend safely.

Impact:

- VGGTO keeps the no-future-geometry invariant.
- The mask is organized around grouped slot/view images rather than Pi3 frame tokens.

### Register Pooling and Writeback

VGGTO pools registers over group slots and views:

- Without `slot_valid_mask`: mean over group slots and views.
- With `slot_valid_mask`: weighted mean over valid slots, including all views for valid slots.

Writeback applies MoT register deltas:

- To every slot/view register when no slot mask exists.
- Only to valid slots when `slot_valid_mask` exists.

Impact:

- Padded geometry slots do not receive MoT register deltas.
- G tokens represent grouped frame-level geometry, not per-slot geometry.

## New VGGTO Depth Loss

File: `wan_va/modules/vggto_loss.py`

### Target Preparation

`normalize_depth_targets(points, valid_mask)` expects:

- `points`: `[B, G, S, V, H, W, 3]`
- `valid_mask`: `[B, G, S, V, H, W]`

It computes:

- `dist = norm(points, dim=-1)`
- per-slot/view scale as the valid mean of `dist`
- target depth as `points[..., 2] / scale`
- invalid pixels set to zero

Impact:

- The trainer now keeps grouped pointcloud labels instead of flattening `G*S` before loss.
- Depth is scale-normalized per grouped slot/view.

### Predicted Input Shapes

The loss accepts predicted depth as either:

- `[B, G*S, V, H, W, 1]`
- `[B, G*S*V, H, W, 1]`

It accepts predicted confidence as either:

- `[B, G*S, V, H, W]`
- `[B, G*S*V, H, W]`

Impact:

- VGGTO dense head can return either frame/view-separated or flattened-view predictions.
- The loss reshapes predictions to match grouped targets internally.

### Loss Terms

`compute_vggto_depth_loss` includes:

- Regression loss on normalized depth difference.
- Confidence-weighted depth loss: `gamma * diff * conf - alpha * log(conf)`.
- Multi-scale gradient loss over normalized depth maps.
- Quantile filtering with `valid_range`.
- Minimum valid pixel guard, defaulting to `100`.

Returned metrics:

- `loss_conf_depth`
- `loss_reg_depth`
- `loss_grad_depth`
- `depth_valid_pixels`

Impact:

- This is a major supervision change from V3 masked point L1.
- V4 supervises depth and confidence, not full xyz point coordinates.
- Loss scale and optimization behavior are expected to differ.

## Initialization Changes

File: `wan_va/modules/mot_init.py`

### Loader Rename and Checkpoint Format

V3:

- `load_pi3_state_dict`
- Loads Pi3 safetensors or state dict.
- Requires Pi3 prefixes such as encoder, decoder, point decoder, point head, and register token.

V4:

- `load_vggto_state_dict`
- Supports directories containing common checkpoint names:
  - `model.safetensors`
  - `pytorch_model.bin`
  - `model.pt`
  - `model.pth`
- Normalizes `module.` and `model.` key prefixes.
- Extracts VGGTO model state from `aggregator.*` and `dense_head.*`.

Impact:

- V4 accepts a broader set of checkpoint file names.
- Required checkpoint key structure is now VGGT-Omega/VGGTO-specific.

### Required Key Validation

V3 required Pi3-related keys:

- `register_token`
- `encoder.*`
- `decoder.*`
- `point_decoder.*`
- `point_head.*`

V4 requires:

- `aggregator.patch_embed.*`
- `aggregator.frame_blocks.*`
- `aggregator.inter_frame_blocks.*`
- `aggregator.register_token`
- `dense_head.*`
- every frame/inter-frame layer up to configured VGGTO depth

Impact:

- Pi3 point prediction modules are no longer required.
- Dense depth head is now required.

### MoT Geometry Stream Initialization

V3:

- Initializes MoT G joint stream from Pi3 odd/global decoder layers.
- Copies Pi3 MLP, norms, Q/K norms, LayerScale, and attention weights.

V4:

- Initializes MoT G joint stream from VGGTO inter-frame blocks.
- Uses source prefix `aggregator.inter_frame_blocks.{min(idx, depth - 1)}`.
- Preserves the head-aware hidden mapping from 1024D geometry to 3072D LingBot/MoT carrier.

Impact:

- Each MoT block is initialized from an odd VGGTO inter-frame layer in the corresponding 2-layer group.
- The LingBot/MoT carrier widening strategy remains conceptually similar.

### Ignored Checkpoint Keys

V3 ignored Pi3 camera/conf keys.

V4 ignores:

- `camera_head.*`
- `text_alignment_head.*`
- `aggregator.camera_token*`

Impact:

- VGGTO keeps geometry and dense depth pieces while dropping camera/text alignment heads.

## Trainer Changes

File: `wan_va/train_mot.py`

### Model Construction

V3:

- Calls `ThreeDVAMOTTransformer3DModel.from_lingbot_and_pi3`.
- Passes LingBot path and Pi3 checkpoint path.
- Logs `loaded_pi3_required_keys` and `ignored_pi3_keys`.

V4:

- Calls `ThreeDVAMOTTransformer3DModel.from_lingbot_and_vggto`.
- Passes LingBot path and VGGTO checkpoint path.
- Logs `loaded_vggto_required_keys` and `ignored_vggto_keys`.

Impact:

- Startup now depends on a VGGTO checkpoint and VGGTO key validation.

### FSDP Sharding

V3 shards:

- `model.front_blocks`
- `model.mot_blocks`
- `model.pi3.local_blocks`
- `model.pi3.global_blocks`
- `model.pi3.encoder`
- `model.pi3.point_decoder`
- `model.pi3.point_head`
- model root

V4 shards:

- `model.front_blocks`
- `model.mot_blocks`
- `model.vggto.frame_blocks`
- `model.vggto.inter_frame_blocks`
- `model.vggto.patch_embed`
- `model.vggto.dense_head`
- model root

Impact:

- FSDP boundaries are adjusted to VGGTO module structure.
- There is no Pi3 point decoder/head sharding because those modules are removed.

### Optimizer Parameter Groups

V3:

- Splits parameters into `lingbot_and_mot` and `pi3_pretrained`.
- Uses `pi3_lr_multiplier`.
- Detects Pi3 pretrained parameters by `name.startswith("pi3.")`.

V4:

- Splits parameters into `lingbot_and_mot` and `vggto_pretrained`.
- Uses `vggto_lr_multiplier`.
- Detects VGGTO pretrained parameters by `name.startswith("vggto.")`.

Impact:

- The lower-LR geometry group is preserved, but now applies to `vggto.*`.
- MoT geometry adapters still train at the main LR because they live under `mot_blocks.*.geometry`.

### Geometry Batch Contract

V3:

- Trainer flattens geometry labels before passing to loss:
  - `geometry_pts3d.flatten(1, 2)`
  - `geometry_point_valid_mask.flatten(1, 2)`
- Shape effectively becomes `[B, G*S, V, H, W, 3]`.

V4:

- Trainer keeps original grouped labels:
  - `geometry_pts3d`
  - `geometry_point_valid_mask`
- Shape remains `[B, G, S, V, H, W, 3]`.

Impact:

- The new VGGTO loss owns flattening and depth normalization.
- Code that reads `input_dict["geometry_dict"]["pts3d"]` must now expect grouped labels.

### Geometry Loss

V3:

- `_masked_point_l1(pred["local_points"], pts3d, valid_mask)`
- Supervises full xyz point prediction.
- Applies mask directly over valid points.
- Returns:
  - `point_loss`
  - `point_loss_raw`
  - `point_loss_weight`

V4:

- `compute_vggto_depth_loss(pred["depth"], pred["depth_conf"], pts3d, valid_mask, ...)`
- Supervises normalized depth and confidence.
- Adds regression, confidence, and gradient terms.
- Returns:
  - `depth_loss`
  - `depth_loss_raw`
  - `depth_loss_weight`
  - `loss_conf_depth`
  - `loss_reg_depth`
  - `loss_grad_depth`
  - `depth_valid_pixels`

Impact:

- Total loss changes from:
  - `latent_loss + action_loss + point_weight * point_loss`
- To:
  - `latent_loss + action_loss + depth_weight * depth_loss`
- Geometry supervision is not numerically comparable across V3 and V4.

### Loss Warmup

V3:

- `point_loss_warmup_steps = 1000`

V4:

- `depth_loss_warmup_steps = 0`

Impact:

- V4 applies full geometry loss from the first training step.
- This can materially change early training stability and gradient balance.

### NaN and Finite-Loss Tracking

V3:

- Tracks `point_loss`, `point_loss_raw`, and `point_loss_weight`.
- Checks finite losses across `loss`, `latent_loss`, `action_loss`, and `point_loss`.

V4:

- Tracks `depth_loss`, `depth_loss_raw`, and `depth_loss_weight`.
- Checks finite losses across `loss`, `latent_loss`, `action_loss`, and `depth_loss`.

Impact:

- Diagnostics now focus on depth supervision instead of point supervision.

### Logging and WandB Metrics

V3 logs:

- `point_loss`
- `point_w`
- `loss_metrics/global_avg_point_loss`
- `loss_metrics/global_max_point_loss`
- `loss_metrics/point_loss_weight`

V4 logs:

- `depth_loss`
- `depth_w`
- `loss_metrics/global_avg_depth_loss`
- `loss_metrics/global_max_depth_loss`
- `loss_metrics/depth_loss_weight`

Impact:

- Existing dashboards comparing V3 point metrics will not line up directly with V4 depth metrics.
- Any downstream metric parser expecting point-loss keys must be updated.

### Evaluation Outputs

V3:

- `_run_pi3_eval` writes predicted point PLY files.
- `_run_va_eval` stores:
  - `action_condition_points.pt`
  - `standalone_points.pt`
- `evaluate_batch` saves predicted action-condition and standalone point clouds through `_save_eval_points`.

V4:

- `_run_vggto_eval` writes:
  - `vggto_depth.pt`
  - `vggto_depth_conf.pt`
- `_run_va_eval` stores:
  - `action_condition_depth.pt`
  - `standalone_depth.pt`
- Predicted action-condition and standalone point PLY export is removed.
- Ground-truth point PLY export remains when `pts3d` and `valid_mask` are present in the batch.

Impact:

- Evaluation artifact format changes.
- V4 no longer produces predicted point cloud PLYs for action-condition and standalone geometry.
- Visual comparison workflows based on predicted PLY files need replacement or adaptation.

## Attention Changes

Files:

- `wan_va/modules/mot_attention.py`
- `wan_va/modules/fa4_attention.py`

### MoT Attention

The code changes are primarily naming and documentation:

- Pi3 references are changed to VGGTO references.
- The G stream invariants remain the same:
  - G only attends to G.
  - G cannot read LingBot video/action/text tokens.
  - G->G uses frame-causal visibility.
  - LingBot VA/action can read G according to existing no-leak rules.

Impact:

- No substantive MoT mask logic change was found in the diff.
- The file now documents VGGTO registers instead of Pi3 registers.

### FA4 Attention

The code changes are primarily naming and error text:

- `FrameCausalMaskSpec` documentation now refers to VGGTO.
- FA4 frame-causal error message now says VGGTO.
- Comments refer to VGGTO registers instead of Pi3 registers.

Impact:

- No substantive FA4 mask logic change was found in the diff.
- The same frame-causal mask rule is retained.

## Tests

### `wan_va/tests/test_mot_v3_attention.py`

Changes:

- Pi3 frame-causal tests are renamed to VGGTO frame-causal tests.
- Test output path changes from `pi3_frame_causal_slot_valid_fa4_flex_pytorch_results.json` to `vggto_frame_causal_slot_valid_fa4_flex_pytorch_results.json`.
- Config import in one H100 test monkeypatches `VGGTO_CHECKPOINT_PATH`.
- Removed Pi3-specific `_frame_token_valid_ids` test.
- Added VGGTO group causal mask test through `build_group_causal_visibility_mask`.

Impact:

- Attention semantic coverage is preserved but retargeted to VGGTO naming and VGGTO mask helper.

### `wan_va/tests/test_mot_v3_trainer.py`

Changes:

- Geometry embedding test now verifies VGGTO group metadata rather than Pi3 slot compaction/restoration.
- Fake model outputs use `depth` and `depth_conf` instead of `local_points`.
- Config-related tests monkeypatch `VGGTO_CHECKPOINT_PATH`.
- Eval tests now expect action/standalone depth tensors instead of point tensors.

Impact:

- Trainer tests now reflect VGGTO depth-output contract.
- The previous explicit Pi3 valid-slot compaction behavior is no longer tested in the same way.

### `wan_va/tests/test_preprocess_pointcloud.py`

Changes:

- Adds regression coverage for stale `_SUCCESS` stores using raw aligned lookup tables.
- Verifies rebuild produces compact local `aligned_to_row` and preserves source aligned frame indices.

Impact:

- New data preprocessing behavior is covered.

### `wan_va/tests/test_vggto_geometry.py`

New test file covering:

- Shared VGGTO registers without camera token behavior.
- VGGTO frame permutation equivariance without causal masking.
- Depth loss zero-valid graph preservation.
- VGGTO depth loss helpers.

Impact:

- Adds direct coverage for the new VGGTO geometry and loss module.

## Removed Pi3 Code

Removed:

- `wan_va/modules/pi3_geometry.py`
- `wan_va/modules/pi3_vendored/**`

Impact:

- Pi3 model-side implementation is no longer present in the working tree.
- Any imports of `wan_va.modules.pi3_geometry` or `wan_va.modules.pi3_vendored` will fail.
- This is consistent with the V4 goal of removing Pi3 model-side code.

## New Vendored VGGTO Code

Added:

- `wan_va/modules/vggto_vendored/heads/**`
- `wan_va/modules/vggto_vendored/layers/**`

These provide:

- Dense depth head.
- Patch embedding.
- Attention blocks.
- RMS/layer norm helpers.
- RoPE position encoding.
- DINO/VGGT-style vision transformer pieces.

Impact:

- V4 carries its own VGGTO/VGGT-Omega-derived model-side dependencies.
- Geometry tower no longer depends on vendored Pi3/DINOv2 pieces.

## Known Compatibility Differences

### Import-Time Config Requirement

Because `VGGTO_CHECKPOINT_PATH` is required at config import time, tests or scripts that import `va_umi_3dwam_train_cfg` must set this environment variable. Some tests were updated with `monkeypatch.setenv`.

### Metric Key Changes

Consumers of logs must update from point keys to depth keys:

- `point_loss` -> `depth_loss`
- `point_loss_weight` -> `depth_loss_weight`
- `global_avg_point_loss` -> `global_avg_depth_loss`
- `global_max_point_loss` -> `global_max_depth_loss`

### Artifact Key Changes

Evaluation artifacts change:

- `action_condition_points.pt` -> `action_condition_depth.pt`
- `standalone_points.pt` -> `standalone_depth.pt`
- predicted PLY output for action-condition/standalone is removed
- `vggto_depth.pt` and `vggto_depth_conf.pt` are added

### Geometry Label Shape

V3 trainer loss path expected flattened geometry labels:

- `[B, G*S, V, H, W, 3]`

V4 depth loss expects grouped labels:

- `[B, G, S, V, H, W, 3]`

## Review Notes

The following areas appear consistent with the intended Pi3-to-VGGTO migration:

- Removing Pi3 vendored code.
- Adding VGGTO geometry tower and vendored VGGTO pieces.
- Loading VGGTO checkpoint state instead of Pi3 state.
- Replacing point supervision with VGGTO depth supervision.
- Updating FSDP and optimizer grouping for `vggto.*`.
- Updating tests and logs from point/Pi3 naming to depth/VGGTO naming.

The following areas are real behavior changes beyond a narrow model replacement:

- MoT layer split changes from `12 front + 18 MoT` to `6 front + 24 MoT`.
- Geometry loss warmup changes from 1000 steps to 0 steps.
- Pointcloud preprocessing now rebuilds stale stores.
- Download script changes dataset date root.
- Evaluation no longer writes predicted point PLY artifacts.

These differences may be intentional for V4, but they should be treated as explicit migration decisions rather than assumed equivalence with V3.
