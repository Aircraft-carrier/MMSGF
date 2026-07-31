# V4 VGGTO Geometry Plan

## Goal

Replace the Pi3 geometry path with a VGGTO-owned geometry path initialized from a VGGT-Omega checkpoint, while preserving the Pi3-like permutation-equivariant geometry idea where it matters.

VGGTO means a VGGT-Omega-derived geometry module. Code should use `VGGTO` / `vggto` naming and should not keep `Omega` as the production model name.

## Architecture Decisions

1. Use VGGTO as the geometry tower.
   - No Pi3 geometry tower.
   - No Pi3 point head.
   - No Pi3 checkpoint initialization path.
   - Delete Pi3 vendored model-side code after replacement.

2. Keep the dataset contract unchanged.
   - Keep `geometry_rgb`.
   - Keep `geometry_pts3d`.
   - Keep `geometry_point_valid_mask`.
   - Keep `geometry_group_valid_mask`.

3. Implement Pi3-like permutation-equivariant geometry inside VGGTO.
   - No camera token.
   - No camera head or camera loss.
   - No reference frame token.
   - No first-frame vs other-frame special-token split.
   - Use one shared register table for every frame/view.
   - Token order is `registers + patches`.
   - `patch_token_start = num_register_tokens`.
   - Initialize the shared register table from the checkpoint first/other register tables by averaging them by default.

4. Preserve anti-leakage training behavior.
   - VGGTO inter-frame attention is chunk-causal during training.
   - Both global inter-frame attention and register-only inter-frame attention are bidirectional inside a chunk and causal across chunks.
   - When causal masking is disabled, the geometry backbone should be permutation-equivariant over frame order.

5. Pair each target VGGTO layer with one LingBot step.
   - Expand the target VGGTO tower from 24 pretrained layers to 30 layers.
   - Each VGGTO layer already contains one frame/local block and one inter-frame/global block.
   - Align all 30 LingBot blocks one-to-one with the 30 VGGTO layers; there are no front-only blocks.
   - At the 15 even indices `(0, 2, ..., 28)`, replace native VGGTO register attention with MoT attention.
   - At the 15 odd indices, keep native VGGTO full-token inter-frame attention and an independent LingBot block.
   - Only VGGTO register tokens are exposed as G tokens.
   - Keep directional visibility: `G_next = f(G)` and `VA_next = f(VA, G)`; G cannot read VA/action/text.
   - Feed the full MoT G output into the VGGTO register path without delta writeback.

6. Use VGGT-style depth supervision.
   - Predict `depth` and `depth_conf`.
   - Target depth is `geometry_pts3d[..., 2]`.
   - Valid mask is `geometry_point_valid_mask`.
   - Normalize predicted and target depth per grouped slot by the valid mean `||geometry_pts3d||`.
   - If there are too few valid pixels, return zero depth loss while preserving the computation graph.
   - Total loss is `latent_loss + action_loss + depth_loss_weight * depth_loss`.

## Configuration

1. Remove `pi3_checkpoint_path`.
2. Add required `vggto_checkpoint_path` from `VGGTO_CHECKPOINT_PATH`.
3. Remove `pi3_lr_multiplier`.
4. Add `vggto_lr_multiplier = 0.1`.
5. Default VGGTO geometry settings:
   - patch size: 16
   - embed dim: 1024
   - target depth: 30
   - register tokens: 16
6. Keep current training resolution at 224.

## Initialization

1. Rename the combined initializer to `from_lingbot_and_vggto`.
2. Require `VGGTO_CHECKPOINT_PATH`; missing values should fail clearly.
3. Load all 30 LingBot blocks one-to-one.
4. Expand VGGTO-24 with target-to-source map `(0,1,2,3,None,4,None,5,6,7,None,8,9,10,None,11,12,13,14,15,16,17,None,18,None,19,20,21,22,23)`.
5. Copy each inserted frame block from its lower-side neighbor and zero both LayerScales so the new layer starts as an exact identity.
6. Move source dense-head cache layers `(4,11,17,23)` to native target layers `(5,15,21,29)` and load the dense head strictly.
7. Initialize every MoT G stream deterministically from its mapped/proxy source inter-frame block, without widening noise.
8. Run VA/action-to-G joint attention and every G-stream geometry update at full strength from initialization, without training or inference warmup state.
9. Ignore and report camera/text-alignment checkpoint keys, average first/other registers into the shared table, and load the VGGT point head strictly from `VGGT_CHECKPOINT_PATH`.

## FSDP, AC, Optimizer

1. Keep repo FSDP training, but remove Pi3-specific shard cases.
2. Shard VGGTO module boundaries: patch embed, VGGTO blocks, dense head, and root.
3. Keep VGGTO/VGGT-style internal execution intact.
4. Apply outer activation checkpointing only to LingBot/MoT blocks.
5. Put VGGTO pretrained modules in a lower-LR optimizer group using `vggto_lr_multiplier`.
6. Keep LingBot, MoT, and newly initialized fusion parameters on the main LR.

## Verification

1. Unit-test no-camera token layout and `patch_token_start`.
2. Unit-test first/other register averaging into shared registers.
3. Unit-test permutation equivariance when causal masking is disabled.
4. Unit-test chunk-causal mask semantics.
5. Unit-test grouped depth scale normalization and zero-valid behavior.
6. Run focused MOT trainer/model tests after integration.
