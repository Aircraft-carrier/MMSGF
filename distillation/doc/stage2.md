# Stage 2: consistency distillation

Stage 2 initializes a trainable autoregressive Video+Action student and a frozen
teacher from published transformer exports. It reuses the native batch builder,
then replaces only noisy/clean Video+Action trajectories for consistency
training.

Self rollout commits history and the known anchor to the incremental MOT cache.
Each later frame is sampled and committed in Video→Action order. Optional
ground-truth replacement restores the corresponding phase checkpoint and
replays downstream action state.

Rollout artifacts contain predicted/target latents, predicted/target actions,
valid masks, video previews, and cache diagnostics.
