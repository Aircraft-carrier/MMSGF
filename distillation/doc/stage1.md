# Stage 1: autoregressive Video+Action training

Stage 1 trains the parameter-compatible autoregressive MOT model using the
`segmented_history_va_v1` profile. The physical batch packing remains identical
to native MOT training; only token order changes from fixed chunks to a stable
history segment followed by frame-wise target order.

The training batch contains video latents, actions, their supervision and valid
masks, text embeddings, and per-view stream IDs. The objective is the weighted
sum of video and action flow losses.

The exported checkpoint contains the transformer config and safetensors plus
metadata recording the pure Video+Action architecture and generation profile.
Later distillation stages validate this profile before loading.
