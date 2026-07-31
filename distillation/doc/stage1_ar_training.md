# Stage 1: 三模态因果 AR Mask 设计

> 范围：本阶段只实现并验证 Video/Action/Geometry (VAG) 的 attention mask。暂不接入
> `forward_train`、trainer、loss、checkpoint 或后续蒸馏阶段。

## 1. 输入与时间轴

每个 batch 必须同时保留 GT 条件、GT target 与 future noisy target：

```text
History:  GT video[12] + GT action[12] + GT geometry[12]
Future:   GT video[12] + GT action[12] + GT geometry[12]
          noisy video[12] + noisy action[12]
```

时间轴是 24 个 latent：

```text
[H00 ... H11] [F00 ... F11]
   12 GT history    12 GT future
```

`latents_per_chunk` 将连续 latent 分成 chunk，默认值为 1。所有 Video、Action、Geometry token
都映射到同一个 `chunk_id`；view、patch、point、register 只增加 token 数，不产生额外因果顺序。

动作 `A_c` 表示从前一视觉 chunk 到视觉 chunk `c` 的 transition。因此“action 能看下一个 chunk
video/geometry”在本设计中是 `A_c -> V_c/G_c`，而不是向未来读取 `V_{c+1}/G_{c+1}`。

## 2. 复用与新增

| 文件 | 复用或新增责任 |
| --- | --- |
| `wan_va/modules/mot_attention.py` | 复用 `MOTMaskMetadata` 的 metadata 思想、dense mask 可读参考、现有 V/A/G 无泄漏语义作为原版对照 |
| `wan_va/modules/model_3dva_mot.py:forward_train` | 暂不修改；未来只消费 adapter 输出的三流字段 |
| `wan_va/train_mot.py` | 暂不修改；未来复用 V/A noise、V/A valid/loss mask、FSDP 和训练生命周期 |
| `distillation/pipeline/causal_chunk_mask.py` | 新增 Stage 1 24-slot VA/VAG metadata 与 dense mask builder |
| `distillation/tests/test_visualize_multimodal_mask.py` | 原版与新 VA/VAG mask 的四图对比、文本审阅与关键 pair 断言 |

## 3. 三层 Mask 合同

| Mask | 作用 | 关键规则 |
| --- | --- | --- |
| Attention | `Bool[Q,K]`，决定 query 是否能读 key | 由 stream、clean/noisy 身份与 chunk 关系共同决定 |
| Validity | 每个 token 是否存在/有效 | invalid query 与 invalid key 的所有边均为 False |
| Loss | 哪些值参与训练损失 | 本次不接 loss；未来仅 future V/A，Geometry 为 condition-only |

这三层不得混用：future GT token 可以 valid，但仍必须因果不可见。

## 4. 统一 Token Metadata

| Stream | 输入来源 | `chunk_id` | token 有效位 |
| --- | --- | --- | --- |
| `CV` | GT history + GT future video | 对应 video chunk | video frame valid |
| `NV` | noisy future video | 对应 future chunk | future 且 video valid；history NV 无效 |
| `CA` | GT history + GT future action | 对应 action transition chunk | action token valid |
| `NA` | noisy future action | 对应 future chunk | future 且 action valid；history NA 无效 |
| `G` | GT history + GT future geometry | 对应 RGB 来源 chunk | geometry slot valid 展开到全部 G token |

物理 token 顺序固定为 `NV, CV, G(optional), NA, CA`，但任何可见性判断只能使用
`stream + chunk_id + token_valid`，不能依赖 token 在序列中的位置。

## 5. 新 Stage 1 Attention 规则

通用前置条件：`same_sample AND valid(query) AND valid(key)`。以下“过去”均指 `key_chunk < query_chunk`。

| Query | 允许读取 | 明确禁止 |
| --- | --- | --- |
| `NV_c` | `NV_c`；过去的 `CV/CA/G` | 当前/未来 clean V/A/G；其他 chunk noisy V/A |
| `CV_c` | `CV_{<=c}`；过去的 `CA/G` | future CV；当前/未来 CA/G；所有 noisy V/A |
| `NA_c` | `NA_c`；过去 `CA`；`CV/G_{<=c}` | `CA_c` 和未来 CA；future CV/G；其他 chunk noisy V/A |
| `CA_c` | `CA_{<=c}`；`CV/G_{<=c}` | future CA/CV/G；所有 noisy V/A |
| `G_c` | `G_{<=c}` | future G；所有 V/A |

关键结果：

```text
NV:F01 -> CV:F01   False    # video 只读过去 GT 条件
NV:F01 -> G:F01    False    # 禁止当前 RGB 派生几何泄漏
NA:F01 -> CV:F01   True     # action 可读 transition 终点的视觉条件
NA:F01 -> G:F01    True     # action 可读同一终点的几何条件
NA:F01 -> CA:F01   False    # 不可读取自身 clean action target
G:F01  -> NV:H11   False    # G 永不反向读取 V/A
```

这与原版的主要差异是：新规则显式使 history noisy V/A 无效，并把 video 对当前 G 的读取收紧为
“仅过去 G”；action 保留当前终点 V/G 条件，从而满足 inverse-dynamics 语义。

## 6. 可视化与验证

运行：

```bash
pytest -q distillation/tests/test_visualize_multimodal_mask.py
```

测试实例化真实的 `H=12, F=12, latents_per_chunk=1`，并输出：

```text
distillation/tests/artifacts/original_va_mask.svg
distillation/tests/artifacts/original_vag_mask.svg
distillation/tests/artifacts/stage1_va_mask.svg
distillation/tests/artifacts/stage1_vag_mask.svg
distillation/tests/artifacts/stage1_mask_original_vs_new.txt
```

四张 SVG 分别是原版 VA、原版 VAG、新 VA、新 VAG。横轴是 key、纵轴是 query；深色为可见、白色为
屏蔽，可直接在浏览器打开。TXT 按 query 列出所有允许的 key，便于逐项人工审阅。

测试还固定断言：video 不能读当前 future GT video/G；action 能读当前 transition endpoint 的 GT
video/G、不能读自身 clean action；G 只读 G；无效 history noisy token 在行和列两个方向都全 False。

## 7. 当前非目标

- 不改现有 `mot_attention.py`、模型 forward 或 attention backend。
- 不实现 adapter、loss、noise、trainer、checkpoint 或 Stage 2/3。
- 不生成或训练 geometry target；G 在 Stage 1 只作为有因果约束的 GT 条件流。
