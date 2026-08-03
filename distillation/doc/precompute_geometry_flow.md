# precompute_geometry：完整运行流程与数据示例

本文详细说明下面这段代码背后的完整运行流程：

    condition = transformer(
        _geometry_input(
            rgb=geometry_rgb,
            stream_ids=batch["stream_ids"].to(device),
            slot_valid_mask=batch["geometry_group_valid_mask"][
                :, :frame_count
            ].to(device, dtype=torch.bool),
            chunk_size=chunk_size,
            window_size=window_size,
            return_points=True,
        ),
        mode="precompute_geometry",
    )

这段代码的作用不是简单地把 RGB 输入 transformer，而是：

    将分组后的多视角 RGB 输入独立的 Geometry/VGGTO 分支，
    经过 geometry-only layers，
    输出 depth、depth confidence、3D points，
    同时缓存后续 Video/Action joint attention 需要的 geometry registers。

完整调用链：

    _geometry_input
          │
          ▼
    生成 geometry input dict
          │
          ▼
    transformer.forward(..., mode="precompute_geometry")
          │
          ▼
    precompute_geometry_condition
          │
          ▼
    _run_grouped_geometry_core
          │
          ├── _embed_geometry
          ├── _prepare_geometry_metadata
          ├── _run_geometry_layers
          │      ├── VGGTO frame block
          │      ├── Geometry register attention
          │      ├── cross-view attention
          │      ├── same-view inter-frame attention
          │      └── cached_outputs / layer_registers
          └── _geometry_predictions
                 ├── depth
                 ├── depth_conf
                 ├── points
                 └── points_conf

相关实现文件：

- inference/mot_inference.py
- wan_va/modules/model_3dva_mot.py
- wan_va/modules/vggto_geometry.py
- wan_va/modules/mot_attention.py

---

## 1. _geometry_input 实际构造的字典

辅助函数逻辑等价于：

    def _geometry_input(
        *,
        rgb,
        stream_ids,
        slot_valid_mask,
        chunk_size,
        window_size,
        return_points,
    ):
        return {
            "rgb": rgb.to(dtype=torch.bfloat16),
            "stream_ids": stream_ids,
            "chunk_size": int(chunk_size),
            "window_size": int(window_size),
            "return_points": bool(return_points),
            "slot_valid_mask": slot_valid_mask,
        }

所以调用后得到：

    {
        "rgb": geometry_rgb.to(torch.bfloat16),
        "stream_ids": stream_ids,
        "chunk_size": 4,
        "window_size": 4,
        "return_points": True,
        "slot_valid_mask": geometry_group_valid_mask,
    }

字段作用：

| 字段 | 作用 |
|---|---|
| rgb | 分组后的 geometry RGB |
| stream_ids | 相机/流 ID，保留在统一输入结构中 |
| chunk_size | geometry 的 chunk 因果粒度 |
| window_size | geometry attention 的 order 可见窗口 |
| return_points | 是否把 depth、points 等最终结果放入返回字典 |
| slot_valid_mask | 每个 geometry slot 是否有效 |

一个实现细节是：当前 geometry-only 路径主要依据 RGB shape、chunk_size 和 slot_valid_mask 工作。stream_ids 会随 input dict 传入，但在 _embed_geometry 和 geometry-only metadata 中不像 Video/Action joint 路径那样参与 V/A/G token 顺序构造。

---

## 2. 模拟真实输入

默认 MOT geometry 输入可以设为：

    B = 1
    G = 8
    S = 4
    V = 2
    C = 3
    H = W = 224

含义：

- B：batch size；
- G：latent geometry group 数；
- S：每个 group 的 geometry slot 数；
- V：同步相机数；
- C：RGB 通道数；
- H、W：VGGTO 输入分辨率。

geometry RGB shape：

    geometry_rgb.shape
    == [1, 8, 4, 2, 3, 224, 224]

维度语义：

    geometry_rgb[
        batch,
        latent_group,
        geometry_slot,
        view,
        rgb_channel,
        height,
        width
    ]

例如：

    geometry_rgb[0, 2, 1, 0]

表示：

    第 0 个样本
    第 2 个 latent group
    第 1 个 geometry slot
    第 0 个相机
    对应的一张 RGB 图片

geometry mask shape：

    geometry_group_valid_mask.shape
    == [1, 8, 4]

一种典型 mask：

    [
        [
            [True,  False, False, False],
            [True,  True,  True,  True],
            [True,  True,  True,  True],
            [True,  True,  True, True],
            [True,  False, False, False],
            [True,  True,  True, True],
            [True,  True,  True, True],
            [True,  True, True, True],
        ]
    ]

第一个 slot 只有一张真实 RGB，其余 slot 是为了统一 geometry group shape 而产生的 padding。

在视频生成阶段，目标 geometry 也可能被屏蔽：

    [
        [True,  False, False, False],
        [True,  True,  True,  True],
        [True,  True,  True,  True],
        [True,  True,  True,  True],
        [True,  False, False, False],
        [False, False, False, False],
        [False, False, False, False],
        [False, False, False, False],
    ]

---

## 3. transformer 的 mode 分发

transformer 是 MOT transformer 模型实例。

它的 forward 会根据 mode 分发：

    if mode == "precompute_geometry":
        return self.precompute_geometry_condition(input_dict)

因此这次调用实际执行：

    transformer.precompute_geometry_condition(geometry_input)

入口逻辑：

    def precompute_geometry_condition(self, input_dict):
        result = self._run_grouped_geometry_core(
            input_dict,
            capture_layer_registers=True,
        )

        layer_registers = result["layer_registers"]

        if set(layer_registers) != self.vggto.register_attention_indices:
            raise RuntimeError(...)

        out = {
            "final_geometry": result["final_geometry"],
            "layer_registers": layer_registers,
            "diagnostics": {
                "register_tokens": self.vggto.patch_start_idx,
                "register_attention_indices": sorted(
                    self.vggto.register_attention_indices
                ),
            },
        }

        if bool(input_dict.get("return_points", True)):
            out.update(
                depth=result["depth"],
                depth_conf=result["depth_conf"],
                points=result["points"],
                points_conf=result["points_conf"],
            )

        return out

这里 capture_layer_registers=True 很重要。

这说明该调用不仅为了得到 depth/points，还要保存每个 joint MOT 层的 geometry register snapshot，供后续 inference_video 或 inference_action 使用。

---

## 4. _run_grouped_geometry_core 总流程

核心函数可以概括为：

    def _run_grouped_geometry_core(
        self,
        input_dict,
        *,
        capture_layer_registers,
    ):
        chunk_size = int(input_dict["chunk_size"])
        window_size = input_dict.get("window_size")

        if window_size is None:
            window_size = 2 * math.ceil(
                input_dict["rgb"].shape[1] / chunk_size
            )

        geometry = self._embed_geometry(
            input_dict,
            chunk_size,
        )

        geometry_meta = self._prepare_geometry_metadata(
            geometry,
            int(window_size),
        )

        geometry, layer_registers = self._run_geometry_layers(
            geometry,
            geometry_meta,
            capture_layer_registers=capture_layer_registers,
        )

        depth, depth_conf, points, points_conf = (
            self._geometry_predictions(geometry)
        )

        return {
            "final_geometry": geometry,
            "layer_registers": layer_registers,
            "depth": depth,
            "depth_conf": depth_conf,
            "points": points,
            "points_conf": points_conf,
        }

可以拆成四步：

    1. 解析 chunk/window 配置
    2. RGB → geometry tokens
    3. 运行 geometry layers
    4. geometry tokens → depth/points

---

## 5. 第一步：_embed_geometry

_embed_geometry 负责：

- 检查 RGB 输入 shape；
- 处理 geometry slot mask；
- 将分组 RGB 展平为 image sequence；
- 通过 VGGTO encoder 产生 tokens；
- 创建 geometry register 的 rotary position embedding；
- 构造 GeometryContext。

### 5.1 检查 RGB shape

代码要求输入是：

    [B, G, S, V, 3, H, W]

本例：

    [1, 8, 4, 2, 3, 224, 224]

得到：

    bsz        = 1
    groups     = 8
    group_size = 4
    views      = 2
    channels   = 3
    height     = 224
    width      = 224

会检查：

    views >= 2
    height == 224
    width == 224
    groups % chunk_size == 0

默认：

    8 % 4 == 0

如果 views=1，会直接报错，因为该 geometry 分支要求同步多视角输入。

### 5.2 处理 slot_valid_mask

输入：

    slot_valid_mask.shape == [B, G, S]
    slot_valid_mask.shape == [1, 8, 4]

如果全部有效：

    if bool(slot_valid_mask.all().item()):
        slot_valid_mask = None

如果存在 invalid slot，则保留 mask，并构造 image-level mask：

    image_valid_mask = (
        slot_valid_mask[:, :, :, None]
        .expand(-1, -1, -1, views)
        .reshape(bsz, groups * group_size * views)
    )

本例总 image 数：

    G × S × V = 8 × 4 × 2 = 64

所以：

    image_valid_mask.shape == [1, 64]

如果 group 0 的 slot mask 是：

    [True, False, False, False]

两个 view 展开后是：

    [
        True,  True,
        False, False,
        False, False,
        False, False,
    ]

第一个 True/True 是 slot 0 的两个 view，后面的 False 是其他 padding slot 的两个 view。

### 5.3 展平 geometry RGB

原始：

    [B, G, S, V, 3, H, W]

执行：

    images = rgb.reshape(
        bsz,
        groups * group_size * views,
        channels,
        height,
        width,
    )

得到：

    [1, 8, 4, 2, 3, 224, 224]
    →
    [1, 64, 3, 224, 224]

64 张图片的顺序大致为：

    group 0:
        slot 0 view 0
        slot 0 view 1
        slot 1 view 0
        slot 1 view 1
        slot 2 view 0
        slot 2 view 1
        slot 3 view 0
        slot 3 view 1

    group 1:
        同样的 8 张

    ...
    
这一步只是把数据展平成 VGGTO 能处理的 image sequence，后续仍然使用 groups、group_size、views 恢复结构关系。

---

## 6. 第二步：VGGTO encode_grouped

_embed_geometry 接着调用：

    state = self.vggto.encode_grouped(
        rgb,
        slot_valid_mask=slot_valid_mask,
    )

对于 7 维输入，encode_grouped 再次展平：

    [B, G, S, V, 3, H, W]
    →
    [B, G×S×V, 3, H, W]

然后调用：

    self.encode(flat)

### 6.1 RGB normalization

VGGTO 首先做 ImageNet 风格归一化：

    normalized = (rgb - mean) / std

例如红色通道：

    mean_R = 0.485
    std_R  = 0.229

如果某个像素：

    rgb_R = 0.7

则：

    normalized_R
    = (0.7 - 0.485) / 0.229
    ≈ 0.939

### 6.2 Patch embedding

默认：

    patch_size = 16

输入图片：

    224 × 224

patch grid：

    224 / 16 = 14

每张图片的 patch token 数：

    14 × 14 = 196

VGGTO 还有：

    num_register_tokens = 16

因此每张图片最终 token 数：

    16 register tokens
    + 196 patch tokens
    = 212 tokens

如果 geometry embed dim 是 1024：

    patch_tokens.shape
    == [B×F, 196, 1024]

本例：

    F = G×S×V = 64
    patch_tokens.shape == [64, 196, 1024]

### 6.3 拼接 register token

模型的共享 register table：

    register_token.shape == [1, 1, 16, 1024]

展开到 64 张图片：

    register.shape == [64, 16, 1024]

拼接：

    tokens = torch.cat(
        [register, patch_tokens],
        dim=1,
    )

得到：

    tokens.shape == [64, 212, 1024]

再 reshape 回 batch/image 维：

    state.tokens.shape == [1, 64, 212, 1024]

其中：

    state.tokens[:, :, :16]
        是 register tokens

    state.tokens[:, :, 16:]
        是 patch tokens

所以：

    registers.shape == [1, 64, 16, 1024]
    patch_tokens.shape == [1, 64, 196, 1024]

### 6.4 VGGTOGeometryState

encode 返回：

    VGGTOGeometryState(
        tokens=...,
        patch_hw=(14, 14),
        image_hw=(224, 224),
        patch_token_start=16,
    )

本例：

    state.tokens.shape == [1, 64, 212, 1024]
    state.patch_hw == (14, 14)
    state.image_hw == (224, 224)
    state.patch_token_start == 16

这里的 64 是：

    8 groups × 4 slots × 2 views

不是 8 个 latent frame。

---

## 7. 创建 geometry rotary embedding

之后生成 geometry register 的 rotary position embedding：

    g_rotary = self.rope(
        self._geometry_register_grid(
            bsz,
            groups,
            group_size,
            views,
            self.vggto.patch_start_idx,
            state.tokens.device,
        )
    )[:, :, None]

geometry register 数：

    G × S × V × R
    = 8 × 4 × 2 × 16
    = 1024

geometry register grid 大致 shape：

    [B, 4, 1024]

每个 register 的位置坐标类似：

    [group_id, -1, -1, 0]

例如：

    group 0: [0, -1, -1, 0]
    group 1: [1, -1, -1, 0]
    ...
    group 7: [7, -1, -1, 0]

geometry register 不使用普通 patch token 的二维空间坐标，而是使用 group-level 时间位置。

---

## 8. GeometryContext 中保存的内容

_embed_geometry 最后返回一个 GeometryContext：

    GeometryContext(
        state=state,
        source_images=images,
        groups=groups,
        group_size=group_size,
        views=views,
        groups_per_chunk=chunk_size,
        register_rotary=g_rotary,
        cached_outputs=[None] * self.vggto.depth,
        slot_valid_mask=slot_valid_mask,
        image_valid_mask=image_valid_mask,
    )

本例：

| 字段 | 示例值/shape | 作用 |
|---|---|---|
| state.tokens | [1,64,212,1024] | 当前 geometry tokens |
| source_images | [1,64,3,224,224] | 展平后的 RGB |
| groups | 8 | latent geometry group 数 |
| group_size | 4 | 每组 slot 数 |
| views | 2 | 相机数 |
| groups_per_chunk | 4 | 一个 causal chunk 的 group 数 |
| register_rotary | 对应 1024 个 geometry registers | register 位置编码 |
| cached_outputs | 长度 30 的 list | depth/point head 中间层缓存 |
| slot_valid_mask | [1,8,4] 或 None | geometry slot validity |
| image_valid_mask | [1,64] 或 None | 展平到 image 的 validity |

---

## 9. 第三步：准备 Geometry metadata

接着执行：

    geometry_meta = self._prepare_geometry_metadata(
        geometry,
        int(window_size),
    )

### 9.1 Geometry token 数量

代码：

    geometry_tokens_per_frame = (
        geometry.group_size
        * geometry.views
        * self.vggto.patch_start_idx
    )

本例：

    4 × 2 × 16 = 128

这里的 per frame 实际上是 per latent geometry group。

8 个 group 总 geometry register token 数：

    8 × 128 = 1024

### 9.2 Geometry token validity

如果存在 slot_valid_mask：

    token_valid_ids = self._geometry_token_valid_ids(
        geometry.slot_valid_mask,
        batch_size=batch_size,
        groups=geometry.groups,
        views=geometry.views,
        register_tokens=self.vggto.patch_start_idx,
        device=...,
    )

shape 变化：

    [B, G, S]
    →
    [B, G, S, V, R]
    →
    [B, G×S×V×R]

本例：

    [1, 8, 4]
    →
    [1, 8, 4, 2, 16]
    →
    [1, 1024]

如果 group 0 的 slot mask 是：

    [True, False, False, False]

则：

    slot 0:
        2 views × 16 registers = 32 个 True

    slot 1:
        32 个 False

    slot 2:
        32 个 False

    slot 3:
        32 个 False

### 9.3 Geometry order

geometry order：

    geometry_order = (
        arange(groups) // groups_per_chunk
    ) * 2

本例：

    group 0~3 → order 0
    group 4~7 → order 2

Geometry-only attention 的 causal 规则：

    当前 group/chunk 可以看自己和过去；
    不能看未来 group/chunk。

---

## 10. 第四步：运行 _run_geometry_layers

核心调用：

    geometry, layer_registers = self._run_geometry_layers(
        geometry,
        geometry_meta,
        capture_layer_registers=True,
    )

逻辑：

    for layer_id, block in enumerate(self.mot_blocks):
        frame_tokens = self.vggto.run_frame_block(
            geometry.state.tokens,
            geometry.state.patch_hw,
            layer_id,
        )

        register_override = None

        if layer_id in register_attention_indices:
            g_register = frame_tokens[:, :, :patch_start_idx]

            layer_registers[layer_id] = g_register.contiguous()

            register_override = block.geometry(
                g_register,
                geometry.register_rotary,
                geometry_meta,
                masked_attn_backend=...,
            )

            register_override = restore_invalid_registers(...)

        geometry = complete_vggto_layer(...)

默认 geometry depth 是 30：

    layer_id = 0, 1, 2, ..., 29

register attention layers：

    0, 2, 4, ..., 28

普通 geometry layers：

    1, 3, 5, ..., 29

---

## 11. 每层先运行 frame block

每层首先执行：

    frame_tokens = self.vggto.run_frame_block(
        state.tokens,
        patch_hw,
        layer_id,
    )

输入：

    [1, 64, 212, 1024]

内部把 64 张图片看成 64 个独立 image sequence：

    [1, 64, 212, 1024]
    →
    [64, 212, 1024]

每张图片独立执行当前 layer 的 frame block：

    同一张图片内部：
        register tokens ↔ patch tokens

这个阶段还没有：

    不同 view 之间的交互
    不同 group/时间之间的交互

输出 reshape 回：

    [1, 64, 212, 1024]

---

## 12. Register attention layer

以 layer 0 为例。

frame tokens：

    [1, 64, 212, 1024]

取前 16 个 register：

    g_register = frame_tokens[:, :, :16]

得到：

    g_register.shape == [1, 64, 16, 1024]

64 个 image 对应：

    8 groups × 4 slots × 2 views

### 12.1 保存 layer_registers

因为 capture_layer_registers=True：

    layer_registers[0] = g_register.contiguous()

最终 cache 类似：

    layer_registers = {
        0:  [1, 64, 16, 1024],
        2:  [1, 64, 16, 1024],
        4:  [1, 64, 16, 1024],
        ...
        28: [1, 64, 16, 1024],
    }

它保存的是：

    每个 joint geometry layer 在 geometry-only 运行时的 register snapshot

后续 Video/Action inference 会读取对应 layer 的 register，并把它插入：

    NV, CV, G, NA, CA

的 joint attention 中。

### 12.2 Geometry register attention

然后：

    register_override = block.geometry(
        g_register,
        geometry.register_rotary,
        geometry_meta,
        masked_attn_backend=...,
    )

这个过程：

1. flatten geometry registers；
2. 生成 Q/K/V；
3. 使用 geometry metadata 进行 masked self-attention；
4. 执行 register residual；
5. 执行 geometry FFN；
6. reshape 回 [B, G×S×V, R, C]。

本例 geometry order：

    group 0~3 → order 0
    group 4~7 → order 2

因此：

    group 0~3 可以互相看；
    group 4~7 可以看 group 0~3 和 group 4~7；
    group 0~3 不能看 group 4~7。

如果 slot invalid，则对应：

    该 slot
      × 所有 view
      × 所有 register

都不会参与有效 attention。

### 12.3 恢复 invalid register

输出后执行：

    register_override = self._restore_invalid_registers(
        register_override,
        g_register,
        geometry.image_valid_mask,
    )

实际逻辑：

    torch.where(
        valid[:, :, None, None],
        updated,
        original,
    )

对于 invalid image/slot：

    使用原始 register
    不使用 attention 更新后的 register

例如 group 0 的 slot 1 无效时：

    slot 1 view 0 registers → 恢复原值
    slot 1 view 1 registers → 恢复原值

这样 padding slot 不会被更新成新的伪 geometry representation。

---

## 13. 普通 geometry layer

以 layer 1 为例，它不在 register attention indices 中。

此时：

    register_override = None

complete_layer 会执行：

    1. cross-view block
    2. same-view inter-frame block

### 13.1 Cross-view block

输入恢复为：

    [B, G, S, V, N, C]

本例：

    [1, 8, 4, 2, 212, 1024]

对每个：

    group × slot

内的不同 view 做 attention。

一个 group/slot 的两个 view 组成：

    view 0: 212 tokens
    view 1: 212 tokens

拼接：

    [424, 1024]

因此这一步实现：

    同一个时间 group、同一个 geometry slot 的多视角融合

例如：

    group 2 slot 1 view 0
             ↕
    group 2 slot 1 view 1

但不会在此阶段把 group 2 slot 1 和 group 2 slot 2 混在一起。

### 13.2 Same-view inter-frame block

cross-view 完成后，tensor 按 view 重新组织：

    batch × view × group × slot × token × channel

然后每个 view 单独沿 group/slot 方向展开：

    [B×V, G×S×N, C]

本例：

    B×V = 2
    G×S×N = 8×4×212 = 6784

因此：

    values.shape == [2, 6784, 1024]

使用 chunk-causal mask。

groups_per_chunk=4：

    group 0~3 → chunk 0
    group 4~7 → chunk 1

因果规则：

    chunk 0 只能看 chunk 0
    chunk 1 可以看 chunk 0 和 chunk 1

这一步实现：

    同一个相机内部，不同时间 geometry group 的因果交互

普通 geometry layer 的整体关系：

    同一 group 内：
        不同 view 交互

    同一 view 内：
        当前 chunk 与历史 chunk 交互

---

## 14. Register layer 为什么没有普通 cross-view/inter-frame block

如果 complete_layer 收到 register_override：

    inter_tokens = torch.cat(
        [
            register_override,
            frame_tokens[:, :, patch_token_start:],
        ],
        dim=2,
    )

也就是：

    更新后的 registers
    +
    当前 frame block 输出的 patch tokens

此时不会再执行：

    cross-view block
    same-view inter-frame block

因为 register layer 的跨 group/view 关系已经通过：

    block.geometry(...)

在 geometry register attention 中处理了。

因此 geometry tower 大致是：

    layer 0:
        frame block
        geometry register attention

    layer 1:
        frame block
        cross-view attention
        same-view inter-frame attention

    layer 2:
        frame block
        geometry register attention

    layer 3:
        frame block
        cross-view attention
        same-view inter-frame attention

    ...

---

## 15. cached_outputs 如何保存

VGGTO 有专门的 cached layer：

    cached_layer_indices = (5, 15, 21, 29)

这些层不能和 register attention layer 重叠。

complete_layer 中：

    if cached_outputs is not None and layer_id in cached_layer_set:
        cached_outputs[layer_id] = torch.cat(
            [frame_tokens, inter_tokens],
            dim=-1,
        )

frame_tokens 和 inter_tokens 都是：

    [1, 64, 212, 1024]

拼接后：

    cached_outputs[layer_id].shape
    == [1, 64, 212, 2048]

因此最终：

    cached_outputs[5]
    cached_outputs[15]
    cached_outputs[21]
    cached_outputs[29]

有值，其他位置为 None。

这些缓存用于：

    vggto.dense_forward(...)
    vggto.point_forward(...)

也就是：

    cached_outputs
        → depth head
        → depth confidence head
        → point head
        → point confidence head

它和 layer_registers 不同：

| 缓存 | 内容 | 后续用途 |
|---|---|---|
| layer_registers | register attention 层的 register snapshot | 后续 Video/Action joint attention |
| cached_outputs | 普通 geometry 层的 full-token features | depth/point head |
| final_geometry | 最终 geometry context | 后续 inference 的 geometry layout/rotary |

---

## 16. 得到最终 geometry state

30 层完成后：

    geometry.state.tokens

就是最终 geometry token 状态。

本例 shape 仍然是：

    [1, 64, 212, 1024]

但是它已经经历：

    每张图片内部的 frame block
    +
    多视角融合
    +
    同视角跨时间 chunk-causal 融合
    +
    geometry register attention

这就是：

    result["final_geometry"]

注意 final_geometry 实际上是一个 GeometryContext，而不只是单独的 tensor。

它包含：

    state.tokens
    patch_hw
    image_hw
    patch_token_start
    source_images
    groups
    group_size
    views
    groups_per_chunk
    register_rotary
    cached_outputs
    slot_valid_mask
    image_valid_mask

后续 Video/Action inference 主要使用：

    final_geometry.register_rotary
    final_geometry.groups
    final_geometry.group_size
    final_geometry.views

---

## 17. 生成 depth、depth_conf、points、points_conf

geometry layers 完成后：

    depth, depth_conf, points, points_conf = (
        self._geometry_predictions(geometry)
    )

内部调用：

    depth, depth_conf = self.vggto.dense_forward(
        geometry.cached_outputs,
        geometry.source_images,
    )

    points, points_conf = self.vggto.point_forward(
        geometry.cached_outputs,
        geometry.source_images,
    )

也就是：

    cached_outputs + source_images
            │
            ├── DenseHead
            │     ├── depth
            │     └── depth_conf
            │
            └── PointHead
                  ├── points
                  └── points_conf

### 17.1 输出 shape

输出会按照：

    output_prefix = (
        bsz,
        geometry.groups * geometry.group_size,
        geometry.views,
    )

组织。

本例：

    geometry.groups × geometry.group_size
    = 8 × 4
    = 32

所以典型 shape 为：

    depth:
        [B, G×S, V, H_d, W_d, 1]

    depth_conf:
        [B, G×S, V, H_d, W_d]

    points:
        [B, G×S, V, H_p, W_p, 3]

    points_conf:
        [B, G×S, V, H_p, W_p]

具体空间大小 H_d/W_d/H_p/W_p 由 geometry head 决定。

本例可以表示成：

    depth.shape
    == [1, 32, 2, H_d, W_d, 1]

    depth_conf.shape
    == [1, 32, 2, H_d, W_d]

    points.shape
    == [1, 32, 2, H_p, W_p, 3]

    points_conf.shape
    == [1, 32, 2, H_p, W_p]

后续 _run_geometry_from_rgb 会将 depth/depth_conf 从 [B, G×S, ...] 变成每个 latent frame 的 representative geometry 结果。

---

## 18. 最终 condition 字典

由于本次调用：

    return_points=True

最终返回：

    {
        "final_geometry": ...,
        "layer_registers": ...,
        "diagnostics": {
            "register_tokens": 16,
            "register_attention_indices": [
                0, 2, 4, ..., 28
            ],
        },
        "depth": ...,
        "depth_conf": ...,
        "points": ...,
        "points_conf": ...,
    }

字段含义：

### final_geometry

最终 GeometryContext：

    包含最终 geometry token、
    rotary position embedding、
    layout 信息、
    cached_outputs、
    slot/image validity 等。

### layer_registers

每个 joint geometry layer 的 register snapshot：

    {
        0:  [B, G×S×V, R, C],
        2:  [B, G×S×V, R, C],
        ...
        28: [B, G×S×V, R, C],
    }

本例每一项：

    [1, 64, 16, 1024]

### diagnostics

例如：

    {
        "register_tokens": 16,
        "register_attention_indices": [
            0, 2, 4, 6, 8, 10,
            12, 14, 16, 18, 20, 22,
            24, 26, 28,
        ],
    }

### depth

每个 geometry slot 的深度预测。

### depth_conf

深度置信度。

### points

每个 geometry slot 的三维点预测，最后一维通常是 3。

### points_conf

三维点置信度。

---

## 19. 一个更小的数值化例子

为了看清 mask 和 shape，使用一个更小的配置：

    B = 1
    G = 2
    S = 2
    V = 2
    R = 2

输入 RGB：

    geometry_rgb.shape
    == [1, 2, 2, 2, 3, 224, 224]

总图片数：

    G × S × V = 2 × 2 × 2 = 8

slot mask：

    [
        [
            [True, False],
            [True, True],
        ]
    ]

解释：

    group 0:
        slot 0 valid
        slot 1 invalid

    group 1:
        slot 0 valid
        slot 1 valid

展开为 image validity：

    group 0:
        slot 0 view 0 = True
        slot 0 view 1 = True
        slot 1 view 0 = False
        slot 1 view 1 = False

    group 1:
        slot 0 view 0 = True
        slot 0 view 1 = True
        slot 1 view 0 = True
        slot 1 view 1 = True

因此：

    image_valid_mask
    == [True, True, False, False, True, True, True, True]

如果每张图有 2 个 register，则 geometry token validity：

    group 0 slot 0:
        2 views × 2 registers = 4 个 True

    group 0 slot 1:
        4 个 False

    group 1 slot 0:
        4 个 True

    group 1 slot 1:
        4 个 True

最终：

    token_valid_ids.shape
    == [1, 2×2×2×2]
    == [1, 16]

如果 chunk_size=1：

    group 0 → geometry order 0
    group 1 → geometry order 2

因此 geometry attention：

    group 0 只能看 group 0
    group 1 可以看 group 0 和 group 1

group 0 的 invalid slot 1：

    RGB 可能仍然是 padding RGB，
    但对应 register 不能有效参与 attention，
    更新后的 register 最后会恢复成原始 register。

一个简化的 condition 结构：

    {
        "final_geometry": {
            "state.tokens": [1, 8, 212, 1024],
            "groups": 2,
            "group_size": 2,
            "views": 2,
        },
        "layer_registers": {
            0: [1, 8, 16, 1024],
            2: [1, 8, 16, 1024],
            ...
        },
        "depth": [1, 4, 2, H_d, W_d, 1],
        "depth_conf": [1, 4, 2, H_d, W_d],
        "points": [1, 4, 2, H_p, W_p, 3],
        "points_conf": [1, 4, 2, H_p, W_p],
    }

这里：

    4 = G × S = 2 × 2
    8 = G × S × V = 2 × 2 × 2

---

## 20. condition 如何被后续复用

视频阶段：

    video_geometry_condition = transformer(
        geometry_input,
        mode="precompute_geometry",
    )

之后每个视频 diffusion step 会传入：

    {
        "geometry_dict": {
            "rgb": video_geometry_rgb,
            "slot_valid_mask": video_geometry_slot_valid,
            "precomputed_condition": video_geometry_condition,
        },
        ...
    }

模型不会再次运行完整 geometry tower，而是读取：

    geometry_condition = input_dict["geometry_dict"][
        "precomputed_condition"
    ]

然后取得：

    layer_registers = geometry_condition["layer_registers"]
    geometry = geometry_condition["final_geometry"]

把对应 layer 的 G registers 插入：

    NV, CV, G, NA, CA

进行 joint masked attention。

动作阶段也一样：

    geometry, geometry_condition = _run_geometry_from_rgb(
        ...,
        geometry_rgb=video.action_geometry_rgb,
    )

之后所有 action diffusion step 复用这一份 geometry_condition。

因此完整推理中有两份不同的 geometry condition：

| condition | 历史 geometry | 目标 geometry | 用途 |
|---|---|---|---|
| video_geometry_condition | GT | 置零/invalid | 生成目标视频 |
| geometry_condition | GT | 生成视频 RGB | 生成目标动作 |

---

## 21. 最终总结

这段调用的真实流程是：

    1. geometry_rgb:
       [B, G, S, V, 3, 224, 224]

    2. _geometry_input:
       RGB 转 bfloat16，
       携带 slot mask、chunk/window 配置

    3. transformer.forward(mode="precompute_geometry"):
       分发到 precompute_geometry_condition

    4. _embed_geometry:
       [B, G, S, V, 3, 224, 224]
       →
       [B, G×S×V, 3, 224, 224]
       →
       VGGTO patch/register tokens

    5. Geometry layers:
       frame block
       + register attention 或 cross-view/inter-frame attention
       + geometry causal mask
       + invalid slot mask

    6. 缓存：
       layer_registers 给后续 Video/Action joint attention
       cached_outputs 给 depth/point heads

    7. Geometry heads:
       cached_outputs
       →
       depth/depth_conf/points/points_conf

    8. 返回 condition:
       final_geometry
       layer_registers
       diagnostics
       depth/depth_conf/points/points_conf

最关键的理解是：

    这个调用既是在预测 geometry，
    也是在准备后续 Video/Action 模态交互所需的 Geometry cache。

其中：

    depth/points
        是最终几何预测结果；

    final_geometry/layer_registers
        是后续 Video/Action inference 真正使用的几何条件。

