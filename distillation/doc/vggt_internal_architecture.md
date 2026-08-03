# VGGT/VGGTO 内部结构与逐层作用

当前仓库中实际使用的不是一个单独叫 VGGT 的完整模型类，而是：

    VGGTOGeometryTower

它是基于 VGGT-Omega/VGGT 风格组件改造的 geometry 分支。

整体结构：

    输入 RGB
      ↓
    DINOv3 风格 patch encoder
      24 层
      ↓
    VGGTO 外层 geometry aggregator
      30 层
      ↓
      ├── DenseHead
      └── PointHead

更具体地说：

    RGB image
      ↓
    PatchEmbed
      ↓
    DINO Vision Transformer，24 层
      ↓
    去掉 DINO 的 CLS/storage tokens，只保留 patch tokens
      ↓
    VGGTO 添加 16 个 geometry registers
      ↓
    30 层 VGGTO geometry tower
      ├── 偶数层：geometry register attention
      └── 奇数层：cross-view + inter-frame attention
      ↓
    从第 5、15、21、29 层提取中间特征
      ↓
    DenseHead：depth/depth_conf
    PointHead：points/points_conf

相关代码：

- wan_va/modules/vggto_geometry.py
- wan_va/modules/vggto_vendored/layers/vision_transformer.py
- wan_va/modules/vggto_vendored/layers/block.py
- wan_va/modules/vggto_vendored/heads/dense_head.py
- wan_va/modules/vggto_vendored/heads/point_head.py

---

## 1. 默认输入和整体 shape

假设 geometry 输入为：

    B = 1
    G = 8
    S = 4
    V = 2
    C = 3
    H = W = 224

其中：

- B：batch size；
- G：latent geometry group 数；
- S：每个 group 的 geometry slot 数；
- V：同步相机数；
- C：RGB 通道数；
- H/W：图像尺寸。

原始 geometry RGB：

    geometry_rgb.shape
    == [1, 8, 4, 2, 3, 224, 224]

它的维度含义是：

    [batch, latent_group, geometry_slot, view, channel, height, width]

8 个 group、每个 4 个 slot、每个 slot 2 个 view，所以总图片数：

    G × S × V
    = 8 × 4 × 2
    = 64

完整 shape 变化：

    [1, 8, 4, 2, 3, 224, 224]
        ↓ reshape
    [1, 64, 3, 224, 224]
        ↓ DINO patch encoder
    [1, 64, 196, 1024]
        ↓ 加 VGGTO registers
    [1, 64, 212, 1024]
        ↓ 30 层 VGGTO geometry tower
    [1, 64, 212, 1024]
        ↓ depth/point heads
    depth / points / confidence

这里：

    196 = 14×14 patch tokens
    212 = 16 registers + 196 patch tokens
    64 = 8 groups × 4 slots × 2 views

---

## 2. 第一层：DINOv3 风格的 24 层 Patch Encoder

VGGTOGeometryTower 内部创建：

    DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        n_storage_tokens=4,
    )

配置：

| 配置 | 值 |
|---|---:|
| 输入图像 | 224×224 |
| patch size | 16 |
| patch 数量 | 14×14=196 |
| embedding dim | 1024 |
| Transformer 层数 | 24 |
| attention heads | 16 |
| storage tokens | 4 |
| FFN ratio | 4 |

### 2.1 PatchEmbed

单张 RGB：

    [3, 224, 224]

PatchEmbed 使用：

    Conv2d(
        in_channels=3,
        out_channels=1024,
        kernel_size=16,
        stride=16,
    )

空间尺寸：

    224 / 16 = 14

patch 数：

    14 × 14 = 196

shape 变化：

    [3, 224, 224]
        ↓
    [1024, 14, 14]
        ↓ flatten
    [196, 1024]

如果一次输入 64 张图片：

    [1, 64, 3, 224, 224]
        ↓ reshape
    [64, 3, 224, 224]
        ↓ patch embed
    [64, 196, 1024]

### 2.2 DINO 内部 token

DINO 内部每张图片有：

    1 CLS token
    4 storage tokens
    196 patch tokens

所以内部序列长度：

    1 + 4 + 196 = 201

内部 shape：

    [64, 201, 1024]

DINO 的 forward 输出字典大致包括：

    {
        "x_norm_clstoken": ...,
        "x_storage_tokens": ...,
        "x_norm_patchtokens": ...,
        "x_prenorm": ...,
        "masks": ...,
    }

VGGTO 只取：

    x_norm_patchtokens

因此 DINO 的 CLS 和 storage token 不会直接进入外层 VGGTO。

输出：

    [64, 196, 1024]

---

## 3. DINO 的每个 block

DINO 的每层都是标准的 pre-norm Transformer block：

    LayerNorm
      ↓
    Multi-Head Self-Attention + RoPE
      ↓
    LayerScale + residual
      ↓
    LayerNorm
      ↓
    MLP/FFN
      ↓
    LayerScale + residual

代码逻辑等价于：

    x1 = x + LayerScale1(
        SelfAttention(
            LayerNorm1(x),
            RoPE
        )
    )

    x2 = x1 + LayerScale2(
        MLP(
            LayerNorm2(x1)
        )
    )

### 3.1 DINO attention

每个 token 被投影成 Q/K/V：

    qkv = self.qkv(x)

默认：

    embedding dim = 1024
    heads = 16
    head dim = 1024 / 16 = 64

每张图片内部的 attention 近似为：

    [16 heads, 201 tokens, 64 dim]

Q/K 使用 2D RoPE。

DINO 内部处理的是单张图片：

    patch ↔ patch
    patch ↔ CLS/storage
    register-like storage ↔ patch

它不会在这 24 层直接处理：

    view 0 ↔ view 1
    group 0 ↔ group 1
    time 0 ↔ time 1

这些关系由外层 VGGTO 30 层处理。

### 3.2 DINO FFN

FFN ratio 为 4：

    输入维度：1024
    hidden 维度：4096
    输出维度：1024

结构：

    Linear(1024 → 4096)
      ↓
    GELU
      ↓
    Linear(4096 → 1024)

### 3.3 DINO 24 层的功能理解

代码中 24 个 block 的结构相同，但功能上可以粗略理解为：

| 层段 | 主要作用 |
|---|---|
| DINO 0~5 | 局部纹理、边缘、颜色和低级视觉特征 |
| DINO 6~11 | 局部区域、物体部件和空间关系 |
| DINO 12~17 | 更高层次的物体/场景区域关系 |
| DINO 18~23 | 最终单图 patch representation |

这是功能上的解释，不是代码中硬编码的层类别。

---

## 4. 第二层：VGGTO 自己添加 Geometry Registers

DINO 输出：

    [64, 196, 1024]

VGGTO 有一个共享的 register table：

    register_token.shape
    == [1, 1, 16, 1024]

扩展到 64 张图片：

    [64, 16, 1024]

与 DINO patch tokens 拼接：

    [64, 16, 1024]
    +
    [64, 196, 1024]
    =
    [64, 212, 1024]

恢复 batch 结构：

    [64, 212, 1024]
        ↓
    [1, 64, 212, 1024]

每张 geometry image 的 token 组成：

    token 0~15:
        VGGTO geometry registers

    token 16~211:
        DINO patch tokens

VGGTO registers 的作用：

- 汇聚当前 image 的 patch 信息；
- 在 geometry group 之间传播信息；
- 在多视角之间传播信息；
- 在后续 Video/Action/Geometry joint attention 中作为 Geometry token；
- 为 depth/point head 提供经过聚合的场景特征。

后续 MOT joint attention 中插入的主要是这些 geometry registers，而不是全部 patch tokens。

---

## 5. VGGTO 30 层 Geometry Aggregator

VGGTO 外层配置：

    depth = 30
    register_attention_indices = (0, 2, 4, ..., 28)
    cached_layer_indices = (5, 15, 21, 29)

每一层都先执行：

    1. per-image frame block

然后根据 layer 类型执行：

    2a. geometry register attention
    或
    2b. cross-view attention + inter-frame attention

所以每层不是只有一个 Transformer block，而是一个组合：

    frame block
      +
    relation block 或 register block

---

## 6. 每层共同的第一步：Frame Block

每层首先调用：

    frame_tokens = self.vggto.run_frame_block(
        state.tokens,
        patch_hw,
        layer_id,
    )

假设：

    state.tokens.shape == [1, 64, 212, 1024]

内部先 reshape：

    [1, 64, 212, 1024]
        ↓
    [64, 212, 1024]

这 64 个 image sequence 分别独立运行当前 layer 的 SelfAttentionBlock。

这一阶段只处理单图内部：

    registers ↔ patch tokens
    patch ↔ patch

暂时不处理：

    view 之间
    group 之间
    时间之间

输出再 reshape 回：

    [1, 64, 212, 1024]

VGGTO frame block 也是：

    LayerNorm
      ↓
    Self-Attention + Q/K normalization + RoPE
      ↓
    LayerScale + residual
      ↓
    LayerNorm
      ↓
    MLP
      ↓
    LayerScale + residual

相比 DINO block，VGGTO frame block 使用 Q/K normalization，并带有 VGGTO-Omega 兼容的 K bias mask。

---

## 7. 偶数层：Geometry Register Attention

偶数层：

    0, 2, 4, ..., 28

以 layer 0 为例。

frame block 输出：

    [1, 64, 212, 1024]

取前 16 个 register：

    g_register = frame_tokens[:, :, :16]

得到：

    [1, 64, 16, 1024]

64 个 image 实际对应：

    8 groups × 4 slots × 2 views

### 7.1 保存 layer_registers

因为 precompute_geometry 使用：

    capture_layer_registers=True

所以会保存：

    layer_registers[0] = g_register.contiguous()

最终：

    layer_registers = {
        0:  [1, 64, 16, 1024],
        2:  [1, 64, 16, 1024],
        4:  [1, 64, 16, 1024],
        ...
        28: [1, 64, 16, 1024],
    }

它们是：

    每个 joint geometry layer 的 register 输入快照

后续 Video/Action inference 会读取对应层的 register，并将其插入：

    NV, CV, G, NA, CA

的 joint attention 中。

### 7.2 Geometry register attention

对 registers 运行：

    block.geometry(
        g_register,
        geometry.register_rotary,
        geometry_meta,
        masked_attn_backend=...,
    )

主要过程：

    geometry registers
        ↓ flatten
    生成 Q/K/V
        ↓
    geometry masked self-attention
        ↓
    residual
        ↓
    geometry FFN
        ↓
    reshape 回 [B, G×S×V, R, C]

默认 8 个 group、每 4 个 group 一个 chunk：

    group 0~3 → order 0
    group 4~7 → order 2

geometry causal 规则：

    当前 group 可以看当前 group
    当前 group 可以看历史 group
    当前 group 不能看未来 group

所以：

    group 0~3：
        可以互相看

    group 4~7：
        可以看 group 0~3 和 group 4~7

    group 0~3：
        不能看 group 4~7

### 7.3 Invalid register 的恢复

register attention 输出后：

    register_override = self._restore_invalid_registers(
        register_override,
        g_register,
        geometry.image_valid_mask,
    )

逻辑：

    valid image/slot：
        使用更新后的 register

    invalid image/slot：
        恢复为原始 register

这样 padding slot 不会通过 attention 混合出新的虚假 geometry representation。

---

## 8. 奇数层：Cross-view Attention

奇数层：

    1, 3, 5, ..., 29

不使用 register joint attention，而是执行普通 geometry relation blocks。

先恢复：

    [B, G, S, V, N, C]

本例：

    [1, 8, 4, 2, 212, 1024]

然后对每个：

    group × slot

内的多个 view 做 attention。

例如：

    group 2, slot 1, view 0:
        [212, 1024]

    group 2, slot 1, view 1:
        [212, 1024]

拼成：

    [424, 1024]

执行 full attention。

这一步的作用：

    同一个时间 group、同一个 geometry slot 的多相机融合

例如：

    左相机看到物体左侧
    右相机看到物体右侧

cross-view attention 让二者形成更一致的 geometry representation。

---

## 9. 奇数层：Same-view Inter-frame Attention

cross-view 完成后，重新按 view 组织：

    [B, V, G, S, N, C]

再对每个 view 单独展开：

    [B×V, G×S×N, C]

本例：

    B×V = 2
    G×S×N = 8×4×212 = 6784

所以输入大致：

    [2, 6784, 1024]

使用 chunk-causal mask。

默认：

    groups_per_chunk = 4

所以：

    group 0~3 → chunk 0
    group 4~7 → chunk 1

可见性：

    chunk 0：
        只能看 chunk 0

    chunk 1：
        可以看 chunk 0 和 chunk 1

这一步的作用：

    同一个相机内部，
    沿 geometry group/时间方向传播场景信息，
    同时保持 chunk-level causal。

普通奇数层完成：

    同一 group、同一 slot：
        多 view 融合

    同一 view：
        跨时间 chunk 融合

---

## 10. 偶数层为什么不再执行普通 relation block

如果 complete_layer 收到：

    register_override

则直接：

    inter_tokens = torch.cat(
        [
            register_override,
            frame_tokens[:, :, patch_token_start:],
        ],
        dim=2,
    )

即：

    更新后的 registers
    +
    当前 frame block 输出的 patch tokens

此时不会再执行：

    cross-view block
    same-view inter-frame block

因为偶数层的跨 view/time register 交互已经由：

    block.geometry(...)

统一处理。

因此 30 层大致交替：

    Layer 0:
        frame block
        geometry register attention

    Layer 1:
        frame block
        cross-view attention
        inter-frame attention

    Layer 2:
        frame block
        geometry register attention

    Layer 3:
        frame block
        cross-view attention
        inter-frame attention

    ...

---

## 11. 30 层逐层表

默认配置：

    geometry depth = 30
    register_attention_indices = 0,2,4,...,28
    cached_layer_indices = 5,15,21,29

| Layer | Frame block | 关系操作 | layer register cache | head feature cache |
|---:|---|---|---|---|
| 0 | 是 | Geometry register attention | 是 | 否 |
| 1 | 是 | Cross-view + inter-frame | 否 | 否 |
| 2 | 是 | Geometry register attention | 是 | 否 |
| 3 | 是 | Cross-view + inter-frame | 否 | 否 |
| 4 | 是 | Geometry register attention | 是 | 否 |
| 5 | 是 | Cross-view + inter-frame | 否 | 是 |
| 6 | 是 | Geometry register attention | 是 | 否 |
| 7 | 是 | Cross-view + inter-frame | 否 | 否 |
| 8 | 是 | Geometry register attention | 是 | 否 |
| 9 | 是 | Cross-view + inter-frame | 否 | 否 |
| 10 | 是 | Geometry register attention | 是 | 否 |
| 11 | 是 | Cross-view + inter-frame | 否 | 否 |
| 12 | 是 | Geometry register attention | 是 | 否 |
| 13 | 是 | Cross-view + inter-frame | 否 | 否 |
| 14 | 是 | Geometry register attention | 是 | 否 |
| 15 | 是 | Cross-view + inter-frame | 否 | 是 |
| 16 | 是 | Geometry register attention | 是 | 否 |
| 17 | 是 | Cross-view + inter-frame | 否 | 否 |
| 18 | 是 | Geometry register attention | 是 | 否 |
| 19 | 是 | Cross-view + inter-frame | 否 | 否 |
| 20 | 是 | Geometry register attention | 是 | 否 |
| 21 | 是 | Cross-view + inter-frame | 否 | 是 |
| 22 | 是 | Geometry register attention | 是 | 否 |
| 23 | 是 | Cross-view + inter-frame | 否 | 否 |
| 24 | 是 | Geometry register attention | 是 | 否 |
| 25 | 是 | Cross-view + inter-frame | 否 | 否 |
| 26 | 是 | Geometry register attention | 是 | 否 |
| 27 | 是 | Cross-view + inter-frame | 否 | 否 |
| 28 | 是 | Geometry register attention | 是 | 否 |
| 29 | 是 | Cross-view + inter-frame | 否 | 是 |

两类缓存不同：

    layer_registers:
        0,2,4,...,28

    cached_outputs:
        5,15,21,29

---

## 12. cached_outputs：给 depth/point head 的特征

在 layer 5、15、21、29：

    cached_outputs[layer_id] = torch.cat(
        [
            frame_tokens,
            inter_tokens,
        ],
        dim=-1,
    )

如果：

    frame_tokens = [1,64,212,1024]
    inter_tokens = [1,64,212,1024]

则：

    cached_outputs[layer_id]
    = [1,64,212,2048]

保存：

    cached_outputs[5]
    cached_outputs[15]
    cached_outputs[21]
    cached_outputs[29]

这些是多尺度中间特征：

    较早层：
        更多局部细节和纹理

    中间层：
        更多多视角/跨时间 geometry 信息

    较深层：
        更多高级场景结构

DenseHead/PointHead 会去掉 register：

    [1,64,212,2048]
        ↓ 去掉前 16 个 register
    [1,64,196,2048]

然后将 196 个 patch tokens reshape 成：

    [64,2048,14,14]

供卷积和多尺度 feature fusion 使用。

---

## 13. DenseHead：depth 如何生成

DenseHead 使用四个缓存层：

    [5, 15, 21, 29]

每层处理过程：

    cached token
      ↓ 去掉 register
    [B×F, 2048, 14, 14]
      ↓ LayerNorm
      ↓ 1×1 Conv
      ↓ 多尺度 resize
    multi-scale feature
      ↓
    DPT 风格 scratch/fusion
      ↓
    prediction conv
      ↓
    pixel shuffle
      ↓
    depth/depth_conf

四个层的 resize scale：

    layer 5:
        ×4

    layer 15:
        ×2

    layer 21:
        ×1

    layer 29:
        ×0.5

然后通过：

    refinenet4
      ↓ 融合 layer 3
    refinenet3
      ↓ 融合 layer 2
    refinenet2
      ↓ 融合 layer 1
    refinenet1

得到融合 feature。

depth 激活：

    depth = exp(depth_logits)

并限制最大 geometry value：

    depth <= 100

confidence 激活：

    depth_conf = bounded_expp1(confidence_logits)

典型输出：

    depth:
        [B, G×S, V, H_d, W_d, 1]

    depth_conf:
        [B, G×S, V, H_d, W_d]

---

## 14. PointHead：3D points 如何生成

PointHead 也使用：

    layer 5
    layer 15
    layer 21
    layer 29

整体流程和 DenseHead 类似：

    patch features
      ↓
    LayerNorm
      ↓
    1×1 Conv
      ↓
    多尺度 resize
      ↓
    feature fusion
      ↓
    output conv

但最后输出 4 个 channel：

    x logit
    y logit
    z logit
    confidence logit

点坐标激活：

    point_magnitude = expm1(abs(point_logits))
    points = sign(point_logits) × point_magnitude

因此：

    正 logit → 正点坐标
    负 logit → 负点坐标

输出：

    points:
        [B, G×S, V, H_p, W_p, 3]

    points_conf:
        [B, G×S, V, H_p, W_p]

---

## 15. 一次完整输入的 shape 变化

使用：

    B=1
    G=8
    S=4
    V=2
    H=W=224
    patch_size=16
    embed_dim=1024
    registers=16

shape 变化：

    输入 geometry RGB:
    [1, 8, 4, 2, 3, 224, 224]

    展平 image:
    [1, 64, 3, 224, 224]

    DINO patch encoder 输入:
    [64, 3, 224, 224]

    DINO patch embedding:
    [64, 196, 1024]

    DINO 内部 token:
    [64, 201, 1024]
    其中：
        1 CLS
        4 storage
        196 patch

    DINO 输出给 VGGTO:
    [64, 196, 1024]

    加 VGGTO registers:
    [64, 212, 1024]

    恢复 batch:
    [1, 64, 212, 1024]

    每个 VGGTO layer:
    [1, 64, 212, 1024]

    layer register snapshot:
    [1, 64, 16, 1024]

    head cached feature:
    [1, 64, 212, 2048]

    去掉 registers:
    [1, 64, 196, 2048]

    depth:
    [1, 32, 2, H_d, W_d, 1]

    depth_conf:
    [1, 32, 2, H_d, W_d]

    points:
    [1, 32, 2, H_p, W_p, 3]

    points_conf:
    [1, 32, 2, H_p, W_p]

其中：

    64 = G×S×V = 8×4×2
    32 = G×S = 8×4
    196 = 14×14
    212 = 16+196

---

## 16. 每一类 layer 的一句话解释

### DINO 24 层

    每张图片内部的 patch token self-attention，
    负责从 RGB 提取单图视觉特征。

### VGGTO frame block

    每张 geometry image 内部继续更新 register/patch 表示。

### VGGTO cross-view block

    同一个 group、同一个 slot 的多相机 token 互相融合。

### VGGTO inter-frame block

    同一个 camera 内，不同 geometry group 按 chunk-causal 规则传播时间信息。

### VGGTO register attention

    处理 geometry registers，
    为后续 Video/Action/Geometry joint attention
    生成可复用的 G token。

### DenseHead

    融合 layer 5/15/21/29 的多尺度 patch features，
    输出 depth 和 depth confidence。

### PointHead

    融合 layer 5/15/21/29 的多尺度 patch features，
    输出 XYZ 点和 point confidence。

---

## 17. 最核心的理解

当前项目中的 VGGT/VGGTO 不是一个普通的单张图片深度估计网络，而是两层结构：

### 第一层：单图视觉编码

DINO 24 层：

    每张图片独立处理
    输出 patch features

### 第二层：结构化 geometry 聚合

VGGTO 30 层：

    处理多 view
    处理多 geometry slot
    处理时间 chunk
    维护 geometry registers
    提取 depth/point head 的多尺度特征

在 MOT 推理中，真正用于跨模态 Video/Action 交互的不是所有 geometry patch token，而主要是：

    VGGTO geometry registers

完整因果链：

    RGB
      ↓
    DINO patch features
      ↓
    VGGTO registers + patch features
      ↓
    多视角/跨时间 geometry aggregation
      ↓
    layer_registers cache
      ↓
    Video/Action joint attention

因此：

    DINO 负责看懂每张图；
    VGGTO 负责把多视角、多时间图像组织成 geometry；
    layer_registers 负责把 geometry 传给 Video/Action；
    DenseHead/PointHead 负责输出 depth 和 3D points。

