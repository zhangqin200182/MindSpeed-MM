# Kimi-K3 NPU 算子分析

本文档面向 MindSpeed-MM 中 Kimi-K3 模型的昇腾 NPU 算子适配，梳理模型涉及的全部定制算子、计算流程、NPU 替换策略，以及各算子在训练/推理链路中的调用关系。

## 模型架构概述

Kimi-K3 的模型实现由以下文件组成（合计约 3300 行）：

| 文件 | 行数 | 职责 |
|------|------|------|
| `mindspeed_mm/fsdp/models/kimi_k3/modeling_kimi_linear.py` | 1438 | 主体：KDA 注意力 + MoE + Norm + DecoderLayer |
| `mindspeed_mm/fsdp/models/kimi_k3/modeling_kimi_k3.py` | 1468 | VLM 包装：Vision Tower + 图文特征融合 + Loss |
| `mindspeed_mm/fsdp/models/kimi_k3/kimi_moe_patch.py` | 394 | MoE NPU 适配补丁（权重重组 + fused 算子） |
| `mindspeed_mm/fsdp/models/kimi_k3/configuration_kimi_k3.py` | — | 模型配置类（HF 仓库文件） |
| `mindspeed_mm/fsdp/models/kimi_k3/tokenization_kimi.py` | — | Tokenizer（HF 仓库文件） |

Kimi-K3 架构上融合了两种注意力机制和两种 FFN 结构：

```
                    ┌──────────────────────────┐
                    │     KimiDecoderLayer      │
                    │                           │
   hidden_states ──►│  ┌─────────────────────┐  │
                    │  │  MLA / KDA Attention │  │  ← 按层配置选择
                    │  └─────────────────────┘  │
                    │  ┌─────────────────────┐  │
                    │  │  MLP / SparseMoE     │  │  ← 按层配置选择
                    │  └─────────────────────┘  │
                    └──────────────────────────┘
```

- **MLA (Multi-Head Latent Attention)**：低秩压缩注意力，适配 DeepSeek-V3 架构
- **KDA (Kimi Delta Attention)**：线性注意力，Kimi-K3 核心创新能力，带 ShortConv + 门控状态更新
- **MLP**：标准 FFN，支持 SituAndMul 激活
- **SparseMoE**：带路由的稀疏专家混合，支持 EP（Expert Parallelism）

## 一、定制算子全景

Kimi-K3 涉及 **12 个关键算子**，按来源分为三层：模型专属算子（6 个）、共享 NPU 融合算子（4 个）、外部 NPU 后端算子（2 个）。

### 1.1 模型专属算子

| 算子 | 所在文件 | NPU 加速 | 用途 |
|------|---------|:---:|------|
| **KimiDeltaAttention** (KDA) | `modeling_kimi_linear.py:559` | ✅ triton_ascend | Delta Rule 线性注意力 |
| **KimiMLAAttention** (MLA) | `modeling_kimi_linear.py:416` | ✅ torch_npu FA | 低秩压缩 Flash Attention |
| **SituAndMul** | `modeling_kimi_linear.py:145` | ❌ PyTorch | 自定义激活函数 |
| **KimiK_3_MoeRMSNormGated** | `modeling_kimi_linear.py:98` | ✅ torch_npu | KDA 输出门控 RMSNorm |
| **KimiRMSNorm** | `modeling_kimi_linear.py:307` | ❌ PyTorch | 标准 RMSNorm |
| **PatchKimiSparseMoeBlock** | `kimi_moe_patch.py` | ✅ torch_npu MoE | NPU 适配版 MoE 块 |

### 1.2 共享 NPU 融合算子

| 算子 | 所在文件 | NPU API | 用途 |
|------|---------|---------|------|
| **Fused MoE Forward** | `ops/npu_patch/npu_fused_operator.py:78` | `npu_moe_token_permute`, `npu_grouped_matmul`, `npu_swiglu` | 完整 MoE 前向链路 |
| **Flash Attention** | `ops/flash_attn/flash_attn.py` | `npu_fusion_attention` | 支持 BNSD/BSND/TND/NTD layout |
| **RoPE** | `ops/npu_patch/npu_fused_operator.py:40` | `npu_rotary_mul` | 文本+视觉旋转位置编码 |
| **SwiGLU / GELU / RMSNorm** | `ops/npu_patch/`, `ops/swiglu.py` | `npu_swiglu`, `npu_gelu`, `npu_rms_norm` | 通用激活与归一化 |

### 1.3 外部 NPU 后端算子

| 算子 | 来源 | 用途 |
|------|------|------|
| **chunk_kda** | `triton_ascend_kernels.attention.fla.kda.chunk` | KDA 融合大算子（Triton → Ascend） |
| **causal_conv1d (AscendC)** | `fla_npu.ops.ascendc` | ShortConv 的 AscendC 实现 |

---

## 二、算子详解

### 2.1 SituAndMul — 自定义激活函数

**位置**：`modeling_kimi_linear.py:145`

**数学定义**：

```
SituAndMul(x) = β · tanh(gate/β) · σ(gate) · φ(up)
其中 φ(u) = linear_beta · tanh(u / linear_beta)   (若配置了 linear_beta)
     φ(u) = u                                        (默认)
```

**设计意图**：
- `β` 控制 gate 的非线性强度（β 越大越接近线性）
- `linear_beta` 对 up 分支施加有界压缩，防止激活值爆炸
- 通过 `ACT2FN["situ"]` 注册为 HuggingFace 官方激活函数

**计算设备**：纯 PyTorch，无 NPU 融合。Gate 和 up 均在 fp32 下计算再 cast 回原 dtype。

### 2.2 KimiRMSNorm / KimiK_3_MoeRMSNormGated — 归一化

**位置**：`modeling_kimi_linear.py:98, 307`

两个 RMSNorm 变体：

| | KimiRMSNorm | KimiK_3_MoeRMSNormGated |
|---|---|---|
| 使用场景 | MLA 子层 Norm | KDA 输出 Norm (`o_norm`) |
| NPU 融合 | ❌ | ✅ `torch_npu.npu_rms_norm` |
| 门控 | 无 | sigmoid(gate) |
| 计算精度 | fp32 | NPU: 硬件精度 / CPU: fp32 |

```python
# KimiK_3_MoeRMSNormGated.forward() — NPU 路径
hidden_states = torch_npu.npu_rms_norm(hidden_states, self.weight, eps)[0]
hidden_states = hidden_states * F.sigmoid(gate)

# CPU/GPU fallback
hidden_states = hidden_states.float()
hidden_states = hidden_states * torch.rsqrt(hidden_states.pow(2).mean(-1, keepdim=True) + eps)
hidden_states = self.weight * hidden_states.to(input_dtype)
hidden_states = hidden_states * F.sigmoid(gate.float()).to(input_dtype)
```

### 2.3 KimiMLAAttention — Multi-Head Latent Attention

**位置**：`modeling_kimi_linear.py:416`

改编自 DeepSeek-V3 MLA，核心特征：

```
hidden_states
  ├─ Q: q_a_proj (↓lora_rank) → RMSNorm → q_b_proj (↑num_heads × q_head_dim)
  │      └─ 拆分为 q_nope + q_rope
  │
  ├─ KV: kv_a_proj_with_mqa → 拆分为 k_compressed + k_rope
  │      └─ k_compressed → RMSNorm → kv_b_proj → 拆分为 k_nope + v
  │
  ├─ RoPE: 仅对 q_rope 和 k_rope 做旋转编码
  │      └─ Kimi-K3 使用 interleave 模式：npu_rotary_mul(x, cos, sin, rotary_mode="interleave")
  │
  └─ Flash Attention → output_gate (可选) → o_proj
```

**定制点**：
- **低秩压缩**：Q 路径 `q_lora_rank` 维压缩，KV 路径 `kv_lora_rank` 维压缩，显著减少 KV cache
- **NoPE 分离**：Q 的 nope/rope 分离 + KV 的 nope/rope/v 三路分离
- **Interleave RoPE**：Kimi-K3 专用 RoPE 布局，区别于 HuggingFace 默认的 half-turn 模式
- **Output Gate**：`g_proj(hidden_states).sigmoid()` 调制输出

**Flash Attention 调用链路**：

```
KimiMLAAttention.forward()
  → ALL_ATTENTION_FUNCTIONS["flash_attention_2"]
  → flash_attention_forward()          # ops/flash_attn/flash_attn.py
  → torch_npu.npu_fusion_attention()   # NPU 融合 FA
```

### 2.4 KimiDeltaAttention — KDA 注意力（核心）

**位置**：`modeling_kimi_linear.py:559`

KDA 是 Kimi-K3 最核心的定制算子，实现了基于 Delta Rule 的线性注意力。完整前向流程：

```
hidden_states [B, T, D]
  │
  ├─ ① Q/K/V 线性投影
  │    q = q_proj(x)          → [B, T, H·K]
  │    k = k_proj(x)          → [B, T, H·K]
  │    v = v_proj(x)          → [B, T, H·V]
  │
  ├─ ② ShortConvolution（时序平滑，kernel_size=4）
  │    q = ShortConv(q)       → causal_conv1d (Triton / AscendC)
  │    k = ShortConv(k)
  │    v = ShortConv(v)
  │
  ├─ ③ Gate & Beta（低秩门控）
  │    g = f_b_proj(f_a_proj(x))     → [B, T, H·K]  # 低秩 gate
  │    beta = b_proj(x).sigmoid()     → [B, T, H]    # 数据依赖衰减
  │
  ├─ ④ chunk_kda() ★ 融合大算子
  │    ├─ l2norm(q), l2norm(k)                       ← fused
  │    ├─ gate transform: -exp(A_log)·softplus(g+dt_bias)  ← fused
  │    ├─ chunk cumsum (chunk_size=64)                ← fused
  │    ├─ intra-chunk: WY recurrence + delta rule     ← fused
  │    └─ inter-chunk: state decay + delta write       ← fused
  │    输出: o [B, T, H, V], recurrent_state [N, H, V, K]
  │
  ├─ ⑤ Output Gate + Norm
  │    g = g_b_proj(g_a_proj(x))   # 或 g_proj(x) for full_rank
  │    o = KimiK_3_MoeRMSNormGated(o, g)
  │
  └─ ⑥ o_proj → [B, T, D]
```

**两种 KDA 实现模式**：

```python
# modeling_kimi_linear.py:692
kda_implementation = getattr(config, "kda_implementation", "fused")

if kda_implementation == "fused":
    # 融合大算子 — triton_ascend_kernels
    o, recurrent_state = chunk_kda(q, k, v, g, beta,
        A_log=A_log, dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        ...
    )
elif kda_implementation == "naive":
    # 小算子参考实现 — chunk_kda_naive（纯 PyTorch）
    o, recurrent_state = chunk_kda_naive(q, k, v, g, beta,
        A_log=A_log, dt_bias=dt_bias,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        ...
    )
```

| 模式 | 实现 | 文件 | NPU | 用途 |
|------|------|------|:---:|------|
| `"fused"` | `chunk_kda` from `triton_ascend_kernels` | `ops/kda/triton_ascend/chunk.py` | ✅ | 训练+推理（默认） |
| `"naive"` | `chunk_kda_naive` | `ops/kda/chunk_kda_naive.py` | ❌ | 精度验证 / 调试 |

**KDA chunk_kda 算子内部算子融合**：

| 子操作 | 是否融合 | 融合位置 |
|--------|:---:|------|
| q/k l2norm | ✅ | kernel 内 `l2norm_fwd` |
| gate transform (`-exp(A_log)·softplus(g+dt_bias)` 或 `lower_bound·sigmoid`) | ✅ | kernel 内 gate cumsum |
| chunk cumsum | ✅ | kernel 内 `chunk_local_cumsum` |
| beta sigmoid | ❌ | kernel 外部预处理（`fused_beta_sigmoid` for fused 模式） |
| intra-chunk WY recurrence | ✅ | kernel 内 forward substitution + matmul |
| inter-chunk recurrence | ✅ | kernel 内 state decay + delta rule |
| GVA (Grouped Value Attention) | ✅ | kernel 内 `HV > H` 时自动处理 |

**ShortConvolution 的双后端**：

```
训练（chunk 模式）
  → ShortConvolution.forward()
  → causal_conv1d (triton)                    ← GPU/Triton 后端
  → causal_conv1d_ascendc (fla_npu AscendC)   ← NPU 后端

推理（step 模式 / decode）
  → ShortConvolution.step()
  → causal_conv1d_update (triton)             ← 逐 token 更新
```

### 2.5 KimiMoEGate + SparseMoeBlock — 稀疏 MoE

**位置**：`modeling_kimi_linear.py:781, 876` + `kimi_moe_patch.py`

**路由逻辑**（改编自 DeepSeek-V3）：

```
hidden_states
  ├─ scoring: weight → sigmoid/softmax（由 scoring_func 控制）
  ├─ top_k 选择 + group 限制
  ├─ routed_scaling_factor 缩放
  └─ 输出: topk_weight, topk_ids
```

**MoE Patch 的关键改造**（`kimi_moe_patch.py`）：

原始 Kimi 模型每个 expert 使用独立 Linear layer，权重布局为：

```
# 原始布局（不兼容 NPU fused MoE）
expert_0.w1, expert_0.w2, expert_0.w3  # 独立的 Linear
expert_1.w1, expert_1.w2, expert_1.w3
...

# Patch 后布局（兼容 NPU fused MoE + EP）
gate_up_proj: [num_experts, hidden_size, 2 × intermediate_size]  # 3D tensor
down_proj:    [num_experts, intermediate_size, hidden_size]       # 3D tensor
```

**Fused MoE Forward 完整链路**（`ops/npu_patch/npu_fused_operator.py:78`）

```
                     ┌─────────────┐
hidden_states ──────►│  permute()  │  torch_npu.npu_moe_token_permute
                     │  tokens 按 expert 重排
                     └──────┬──────┘
                            ▼
                     ┌──────────────┐
                     │ grouped_matmul│  torch_npu.npu_grouped_matmul
                     │ gate_up_proj  │  [M, H] @ grouped([H, 2×I])
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │   swiglu()   │  torch_npu.npu_swiglu
                     │   SiLU·gate  │
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ grouped_matmul│  torch_npu.npu_grouped_matmul
                     │  down_proj    │
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │ unpermute()  │  torch_npu.npu_moe_token_permute
                     │ tokens 恢复   │  + routing_weights 加权
                     │ 原始顺序      │
                     └──────┬───────┘
                            ▼
                       output
```

**共享 Expert（Shared Expert）**：与 routed experts 并行计算，结果相加。在 EP 场景下通过 `all2all_grouped_matmul` / `grouped_matmul_all2all` 实现跨卡通信+计算融合（`ops/moe_ops/gemm_mc2.py`）。

### 2.6 Flash Attention — NPU 融合实现

**位置**：`ops/flash_attn/flash_attn.py`

Kimi-K3 的 MLA 分支和 VLM 交叉注意力均走此实现。核心逻辑：

```python
# flash_attn.py:501 — 主调用路径
if IS_NPU_AVAILABLE and module.config._attn_implementation in ["flash_attention_2", "flash_attention_3"]:
    # NPU 融合 FA
    attn_output = torch_npu.npu_fusion_attention(
        query, key, value, head_num,
        input_layout=layout,          # BNSD / BSND / TND / NTD
        atten_mask=attention_mask,
        scale=scaling,
        sparse_mode=3 if is_causal else 0,
        ...
    )
```

**支持的 Layout**：

| Layout | 形状 | 使用场景 |
|--------|------|---------|
| `BNSD` | [Batch, Heads, Seq, Dim] | 标准批量处理 |
| `BSND` | [Batch, Seq, Heads, Dim] | HuggingFace 默认 |
| `TND` | [Total_tokens, Heads, Dim] | Packed / varlen 序列 |
| `1TND` | [1, Tokens, Heads, Dim] | 单 batch packed |
| `1NTD` | [1, Heads, Tokens, Dim] | 单 batch packed (head-first) |

**Context Parallel (Ring Attention) 支持**：仅在 NPU + FA2/FA3 下可用，结合 Ulysses SP 实现混合并行：

```
CP (Ring Attention)         SP (Ulysses)
  ┌─────┐  ┌─────┐         ┌─────┐  ┌─────┐
  │ rank│  │ rank│         │ rank│  │ rank│
  │  0  │  │  1  │         │  0  │  │  1  │
  └──┬──┘  └──┬──┘         └──┬──┘  └──┬──┘
     │ ring   │               │ all2all│
     ▼        ▼               ▼        ▼
  序列维度切分              注意力头维度切分
```

### 2.7 RoPE — 旋转位置编码

**位置**：`ops/npu_patch/npu_fused_operator.py:40`

Kimi-K3 使用两种 RoPE 模式：

| API | 模式 | 使用场景 | 文件 |
|-----|------|---------|------|
| `npu_rotary_mul(q, cos, sin)` | half-turn（默认） | MLA 文本部分 | `npu_fused_operator.py:45` |
| `npu_rotary_mul(q, cos, sin, rotary_mode="interleave")` | interleave | Kimi-K3 Vision RoPE | `modeling_kimi_k3.py:274` |
| `npu_rotary_mul(q_4d, cos_4d, sin_4d)` | 4D vision half-turn | 视觉 RoPE（Qwen3VL/Omni） | `npu_fused_operator.py:63` |

Interleave 模式下，cos/sin 以交替方式与 hidden dims 交互，区别于标准的 half-turn 拼接方式：

```
half-turn:    [x1, x2] → [x1*cos - x2*sin, x2*cos + x1*sin]
interleave:   [x1, x2, x3, x4, ...] → [x1*c1, x2*s1, x3*c2, x4*s2, ...]
```

**性能优化**：小序列时走 `npu_rotary_mul` 融合路径；大序列时退化为显式复数乘法（避免 kernel launch 开销 > 计算收益）：

```python
if q.shape[0] * q.shape[1] <= 128:
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)    # 融合
else:
    q_embed = (q * cos) + (_rotate_half(q) * sin)       # 显式
```

---

## 三、计算流程总图

### 3.1 KimiDecoderLayer 完整前向（训练模式）

```
Input: hidden_states [B, T, D]

┌─────────────────────────────────────────────────────────┐
│ ① Pre-Attention Norm                                    │
│    KimiRMSNorm                                          │
├─────────────────────────────────────────────────────────┤
│ ② Attention（按层配置二选一）                              │
│                                                         │
│  ┌─ MLA Path ─────────────────────────────────────┐     │
│  │ Q/KV 低秩投影 → RMSNorm → 拆 nope/rope         │     │
│  │ RoPE (interleave) → Flash Attention (NPU FA)   │     │
│  │ → output gate → o_proj                          │     │
│  └────────────────────────────────────────────────┘     │
│  ┌─ KDA Path ─────────────────────────────────────┐     │
│  │ Q/K/V 投影 → ShortConv (AscendC)               │     │
│  │ Gate/Beta → chunk_kda (triton_ascend)          │     │
│  │ → GatedRMSNorm (NPU) → o_proj                   │     │
│  └────────────────────────────────────────────────┘     │
├─────────────────────────────────────────────────────────┤
│ ③ Post-Attention Norm + Residual                        │
│    KimiRMSNorm                                          │
├─────────────────────────────────────────────────────────┤
│ ④ FFN（按层配置二选一）                                    │
│                                                         │
│  ┌─ MLP Path ──────────────────────────────────────┐    │
│  │ gate_proj/up_proj → SituAndMul/silu → down_proj │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌─ MoE Path ──────────────────────────────────────┐    │
│  │ Route → permute (NPU) → grouped_gemm (NPU)     │    │
│  │ → swiglu (NPU) → grouped_gemm (NPU)            │    │
│  │ → unpermute (NPU) + shared_expert              │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘

Output: hidden_states [B, T, D]
```

### 3.2 VLM 多模态流程

```
Input: pixel_values + input_ids

┌──────────────────────────────────────────────┐
│ Vision Tower                                  │
│   pixel_values → ViT → image_features         │
├──────────────────────────────────────────────┤
│ Projector                                     │
│   image_features → merge 到文本 embedding 中    │
├──────────────────────────────────────────────┤
│ KimiLinearForCausalLM (Language Model)         │
│   merged_inputs_embeds                        │
│   → KimiLinearModel.forward()                 │
│     → KimiDecoderLayer × N (上述 Layer 流程)    │
│   → lm_head                                    │
├──────────────────────────────────────────────┤
│ Loss                                          │
│   build_loss_func → ChunkLoss / CrossEntropy  │
└──────────────────────────────────────────────┘
```

Vision Tower 中涉及的特殊适配：
- `_merge_input_ids_with_image_features`：图像 token 与文本 token 的精细合并
- Vision RoPE（interleave 模式）：`npu_rotary_mul` 对视觉位置编码
- Ungrouped/Rebalanced 序列长度处理（CP 场景）

---

## 四、算子 → 设备映射

| 算子 | NPU (Ascend) | CPU/GPU | 备注 |
|------|:---:|:---:|------|
| **chunk_kda** (fused) | `triton_ascend_kernels` | — | 仅支持 Ascend；GPU 侧可用上游 fla |
| **chunk_kda_naive** | — | PyTorch | 纯 eager，训练慢，精度对齐用 |
| **ShortConvolution** | `fla_npu` AscendC | Triton | 通过 `causal_conv1d_ascendc` 切换 |
| **Flash Attention** | `npu_fusion_attention` | `flash_attn` | layout 自动适配 |
| **Ring Attention (CP)** | ✅ 仅 NPU | ❌ | CP 功能仅在 NPU 可用 |
| **KimiK_3_MoeRMSNormGated** | `npu_rms_norm` | fp32 eager | KDA 输出专用 |
| **KimiRMSNorm** | — | fp32 eager | MLA 子层 Norm |
| **Fused MoE (permute/GMM/swiglu/unpermute)** | `npu_moe_token_permute`, `npu_grouped_matmul`, `npu_swiglu` | eager fallback | `IS_NPU_AVAILABLE` 自动切换 |
| **AllToAll GMM (EP)** | `npu_alltoallv_gmm` | — | Expert Parallel 专用 |
| **RoPE (interleave)** | `npu_rotary_mul` | eager | big seq 回退显式乘法 |
| **SituAndMul** | — | PyTorch fp32 | 无 NPU 融合 |
| **SwiGLU / GELU** | `npu_swiglu`, `npu_gelu` | eager | 通用激活 |

---

## 五、关键配置项

### 5.1 KDA 相关

```yaml
# config.json 或 kimik3_config.yaml
linear_attn_config:
  head_dim: 128              # K 维度
  num_heads: 16              # 注意力头数
  short_conv_kernel_size: 4  # ShortConv 卷积核大小
  gate_lower_bound: -5       # 遗忘门下界（safe_gate 模式），null 表示 softplus

kda_implementation: "fused"  # "fused" = triton_ascend 融合算子, "naive" = PyTorch 小算子
skip_kda_recompute: false    # 跳过 KDA 重计算（内存换速度）
```

### 5.2 MoE 相关

```yaml
num_experts: 128                # 路由专家数
num_experts_per_token: 8        # 每个 token 激活的专家数
expert_parallel_size: 1         # EP 并行度，1 = 不开启
moe_router_activation_func: "sigmoid"  # 路由激活函数
routed_scaling_factor: 1.0      # 路由权重缩放

# MoE Patch 环境变量
MM_FORCE_EP_BALANCE: "0"       # 强制 EP 负载均衡
```

### 5.3 注意力相关

```yaml
_attn_implementation: "flash_attention_2"  # 触发 NPU FA 路径
skip_flash_attn_recompute: false           # 跳过 FA 重计算
```

### 5.4 激活函数

```yaml
hidden_act: "situ"              # 使用 SituAndMul；也可配 "silu"
activation_situ_beta: 1.0       # Situ beta 参数
activation_situ_linear_beta: null  # Situ linear_beta 参数
```

---

## 六、开发与调试

### 6.1 KDA 算子切换

训练时默认使用 `fused` 模式（`triton_ascend_kernels`）。如需做精度对比或调试，可通过 config 切换为 `naive`：

```python
# 在模型加载前
config.kda_implementation = "naive"
```

两个实现在数值上不等价（fp32 vs kernel 内部精度差异），但误差应在 `1e-3` 量级。

### 6.2 NPU 算子开关

所有 `IS_NPU_AVAILABLE` 分支均可通过环境变量控制回退：

```python
# ops/moe_ops/permute.py
def permute(tokens, indices, num_out_tokens=None, fused=True):
    if fused and IS_NPU_AVAILABLE:
        return fused_permute(tokens, indices, num_out_tokens)  # NPU
    else:
        return eager_permute(tokens, indices, num_out_tokens)  # PyTorch
```

调用时传入 `fused=False` 强制回退到 PyTorch eager 实现，用于性能对比和精度排查。

### 6.3 关键日志

训练启动时关注以下日志确认 NPU 算子已生效：

```
# Kimi RMSNorm
KimiK_3_MoeRMSNormGated use NPU fused ops

# MoE patch
Applying Kimi-K3 MoE patch for NPU fused operators...

# Flash Attention
flash_attention_forward: use_npu_fusion_fa=True, layout=BNSD
```

### 6.4 性能分析建议

Kimi-K3 训练的性能瓶颈主要在以下算子：

| 优先级 | 算子 | 原因 |
|--------|------|------|
| P0 | `chunk_kda` | KDA 整个 chunk 计算在单个 kernel 内，是最重的大算子 |
| P1 | `npu_fusion_attention` (MLA) | MLA 层的标准注意力 |
| P2 | Fused MoE (permute → GMM → swiglu → unpermute) | 通信+计算，EP 场景尤其关键 |
| P3 | `causal_conv1d` (ShortConv) | 逐 token 因果卷积 |

建议使用 `torch_npu.profiler` 或 MindStudio 对这 4 类算子做详细的算子级 profiling。
