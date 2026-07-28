# Kimi-K3 CANN 算子依赖对比

> 对比 MindSpeed-MM (训练) 与 vllm-ascend (推理) 对 Kimi-K3 的 CANN 算子依赖
> 分析日期: 2026-07-28

---

## 1. MindSpeed-MM 训练侧

### 1.1 torch_npu CANN 算子（10 个）

| 算子 | 用途 | 调用位置 |
|------|------|------|
| `npu_rms_norm` | KDA 输出 GatedRMSNorm; 通用 RMSNorm 融合 | `modeling_kimi_linear.py:110`; `npu_fused_operator.py:37` |
| `npu_fusion_attention` | MLA Flash Attention (KimiMLA + VLM 交叉注意力) | `flash_attn.py:511`; `modeling_kimi_k3.py:154` |
| `npu_rotary_mul` | RoPE 旋转位置编码 (interleave 模式) | `npu_fused_operator.py:45,63`; `modeling_kimi_k3.py:274` |
| `npu_swiglu` | SwiGLU 激活 (MLP + MoE SiLU 路径) | `swiglu.py:16`; `npu_fused_operator.py:75` |
| `npu_gelu` | GELU 激活 (通用) | `npu_fused_operator.py:27` |
| `npu_moe_token_permute` | MoE token 按 expert 重排 | `permute.py:20` |
| `npu_moe_token_unpermute` | MoE token 恢复原始顺序 | `unpermute.py:20` |
| `npu_grouped_matmul` | MoE 分组矩阵乘 (gate/up/down projections) | `gemm.py:16` |
| `npu_alltoallv_gmm` | EP AllToAll + 分组矩阵乘融合 | `gemm_mc2.py:21` |
| `npu_gmm_alltoallv` | EP 分组矩阵乘 + AllToAll 融合 | `gemm_mc2.py:66` |

### 1.2 外部 Ascend 后端算子（2 个）

| 算子 | 来源 | 用途 |
|------|------|------|
| `chunk_kda` | `triton_ascend_kernels` 外部包 | KDA chunk attention 融合大算子 (l2norm + gate + cumsum + WY recurrence) |
| `causal_conv1d` (AscendC) | `fla_npu.ops.ascendc` | KDA ShortConvolution (kernel_size=4, depthwise) |

### 1.3 依赖关系图

```
Kimi-K3 DecoderLayer (训练)
│
├── Attention
│   ├── MLA ──► npu_fusion_attention, npu_rotary_mul
│   └── KDA ──► triton_ascend_kernels.chunk_kda  ← 融合大算子
│                └── internal: l2norm + gate + cumsum + WY + recurrence
│               ──► fla_npu.causal_conv1d (AscendC)  ← ShortConv
│               ──► npu_rms_norm (GatedRMSNorm)
│
├── FFN
│   ├── MLP ──► npu_swiglu / npu_gelu
│   └── MoE ──► npu_moe_token_permute / unpermute
│               ──► npu_grouped_matmul
│               ──► npu_swiglu
│               ──► [EP] npu_alltoallv_gmm / npu_gmm_alltoallv
│
└── Norm ──► (fp32 eager KimiRMSNorm)
```

---

## 2. vllm-ascend 推理侧

### 2.1 torch.ops._C_ascend 自研 AscendC 算子（6 个 Kimi-K3 专属）

| 算子 | 用途 | 调用位置 |
|------|------|------|
| `chunk_kda_fwd` | KDA Prefill chunk 并行扫描 (4-Phase) | `kimi_kda.py:373` |
| `kda_gate_cumsum` | KDA Prefill 前置: gate transform + segment cumsum | `kimi_kda.py:353` |
| `recurrent_kda` | KDA Decode 逐 token 循环状态更新 | `kimi_kda.py:282` |
| `dequant_situ_quant` | A2/A3: 反量化(INT32) → SiTU → INT8 量化 | `kimi_k3.py` MoE path |
| `situ_mx_quant` | A5: SiTU → MXFP8 量化 | `kimi_k3.py` MoE path |
| `npu_causal_conv1d_custom` | KDA ShortConvolution (合并 Q/K/V, 单次调用) | `kimi_kda.py:231` |

### 2.2 torch_npu CANN 算子（17 个，含通用推理算子）

**Kimi-K3 直接依赖:**

| 算子 | 用途 |
|------|------|
| `npu_swiglu` | SwiGLU 激活 (非 SiTU 的 MLP 层) |
| `npu_grouped_matmul` | MoE 分组矩阵乘 |
| `npu_moe_token_permute` | MoE token permute |
| `npu_moe_token_unpermute` | MoE token unpermute |
| `npu_moe_distribute_dispatch` / `npu_moe_distribute_dispatch_v2` | MoE token dispatch (EP) |
| `npu_moe_distribute_combine` / `npu_moe_distribute_combine_v2` | MoE token combine (EP) |
| `npu_moe_finalize_routing` | MoE routing 后处理 |
| `npu_moe_init_routing_v2` | MoE routing 初始化 |
| `npu_fusion_attention` | Flash Attention (MLA / VLM) |
| `npu_rms_norm` | RMSNorm |
| `npu_fast_gelu` | Fast GELU |
| `npu_dynamic_quant` / `npu_dynamic_mx_quant` | 动态量化 |
| `npu_format_cast` | 格式转换 (ND → NZ) |
| `npu_quant_matmul` | 量化矩阵乘 |
| `_npu_group_topk` | MoE gating top-k 选择 |

**Kimi-K3 MoE 量化融合路径 (vllm-ascend 自研):**

| 算子 | 用途 |
|------|------|
| `grouped_matmul_swiglu_quant_v2` | GMM + SwiGLU + 量化 融合 |
| `npu_dequant_swiglu_quant` | 反量化 + SwiGLU + 量化 融合 |
| `npu_swiglu_group_quant` | SwiGLU + group 量化 |
| `dispatch_gmm_combine_decode` | Decode: dispatch + GMM + combine 融合 |

### 2.3 依赖关系图

```
Kimi-K3 DecoderLayer (推理)
│
├── Attention
│   ├── MLA ──► npu_fusion_attention (via vllm MLA backend)
│   └── KDA
│       ├── Prefill:
│       │   ├── npu_causal_conv1d_custom  ← ShortConv (QKV concat)
│       │   ├── kda_gate_cumsum            ← gate + cumsum (AscendC)
│       │   └── chunk_kda_fwd              ← chunk scan (AscendC)
│       └── Decode:
│           ├── npu_causal_conv1d_custom   ← ShortConv
│           └── recurrent_kda              ← 逐 token 循环 (AscendC)
│
├── FFN
│   ├── MLP ──► npu_swiglu / npu_fast_gelu
│   └── MoE
│       ├── Router ──► _npu_group_topk
│       ├── EP dispatch ──► npu_moe_distribute_dispatch
│       ├── GMM1 ──► npu_grouped_matmul
│       ├── [SiTU] ──► dequant_situ_quant (A2/A3) / situ_mx_quant (A5)
│       │             或 grouped_matmul_swiglu_quant_v2 (SiLU路径)
│       ├── GMM2 ──► npu_grouped_matmul / npu_quant_matmul
│       └── EP combine ──► npu_moe_distribute_combine
│
└── Norm ──► npu_rms_norm
```

---

## 3. 并排对比

### 3.1 KDA Attention 算子

| 功能 | MindSpeed-MM (训练) | vllm-ascend (推理) |
|------|------|------|
| **Chunk KDA** | `triton_ascend_kernels.chunk_kda` (1 融合 kernel) | `kda_gate_cumsum` + `chunk_kda_fwd` (2 分离 AscendC kernel) |
| **Recurrent KDA** | ❌ (训练无 decode) | `recurrent_kda` (AscendC) |
| **ShortConv** | `fla_npu.ops.ascendc.causal_conv1d` (3 次调用) | `npu_causal_conv1d_custom` (1 次合并调用) |
| **GatedRMSNorm** | `npu_rms_norm` + Python sigmoid | Triton `rms_norm_gated` (norm + gate + residual 融合) |

### 3.2 MoE 算子

| 功能 | MindSpeed-MM (训练) | vllm-ascend (推理) |
|------|------|------|
| **Token Permute** | `npu_moe_token_permute` | `npu_moe_token_permute` ✅ 相同 |
| **Token Unpermute** | `npu_moe_token_unpermute` | `npu_moe_token_unpermute` ✅ 相同 |
| **Grouped MatMul** | `npu_grouped_matmul` | `npu_grouped_matmul` ✅ 相同 |
| **EP AllToAll** | `npu_alltoallv_gmm` / `npu_gmm_alltoallv` | `npu_moe_distribute_dispatch` / `npu_moe_distribute_combine` |
| **Activation (SiLU)** | `npu_swiglu` | `npu_swiglu` ✅ 相同 |
| **Activation (SiTU)** | PyTorch `SituAndMul` (无融合) | `dequant_situ_quant` / `situ_mx_quant` (融合量化) |
| **Routing** | PyTorch topk | `_npu_group_topk` |

### 3.3 通用算子

| 功能 | MindSpeed-MM | vllm-ascend |
|------|:---:|:---:|
| **Flash Attention** | `npu_fusion_attention` | `npu_fusion_attention` ✅ 相同 |
| **RMSNorm** | `npu_rms_norm` / fp32 eager | `npu_rms_norm` ✅ 相同 |
| **SwiGLU** | `npu_swiglu` | `npu_swiglu` ✅ 相同 |
| **RoPE** | `npu_rotary_mul` | vllm 标准 MLA RoPE (无 NPU 融合) |

---

## 4. 差异总结

### 4.1 算子数量

| 分类 | MindSpeed-MM | vllm-ascend | 差异 |
|------|:---:|:---:|------|
| **KDA 专属 AscendC** | 0 (依赖外部包) | 3 (`chunk_kda_fwd`, `kda_gate_cumsum`, `recurrent_kda`) | vllm-ascend 自研 |
| **KDA Triton/Ascend** | 2 (`triton_ascend_kernels.chunk_kda`, `fla_npu.causal_conv1d`) | 1 (`npu_causal_conv1d_custom`) | 来源不同 |
| **SiTU 量化融合** | 0 | 2 (`dequant_situ_quant`, `situ_mx_quant`) | vllm-ascend 独有 |
| **torch_npu 基础算子** | 10 | 17 | vllm-ascend 额外多了量化/EP dispatch/routing 算子 |

### 4.2 关键差异

**A. KDA 算子的归属不同**:

```
MindSpeed-MM:
  chunk_kda → triton_ascend_kernels (外部包, Triton → Ascend 编译)
  causal_conv1d → fla_npu (外部包, AscendC)

vllm-ascend:
  chunk_kda_fwd → torch.ops._C_ascend (自研 AscendC)
  kda_gate_cumsum → torch.ops._C_ascend (自研 AscendC)
  recurrent_kda → torch.ops._C_ascend (自研 AscendC)
  causal_conv1d → torch.ops._C_ascend.npu_causal_conv1d_custom (自研 AscendC)
```

**B. SiTU 激活**:
- MindSpeed-MM: 纯 PyTorch, 不走 CANN
- vllm-ascend: 自研 AscendC fusion kernel (`dequant_situ_quant` / `situ_mx_quant`)

**C. 共享的 torch_npu 算子 (3 个完全一致)**:
- `npu_swiglu`
- `npu_grouped_matmul`

---

## 5. 平台兼容性 (A2 / A3 / A5)

### 5.1 训练算子 (MindSpeed-MM)

| 算子 | 来源 | A2 (`910B`) | A3 (`910_93`) | A5 (`950`) | 说明 |
|------|------|:---:|:---:|:---:|------|
| `npu_rms_norm` | torch_npu | ✅ | ✅ | ✅ | CANN 通用 API |
| `npu_fusion_attention` | torch_npu | ✅ | ✅ | ✅ | CANN 通用 API |
| `npu_rotary_mul` | torch_npu | ✅ | ✅ | ✅ | CANN 通用 API |
| `npu_swiglu` | torch_npu | ✅ | ✅ | ✅ | CANN 通用 API |
| `npu_gelu` | torch_npu | ✅ | ✅ | ✅ | CANN 通用 API |
| `npu_moe_token_permute` | torch_npu | ✅ | ✅ | ✅ | CANN 通用 API |
| `npu_moe_token_unpermute` | torch_npu | ✅ | ✅ | ✅ | CANN 通用 API |
| `npu_grouped_matmul` | torch_npu | ✅ | ✅ | ✅ | CANN 通用 API |
| `npu_alltoallv_gmm` | torch_npu | ✅ | ✅ | ✅ | CANN 通用 API |
| `npu_gmm_alltoallv` | torch_npu | ✅ | ✅ | ✅ | CANN 通用 API |
| `chunk_kda` (fused) | `triton_ascend_kernels` | ✅ | ✅ | ✅ | Triton-Ascend 后端, CHIP_TYPE=910b/A3/A5 |
| `causal_conv1d` (AscendC) | `fla_npu` | ✅ | ✅ | ✅ | AscendC arch32/35 |

**外部依赖的平台要求**:

- **`torch_npu`** (CANN): 所有 10 个算子均为 CANN 通用 API，与 SoC 平台无关。A2/A3/A5 均可使用，只需安装对应平台的 CANN 版本即可。
- **`Triton-Ascend`** + **`triton-ascend-kernels`**: `chunk_kda` 融合大算子的运行时。Triton-Ascend 支持 A2 (910B) / A3 (910_93) / A5 (950)，通过 `CHIP_TYPE` 编译选项选择目标平台。A2 设置 `CHIP_TYPE=910b`。
- **`fla_npu`**: `causal_conv1d` 的 AscendC 实现。AscendC 的 `arch32` 编译目标同时覆盖 A2 和 A3，与 vllm-ascend 中的 `AddConfig("ascend910b")` 一致。

> **结论**: 训练侧所有 12 个算子均支持 A2/A3/A5 三平台。唯一的平台差异是需要安装对应 SoC 版本的 CANN、Triton-Ascend 和 fla_npu 包。

### 5.2 推理算子 (vllm-ascend)

| 算子 | 来源 | A2 | A3 | A5 | 说明 |
|------|------|:---:|:---:|:---:|------|
| `chunk_kda_fwd` | `_C_ascend` (AscendC) | ✅ | ✅ | ✅ | arch32/35 |
| `kda_gate_cumsum` | `_C_ascend` (AscendC) | ✅ | ✅ | ✅ | arch32/35 |
| `recurrent_kda` | `_C_ascend` (AscendC) | ✅ | ✅ | ✅ | 公共 AIV kernel |
| `npu_causal_conv1d_custom` | `_C_ascend` (AscendC) | ✅ | ✅ | ✅ | 已有复用算子 |
| `dequant_situ_quant` | `_C_ascend` (AscendC) | ✅ | ✅ | ❌ | A2/A3 INT8 路径 |
| `situ_mx_quant` | `_C_ascend` (AscendC) | ❌ | ❌ | ✅ | A5 专用 MXFP8 |
| `npu_swiglu` 等共享算子 | torch_npu | ✅ | ✅ | ✅ | CANN 通用 API |

### 5.3 训练 vs 推理平台差异

```
训练 (MindSpeed-MM):
  A2 ✅ 全算子支持 (12/12)
  A3 ✅ 全算子支持 (12/12)
  A5 ✅ 全算子支持 (12/12)
  关键依赖: CANN + Triton-Ascend (CHIP_TYPE) + fla_npu

推理 (vllm-ascend):
  A2 ✅ KDA 全支持, SiTU 走 dequant_situ_quant (INT8)
  A3 ✅ KDA 全支持, SiTU 走 dequant_situ_quant (INT8)
  A5 ✅ KDA 全支持, SiTU 走 situ_mx_quant (FP8)
  关键依赖: CANN + 自研 AscendC 算子编译 (arch32/35)
```
- `npu_moe_token_permute` / `npu_moe_token_unpermute`
- `npu_fusion_attention`
- `npu_rms_norm`
