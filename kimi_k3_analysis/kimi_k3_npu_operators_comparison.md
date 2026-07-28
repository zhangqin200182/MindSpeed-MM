# Kimi-K3 NPU 算子对比分析：MindSpeed-MM(训练) vs vllm-ascend(推理)

> 分析日期: 2026-07-28
> MindSpeed-MM 分支: HEAD
> vllm-ascend 分支: `feature/kimi-k3-release-v0.23.0`
> PR: [#12951](https://github.com/vllm-project/vllm-ascend/pull/12951)

---

## 1. 概述

两个项目分别针对 Kimi-K3 在昇腾 NPU 上的**训练**和**推理**场景进行了算子适配。由于训练和推理在计算模式、显存约束、精度要求上的本质差异，两者的算子设计策略有显著不同：

| 维度 | MindSpeed-MM (训练) | vllm-ascend (推理) |
|------|:---:|:---:|
| 核心场景 | 大规模分布式训练 (FSDP2 + EP + CP) | 在线推理服务 (Prefill + Decode 分离) |
| KDA chunk_kda 形态 | **1 个融合大算子** (`triton_ascend_kernels`) | **2 个分离算子** (`kda_gate_cumsum` + `chunk_kda_fwd`) |
| KDA Decode | 复用 `chunk_kda` / `fused_recurrent_kda` | **1 个独立算子** (`recurrent_kda`) |
| SiTU 激活 | **纯 PyTorch** (`SituAndMul`，无 NPU 融合) | **2 个量化融合算子** (`dequant_situ_quant` / `situ_mx_quant`) |
| 量化 | ❌ 不支持 | ✅ INT8 (A3) / MX FP8 (A5) |
| 状态管理 | `KimiDynamicCache` 简单追踪 | 容量型状态池 + `ssm_state_indices` 索引 |
| ShortConv 后端 | `causal_conv1d_ascendc` (AscendC) | `npu_causal_conv1d_custom` (AscendC) |
| MoE 结构 | 原始 expert 权重 → 3D tensor 重组 | Latent MoE (compressed + norm + decompress) |

---

## 2. KDA Attention 算子对比

这是差异最大的领域。训练时所有 token 等长、需梯度反向传播；推理时分 Prefill（长序列）和 Decode（逐 token），且无需反向。

### 2.1 算子拆分策略

```
MindSpeed-MM (训练):
  ┌─────────────────────────────────────────────┐
  │              chunk_kda()                     │
  │  (triton_ascend_kernels, 单个融合大算子)      │
  │                                             │
  │  l2norm(q,k) ─┬─ gate transform ─┬─ cumsum  │
  │               │                  │          │
  │               ├─ intra-chunk ────┤          │
  │               │  WY recurrence   │          │
  │               ├─ inter-chunk ────┘          │
  │               │  state update               │
  │               └─ output projection          │
  └─────────────────────────────────────────────┘

vllm-ascend (推理):
  ┌──────────────────────┐  ┌──────────────────────┐  ┌───────────────────┐
  │   kda_gate_cumsum    │  │    chunk_kda_fwd      │  │   recurrent_kda   │
  │   (kernel 1)         │  │    (kernel 2)          │  │   (kernel 3)      │
  │                      │  │                        │  │                   │
  │  Prefill 前置        │  │  Prefill 主计算         │  │  Decode 逐 token  │
  │  gate+cumsum 融合    │  │  4-Phase chunk 扫描    │  │  单 kernel 融合   │
  └──────────────────────┘  └──────────────────────┘  └───────────────────┘
```

**差异原因**：
- 训练时 chunk_kda 必须是一个完整的 `autograd.Function`，前后向成对出现，拆分反而增加显存（需要保存中间结果给反向）。
- 推理时 Prefill/Decode 访问模式完全不同（连续大块 vs. 离散小 token），拆分为 3 个独立 kernel 可以利用不同的硬件并行策略：Prefill 用 compute-bound chunk 并行，Decode 用 memory-bound 单 kernel 融合。

### 2.2 chunk_kda 对比

| 特性 | MindSpeed-MM `chunk_kda` (`triton_ascend`) | vllm-ascend `chunk_kda_fwd` (AscendC) |
|------|------|------|
| **来源** | `triton_ascend_kernels` 外部包 | `csrc/attention/chunk_kda_fwd/` 自研 AscendC |
| **gate transform** | kernel 内部融合 | **分离** → 由 `kda_gate_cumsum` 前置完成 |
| **l2norm** | kernel 内部融合 (`use_qk_l2norm_in_kernel=True`) | Python 侧显式调用 `l2norm_fwd` |
| **beta sigmoid** | Python 侧预处理 (`use_beta_sigmoid_in_kernel=False`) | Python 侧预处理 |
| **返回值数量** | 2 (`o`, `final_state`) | **12** 个（含全部中间张量，供调试/重计算） |
| **chunk 内部阶段** | 单一 kernel 内完成 (WY + recurrence) | 4 个显式 Phase (scaled_dot_kkt → solve_tril → recompute_w_u → gdn_fwd_h → gla_fwd_o) |
| **反向传播** | ✅ 完整 backward kernel | ❌ 不需要（推理） |
| **cu_seqlens** | 支持 host 端 python list | 支持 host 端 python list |
| **state layout** | `[N, H, V, K]` (transpose_state_layout=True) | `[N, H, K, V]` 输入，内部转置 |

**关键发现**：MindSpeed-MM 的 `chunk_kda` 是 **1 个融合 kernel**，而 vllm-ascend 的 `chunk_kda_fwd` 内部显式拆分为 **4 个计算 Phase**（scaled_dot_kkt → solve_tril → recompute_w_u → gdn_fwd_h → gla_fwd_o），每个 Phase 可能对应一个或多个独立 kernel launch。这是因为：
1. 推理不需要保存中间结果用于反向
2. 分阶段可以利用 vllm 的 CUDA Graph / ACL Graph 图编译进行 kernel 融合优化
3. 返回 12 个张量便于调试和适配不同推理框架

### 2.3 gate transform 对比

| 特性 | MindSpeed-MM | vllm-ascend |
|------|------|------|
| **Kimi-K3 bounded sigmoid** | kernel 内部融合 (`safe_gate=True, lower_bound=-5`) | `kda_gate_cumsum` 中实现 + 也有 Triton `fused_kda_gate` fallback |
| **标准 KDA softplus** | kernel 内部融合 (`safe_gate=False`) | `kda_gate_cumsum` cumsum_only 模式 + Triton gate |
| **算子形式** | kernel 内部子操作，不独立暴露 | 独立注册算子 `torch.ops._C_ascend.kda_gate_cumsum` |
| **额外用途** | 无（仅 KDA 内用） | 可被其他模型（Kimi K2.5）复用 |

`kda_gate_cumsum` 独立出来的优势：
- 支持两种模式（cumsum only / gate+cumsum），兼容标准 KDA 和 Kimi-K3 bounded sigmoid
- Segment-wise cumsum 是 memory-bound 操作，独立 kernel 可以用更小的 block size 和更激进的 cache 策略
- 可复用于其他 linear attention 模型

### 2.4 recurrent_kda (Decode) 对比

这是 vllm-ascend **独有的算子**，MindSpeed-MM 没有对应实现。

| 特性 | MindSpeed-MM | vllm-ascend |
|------|:---:|------|
| **Decode 专用 kernel** | ❌ 不存在（训练无 decode 阶段） | ✅ `torch.ops._C_ascend.recurrent_kda` |
| **融合范围** | — | 状态衰减 + delta update + 输出 + l2norm + gate transform |
| **状态更新** | — | **In-place** 更新 state pool |
| **序列长度** | — | ≤ 8 tokens (含 speculative decode) |
| **状态池** | — | `[state_capacity, HV, V, K]` 容量池 + `ssm_state_indices` 二维索引 |
| **SpecDec 支持** | — | ✅ `num_accepted_tokens` 参数 |

**设计要点**：
- **单 AI Core 内完成**：状态读取→计算→状态写回全在 UB 内，不写 DRAM 中间结果
- **In-place state**：`initial_state` 是 mutable input，命中槽位原地更新，兼容 AOT functionalization
- **按需索引**：通过 `ssm_state_indices` 从容量池中只读取活跃序列的状态，避免全量读写

---

## 3. ShortConvolution (causal_conv1d) 对比

| 特性 | MindSpeed-MM | vllm-ascend |
|------|------|------|
| **实现位置** | `ops/kda/short_conv.py` → `causal_conv1d_ascendc` | `_run_causal_conv1d` → `npu_causal_conv1d_custom` |
| **后端** | `fla_npu.ops.ascendc.causal_conv1d` (AscendC) | `torch.ops._C_ascend.npu_causal_conv1d_custom` (AscendC) |
| **Q/K/V 处理** | 分开 3 次调用 | **Concat Q/K/V → 1 次调用** |
| **权重格式** | `[D, W]` 经 rearrange 为 `[W, D]` | Concat 3 组权重为 `[W, 3D]` |
| **State 管理** | `KimiDynamicCache.conv_states[layer_idx]` | `conv_state` 由上层 cache 管理 |
| **Decode step** | `causal_conv1d_update` (Triton) | `npu_causal_conv1d_custom(run_mode=1)` |

**关键差异**：vllm-ascend 将 Q/K/V 的卷积权重 concat 为单次调用，减少 3× kernel launch 开销（对 decode 阶段的小 batch 尤为重要）。MindSpeed-MM 保持 3 次独立调用，因为训练时 sequence 长，3 次调用的额外 launch 开销占比可忽略，且独立调用更利于 gradient checkpointing。

---

## 4. SiTU 激活函数对比

### 4.1 算子实现

| 特性 | MindSpeed-MM | vllm-ascend |
|------|------|------|
| **核心实现** | `SituAndMul` (`modeling_kimi_linear.py:145`) | `SituActivationConfig` 配置类 + 融合量化算子 |
| **数学公式** | `β·tanh(gate/β)·σ(gate) · φ(up)` | **完全相同** |
| **NPU 融合** | ❌ 纯 PyTorch fp32 | ✅ **反量化 → SiTU → 量化** 融合 |
| **量化输出** | ❌ 不涉及 | ✅ INT8 (A3) / MXFP8 (A5) |
| **计算精度** | fp32 → cast back | BF16 (输入) → FP32 (中间) → INT8/FP8 (输出) |

### 4.2 融合量化算子详解

vllm-ascend 独有的 2 个 SiTU 融合算子：

**`dequant_situ_quant` (A3 / Ascend 910_93)**:
```
输入 INT32 [M, 2H]         (QMM 累加器)
  → dequant(weight_scale, activation_scale)
  → BF16 gate/up
  → SiTU: β·tanh(gate/β)·σ(gate) · φ(up)
  → dynamic INT8 quant: clamp(round(situ/scale), -128, 127)
输出: INT8 [M, H] + FP32 scale [M]
```

**`situ_mx_quant` (A5 / Ascend 950)**:
```
输入 BF16 [M, 2H]
  → SiTU: β·tanh(gate/β)·σ(gate) · φ(up)
  → MXFP8 block quant: per-64-element E8M0 scale
输出: FP8 E4M3FN [M, H] + FP8 E8M0 scale [M, ceil(H/64), 2]
```

### 4.3 使用场景

| 场景 | MindSpeed-MM | vllm-ascend |
|------|------|------|
| **Routed Experts (MoE)** | `SituAndMul` → BF16 输出 | `dequant_situ_quant` / `situ_mx_quant` → INT8/FP8 |
| **Shared Experts** | `SituAndMul` → BF16 输出 | A3: `dequant_situ_quant(INT32→SiTU→INT8)`, A5: `situ_mx_quant(BF16→SiTU→FP8)` |
| **Dense MLP** | `SituAndMul` → BF16 输出 | `SituActivationConfig` 配置, 但运行时抛异常（必须走融合路径） |

**差异原因**：推理需要量化来降低显存带宽压力（MoE 专家数量 896，权重大，量化收益显著）；训练不需要量化（需要 fp32 梯度精度），且 SiTU 计算量相比 KDA/MoE-GEMM 可忽略。

---

## 5. MoE 结构对比

### 5.1 架构差异

```
MindSpeed-MM (训练):
  hidden_states [B, T, 7168]
    ├─ Router → topk_ids, topk_weights
    ├─ Fused MoE:
    │   ├─ permute (NPU)
    │   ├─ grouped_matmul: gate_up_proj [num_experts, 7168, 6144]
    │   ├─ swiglu / SituAndMul (no NPU fusion for SiTU)
    │   ├─ grouped_matmul: down_proj [num_experts, 3072, 7168]
    │   └─ unpermute (NPU) + shared_expert
    └─ output

vllm-ascend (推理):
  hidden_states [M, 7168]
    ├─ Router(ReplicatedLinear) → router_logits [M, 896]
    ├─ routed_expert_down_proj: [M, 7168] → [M, 3584]  ← Latent 压缩!
    ├─ FusedMoE (hidden=3584, inter=3072, EP=64)
    │   ├─ dispatch
    │   ├─ GMM1(gate+up) → BF16 [M, 6144]
    │   ├─ [A3] dequant_situ_quant → INT8 [M, 3072]
    │   │   [A5] situ_mx_quant → FP8 [M, 3072]
    │   ├─ GMM2(down) → BF16 [M, 3584]
    │   └─ combine
    ├─ routed_expert_norm (可选 RMSNorm)
    ├─ routed_expert_up_proj: [M, 3584] → [M, 7168]  ← Latent 解压!
    └─ + shared_expert output
```

**关键差异 — Latent MoE**：
- MindSpeed-MM：Expert 的 hidden_size = 7168, intermediate = 3072（标准 MoE）
- vllm-ascend：Expert 的 hidden_size = **3584**, intermediate = 3072（Latent MoE，先压缩再解压）

vllm-ascend 的 MoE 实现中，routed experts 先经过 `routed_expert_down_proj` 将 hidden 从 7168 压缩到 3584，MoE 计算在 3584 维空间进行，最后通过 `routed_expert_up_proj` 解压回 7168。这是 Kimi-K3 checkpoint 的原生设计，用于降低 MoE 参数量和计算量。

### 5.2 MoE 激活算子对比

| 特性 | MindSpeed-MM | vllm-ascend |
|------|------|------|
| **Ruoted expert 激活** | `SituAndMul` (PyTorch, no fusion) | `dequant_situ_quant` / `situ_mx_quant` (NPU fused) |
| **SiTU 是否 fuse SiLU** | ❌ SiTU 不是 SiLU，走 PyTorch | ✅ 融合（SiTU + quant） |
| **SiLU 是否 fuse** | ✅ `torch_npu.npu_swiglu` (kimi_moe_patch) | ✅ `torch_npu.npu_swiglu` (AscendSiluAndMul) |
| **量化** | ❌ | ✅ 动态 INT8 / MX FP8 |
| **Weight layout** | 3D: `[num_experts, H, 2×I]` + `[num_experts, I, H]` | W4A8 / MXFP8 量化权重 |

---

## 6. 辅助算子对比

### 6.1 RMSNorm / GatedRMSNorm

| 特性 | MindSpeed-MM | vllm-ascend |
|------|------|------|
| **MLA 子层 Norm** | `KimiRMSNorm` (纯 PyTorch fp32) | 标准 `RMSNorm` (vllm 内置) |
| **KDA 输出 Norm** | `KimiK_3_MoeRMSNormGated` → `torch_npu.npu_rms_norm` + sigmoid | Triton `layer_norm_gated_fwd` (RMS + sigmoid/swish gate) |
| **融合方式** | NPU API + Python gate | 单 Triton kernel (norm + gate + residual) |

vllm-ascend 的 Triton kernel 额外融合了 residual add（`HAS_RESIDUAL` 分支），将 RMSNorm + gate + residual 合并为 1 个 kernel。

### 6.2 Flash Attention

| 特性 | MindSpeed-MM | vllm-ascend |
|------|------|------|
| **MLA FA 调用** | `flash_attention_forward` → `torch_npu.npu_fusion_attention` | vllm 标准 MLA backend |
| **Layout** | BNSD/BSND/1TND/1NTD 四选一 | vllm 内部 layout |
| **Ring Attention (CP)** | ✅ 支持 Ulysses + Ring 混合 | ❌ KDA prefill 暂不支持 PCP |
| **skip_recompute** | ✅ `skip_recompute_flash_attention` | ❌ 推理不需要 |

### 6.3 RoPE

| 特性 | MindSpeed-MM | vllm-ascend |
|------|------|------|
| **MLA RoPE** | interleave 模式 `npu_rotary_mul` | `apply_rope` (vllm 标准) |
| **Vision RoPE** | `npu_rotary_mul` + 4D 格式 | `Rope2DPosEmbRepeated` + `apply_rope` (ViT 内) |
| **NPU 融合** | ✅ 小 seq 走 `npu_rotary_mul`; big seq 回退显式乘法 | 标准 PyTorch |

---

## 7. 状态管理对比

| 特性 | MindSpeed-MM | vllm-ascend |
|------|------|------|
| **状态容器** | `KimiDynamicCache` (per-batch) | 容量型 state pool + `ssm_state_indices` |
| **Conv state shape** | `[B, D, W-1]` | `[local_conv_dim, state_len]` (dim first) |
| **Recurrent state shape** | `[N, HV, K, V]` 或 `[N, HV, V, K]` | `[state_capacity, HV, V, K]` |
| **Multi-batch** | 按 batch 维度索引 | 通过 `cu_seqlens` + `ssm_state_indices` 按需索引 |
| **SpecDec 支持** | ❌ | ✅ `num_accepted_tokens` + 二维 state indices |

vllm-ascend 的状态池设计是为了支持推理场景下动态 batch（请求到达/离开时状态增删），通过容量池 + 索引实现 O(1) 的状态分配/释放，避免 tensor copy。

---

## 8. 完整算子集对比矩阵

| 算子 | MindSpeed-MM (训练) | vllm-ascend (推理) | 差异摘要 |
|------|:---:|:---:|------|
| **chunk_kda** | ✅ 1 融合 kernel (triton_ascend) | ✅ 2 分离 kernel (gate_cumsum + fwd) | 训练融合 > 推理分离 |
| **recurrent_kda** | ❌ | ✅ AscendC 独立 kernel | 推理独有 |
| **kda_gate_cumsum** | ❌ (融合在 chunk_kda 内) | ✅ AscendC 独立 kernel | 推理独有 |
| **kda_layout_swap12** | ❌ | ✅ 辅助算子 | 推理独有 |
| **causal_conv1d** | ✅ AscendC (fla_npu) | ✅ AscendC (npu_causal_conv1d_custom) | 不同后端，vllm-ascend 合并 QKV |
| **dequant_situ_quant** | ❌ | ✅ AscendC (A3) | 推理独有 (量化) |
| **situ_mx_quant** | ❌ | ✅ AscendC (A5) | 推理独有 (MX FP8) |
| **SituAndMul (纯计算)** | ✅ PyTorch | ✅ 仅配置 (必须走融合) | 训练: eager; 推理: fused quant |
| **SiLU (SwiGLU)** | ✅ `npu_swiglu` | ✅ `npu_swiglu` | 相同 |
| **Fused MoE forward** | ✅ permute-GMM-swiglu-unpermute | ✅ FusedMoE with SiTU quant | 结构相同，激活不同 |
| **RMSNorm** | `npu_rms_norm` / fp32 eager | Triton fused + residual | 不同融合策略 |
| **Flash Attention** | `npu_fusion_attention` | vllm standard MLA backend | 相同底层 API |
| **Ring Attention (CP)** | ✅ Ulysses + Ring | ❌ (KDA prefill 暂不支持) | 训练独有 |
| **AllToAll GMM (EP)** | ✅ `npu_alltoallv_gmm` | ✅ (通过 MoE dispatcher) | 功能等价 |

---

## 9. 总结

### 9.1 两个项目 NPU 算子的共同点

1. **KDA chunk 计算**：核心算法逻辑一致（WY recurrence + chunk scan + delta rule），都基于 `fla` (flash-linear-attention) 的数学框架
2. **MoE GEMM**：都使用 `npu_grouped_matmul` 做 expert 分组矩阵乘
3. **Flash Attention**：都调用 `npu_fusion_attention` 底层 API
4. **RoPE / SwiGLU / 基础 Norm**：都使用 `torch_npu` 融合 API

### 9.2 关键差异根源

| 差异 | 根源 |
|------|------|
| KDA 拆分 vs 融合 | **训练需要反向图** → 1 个 autograd.Function；**推理不需要** → 按 Prefill/Decode 拆为独立 kernel 优化 latency |
| gate_cumsum 独立 | 推理中该操作 memory-bound，独立可复用；训练中隐藏在 compute-bound kernel 内 |
| SiTU 量化融合 | 推理需要量化减带宽；训练需要 fp32 精度 |
| Latent MoE | vllm-ascend 遵循 checkpoint 原生设计；MindSpeed-MM 当前未实现 latent 压缩 |
| 状态池 vs 简单 cache | 推理需动态 batch；训练固定 batch |
| recurrent_kda | 推理 decode 的核心路径；训练无此概念 |

### 9.3 算子数量总结

| 类别 | MindSpeed-MM | vllm-ascend |
|------|:---:|:---:|
| KDA 定制 NPU 算子 | 1 (chunk_kda via triton_ascend) | 4 (chunk_kda_fwd, kda_gate_cumsum, recurrent_kda, kda_layout_swap12) |
| SiTU 量化算子 | 0 | 2 (dequant_situ_quant, situ_mx_quant) |
| 共享 NPU 算子 (torch_npu) | 6 | 6 |
| AscendC 独立实现算子 | 1 (causal_conv1d) | 2 (causal_conv1d_custom, chunk_gated_delta_rule_fwd_h) |
| **新注册 torch.ops._C_ascend** | **0** | **6** |
