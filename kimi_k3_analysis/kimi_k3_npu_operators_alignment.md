# Kimi-K3 训推精度对齐分析

> 分析目标: 识别 vllm-ascend 推理侧为实现与 MindSpeed-MM 训练侧数值对齐所需的修改点
> 分析日期: 2026-07-28

---

## 1. 对齐目标与前提

### 1.1 对齐范围

训推精度对齐的核心诉求是：**相同输入下，vllm-ascend 推理的 logits 与 MindSpeed-MM 训练的 eval forward 输出在可接受误差范围内一致**。

可接受的误差量级参考：
- BF16 下逐 token logits 的 max diff < 1e-2, mean diff < 1e-4
- KDA attention output 的 max diff < 5e-3

### 1.2 根本性差异（无法也不应对齐）

以下差异源自训练/推理的本质不同，**不应强行对齐**：

| 差异项 | MindSpeed-MM (训练) | vllm-ascend (推理) | 对齐策略 |
|--------|:---:|:---:|------|
| Decode 阶段 | ❌ 不存在 | ✅ `recurrent_kda` 逐 token | 需要新增参考实现 |
| 量化 (SiTU + quant) | ❌ | ✅ INT8/FP8 | **推理需要 dequant→SiTU→BF16 路径** |
| Batch 动态变化 | ❌ 固定 | ✅ 状态池 | 不影响数值精度 |
| 反向传播 | ✅ | ❌ | 不影响 forward 精度 |
| Gradient Checkpointing | ✅ skip_recompute | ❌ | 不影响 forward 精度 |

---

### 1.3 平台兼容性 (A2 / A3 / A5)

Kimi-K3 推理对昇腾各平台的算子兼容性如下：

| 算子 | A2 (`ascend910b`) | A3 (`ascend910_93`) | A5 (`ascend950`) | 备注 |
|------|:---:|:---:|:---:|------|
| `recurrent_kda` | ✅ | ✅ | ✅ | 公共 AIV kernel，三平台无算法差异 |
| `chunk_kda_fwd` | ✅ | ✅ | ✅ | `arch32` 覆盖 A2/A3；A5 独立 `arch35` |
| `kda_gate_cumsum` | ✅ | ✅ | ✅ | `arch32` 覆盖 A2/A3 |
| `kda_layout_swap12` | ✅ | ✅ | ✅ | 公共实现 |
| `dequant_situ_quant` | ✅ | ✅ | ❌ | A2/A3 共用 INT8 量化路径 |
| `situ_mx_quant` | ❌ | ❌ | ✅ | A5 专用 MX FP8 量化 |
| `npu_causal_conv1d_custom` | ✅ | ✅ | ✅ | 已有复用算子 |

**架构映射**（`csrc/CMakeLists.txt`）：

```
ascend310p   → arch22
ascend910b   → arch32    ← A2
ascend910_93 → arch32    ← A3 (与 A2 共享)
ascend950    → arch35    ← A5
kirinx90     → arch32
```

A2 与 A3 共用 `arch32` 编译目标，KDA 三个核心算子在 A2 上已通过精度验证：

```cpp
// chunk_kda_fwd: A2 已注册
this->AICore().AddConfig("ascend910b", aicoreConfig);   // chunk_kda_fwd_def.cpp:89

// kda_gate_cumsum: A2 已注册
this->AICore().AddConfig("ascend910b", aicoreConfig);   // kda_gate_cumsum_def.cpp:53
```

`recurrent_kda` 设计文档（`csrc/attention/recurrent_kda/docs/design.md:148-153`）明确列出三平台均为公共 AIV kernel，且在 A2/A5 精度验证中修复了 `scale/lower_bound` 的 fp64→fp32 cast、l2norm 后的 V→S 同步等问题。

**硬件限制**：

| 项目 | 要求 | 说明 |
|------|------|------|
| 卡数 | **≥ 64 张** (EP=64) | Expert Parallelism 切分 896 个 expert |
| 单卡 HBM | 64 GB (910B) / 64 GB (910_93) | W4A8 量化后足够；非量化需更多 |
| KDA 约束 | `K=V=128`, `chunk_size=64` | 三平台一致 |

A2 跑 Kimi-K3 推理算子层面无 blocker，与 A3 在精度特性上完全等价（共享 `arch32` 二进制）。以下章节的精度分析和修改建议对 A2/A3 均适用。

---

## 2. 精度差异点分析（按优先级排序）

### 2.1 P0 — KDA chunk_kda 算子内部计算差异

这是**影响最大**的精度差异来源。

#### 2.1.1 l2norm 执行位置不同

```
MindSpeed-MM (fused):
  chunk_kda(use_qk_l2norm_in_kernel=True)
    └─ l2norm_fwd(q), l2norm_fwd(k)    ← 融合在 triton_ascend kernel 内

vllm-ascend (prefill):
  q = l2norm_fwd(q.contiguous())        ← Python 侧，独立的 Triton kernel
  k = l2norm_fwd(k.contiguous())        ← input dtype=Bf16, compute=float32
  chunk_kda_fwd(q, k, ...)              ← AscendC kernel 接收已归一化的 q,k
```

**差异**：虽然公式相同（`x / sqrt(sum(x²) + 1e-6)`），但 f32 reduction 顺序不同（kernel 内 tiling vs Triton block tiling），导致末位精度差异。

**修改建议**：
```python
# vllm_ascend/ops/kimi_kda.py → AscendKimiGatedDeltaNetAttention._run_prefill

# 当前实现:
q = l2norm_fwd(q.contiguous())
k = l2norm_fwd(k.contiguous())

# 对齐方案 A（推荐）: 将 l2norm 从 Python 侧移入 chunk_kda_fwd kernel
#   → 修改 AscendC kernel，使用内部统一的 l2norm reduction order
#   → 新增 use_qk_l2norm_in_kernel 参数，传入未归一化的 q/k
# 对齐方案 B: 使用 MindSpeed-MM 同款 l2norm 实现做 Python 侧预处理
#   → 复用 mindspeed_mm.fsdp.ops.kda.chunk_kda_naive.l2norm
```

#### 2.1.2 gate transform 拆分 vs 融合

```
MindSpeed-MM (fused):
  chunk_kda(g=raw_gate, use_gate_in_kernel=True, safe_gate=True, lower_bound=-5)
    ├─ gate = lower_bound * sigmoid(exp(A_log) * (g + dt_bias))   ← kernel 内 fp32
    └─ cumsum(gate, chunk_size)                                     ← kernel 内 fp32

vllm-ascend (prefill):
  gate_cumsum = kda_gate_cumsum(raw_gate, chunk_size,
      A_log=A_log, dt_bias=dt_bias,
      use_gate_in_kernel=True, safe_gate=True, lower_bound=-5.0)   ← AscendC kernel
  chunk_kda_fwd(q, k, v, gate_cumsum, beta, scale, ...)           ← 接收 gate_cumsum
```

**差异**：两段计算在不同的硬件单元间传输（gate_cumsum 输出 → chunk_kda_fwd 输入需写 DRAM），中间结果的 rounding 方式可能不同。训练 kernel 在 UB 内完成全流程。

**修改建议**：
```python
# 在 MindSpeed-MM 的 fused 模式下，对同一组输入并行运行:
#   A: chunk_kda (训练 fused kernel)
#   B: kda_gate_cumsum + chunk_kda_fwd (推理 split kernel)
# 对比 (A) vs (B) 的输出差异，profile gate_cumsum 中间精度
# 如果差异 > 1e-4: 需要 AscendC kernel 内部对齐 rounding 策略
```

#### 2.1.3 chunk 内部计算 4-Phase vs Fused

MindSpeed-MM 的 `triton_ascend_kernels.chunk_kda` 是一个 **single launch kernel**，内部 Phase 1-4 在 UB 内完成，不写 DRAM 中间结果。

vllm-ascend 的 AscendC `chunk_kda_fwd` 返回 **12 个张量**，内部显式分为 4 个 Phase（scaled_dot_kkt → solve_tril → recompute_w_u → gdn_fwd_h → gla_fwd_o），每个 Phase 产生中间结果写 DRAM：

| Phase | 中间输出 | 精度 |
|-------|---------|------|
| Phase 1 | `Aqk`, `Akk` | BF16/FP32 |
| Phase 2 | `w`, `u`, `qg`, `kg` | BF16/FP32 |
| Phase 3 | `h`, `v_new` | BF16/FP32 |
| Phase 4 | `o` | BF16 |

这导致每个 Phase 边界都有一次 **fp32 → BF16 rounding**，而训练 kernel 在 UB 内全程 fp32 直到最终输出才 cast。

**修改建议**：
```python
# AscendC kernel 改造 (P0 优先级):
# 1. 将 Phase 1-4 合并为单次 kernel launch
# 2. 中间结果 (Aqk, A, w, u, h, v_new) 在 UB 内以 fp32 传递
# 3. 仅在 final output (o) 处 cast 到 BF16
# 
# 如果无法合并为单 kernel:
# 备选方案：将中间结果以 fp32 存储，在各 Phase 入口处加载 fp32
# → DRAM 带宽增加, 但精度对齐
```

#### 2.1.4 chunk scan 的 log2 空间计算

MindSpeed-MM `chunk_kda_naive` 文档明确说明：
> The triton kernel works in log2 space (scale=RCP_LN2 + exp2), which is mathematically equivalent to exp; plain exp is used here.

vllm-ascend Triton `chunk_kda_fwd` 也使用 `RCP_LN2` + `exp2`：
```python
# kda.py:1124
g = g * RCP_LN2
# kda.py:533
b_k = tl.load(p_k, ...) * tl.exp2(b_g - b_gn[None, :])
```

**差异**：`exp2(log2_e * x)` 与 `exp(x)` 在 fp32 下有微小差异（e 的表示误差）。训练 kernel 和推理 kernel 如果使用不同的 exp 实现（如硬件 exp vs 查表多项式），差异会放大。

**修改建议**：
```python
# 验证 AscendC kernel 中 exp 实现的查表精度是否与 Triton tl.exp2 一致
# 如有差异：统一使用硬件 expf 指令（Ascend C 的 exp() 内置函数）
```

### 2.2 P0 — recurrent_kda (Decode) 参考实现

MindSpeed-MM 不包含 decode 阶段的实现，因此无法直接对比精度。但若不验证，递归解码的误差累积会导致多轮对话后 logits 漂移。

**修改建议**：
```python
# 1. 基于 MindSpeed-MM 的 chunk_kda_naive 实现 recurrent reference
# 在 kimi_kda.py 中新增 _run_recurrent_reference():

def _run_recurrent_reference(self, q, k, v, raw_gate, beta, 
                               recurrent_state, cu_seqlens, state_indices):
    """Reference decode using chunk_kda_naive with chunk_size=1."""
    from mindspeed_mm.fsdp.ops.kda.chunk_kda_naive import chunk_kda_naive
    # 逐 token 调用 chunk_kda_naive(chunk_size=1) → 与 fused recurrent 对比

# 2. 对比测试:
#    - 同一组 initial_state + 单 token 输入
#    - 运行 reference vs AscendC recurrent_kda
#    - 验证 max diff < 1e-3

# 3. 如果差异超限:
#    - 检查 recurrent_kda kernel 的 state update 顺序与 reference 是否一致
#    - 检查 l2norm/gate/beta 在 iterative 模式下的累积误差
```

### 2.3 P1 — ShortConvolution 不同后端

```
MindSpeed-MM:
  q_conv1d, k_conv1d, v_conv1d (3 次独立调用)
    → causal_conv1d (triton) 或 causal_conv1d_ascendc (fla_npu)

vllm-ascend:
  torch.cat([q, k, v], dim=-1) (1 次合并调用)
    → npu_causal_conv1d_custom (AscendC)
```

**差异**：
1. Concat 后单次调用 vs 3 次独立调用 — 虽然语义等价，但权重 concat 后 stride 不同可能影响内存对齐和向量化 rounding
2. `fla_npu` AscendC vs `npu_causal_conv1d_custom` AscendC — 两个独立的 AscendC 实现

**修改建议**：
```python
# 方案 A: 验证 concat vs 3次调用的差异
#   对同一输入分别运行:
#     torch.cat 方式 (vllm-ascend)
#     3次独立调用 (MindSpeed-MM 方式)
#   验证输出 max diff < 1e-5
#   如果差异在可接受范围 → 无需修改
#   如果差异超限 → 回退到 3 次独立调用

# 方案 B: 端到端对比
#   在 KDA forward 中, causal_conv1d 之后立即对比 q/k/v 的 hidden_states
#   使用 MindSpeed-MM 同款 fla_npu backend 运行相同权重
```

### 2.4 P1 — O-Norm (KimiK_3_MoeRMSNormGated) 精度

```
MindSpeed-MM:
  o = KimiK_3_MoeRMSNormGated(o, g)
    NPU:  o = npu_rms_norm(o, weight, eps) * sigmoid(gate_float)
    CPU:  o = rms_norm_fp32(o) * sigmoid(gate_float)
    key:  gate 在 fp32 下计算 sigmoid

vllm-ascend:
  o = self.o_norm(core_attn_out, output_gate)
    → rms_norm_gated(x, g, weight, activation="sigmoid")
    → Triton layer_norm_gated_fwd (RMS + sigmoid gate 融合)
    key: gate sigmoid 在 triton kernel 内以 float32 计算
```

**差异**：
- `npu_rms_norm` (硬件指令) vs Triton RMS (软件 tiling) — reduction tree 不同
- MindSpeed-MM: norm → sigmoid(gate)，两步分离
- vllm-ascend: norm + sigmoid(gate) 融合在 Triton kernel，但 sigmoid 的输入在 kernel 内部加载，其 f32 cast 时机可能不同

**修改建议**：
```python
# 验证方案:
#   MindSpeed-MM 方式: y = npu_rms_norm(x) * sigmoid(g.float())
#   vllm-ascend 方式: y = rms_norm_gated(x, g, w, activation="sigmoid")
#   对比 max diff, 如果 > 1e-4:
#     修改 vllm-ascend: 分离 norm 和 gate
#     y = rms_norm(x, w, eps) * torch.sigmoid(g.float())
```

### 2.5 P1 — SiTU 激活精度

```
MindSpeed-MM:
  SituAndMul.forward():
    gate = x[..., :d].to(torch.float32)
    up = x[..., d:].to(torch.float32)
    situ_a = beta * tanh(gate / beta) * sigmoid(gate)
    if linear_beta:
        up = linear_beta * tanh(up / linear_beta)
    return (situ_a * up).to(x.dtype)

vllm-ascend:
  routed experts: dequant_situ_quant → INT8   (A2/A3)
                  situ_mx_quant → MXFP8       (A5 only)
  shared experts: dequant_situ_quant(INT32→SiTU→INT8)  (A2/A3)
                  situ_mx_quant(BF16→SiTU→FP8)         (A5 only)
```

> **平台说明**: A2 和 A3 使用 `dequant_situ_quant`（INT8 量化），A5 使用 `situ_mx_quant`（MX FP8 量化）。训推精度对齐主要关注 A2/A3 的 INT8 量化误差；A5 的 MXFP8 路径精度特性不同，需独立评估。

**差异**：量化引入的误差远大于算子级精度差异（INT8 的 step size 约为 max(|x|)/127，相对误差 ~1% 量级）。

**修改建议**：
```python
# 关键修改: 新增 SiTU 的 BF16 非量化路径用于精度对比

# vllm_ascend/ops/activation.py 新增:
class AscendSituAndMul(nn.Module):
    """MindSpeed-MM 对齐版 SiTU，无量化"""
    def __init__(self, beta=4.0, linear_beta=25.0):
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)

# kimi_k3.py 中用于精度对齐的 MoE forward:
#   if self.precision_align_mode:
#       gate_up = GMM1(...)
#       situ_out = AscendSituAndMul(beta=4.0, linear_beta=25.0)(gate_up)
#       output = GMM2(situ_out)
#   else:
#       gate_up = GMM1(...)
#       situ_out, scale = dequant_situ_quant(gate_up, ...)
#       output = GMM2(situ_out, scale)
```

### 2.6 P2 — RMSNorm 精度

```
MindSpeed-MM (KimiRMSNorm):
  forward(x):
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(dtype)

vllm-ascend (RMSNorm, vllm 内置):
  标准 RMSNorm，可能与 MindSpeed-MM 的 fp32 eager 实现一致
```

**差异**：如果 vllm 的 RMSNorm 使用了不同的 eps 值或不同的中间精度策略，会有微小差异。

**修改建议**：
```python
# 验证 KimiRMSNorm.eps 是否与 config.rms_norm_eps 一致
# MindSpeed-MM: eps=1e-6 (默认), weight 在最后 cast
# vllm: 确认 eps 来源 (config.rms_norm_eps)
# 如果 max diff > 1e-5: 统一为 fp32 reduction
```

### 2.7 P3 — Vision Tower 差异

MindSpeed-MM VLM forward 中涉及 `_merge_input_ids_with_image_features` 的复杂 token 合并逻辑。vllm-ascend 中通过 `KimiK3MultiModalProcessor` 和 `embed_multimodal` 处理。

**差异**：两者的 pixel_values → image_features → input_embeds 流程可能有不同的 pad token 处理、image token 替换逻辑。

**修改建议**：
```python
# 端到端对比:
#   同一张图片 → MindSpeed-MM forward → hidden_states after embedding
#            → vllm-ascend embed_multimodal → inputs_embeds
#   验证 max diff < 1e-6
```

---

## 3. 修改优先级与实施路线

### 3.1 第一阶段: 精度基准建立 (P0)

**目标**：建立 MindSpeed-MM training forward 和 vllm-ascend prefill forward 的数值基准

```python
# 新增文件: tests/precision/test_kda_alignment.py
# 
# 1. 在 MindSpeed-MM 侧: 对单层 KimiDeltaAttention 运行 forward,
#    保存所有中间张量 (q, k, v, gate, beta, conv_q/k/v, chunk_kda_in/out,
#    o_norm_in/out, o_proj_out)
#
# 2. 在 vllm-ascend 侧: 加载相同权重和输入, 运行相同 layer 的 forward,
#    保存对应中间张量
#
# 3. 对比报告:
#    | Layer | Tensor | Max Diff | Mean Diff | Norm Diff |
#    |-------|--------|----------|-----------|-----------|
#    | q_proj | q | 1e-7 | 3e-9 | ... |
#    | causal_conv1d | q_conv | 2e-6 | 5e-8 | ... |
#    | l2norm | q_norm | 3e-6 | 1e-7 | ... |
#    | chunk_kda | o | 5e-3 | 2e-4 | ... |  ← expected largest
#    | o_norm | o_gated | 1e-4 | 1e-5 | ... |
```

### 3.2 第二阶段: KDA 精度修正 (P0)

基于第一阶段基准，按优先级修正：

| 序号 | 修改项 | 文件 | 难度 |
|------|--------|------|:--:|
| 1 | **AscendC chunk_kda_fwd 内部 fp32 中间传递** | `csrc/attention/chunk_kda_fwd/` | 高 |
| 2 | **l2norm 移入 AscendC kernel (或统一 Python 实现)** | `csrc/attention/chunk_kda_fwd/` + `kimi_kda.py` | 中 |
| 3 | **gate_cumsum + chunk_kda_fwd 合并为单 kernel** | `csrc/attention/` | 高 |
| 4 | **recurrent_kda 参考实现** | `kimi_kda.py` + tests | 中 |

### 3.3 第三阶段: 辅助算子修正 (P1)

| 序号 | 修改项 | 文件 | 难度 |
|------|--------|------|:--:|
| 5 | **SiTU BF16 非量化路径** | `vllm_ascend/ops/activation.py` + `kimi_k3.py` | 低 |
| 6 | **O-Norm 分离为 norm + gate** | `kimi_kda.py` 或 `kda.py` | 低 |
| 7 | **ShortConv 后端对齐验证** | `kimi_kda.py` + tests | 低 |
| 8 | **RMSNorm eps 对齐** | `kimi_k3.py` | 低 |

### 3.4 第四阶段: 端到端验证 (P2)

| 序号 | 修改项 |
|------|--------|
| 9 | Decode 多步精度累积测试 (1→10→100 步) |
| 10 | MoE routing + SiTU 端到端对比 |
| 11 | Vision Tower embedding 对比 |
| 12 | Full model logits 对比 (同一 prompt) |

---

## 4. 实现建议

### 4.1 精度对齐开关 (Precision Align Mode)

在 vllm-ascend 中新增全局精度对齐模式，一键切换到与训练对齐的算子路径：

```python
# vllm_ascend/ascend_config.py
class AscendConfig:
    precision_align_mode: bool = False  # 默认关闭，线上推理不需要

# 启用方式:
# 配置 YAML: ASCEND_PRECISION_ALIGN_MODE=1
# 代码: get_ascend_config().precision_align_mode = True

# kimi_kda.py 中的使用:
if get_ascend_config().precision_align_mode:
    # 使用与 MindSpeed-MM 对齐的算子路径
    q = l2norm_align(q)          # fp32 mean reduction
    k = l2norm_align(k)
    o = chunk_kda_naive(q, k, v, g, beta, ...)  # pure PyTorch path
else:
    # 生产推理路径
    q = l2norm_fwd(q)
    k = l2norm_fwd(k)
    gate_cumsum = kda_gate_cumsum(...)
    o = chunk_kda_fwd(q, k, v, gate_cumsum, ...)
```

### 4.2 关键算子统一策略

对于差异最大的 KDA prefill，优先采用以下策略：

```
┌─────────────────────────────────────────────────────────┐
│ 策略 A (短期):                                           │
│   Python 侧对齐预处理 (l2norm, gate, beta)                │
│   → AscendC kernel 只做 chunk scan                       │
│   → 降低 AscendC kernel 修改成本                         │
│   缺点: 多次 kernel launch，性能下降                      │
├─────────────────────────────────────────────────────────┤
│ 策略 B (长期, 推荐):                                     │
│   AscendC kernel 内部实现与 triton_ascend 一致的全融合    │
│   → 单 kernel, 内部 fp32, 仅输出 cast BF16               │
│   → 精度+性能兼得                                       │
│   需要: AscendC kernel 重写                             │
└─────────────────────────────────────────────────────────┘
```

### 4.3 recurrent_kda 验证方法

因训练无 decode 阶段，recurrent_kda 的精度验证需通过**自洽性测试**：

```python
# tests/precision/test_recurrent_kda_self_consistency.py

def test_recurrent_vs_chunk_consistency():
    """同一序列，chunk_kda(全部 token) vs recurrent_kda(逐 token) 对比"""
    # 1. 准备长度为 L=64 的序列
    # 2. chunk_kda 处理全部 64 token → out_chunk, final_state_chunk
    # 3. recurrent_kda 逐 token 处理 64 步 → out_recurrent, final_state_recurrent
    # 4. 验证:
    #    - out_recurrent 每步结果与 out_chunk 对应位置一致 (max diff < 1e-3)
    #    - final_state 一致 (max diff < 1e-4)
    # 5. 如果差异超限: 检查 recurrent_kda kernel 的 state update 顺序是否
    #    与 chunk_kda 的 inter-chunk recurrence 一致
```

---

## 5. 总结

| 优先级 | 差异点 | 根因 | 修改方向 |
|:---:|--------|------|------|
| **P0** | chunk_kda 内部 4-Phase DRAM 中间结果 | AscendC 拆算子 vs 训练单融合 kernel | AscendC kernel 内部 fp32 中间传递, 最终合并为单 kernel |
| **P0** | l2norm Python 侧 vs kernel 内 | 执行位置不同导致 reduction 顺序不同 | 移入 AscendC kernel 或统一 Python 实现 |
| **P0** | recurrent_kda 无参考实现 | 训练无 decode 阶段 | 基于 chunk_kda_naive 实现参考路径 + 自洽性测试 |
| **P1** | ShortConv concat vs 3次调用 | 不同实现后端 | 验证差异量级，必要时回退到 3 次独立调用 |
| **P1** | O-Norm norm+gate 融合顺序 | Triton vs npu_rms_norm 实现不同 | 分离 norm 和 gate，统一 eps 和中间精度 |
| **P1** | SiTU + 量化精度损失 | INT8/FP8 量化 | 新增 BF16 非量化 SiTU 路径用于精度对齐 |
| **P2** | RMSNorm eps / fp32 中间精度 | 默认参数差异 | 统一 eps 和 weight 应用顺序 |
| **P2** | Vision Tower token 合并 | 实现差异 | 端到端对比验证 |

**最小可行方案**（仅 P0 级别修改）：
1. 在 Python 侧统一 l2norm 实现（直接用 MindSpeed-MM `chunk_kda_naive.l2norm`）
2. AscendC `chunk_kda_fwd` 内部改用 fp32 存储中间张量
3. 新增 `recurrent_kda` 的 `chunk_kda_naive` 参考路径用于精度验证
