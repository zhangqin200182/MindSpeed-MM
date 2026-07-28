# Kimi-K3 并行策略与资源配置指南

> 分析日期: 2026-07-28
> 数据来源: [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) config.json + safetensors 文件清单

---

## 1. 模型规模概览

### 1.1 架构参数 (来自 config.json)

| 参数 | 数值 | 说明 |
|------|------|------|
| `hidden_size` | 7168 | 主隐藏维度 |
| `num_hidden_layers` | **93** | 69 KDA + 24 MLA |
| `num_attention_heads` | 96 | 注意力头数 |
| `kv_lora_rank` | 512 | MLA KV 低秩压缩维度 |
| `q_lora_rank` | 1536 | MLA Q 低秩压缩维度 |
| `intermediate_size` | **33792** | Dense MLP 中间维度 |
| `moe_intermediate_size` | 3072 | MoE Expert 中间维度 |
| `num_experts` | 896 | 路由专家总数 |
| `num_experts_per_token` | 16 | 每个 token 激活的专家数 |
| `num_shared_experts` | 2 | 共享专家 |
| `vocab_size` | **163840** | 词表大小 (16 万) |
| `max_position_embeddings` | 1,048,576 | 最大上下文 1M |
| `hidden_act` | `situ` | SiTU 自定义激活 (beta=4.0, linear_beta=25.0) |
| KDA head_dim (K) | 128 | Delta Attention 头维度 |
| KDA conv_kernel_size | 4 | ShortConvolution 核大小 |

### 1.2 参数量与权重

| 指标 | 数值 |
|------|------|
| 总参数量 | **2.78T** |
| 激活参数量 (per token) | **~104B** |
| 权重文件总大小 (BF16/MXFP mixed) | **1560 GB** (96 个 safetensors shard) |
| 量化版本大小 (MXFP4) | **~594 GB** |

---

## 2. 并行策略总览

MindSpeed-MM 提供了以下可组合的并行策略：

```
┌─────────────────────────────────────────────────────┐
│                  Kimi-K3 并行维度                      │
│                                                     │
│  FSDP2 (ZeRO-2 风格数据并行)                          │
│    └─ fully_shard_parallel_size: auto               │
│    └─ 参数/梯度/优化器状态均匀分片到所有卡                │
│                                                     │
│  EP (Expert Parallel)                               │
│    └─ expert_parallel_size: N                       │
│    └─ 896 experts 分到 N 组, 每组 896/N experts      │
│    └─ dispatcher: alltoall / allgather              │
│                                                     │
│  CP (Context Parallel)                              │
│    └─ ulysses_parallel_size: N                      │
│    └─ 长序列 attention 计算切分                        │
│    └─ Ring Attention (NPU FA2/FA3 only)             │
│                                                     │
│  显存优化 (辅助)                                      │
│    └─ recompute (重计算)                              │
│    └─ activation offload (激活值 CPU 卸载)             │
│    └─ chunk_loss / chunk_mbs (分块计算)                │
└─────────────────────────────────────────────────────┘
```

### 2.1 FSDP2 (必选)

所有场景都需要。参数、梯度、优化器状态均匀分片到所有卡。

### 2.2 EP (Expert Parallel, 必选)

896 个 expert, 每个约 22M 参数 (3584×3072×2), 不切分单卡放不下。EP 组内每卡持有 `896 / expert_parallel_size` 个 expert。

dispatcher 选择规则:
- `expert_parallel_size > topk(16)` → `alltoall` (推荐, 通信量更小)
- `expert_parallel_size < topk(16)` → `allgather` (每卡需要完整 token 集)

### 2.3 CP (Context Parallel, 长序列时推荐)

`cutoff_len > 4096` 时建议开启。减少每个 attention head 处理的序列长度, 从而降低激活显存。

---

## 3. 资源配置

### 3.0 显存分析基础

**单卡放权重的下限计算** (不考虑优化器):

```
BF16 权重总量: 1560 GB
MXFP4 量化权重: 594 GB

单卡仅放权重 (BF16):
  64GB 卡: 1560 / 64 ≈ 25 卡
  128GB 卡: 1560 / 128 ≈ 13 卡

单卡仅放权重 (MXFP4):
  64GB 卡: 594 / 64 ≈ 10 卡
```

训练需要额外存放优化器状态 (AdamW: 参数 × 3 × 4 bytes ≈ 参数 × 12 bytes) 和梯度 + 激活值, 所需的卡数远多于纯推理。

### 3.1 全量微调 (Full Fine-tuning)

**场景**: 全部 ~2.78T 参数参与训练, BF16 混合精度

**单卡显存逐项估算** (以 A2 64GB 为例):

| 项目 | 公式 | 32 卡 (EP=8) | 64 卡 (EP=16) | 128 卡 (EP=16) |
|------|------|:--:|:--:|:--:|
| Expert 权重 | `1560 × (896/EP) / 896 / DP` GB | 49 / 4 = 12.2 | 23 / 4 = 5.7 | 23 / 8 = 2.9 |
| 非 Expert 权重 | `~60 / total_cards` GB | 1.9 | 0.9 | 0.5 |
| 优化器状态 | `~2 × 权重` GB (FSDP2 shard) | 28.2 | 13.2 | 6.8 |
| 梯度 | `~1 × 权重` GB | 14.1 | 6.6 | 3.4 |
| 激活值 | 重计算+卸载后 | ~15 | ~10 | ~8 |
| **合计** | | **~71.4 GB** ❌ | **~36.4 GB** ✅ | **~21.6 GB** ✅ |

> 注: 实际显存分布受 FSDP2 sharding plan、EP group topology、micro_batch_size 影响, 以上为数量级估算。

**结论**:

| 平台 | 最少卡数 | 推荐卡数 | EP | 每节点×节点 |
|------|:--:|:--:|:--:|------|
| **A2 (64GB)** | **64** | 128 | 16 | 8×8 或 8×16 |
| **A3 (64GB)** | **64** | 64 | 16 | 8×8 |
| **A5 (128GB)** | **32** | 64 | 8 | 8×4 或 8×8 |

> ⚠️ **A2 单卡 64GB 在 EP=8 时显存约 71GB, 放不下!** 需要 EP≥16 或 开启 CPU offload + chunk_mbs 等全部优化。32 卡 A2 全量微调不可行。

**A2 推荐配置** (64 卡):

```yaml
parallel:
  fully_shard_parallel_size: auto
  expert_parallel_size: 16
  ep_plan:
    dispatcher: alltoall
    use_npu_fused_ops: true

training:
  micro_batch_size: 1
  gradient_accumulation_steps: 16

features:
  recompute: true
  enable_activation_offload: true
  enable_chunk_loss: true
  enable_chunk_mbs: true      # 必须, 减少激活值尖刺

model:
  skip_flash_attn_recompute: true
  skip_kda_recompute: true
  use_grouped_expert_matmul: true
```

### 3.2 LoRA 微调

**场景**: 仅训练 LoRA adapter (~200M 参数), base 模型 2.78T 冻结

**关键差异**: 
- 冻结权重不需要优化器状态
- 优化器仅用于 LoRA 参数 (~200M × 12 bytes ≈ 2.4GB, 可忽略)
- 主要瓶颈: 前向/反向的激活显存 + base 权重读取

**单卡显存估算** (A2):

| 项目 | 16 卡 (EP=4) | 32 卡 (EP=8) |
|------|:--:|:--:|
| Base 权重 (FSDP2 分片) | ~97 GB / 16 ≈ 6.1 | ~97 GB / 32 ≈ 3.0 |
| Expert 权重 (EP 分片) | ~50 GB / 4 ≈ 12.5 | ~50 GB / 8 ≈ 6.25 |
| 激活值 (重计算+卸载后) | ~20 | ~12 |
| LoRA 优化器 | ~0.2 | ~0.2 |
| **合计** | **~38.8 GB** ✅ | **~21.5 GB** ✅ |

> Expert 权重和 base 权重的拆分: 896 experts 的 gate_up + down 约 2.6T 参数, 其余 (attention, embedding, shared, lm_head, norm) 约 0.18T。Expert 由 EP 分片, 其余由 FSDP2 分片。

**结论**:

| 平台 | 最少卡数 | 推荐卡数 | EP | 备注 |
|------|:--:|:--:|:--:|------|
| **A2 (64GB)** | **16** | 32 | 4 | 需 LoRA 功能适配 (见开发方案) |
| **A3 (64GB)** | 16 | 16 | 4 | |
| **A5 (128GB)** | 8 | 8 | 4 | |

> ⚠️ **我之前估算的 A2 8 卡 LoRA 是错误的。** 激活值 + Expert 权重在 EP=4 时已达 38.8GB, 8 卡意味着 expert 权重翻倍 ~25GB, 加上激活值 20GB, 合计 ~48GB — 边界, 需要非常激进的优化才能跑。

**A2 LoRA 推荐配置** (32 卡):

```yaml
parallel:
  fully_shard_parallel_size: auto
  expert_parallel_size: 8
  ep_plan:
    dispatcher: alltoall
    use_npu_fused_ops: true

training:
  lora:
    enable: true
    rank: 8
    alpha: 16
    target_modules:  # 见 LoRA 开发方案文档

features:
  recompute: true
  enable_activation_offload: true
  enable_chunk_mbs: true
```

### 3.3 推理 (vllm-ascend)

**场景**: 在线推理服务, MXFP4 量化权重 ~594 GB

| 平台 | EP | 最少总卡数 | 单卡显存占用 | 备注 |
|------|:--:|:--:|:--:|------|
| **A2 (64GB)** | 64 | **64** | `594/64 + KV Cache ≈ 9.3 + ~30 = ~39 GB` | EP=64 硬约束 |
| **A3 (64GB)** | 64 | 64 | 同上 | |
| **A5 (128GB)** | 64 | 64 | `594/64 + KV Cache ≈ 9.3 + ~60 = ~69 GB` | KV Cache 余量更大 |

> EP=64 是推理的最低要求。896 experts / 14 per card = 64 卡。MQFP4 量化权重 594GB, 64 卡每卡 ~9.3GB。剩余空间用于 KV cache + KDA recurrent states + Conv states。

---

## 4. 资源需求汇总 (修正版)

| 场景 | A2 最小 | A2 推荐 | A3 最小 | A5 最小 | 关键约束 |
|------|:--:|:--:|:--:|:--:|------|
| **全量微调** | **64** | 128 | **64** | **32** | EP≥16(A2), EP≥8(A5) |
| **LoRA 微调** | **16** | 32 | **16** | **8** | EP≥4, 需重计算+卸载 |
| **推理** | **64** | 128 | **64** | **64** | EP=64 硬约束 |

### 4.1 与之前估算的差异

| 项目 | 之前估算 | 修正后 | 偏差原因 |
|------|:--:|:--:|------|
| 全量微调 A2 最少 | 32 | **64** | 之前用的 ~48 layers / 50B params, 实际 93 layers / 2.78T params |
| LoRA A2 最少 | 8 | **16** | 之前低估了激活值 + 93 层 attention 的显存 |
| 权重大小 | ~100 GB | **1560 GB** | 之前把 active params 当成了 total params |
| dense MLP intermediate | ~3072 | **33792** | 之前没拿到真实 config |
| vocab_size | ~128K | **163840** | embedding/lm_head 比预期大 27% |

---

## 5. 之前其他文档中的修正项

| 文档 | 需要修正的内容 |
|------|------|
| `kimi_k3_npu_operators.md` | 无 (算子分析不涉及资源估算) |
| `kimi_k3_npu_operators_comparison.md` | 无 (算子对比) |
| `kimi_k3_npu_operators_alignment.md` | 1.3 节: A2 全量微调最少 64 卡 (原写 16) |
| `kimi_k3_lora_development.md` | 4.3 节: A2 LoRA 最少 16 卡 (原写 8) |
| `kimi_k3_cann_operators.md` | 无 (算子依赖) |
