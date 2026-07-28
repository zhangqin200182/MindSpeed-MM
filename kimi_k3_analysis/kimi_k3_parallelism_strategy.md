# Kimi-K3 并行策略与资源配置指南

> 分析日期: 2026-07-28

---

## 1. 模型规模概览

| 参数 | 数值 | 说明 |
|------|------|------|
| Hidden Size | 7168 | 主隐藏维度 |
| Routed Expert Hidden | 3584 | Latent MoE 压缩后的专家隐藏维度 |
| Routed Expert Intermediate | 3072 | 专家 FFN 中间维度 |
| Num Experts | 896 | 路由专家数 |
| Top-K | 16 | 每个 token 激活的专家数 |
| Shared Experts | 2 | 共享专家数 |
| Shared Expert Intermediate | 6144 | 2×3072 |
| Num Layers | ~48 | 包含 MLA + KDA 混合 attention |
| KDA Head Dim (K/V) | 128 | Delta Attention 头维度 |
| SiTU Beta / Linear Beta | 4.0 / 25.0 | 自定义激活参数 |

**模型总参数量估算**: ~45-50B (BF16 约 90-100GB)

---

## 2. 并行策略总览

MindSpeed-MM 提供了以下可组合的并行策略：

```
┌─────────────────────────────────────────────────────┐
│                  Kimi-K3 并行维度                      │
│                                                     │
│  FSDP2 (数据并行)                                     │
│    └─ fully_shard_parallel_size: auto               │
│    └─ 参数/梯度/优化器分片到所有卡                       │
│                                                     │
│  EP (Expert Parallel)                               │
│    └─ expert_parallel_size: N                       │
│    └─ 896 experts 分到 N 个 EP 组                     │
│    └─ dispatcher: alltoall / allgather              │
│                                                     │
│  CP (Context Parallel)                              │
│    └─ ulysses_parallel_size: N                      │
│    └─ 长序列的注意力计算分到 N 个 CP 组                  │
│    └─ Ring Attention (NPU FA2/FA3)                  │
│                                                     │
│  显存优化 (辅助)                                      │
│    └─ recompute (重计算)                              │
│    └─ activation offload (激活 CPU 卸载)               │
│    └─ chunk_loss / chunk_mbs (分块计算)                │
└─────────────────────────────────────────────────────┘
```

### 2.1 FSDP2 (必选)

所有场景都需要。参数、梯度、优化器状态均匀分片到所有卡。

```yaml
parallel:
  fully_shard_parallel_size: auto   # 自动计算
```

### 2.2 EP (Expert Parallel, 必选)

896 个 expert 需要 EP 切分，否则单卡放不下。EP 组内的卡各持有 `896 / expert_parallel_size` 个 expert。

```yaml
parallel:
  expert_parallel_size: 4   # 或 8
  ep_plan:
    apply_modules:
      - language_model.model.layers.{*}.block_sparse_moe.experts
    dispatcher: alltoall     # EP > topk(16) 时推荐
    use_npu_fused_ops: true
```

dispatcher 选择：
- `expert_parallel_size > topk(16)` → `alltoall` (推荐)
- `expert_parallel_size < topk(16)` → `allgather`

### 2.3 CP (Context Parallel, 长序列时推荐)

当 `cutoff_len > 4096` 时建议开启，缓解 attention 计算的显存压力。

```yaml
parallel:
  ulysses_parallel_size: 2   # 或更大
```

---

## 3. 推荐部署配置

### 3.1 全量微调 (Full Fine-tuning)

**场景**: 最大训练能力，全部参数参与训练

| 平台 | EP | 每节点卡数 | 最少节点 | 总卡数 | 备注 |
|------|:--:|:--:|:--:|:--:|------|
| **A5 (950, 128GB)** | 4 | 8 | 2 | 16 | 示例配置，EP=4 |
| **A3 (910_93, 64GB)** | 8 | 8 | 2 | 16 | 需开启重计算+CPU卸载 |
| **A2 (910B, 64GB)** | 8 | 8 | 4 | 32 | 需开启全部显存优化 |

**A2 推荐配置** (`kimik3_config.yaml`):

```yaml
parallel:
  fully_shard_parallel_size: auto
  expert_parallel_size: 8          # A2 显存紧张, EP 切大点
  ulysses_parallel_size: 1         # 短序列可关闭
  ep_plan:
    apply_modules:
      - language_model.model.layers.{*}.block_sparse_moe.experts
    dispatcher: alltoall
    use_npu_fused_ops: true

training:
  micro_batch_size: 1
  gradient_accumulation_steps: 16   # 增大 GA 补偿小 microbatch

features:
  recompute: true                   # 必须开启
  enable_activation_offload: true    # 必须开启
  enable_chunk_loss: true           # 推荐
  enable_chunk_mbs: true            # 推荐
  chunkmbs_plan:
    chunk_mbs: 1

model:
  skip_flash_attn_recompute: true   # 跳过 FA 重计算
  skip_kda_recompute: true          # 跳过 KDA 重计算
  use_grouped_expert_matmul: true   # NPU 融合 MoE
```

**A2 单卡显存估算** (EP=8, FSDP2 32 卡):

| 项目 | 大小 |
|------|------|
| Expert 权重 (112 experts, BF16) | ~5 GB |
| 非 Expert 权重 (BF16) | ~3 GB |
| 优化器状态 (AdamW fp32) | ~24 GB |
| 梯度 + 激活值 (已开重计算) | ~15 GB |
| **合计** | **~47 GB < 64 GB** ✅ |

### 3.2 LoRA 微调

**场景**: 仅训练小量 adapter，base 模型冻结

| 平台 | EP | 每节点卡数 | 最少节点 | 总卡数 | 备注 |
|------|:--:|:--:|:--:|:--:|------|
| **A2/A3/A5** | 4 | 8 | 1 | 8 | 激活占用是瓶颈 |

```yaml
parallel:
  expert_parallel_size: 4           # LoRA 场景 EP 可减小
  ep_plan:
    dispatcher: alltoall
    use_npu_fused_ops: true

training:
  lora:
    enable: true
    rank: 8
    alpha: 16
    target_modules:
      - "language_model.model.layers.{*}.self_attn.q_a_proj"
      - "language_model.model.layers.{*}.self_attn.q_b_proj"
      - "language_model.model.layers.{*}.self_attn.kv_a_proj_with_mqa"
      - "language_model.model.layers.{*}.self_attn.kv_b_proj"
      - "language_model.model.layers.{*}.self_attn.o_proj"
      - "language_model.model.layers.{*}.self_attn.q_proj"
      - "language_model.model.layers.{*}.self_attn.k_proj"
      - "language_model.model.layers.{*}.self_attn.v_proj"

features:
  recompute: true
  enable_activation_offload: true
```

> **注意**: LoRA 功能当前为实验特性，Kimi-K3 的 LoRA 适配需参考 [`kimi_k3_lora_development.md`](kimi_k3_lora_development.md)。

### 3.3 推理 (vllm-ascend)

**场景**: 在线推理服务

| 平台 | EP | 最少总卡数 | 量化 | 备注 |
|------|:--:|:--:|------|------|
| **A2/A3** | 64 | 64 | W4A8 INT8 | 896/14=64 卡, 每卡 14 experts |
| **A5** | 64 | 64 | W4A8 MXFP8 | 同上 |

```bash
vllm serve moonshotai/Kimi-K3 \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --expert-parallel-size 64 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9
```

> EP=64 是推理的最低要求。896 个 expert / 14 个 per card = 64 卡。

---

## 4. 资源需求汇总

| 场景 | 最小卡数 (A2) | 推荐卡数 (A2) | 关键约束 |
|------|:--:|:--:|------|
| **全量微调** | 32 (4 节点×8) | 64 (8 节点×8) | 需开启重计算+CPU卸载+chunk_loss |
| **LoRA 微调** | 8 (1 节点×8) | 16 (2 节点×8) | 需开启重计算+CPU卸载 |
| **推理** | 64 (8 节点×8) | 128 (16 节点×8) | EP=64 硬约束 |

### 4.1 为什么 A2 全量微调最少 32 卡？

```
EP=8: Expert 权重切分到 8 卡, 每卡 112 experts ≈ 5GB
FSDP2: 非 expert 参数在 EP 组内再切分
  优化器状态: ~200GB / 32 卡 ≈ 6.25GB/卡
  梯度: ~100GB / 32 卡 ≈ 3.1GB/卡
  激活值: ~15GB/卡 (开重计算+CPU卸载)
  ─────────────────
  合计: 5 + 3 + 6.25 + 3.1 + 15 ≈ 32.4GB < 64GB ✅
```

如果只用 16 卡 (EP=4):
```
  Expert 权重: 224 experts ≈ 10GB
  优化器: ~200GB / 16 卡 ≈ 12.5GB/卡
  梯度: ~100GB / 16 卡 ≈ 6.25GB/卡
  激活值: ~15GB/卡
  ─────────────────
  合计: 10 + 3 + 12.5 + 6.25 + 15 ≈ 46.75GB — 边界，需更激进优化
```

### 4.2 为什么推理 EP=64 是硬约束？

- 896 experts，每个 expert 的权重约 22M 参数 (W4A8 量化后约 5.5MB)
- 每卡放 14 个 experts: 14 × 5.5 ≈ 77MB — 完全可以
- 但非 expert 权重 (attention, embedding, shared) 较大，需 EP=64 来分摊 KV Cache
- EP=32 时每卡 28 experts，KV Cache 可用空间减半，max_model_len 会受限制

---

## 5. 循序渐进部署路线

```
第 1 步: A5 验证
  └─ 16 卡, EP=4, 跑通全量微调, 验证 loss 收敛

第 2 步: A3 对齐
  └─ 16 卡, EP=8, 开启重计算+CPU卸载, 验证精度对齐 A5

第 3 步: A2 对齐
  └─ 32 卡, EP=8, 全部显存优化开启, 验证精度对齐

第 4 步: A2 LoRA
  └─ 8 卡, EP=4, 跑通 LoRA 微调, 验证 loss 收敛

第 5 步: A2 推理
  └─ 64 卡, EP=64, 跑通 vllm-ascend 推理服务
```

---

## 6. 环境变量

训练启动前确保设置:

```bash
# CANN 任务队列优化
export TASK_QUEUE_ENABLE=1

# HCCL 超时 (多机必须)
export HCCL_CONNECT_TIMEOUT=1800

# 虚拟内存 (大模型推荐)
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Triton 编译缓存 (避免反复编译)
export TRITON_ALWAYS_COMPILE=0
```
