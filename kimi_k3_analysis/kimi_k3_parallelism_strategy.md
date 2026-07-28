# Kimi-K3 并行策略与资源配置指南

> 分析日期: 2026-07-28
> 数据来源: [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) config.json + safetensors 文件清单

---

## 1. 模型规模概览

### 1.1 架构参数 (来自 config.json)

| 参数 | 数值 | 说明 |
|------|------|------|
| `hidden_size` | 7168 | 主隐藏维度 |
| `num_hidden_layers` | 93 | 69 KDA + 24 MLA |
| `num_attention_heads` | 96 | 注意力头数 |
| `kv_lora_rank` | 512 | MLA KV 低秩压缩 |
| `q_lora_rank` | 1536 | MLA Q 低秩压缩 |
| `intermediate_size` | 33792 | Dense MLP 中间维度 |
| `moe_intermediate_size` | 3072 | MoE Expert 中间维度 |
| `num_experts` | 896 | 路由专家总数 |
| `num_experts_per_token` | 16 | 每个 token 激活的专家数 |
| `num_shared_experts` | 2 | 共享专家 |
| `vocab_size` | 163840 | 词表大小 |
| `max_position_embeddings` | 1,048,576 | 最大上下文 1M |
| `hidden_act` | `situ` | SiTU 自定义激活 |
| KDA head_dim (K) | 128 | Delta Attention 头维度 |

### 1.2 权重存储格式 — 关键约束

HuggingFace 发布的 Kimi-K3 checkpoint **是 MXFP4 + BF16 混合精度，不是纯 BF16**:

| 组成部分 | 存储格式 | 存储大小 | 等效参数量 |
|------|------|:--:|:--:|
| Expert 权重 | **MXFP4** (4-bit, 0.5 bytes/param) | ~1300 GB | ~2.6T |
| 非 Expert (attention, embedding, norm, shared, lm_head) | BF16 (2 bytes/param) | ~260 GB | ~0.13T |
| **合计** | 混合精度 | **~1560 GB** | **~2.73T** |

推演验证:

```
Expert  MXFP4: 2.6T × 0.5 = 1300 GB
非 Expert BF16: 0.13T × 2 = 260 GB
合计: 1300 + 260 = 1560 GB ✓ 匹配 HF 仓库大小
```

HuggingFace 上有两个版本:
- `moonshotai/Kimi-K3` — 1560 GB, 混合精度 (发布版本)
- `moonshotai/Kimi-K3-MXFP4` — 594 GB, 全量 MXFP4

### 1.3 昇腾平台 MXFP4 支持

| 平台 | MXFP4 原生计算 | MXFP4 格式 | 训练可用格式 | 加载 MXFP4 权重的处理 |
|------|:---:|:---:|------|------|
| **A2 (910B)** | ❌ | ❌ | BF16, FP16, FP32, INT8 | **必须反量化 MXFP4→BF16** (4×膨胀) |
| **A3 (910_93)** | ❌ | ❌ | BF16, FP16, FP32, INT8 | **必须反量化 MXFP4→BF16** (4×膨胀) |
| **A5 (950)** | ✅ | ✅ | BF16 + MX 全系列 | 可直接使用 MXFP4 权重 |

> **这是 A2/A3 和 A5 之间最关键的区别。** A2/A3 加载 Expert 权重时需将 MXFP4 反量化为 BF16: 1300 GB → 5200 GB (4×)。全 BF16 总权重 **~5460 GB**。

---

## 2. 并行策略

### 2.1 策略总览

```
┌─────────────────────────────────────────────────────┐
│              Kimi-K3 并行维度                         │
│                                                     │
│  FSDP2 ──► 非 Expert 参数/梯度/优化器 均匀分片到 N 卡    │
│  EP    ──► 896 个 Expert 分到 E 个 EP 组              │
│           └─ 每卡持有 896/E 个 Expert                 │
│           └─ Expert 权重/梯度/优化器 仅在 EP 组内分片    │
│  CP    ──► 长序列 attention 切分 (可选)                 │
└─────────────────────────────────────────────────────┘
```

### 2.2 单卡显存公式

设 N=总卡数, E=expert_parallel_size:

**全量微调:**
```
Expert 权重 (BF16):      5200/E  GB     ← A2/A3 反量化后
Expert 优化器 (fp32×3):  15600/E GB     ← AdamW: param + m + v
Expert 梯度 (BF16):      5200/E  GB
非 Expert 权重 (BF16):   260/N   GB
非 Expert 优化器:        780/N   GB
非 Expert 梯度:          260/N   GB
激活值 (重计算+卸载后):   ~15    GB

单卡合计: 33800/E + 1300/N + 15
```

**LoRA 微调 (base 冻结, 仅 LoRA adapter 有优化器):**
```
Expert 权重 (BF16):   5200/E  GB     ← 冻结, 仅前向读取
非 Expert 权重 (BF16): 260/N  GB     ← 冻结
激活值:                ~15    GB
LoRA 优化器:           ~3     GB     ← ~200M params × 12 bytes

单卡合计: 5200/E + 260/N + 18
```

**推理 (量化权重, 无优化器):**
```
量化权重 (W4A8):      594/64  GB     ← 保持量化存储
KV Cache:             ~30    GB     ← 取决于 max_model_len
KDA States:           ~2     GB

单卡合计: ~42 GB
```

### 2.3 为何 Expert 优化器是瓶颈

Expert 参数 (2.6T) 只被 EP 分片, FSDP2 不分片 Expert。这意味着 Expert 的优化器状态集中在 EP 组内:

```
EP=4:  15600/4  = 3900 GB/卡   ← 仅优化器就远超任何单卡 HBM
EP=64: 15600/64 = 244 GB/卡    ← 仍远超
EP=128:15600/128= 122 GB/卡    ← 接近 A5 128GB
EP=256:15600/256= 61 GB/卡     ← A5 可行
EP=896:15600/896= 17.4 GB/卡  ← 任何平台可行
```

---

## 3. A2 资源配置

### 3.1 全量微调: ❌ 不可行

```
单卡要求: 33800/E + 1300/N + 15 < 64

EP=64:  33800/64  = 528 GB/卡  ← 仅 optimizer 就爆了
EP=256: 33800/256 = 132 GB/卡  ← 仍远超 64GB
EP=896: 33800/896 = 37.7 GB/卡 ← optimizer 勉强, 但权重 + 激活 = 37.7 + 6 + 15 = 58.7 GB
```

即使 EP=896 (每卡 1 个 expert), 边界状态。且 896 张 A2 卡的全量微调在通信开销上不可接受。

**结论: A2/A3 全量微调 Kimi-K3 不可行。** 根本原因是 MXFP4 → BF16 反量化导致 Expert 权重膨胀 4×，进而优化器膨胀到单卡无法承受。

### 3.2 LoRA 微调: 128 卡起步

```
单卡要求: 5200/E + 260/N + 18 < 64 → 5200/E + 260/N < 46
```

| N | EP | Expert/卡 | Expert 权重 | 非Expert权重 | +激活+LoRA | 合计 | 状态 |
|--:|:--:|:--|:--|:--|:--|:--|:--:|
| 64 | 64 | 14 | 81.3 | 4.1 | 18 | 103 GB | ❌ |
| 64 | 128 | 7 | 40.6 | 4.1 | 18 | 62.7 GB | 边界 |
| 128 | 128 | 7 | 40.6 | 2.0 | 18 | 60.6 GB | ✅ |
| 192 | 128 | 7 | 40.6 | 1.4 | 18 | 60.0 GB | ✅ |
| 256 | 128 | 7 | 40.6 | 1.0 | 18 | 59.6 GB | ✅ |
| 256 | 256 | 3.5 | 20.3 | 1.0 | 18 | 39.3 GB | ✅ 充裕 |

> EP=64 (14 experts/卡) → Expert 权重 81 GB 直接爆。**必须 EP ≥ 128。**

**A2 LoRA 最少: 128 卡 (EP=128, N=128)。**
**推荐: 256 卡 (EP=256, N=256)** — 充裕, 且 EP=256 时每卡仅 3.5 个 expert, 通信开销更小。

```yaml
# A2 LoRA 推荐配置 (256 卡, EP=256)
parallel:
  fully_shard_parallel_size: auto
  expert_parallel_size: 256
  ep_plan:
    apply_modules:
      - language_model.model.layers.{*}.block_sparse_moe.experts
    dispatcher: alltoall
    use_npu_fused_ops: true

training:
  micro_batch_size: 1
  gradient_accumulation_steps: 32
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
  enable_chunk_loss: true
  enable_chunk_mbs: true

model:
  skip_flash_attn_recompute: true
  skip_kda_recompute: true
  use_grouped_expert_matmul: true
```

### 3.3 推理: 64 卡 (EP=64)

推理时权重保持量化存储 (W4A8), 无需反量化:

```
量化权重: ~594 GB / 64 = 9.3 GB/卡
KV Cache: ~30 GB  (max_model_len=32768)
KDA States: ~2 GB
─────────────────────────
单卡合计: ~42 GB < 64 GB ✅
```

| 平台 | EP | 最少卡数 | 单卡显存 |
|------|:--:|:--:|:--|
| A2 | 64 | 64 | ~42 GB |
| A3 | 64 | 64 | ~42 GB |
| A5 | 64 | 64 | ~35 GB (余量更大) |

---

## 4. A5 资源配置 (参考)

### 4.1 全量微调

A5 支持 MXFP4 原生计算, Expert 权重无需反量化:

```
单卡要求: 15600/E + 780/N + 15 < 128   (A5 128GB)

EP=64:  15600/64 = 244 GB/卡  ← 仅 optimizer 就超了
EP=128: 15600/128 = 122 GB/卡 ← 边界
EP=256: 15600/256 = 61 GB/卡  ← 可行
```

**A5 全量微调最少: ~256 卡 (EP=256),** 或使用 CPU offload + 8-bit Adam 等优化降低 EP 需求。具体需结合实际训练脚本验证。

### 4.2 LoRA 微调

A5 同样可用 MXFP4 存储 Expert, 资源需求远小于 A2:

```
单卡: 1300/E + 260/N + 18   (Expert 权重 MXFP4, 不需要反量化)

EP=32, N=32:  40.6 + 8.1 + 18 = 66.7 GB  ← A5 128GB 轻松
EP=16, N=16:  81.3 + 16.3 + 18 = 115.6 GB ← 同样可行
```

---

## 5. 资源需求汇总

| 场景 | A2 最少 | A2 推荐 | A3 最少 | A5 最少 | 关键约束 |
|------|:--:|:--:|:--:|:--:|------|
| **全量微调** | ❌ 不可行 | — | ❌ 不可行 | ~256 (EP=256) | A2/A3 MXFP4→BF16 反量化导致优化器爆炸 |
| **LoRA 微调** | **128** (EP=128) | **256** (EP=256) | 128 | 16 | base 冻结, 仅 LoRA 有优化器 |
| **推理** | **64** (EP=64) | 128 | 64 | 64 | 量化权重存储, 无优化器 |

### 5.1 A2 配置速查

| 场景 | 卡数 | EP | Dispatcher | 必须开启的优化 |
|------|:--:|:--:|------|------|
| LoRA 128 卡 | 128 | 128 | alltoall | recompute + activation_offload + chunk_mbs |
| LoRA 256 卡 | 256 | 256 | alltoall | recompute (chunk_mbs 可选) |
| 推理 | 64 | 64 | — | — |

---

## 6. 环境变量

```bash
# CANN 任务队列
export TASK_QUEUE_ENABLE=1

# HCCL 超时 (多机必须)
export HCCL_CONNECT_TIMEOUT=1800

# 虚拟内存 (大模型推荐)
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# Triton 缓存
export TRITON_ALWAYS_COMPILE=0
```
