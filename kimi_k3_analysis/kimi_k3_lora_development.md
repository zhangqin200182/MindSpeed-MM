# Kimi-K3 LoRA 微调开发方案

> 目标: 在 MindSpeed-MM FSDP2 训练框架中支持 Kimi-K3 模型的 LoRA 微调
> 分析日期: 2026-07-28

---

## 1. 现状分析

### 1.1 已有的基础设施

MindSpeed-MM FSDP2 trainer **已经内置了完整的 LoRA 框架**, 无需从头构建：

```
mindspeed_mm/fsdp/
├── train/trainer.py                 # get_model() → enable_lora() → FSDP2
├── utils/lora_utils.py              # add_lora_to_model() via PEFT
├── utils/lora_weight_manager.py     # 权重保存/加载/合并
├── params/lora_args.py             # LoRA 配置 dataclass
└── params/training_args.py         # training.lora.* 参数解析
```

核心流程 (`trainer.py:188-216`):

```python
model = self.get_foundation_model()            # 加载 Kimi-K3
if args.training.lora.enable:
    model = self.enable_lora(model)             # PEFT inject_adapter_in_model
model = self.model_parallel_applier(model)      # FSDP2 sharding
```

`enable_lora()` 做了三件事 (`trainer.py:223-302`):

1. `match_target_modules(model, patterns)` — 通配符匹配 `nn.Linear` 子模块
2. `freeze_parameters(model)` — 冻结基础模型
3. `add_lora_to_model()` — 调用 PEFT 注入 LoRA adapter

### 1.2 Kimi-K3 当前不支持的原因

PEFT 的 `inject_adapter_in_model` 对 Kimi-K3 有 **3 个技术阻塞点**:

| # | 阻塞点 | 根因 | 影响 |
|---|--------|------|------|
| ① | **MoE expert 权重是 3D tensor** | `PatchKimiMoeExperts` 的 `gate_up_proj: [E, H, 2I]`, `down_proj: [E, I, H]` 不是 `nn.Linear` | PEFT 无法识别, 不会注入 LoRA (无需处理, 本来就是冻结的) |
| ② | **ShortConvolution 是 depthwise Conv1d** | `q_conv1d/k_conv1d/v_conv1d` 继承 `nn.Conv1d(groups=hidden)`, 不是 `nn.Linear` | PEFT 默认不支持 Conv1d 的 LoRA |
| ③ | **KDA attention 有低秩 gating 投影** | `f_a_proj/f_b_proj/g_a_proj/g_b_proj` 是实现 gate 计算的中间层, 不是标准的 Q/K/V/O 投影 | `target_modules` 配置需要区分哪些层值得加 LoRA |

其中 ① 不是问题 (Expert 本来就该冻结); ② 需要少量适配; ③ 是配置层面的问题。

---

## 2. 开发任务

### 任务 1: Kimi-K3 target_modules 映射 (P0, ~1天)

#### 1.1 模型层级结构

Kimi-K3 的完整模块路径前缀 (与 FSDP2 plan 对齐):

```
KimiK3ForConditionalGeneration
├── vision_tower                    # ViT (冻结, 不加 LoRA)
├── mm_projector                    # Vision → Text 投影 (冻结)
└── language_model                  # KimiLinearForCausalLM
    ├── model.embed_tokens          # Embedding (冻结)
    ├── model.layers.{N}            # KimiDecoderLayer
    │   ├── self_attn               # MLA 或 KDA (混合)
    │   ├── mlp / block_sparse_moe  # FFN / MoE
    │   ├── input_layernorm
    │   ├── post_attention_layernorm
    │   ├── pre_ffn_layernorm (MoE层)
    │   ├── self_attention_res_proj
    │   └── mlp_res_proj
    └── lm_head                     # (冻结)
```

#### 1.2 推荐的 target_modules

Kimi-K3 有两种 attention 层 (MLA 和 KDA), 混合排列。建议对 **attention 的 Q/K/V/O 投影** 加 LoRA:

```yaml
training:
  lora:
    enable: true
    rank: 8
    alpha: 16
    target_modules:
      # MLA attention 层 (标准 Q/KV/O 投影)
      - "language_model.model.layers.{*}.self_attn.q_a_proj"
      - "language_model.model.layers.{*}.self_attn.q_b_proj"
      - "language_model.model.layers.{*}.self_attn.kv_a_proj_with_mqa"
      - "language_model.model.layers.{*}.self_attn.kv_b_proj"
      - "language_model.model.layers.{*}.self_attn.o_proj"
      # KDA attention 层 (Q/K/V 投影)
      - "language_model.model.layers.{*}.self_attn.q_proj"
      - "language_model.model.layers.{*}.self_attn.k_proj"
      - "language_model.model.layers.{*}.self_attn.v_proj"
      - "language_model.model.layers.{*}.self_attn.o_proj"
    dropout: 0.0
```

> **注意**: 不推荐对 MoE gate/router (`block_sparse_moe.gate`)、KDA gating 层 (`f_a_proj`, `b_proj`, `g_proj`)、ShortConvolution 加 LoRA。这些层的参数已经很小, 加 LoRA 收益有限且可能破坏 gate 计算的数值特性。

#### 1.3 通配符匹配的自动化

`match_target_modules` (`lora_utils.py:48`) 已经支持 `{*}` 通配符, 会用 `fnmatch.fnmatch` 匹配所有 decoder layer。上述 `target_modules` 会自动命中有 MLA 或 KDA 的层 — PEFT 对不存在的模块 (如 KDA 层的 `kv_a_proj_with_mqa`) 会静默跳过。

> **实际测试时需注意**: PEFT `inject_adapter_in_model` 在找不到 target_module 时会 **抛出 KeyError**。需要在 `add_lora_to_model` 中先过滤掉不存在的 module name pattern, 只保留 `match_target_modules` 返回的命中列表。当前代码已经这样做了 (`trainer.py:257`)。

### 任务 2: ShortConvolution LoRA 适配 (P1, ~2天)

#### 2.1 问题

`ShortConvolution` 继承 `nn.Conv1d`, 但 `groups=hidden_size` (depthwise), 且 kernel_size=4。PEFT 的默认 LoRA 不支持 `nn.Conv1d` — 需要手动注册。

#### 2.2 方案选择

| 方案 | 工作量 | 风险 | 推荐 |
|------|:--:|------|:--:|
| A. 跳过 ShortConv, 只对 Linear 加 LoRA | 0 | 低 — ShortConv 参数极少 (3 × 4 × 128 = 1536 个) | ✅ |
| B. 注册 PEFT Conv1d LoRA 支持 | 中 | 中 — depthwise conv 的 LoRA 语义需要定义清楚 | ❌ |
| C. 在 ShortConv 内部插入 Linear adapter | 大 | 高 — 改变 forward 语义 | ❌ |

**推荐方案 A**。ShortConvolution 的总参数量:

```
q_conv1d: depthwise, kernel=4, in_channels=H*K, groups=H*K
  → weight: [H*K, 1, 4] = [16*128, 1, 4] = 8192 floats
k_conv1d: 同 q_conv1d, 8192 floats
v_conv1d: 同 (但 V 维度可能不同), ~8192 floats

总计: ~25K 参数 / 层, 仅占单层总参数的 <0.01%
```

不需要加 LoRA。在 `lora_utils.py` 中加一行排除逻辑即可:

```python
# lora_utils.py → match_target_modules()
# 新增: 排除 ShortConvolution 模块
EXCLUDED_MODULE_TYPES = (nn.Conv1d,)  # 或直接靠 target_modules 不命中
```

#### 2.3 实现

当前 `target_modules` 使用名称后缀匹配 (`"q_proj"` 匹配所有以 `.q_proj` 结尾的 `nn.Linear`)。ShortConvolution 的属性名是 `q_conv1d` / `k_conv1d` / `v_conv1d`, 不会被 `"q_proj"` 等命中。**无需额外开发**。

### 任务 3: PEFT 模型兼容性验证 (P0, ~1天)

#### 3.1 PEFT `inject_adapter_in_model` 兼容性

验证项:

```python
# 测试脚本: tests/test_kimi_k3_lora.py

from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoConfig
from mindspeed_mm.fsdp.models.kimi_k3 import KimiK3ForConditionalGeneration

# 1. 加载模型
config = AutoConfig.from_pretrained("mindspeed_mm/fsdp/models/kimi_k3", trust_remote_code=True)
model = KimiK3ForConditionalGeneration(config)

# 2. 注入 LoRA
lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj"],
    bias="none",
)
model = inject_adapter_in_model(lora_config, model)

# 3. 验证
for name, module in model.named_modules():
    if hasattr(module, 'lora_A'):
        print(f"LoRA injected: {name}")

# 4. 前向测试
out = model(input_ids=..., pixel_values=...)
loss = out.loss
loss.backward()  # 验证梯度流
```

#### 3.2 潜在问题

| 问题 | 可能性 | 处理 |
|------|:--:|------|
| `KimiMLAAttention` 有 `q_proj` 但 MLA 模式下 `q_lora_rank` 存在时用 `q_a_proj/q_b_proj` 替代 | 高 | `target_modules` 同时包含 `q_proj` 和 `q_a_proj/q_b_proj`, PEFT 对不存在的跳过 |
| `KimiSparseMoeBlock.experts` 下的 `PatchKimiMoeExperts` 不是 `nn.Linear` | 低 | PEFT 自动跳过 |
| `KimiDeltaAttention` 的 `f_a_proj/f_b_proj` 是 gate 投影, 加了 LoRA 可能破坏 gate 语义 | 低 | 不加到 `target_modules` 中 |
| FSDP2 后 `DTensor` 与 `PeftModel` 的交互 | 中 | 已由 `LoraWeightManager._gather_dtensor` 处理 |

### 任务 4: FSDP2 Sharding Plan 对齐 (P1, ~1天)

#### 4.1 问题

PEFT `inject_adapter_in_model` 会在目标 Linear 外层包裹 `lora.Linear`, 模块层次变为:

```
原始: self_attn.q_proj (nn.Linear)
注入后: self_attn.q_proj (lora.Linear)
            ├── base_layer (nn.Linear)     ← 原始权重, frozen
            ├── lora_A (nn.Linear)         ← LoRA A, trainable
            ├── lora_B (nn.Linear)         ← LoRA B, trainable
            └── lora_dropout (nn.Dropout)
```

FSDP2 的 `fsdp_plan.apply_modules` 通过模块路径匹配。LoRA 注入后的子模块路径包含了 `lora_A` / `lora_B` / `base_layer` 等新名, 需要确认 FSDP2 的 `fully_shard` 是否能正确识别。

#### 4.2 处理

FSDP2 `fully_shard` 的策略是对 `apply_modules` 列表中的每个模块做 `fully_shard`。LoRA 注入后:

- `self_attn.q_proj` 从 `nn.Linear` 变为 `lora.Linear` — FSDP2 仍然能看到这个模块, 按 policy 做 shard
- `lora_A.weight` 和 `lora_B.weight` 是 `nn.Parameter`, 在 FSDP2 下会被当作普通参数处理
- `base_layer.weight` 是 frozen parameter (`requires_grad=False`)

需要在 `fsdp_plan` 中新增 LoRA 参数的 sharding 策略。通常 LoRA 参数很小 (rank × (d_in+d_out)), **不强耦合到 FSDP2 sharding** 也可以 (DDP 复制)。如果要省显存, 可加入:

```yaml
parallel:
  fsdp_plan:
    apply_modules:
      # ... 原有的 ...
      # LoRA adapter 参数也加入 FSDP2 shard
      - language_model.model.layers.{*}.self_attn.*.lora_A
      - language_model.model.layers.{*}.self_attn.*.lora_B
```

但更简单的方式是: **不在 `apply_modules` 中加 LoRA 子模块**, 让 FSDP2 跳过它们 (自动复制到所有 rank)。LoRA 参数量小, 复制开销可忽略。

#### 4.3 验证

```python
# 在 enable_lora() 之后, FSDP2 sharding 之前, 打印模块树:
for name, module in model.named_modules():
    if 'lora' in name:
        print(name, type(module), sum(p.numel() for p in module.parameters()))
```

确认:
1. LoRA 参数的 `requires_grad=True`, base 的 `requires_grad=False`
2. FSDP2 后 LoRA 参数在 optimizer 中可见
3. 分布式 checkpoint save/load 正确 (已由 `LoraWeightManager` 处理)

### 任务 5: 配置文件与启动脚本 (P1, ~0.5天)

#### 5.1 YAML 配置

```yaml
# examples/kimi_k3/kimik3_lora_config.yaml

parallel:
  fully_shard_parallel_size: auto
  fsdp_plan:
    apply_modules:
      - vision_tower
      - vision_tower.encoder.blocks.{*}
      - mm_projector
      - language_model.model
      - language_model.model.embed_tokens
      - language_model.model.layers.{*}
      - language_model.model.layers.{*}.block_sparse_moe.experts
      - language_model.lm_head
    hook_modules:
      - language_model.model.layers.{*}
    num_to_forward_prefetch: 1
    num_to_backward_prefetch: 1
    param_dtype: bf16
    reduce_dtype: fp32
  ulysses_parallel_size: 1
  expert_parallel_size: 4          # A2 建议 8
  ep_plan:
    apply_modules:
      - language_model.model.layers.{*}.block_sparse_moe.experts
    dispatcher: alltoall
    use_npu_fused_ops: true

data:
  dataset_param:
    dataset_type: huggingface
    attr:
      images: images
      messages: messages
    preprocess_parameters:
      model_name_or_path: &HF_MODEL_LOAD_PATH mindspeed_mm/fsdp/models/kimi_k3
      trust_remote_code: true
      image_max_pixels: 262144
      image_min_pixels: 1024
    basic_parameters:
      cutoff_len: 1024
      template: kimi_k3
      enable_thinking: false

training:
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  lr: 1.0e-4
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
    dropout: 0.0
    init_lora_weights: true
    # lora_target_modules_support 留空, 不校验白名单
  model_id: kimi_k3
```

#### 5.2 启动脚本

```bash
# examples/kimi_k3/finetune_kimik3_lora.sh

BASEPATH=$(cd "$(dirname "$0")/../.."; pwd)

# A2 平台建议 EP=8 (若卡数有限可用 EP=4 + 更小 cutoff_len)
export EXPERT_PARALLEL_SIZE=4

torchrun --nproc_per_node=8 \
  --nnodes=$NNODES \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR \
  --master_port=29500 \
  $BASEPATH/mindspeed_mm/fsdp/train/train_entry.py \
    examples/kimi_k3/kimik3_lora_config.yaml \
    2>&1 | tee "logs/train_kimi_k3_lora_${logfile}.log"
```

### 任务 6: 测试与验证 (P0, ~2天)

#### 6.1 单元测试

| 测试项 | 内容 | 预期 |
|--------|------|------|
| `test_target_module_match` | `match_target_modules` 命中所有 MLA/KDA 投影层 | 命中数 = N_layers × (MLA层数×5 + KDA层数×4) |
| `test_lora_injection` | `inject_adapter_in_model` 成功, 无报错 | 所有 target Linear 被包装 |
| `test_freeze_verify` | base params `requires_grad=False`, lora params `requires_grad=True` | 冻结正确 |
| `test_forward_backward` | 随机输入前向+反向, 梯度非零 | loss 正常, grad norm > 0 |
| `test_fsdp2_compat` | FSDP2 sharding 后参数分布正确 | DTensor 分片正确 |

#### 6.2 端到端测试

| 测试项 | 内容 | 验收标准 |
|--------|------|------|
| Single node, 8-card | 小数据集, 10 steps | loss 下降, 无 OOM |
| Multi node, 16-card | 正式数据, 100 steps | loss 收敛正常 |
| Checkpoint save/load | 保存 LoRA → 加载续训 | loss 连续 |
| LoRA merge | `merge_and_unload` → HF 推理 | 输出合理 |

#### 6.3 精度验证

对比 LoRA fine-tune vs. 全量 fine-tune 在同一小数据集上的 loss 曲线:

```python
# 预期: LoRA (rank=8) 的 loss 下降速度接近全量微调
# rank 越大, 越接近全量微调效果
```

---

## 3. 开发清单与工时估算

| # | 任务 | 难度 | 工时 | 依赖 |
|---|------|:--:|:--:|---|
| 1 | `target_modules` 映射配置 | 低 | 1d | — |
| 2 | ShortConv LoRA 跳过确认 | 低 | 0.5d | 任务1 |
| 3 | PEFT 兼容性验证 (前向+反向) | 中 | 1d | 任务1 |
| 4 | FSDP2 sharding plan 对齐 | 中 | 1d | 任务3 |
| 5 | 配置文件 + 启动脚本 | 低 | 0.5d | 任务1 |
| 6 | 测试与验证 (单元+端到端) | 中 | 2d | 任务3,4 |
| **合计** | | | **~6 工作日** | |

---

## 4. 风险与注意事项

### 4.1 KDA 层的 Gating 投影

`KimiDeltaAttention` 中的 `f_a_proj/f_b_proj` (gate 输入), `g_a_proj/g_b_proj` 或 `g_proj` (output gate), `b_proj` (beta) 都是 `nn.Linear`。如果 `target_modules` 配置不当, 可能会命中这些层。

**不建议**对这些 gate 层加 LoRA, 原因:

1. `f_a_proj: 7168→128` (head_dim) — 只有 ~1M 参数, 不需要 LoRA
2. `g_a_proj/g_b_proj` — output gate 的数值范围 (sigmoid 输出) 限制了 LoRA 的有效性
3. `b_proj: 7168→16` (num_heads) — 参数极少

构造 `target_modules` 时精确匹配 `q_proj/k_proj/v_proj/o_proj` 可避免命中。

### 4.2 MoE Expert 的 `routed_expert_down_proj/up_proj`

`KimiSparseMoeBlock` 中的 `routed_expert_down_proj` 和 `routed_expert_up_proj` 是 `nn.Linear` (注意不是 PatchKimiMoeExperts 的 3D tensor), 它们是全量 replicate 的线性层 (压缩/解压)。如果 `target_modules` 中有 `"down_proj"` 或 `"up_proj"`, 会命中。

这些层参数不大 (7168×3584 ≈ 25M), 可加 LoRA 也可不加。取决于需求。

### 4.3 平台卡数要求 (基于真实模型规模)

Kimi-K3 真实参数规模 (来自 HuggingFace config.json):
- 总参数: 2.78T, 激活参数: ~104B
- 93 层 (69 KDA + 24 MLA), hidden_size=7168
- BF16 权重: 1560 GB

LoRA 微调下的单卡显存估算 (A2 64GB):

| 项目 | 16 卡 (EP=4) | 32 卡 (EP=8) |
|------|:--:|:--:|
| Expert 权重 (EP 分片) | ~12.5 GB | ~6.25 GB |
| 非 Expert 权重 (FSDP2 分片) | ~6.1 GB | ~3.0 GB |
| 激活值 (重计算+CPU卸载后) | ~20 GB | ~12 GB |
| LoRA 优化器 | ~0.2 GB | ~0.2 GB |
| **合计** | **~38.8 GB** ✅ | **~21.5 GB** ✅ |

**LoRA 微调最少需要**:
- A2: **16 卡** (EP=4), 推荐 32 卡
- A3: **16 卡** (EP=4), 推荐 16 卡
- A5: **8 卡** (EP=4), 推荐 8 卡

> ⚠️ 8 卡 A2 放不下 — Expert 权重 ~25GB + 激活 ~20GB + 非 Expert ~12GB ≈ 57GB, 边界风险。建议至少 16 卡。

### 4.4 FP32 LoRA 参数精度

当前 `add_lora_to_model()` 将 LoRA 参数 cast 到 `float32` (`lora_utils.py:247-251`)。这对精度有利但对 A2 显存不利 (LoRA 参数 ×4 bytes)。

如果显存紧张, 可考虑:

```python
# 将 LoRA 参数保持 bf16
for param in model.parameters():
    if param.requires_grad and "lora" in param_name:
        param.data = param.data.to(torch.bfloat16)
```

但在混合精度训练 (AMP) 中, 梯度累积可能在 fp32, 需要测试数值稳定性。

## 5. 参考资料

- [FSDP2 LoRA 文档](../../docs/zh/features/lora_finetune_fsdp2.md)
- [Kimi-K3 NPU 算子分析](kimi_k3_npu_operators_analysis.md)
- [Kimi-K3 训推精度对齐分析](kimi_k3_npu_operators_alignment.md)
- [PEFT LoRA API](https://huggingface.co/docs/peft/developer_guides/lora)
