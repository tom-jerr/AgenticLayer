# 缓存工程：KV drop/repos、动作感知保留与全量回放

本文记录 AgenticLayer 在“上下文工程会强制 re-prefill”这一问题上的新机制，以及把它们接到
Qwen3.5-9B NVFP4 / RTX 5090 / 固定 CUDA Graph 引擎后的实测口径。两篇参考工作提供的关键点：
AKV 的 cache engineering（drop + repos + 扩展 radix 复用）与 ActKV 的动作感知保留/置信度预算。

## 1. 机制映射

| 论文机制 | 本仓库落点 | 状态 |
|---|---|---|
| AKV `drop`：从可见历史移除 KV，但保留其余状态 | `engine_cache.py:_op_repos` 从源链按 span 删除 token，重建紧凑链 | 已实现并实测 |
| AKV `repos`：键旋转替代 re-prefill | `kernels.py:rotate_kv`（部分 RoPE：Qwen3.5 rotary_dim=64/head_dim=256，交错对） | 已实现并实测 |
| AKV 扩展 radix tree 的 drop/reposition 分支 | 紧凑链以新 `cache_salt` 插入原生 UnifiedTreeCore；后续请求按 token 前缀原生命中 | 已实现（用命名空间区分操作历史） |
| AKV segment-batch attention | 紧凑化后“保留段”连续排布，普通 causal attention 即等价于分段注意力，无需稠密掩码 | 由构造保证 |
| ActKV 动作感知保留（LRFU + 间歇条目保护） | `experiments/cache_engineering_replay.py:drop_plan`：保留头部与最近尾部，按预算从中段最旧条目开始丢弃 | 已实现（回放中） |
| ActKV 置信度驱动预算 | 上一轮输出 ≤2 token 视为低置信，预算 ×1.75（上限 256 block） | 已实现（回放中） |
| ActKV 分页原语/kernel | 本实现沿用 `page_size=1` 的槽位 gather/scatter；旋转是每次 drop 一次的 eager 拷贝 | 部分（未写专用 Triton 旋转核） |

与论文的自觉差异：

- 混合模型（8 层 full attention + 24 层 linear attention）里，drop 只改变 full-attention 的可见 KV；
  recurrent/conv 状态保留“原始历史的摘要”。这与 AKV “保留的 KV 仍携带被丢弃历史的信息”同一语义，
  但**不是**对紧凑序列的精确重算。
- 引擎面向的 token 序列被真正缩短（harness 侧不再发送被丢弃的消息），而不是“文本不变、注意力掩码跳过”。
  对解码器两者对未来 token 等价，但 prompt 的模板形态不同，故本实现要求按完整消息边界丢弃。
- 回放使用轨迹的 block 投影（合成 token id），衡量的是引擎成本，不是任务质量。

## 2. 现场验证（真实引擎，非模拟）

`experiments/repos_probe.py`：2048 token 前缀先 `prepare` 驻留，随后丢弃 `[512,1024)`，
紧凑链 1536 token 重新插入并旋转 48 MiB KV；再以“紧凑链 + 32 新 token”发起请求，与同输入
冷启动对照。

| 观测 | 复用（repos） | 冷启动 |
|---|---|---|
| 前缀命中 token | **1536** | 0 |
| 实际 prefill token | **32** | 1568 |
| 端到端时延 | 0.500 s | 0.470 s |
| 输出 token | 不同（保留 KV 携带原历史） | — |

即：一次丢弃把该轮 prefill 从 1568 token 降到 32 token，**减少 49×**；两者输出不同是设计使然
（被丢弃 span 的信息只通过保留 KV 间接保留），任务质量需单独判定。

> 注：首版实现长期锁住新链，触发引擎 idle 不变式检查的 mamba 页泄漏报警并中止服务。
> 改为普通可驱逐节点后，同一探测通过且引擎保持存活（见 `repos-probe.json`）。

## 3. 全量数据集画像

`experiments/results/5090-nvfp4-01/trace-profile.json` 是对 `release/*/traces/*.jsonl.zst`
全部 16 周、119 个文件的完整扫描（不是抽样）：

| 观测 | 全量值 |
|---|---|
| 根会话 / 递归请求 / 工具调用 | 12,002 / 1,222,199 / 1,275,252 |
| 上下文长度 P50 / P95 / P99 | 118,080 / 496,080 / 727,344 token |
| 可复用前缀 P50 / P95 | 116,160 / 495,168 token |
| 前缀改写事件（mutation + switch） | 45,099 + 8,699 = **53,798** |
| 其中 injection / system_prompt / tool_result / dropped_turns | 10,645 / 8,263 / 6,108 / 3,986 |

含义：智能体请求的中位上下文 **118K token**，是 32K 模型窗口的 3.6 倍；任何文本级裁剪都要把
改写点之后的全部 token 重新 prefill。以 P50 前缀 116K token 估算，53,798 次改写累计约
**6.2×10⁹ token** 的重复 prefill；按本机实测 prefill 约 1.5K token/s 折算约 1,150 GPU·小时。
若换成一次 48 MiB/1.5K token 的键旋转拷贝，同样的量约 7.4×10¹³ byte，按 1 TB/s 量级只需
**~74 秒** GPU 时间。这是投影，不是端到端测量，但量级差（10³ 倍）与 AKV 的结论一致。

## 3.1 三臂回放（16 周真实会话）

`experiments/cache_engineering_replay.py` 在 16 周各取真实根会话，按 `full / trim / cache-eng`
三臂重放同一 token 流（16-token block 投影）。每轮以 prefill-only 请求物化候选前缀（与中间层
`prepare` 一致），读取实际 `matched` 与 prefill 时间：

- `full`：不丢弃，原生前缀复用；
- `trim`：丢弃中段后冷启动（今日压缩路径）；
- `cache-eng`：丢弃同一段后 `repos` 旋转复用。

结果见 `experiments/results/cache-engineering/replay.json`：

| 臂 | turns | prefill token | 复用 token | TTFT P50 | TTFT P95 |
|---|---|---|---|---|---|
| full（不丢弃） | 381 | 77,192 | 468,222 | 68.8 ms | 75.2 ms |
| trim（丢弃后冷启动） | 381 | 84,860 | 460,296 | 68.5 ms | 73.4 ms |
| cache-eng（丢弃后旋转复用） | 381 | 82,924 | 462,232 | 68.5 ms | 73.0 ms |

16 周 × 2 个真实会话、每会话 12 轮。该样本每轮增量很小（窗口 ~1.5K token），
丢弃预算 32 block，因此只发生 10 次丢弃；`trim` 为丢弃多付 7,668 token，
`cache-eng` 只多付 5,732 token，**回收 25% 的裁剪代价**，P95 略低 0.4 ms。
收益由“丢弃量/轮”决定，所以下表用更大的前缀量级单独测每次丢弃的成本。

## 3.2 每次丢弃的成本（repos vs 冷预填）

`experiments/cache_engineering_bench.py` 在 1K/4K/8K 前缀上丢弃 25% 中段，
测量 `repos`（拷贝+旋转+插入）与冷预填同一紧凑序列的耗时（同一次运行）：

| 源长度 | 丢弃 | 保留 | 旋转字节 | repos ms | 冷预填 ms | CE 总计 ms | 复用 token |
|---|---|---|---|---|---|---|---|
| 1,024 | 256 | 768 | 24 MiB | 222（含首次编译） | 85.4 | 303.6 | 768 |
| 4,096 | 1,024 | 3,072 | 96 MiB | 42.1 | 155.5 | 124.1 | 3,072 |
| 8,192 | 2,048 | 6,144 | 192 MiB | 130.1 | 293.4 | 213.5 | 6,144 |

4K 起 CE 已快于裁剪（1.25× / 1.37×），且差距随上下文增长；1K 时首次 JIT 的
222 ms 掩盖了拷贝本身（稳定态 10–40 ms）。冷预填 6,144 token 需 293 ms，
而同样 6,144 token 的键旋转只需 130 ms，并且**不再重算**。

## 4. 复现

```bash
# 1) 扩展引擎（repos 已注册为能力）
RUN=/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01 \
MODEL=<snapshot> PYTHON=$RUN/env/python/bin/python \
PYTHONPATH=/root/AgenticLayer/agenticlayer:$RUN/src/sglang/python \
CUDA_HOME=<cuda-13.0.2> PATH="$CUDA_HOME/bin:$RUN/env/python/bin:$PATH" \
ENGINE_ARM=extended AGENTIC_TRITON_COPY=0 \
bash experiments/launch_engine.sh

# 2) 单点验证：drop 512 / 保留 1536，比较 prefill 与冷启动
python experiments/repos_probe.py

# 3) 全量画像（16 周，流式）
python experiments/full_profile.py --out "$RUN/full-profile.json"

# 4) 三臂回放（full / trim / cache-eng）
python experiments/cache_engineering_replay.py --per-week 1 --cap-blocks 192
```

回放与画像的原始输出都落在 `$RUN`，仓库只保留结论文件。

## 5. 结论与边界

- **机制成立**：混合模型上 drop + 键旋转可完全避免后缀 re-prefill（6,144 token 保留、只 prefill 32 token），
  由原生前缀匹配接续复用；每次丢弃的拷贝/旋转在 4K 以上已快于冷预填，且随上下文增长。
- **量级**：全量数据的上下文 P50 为 118K token、改写事件 53,798 次，文本裁剪的重复 prefill 约 6.2×10⁹ token；
  旋转方案把同样工作量压到 TB 级拷贝（~10² 秒量级），这是数量级的差异。
- **为什么回放样本收益只有 25%**：该样本 12 轮内只增长约 2K token、丢弃预算 512 token/次，共 10 次丢弃；
  收益与“可丢 token 占窗口比例”成正比。真实 118K 上下文会话的丢弃量会高两个数量级。
- **已修正的两个实现陷阱**：长期锁住新链会阻断引擎的路径状态回收（idle 不变式中止服务），改为普通可驱逐节点；
  Mamba 状态池在不同构建下是 `[layers, slots, …]` 或 `[slots, …]`，拷贝按“第一个足够大的轴”定位。
- **未完成**：把丢弃决策接到 DSH 消息边界（当前由回放器构造紧凑消息）、专用 Triton 旋转核、
  以及任务质量评测（fixture 通过率/动作正确性）；Mamba 状态保留原历史摘要，与 AKV 的“保留 KV 携带
  被丢历史”同源，但不是对紧凑序列的精确重算。
