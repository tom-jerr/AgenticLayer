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
| ActKV 动作感知保留（LRFU + 间歇条目保护） | `policy.py:select_drop_messages`：按消息边界丢弃——任务锚点与最新轮次保留，中段最旧消息按预算丢弃，从不切开一条消息 | 已实现（回放 + 中间层共用） |
| ActKV 置信度驱动预算 | `policy.py:retention_budget`：上一轮低置信时预算 ×1.75（上限 256 block）；回放轨迹无模型反馈，用锚点 + 最新轮次的全保留语义 | 已实现（中间层） |
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

## 3.1 三臂回放（16 周真实会话，真实尺度窗口）

`experiments/cache_engineering_replay.py` 在 16 周各取真实根会话，按 `full / trim / cache-eng`
三臂重放同一 token 流（8 ids/block 投影）。窗口即真实约束：初始视图 = 会话最后 2,750 block
（22K token，轨迹 16-token block 压缩投影），引擎上限内窗口滑落保护 32K token，每会话最多 30 轮。
**丢弃对齐 DSH 消息边界**：每个轨迹请求是一条消息，`select_drop_messages` 只整消息丢弃
（任务锚点 + 最新轮次保留、中段最旧消息按预算丢），保留视图即 DSH 消息级丢弃会发出的紧凑序列。
每轮以 prefill-only 请求物化候选前缀（与中间层 `prepare` 一致），收据的 `full_kv_hit_length`
是实际命中，其余是真实 prefill：

- `full`：不丢弃、原生前缀复用；窗口溢出时滑落（guard 截头 → 整窗冷预填）；
- `trim`：丢弃中段后冷启动（今日压缩路径，每轮重预填保留视图）；
- `cache-eng`：丢弃同一段后 `repos` 旋转复用（旋转 ~530 MiB KV，重预填仅增量）。

真实尺度回放（16 周 × 每周 8 会话 × 最多 30 轮，128 会话，三臂各 3,145 轮，
原始数据 `experiments/results/cache-engineering/replay.json`）：

| 臂 | turns | prefill token | 复用 token | prefill 墙钟 | TTFT P50 | TTFT P95 |
|---|---|---|---|---|---|---|
| full（不丢弃，窗口滑落） | 3,145 | 26,063,160 | 44,362,000 | 1,580.1 s | 74.0 ms | 1,715.2 ms |
| trim（丢弃后冷启动） | 3,145 | 33,166,136 | 22,793,992 | 1,648.7 s | 154.7 ms | 1,383.2 ms |
| cache-eng（丢弃后旋转复用） | 3,145 | **9,592,952** | 46,367,176 | **687.9 s** | **72.0 ms** | **955.3 ms** |

- **re-prefill：cache-eng 比 full 少 2.7×、比 trim 少 3.5×**；1,120 次消息级丢弃全部经 `repos`
  旋转复用，无一回退（丢弃轮实际命中保留视图、只预填增量）。
- **TTFT P95：955.3 ms vs full 1,715.2 ms（1.8×）vs trim 1,383.2 ms（1.4×）**；prefill 墙钟
  687.9 s vs 1,580.1 s（2.3×）vs 1,648.7 s（2.4×）。full 的 P95 由窗口滑落冷预填主导
  （811/3,145 轮整窗 32K 重预填），这正是 AKV 观察到的“改前缀即全量 re-prefill”；
  cache-eng 靠键旋转把同一滑落变成拷贝。
- trim 每轮丢弃都要把保留视图重新 prefill，墙钟是 cache-eng 的 2.4 倍；cache-eng 的旋转随上下文
  增长的字节拷贝远便宜于重算（§3.2），且保留视图不再重算。

> 修正记录：真实尺度下 `repos` 的目标槽位分配在竞技场填满时失败（旧链按新 salt 累积），
> `_op_repos` 已改为分配失败先按 LRU 驱逐陈旧命名空间再重试（对齐 `_op_bind`）；首次全量冒烟
> 中 11/12 会话回退冷启动，修复后 1,120/1,120 成功。

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

# 4) 三臂回放（full / trim / cache-eng，真实尺度窗口 + 消息边界丢弃）
python experiments/cache_engineering_replay.py --per-week 8 --cap-blocks 2750 --turns 30
```

回放与画像的原始输出都落在 `$RUN`，仓库只保留结论文件。

## 5. 结论与边界

- **机制成立**：混合模型上 drop + 键旋转可完全避免后缀 re-prefill（6,144 token 保留、只 prefill 32 token），
  由原生前缀匹配接续复用；每次丢弃的拷贝/旋转在 4K 以上已快于冷预填，且随上下文增长。
- **真实尺度回放**：16 周 × 8 真实会话、22K 初始窗口、30 轮消息级丢弃，cache-eng 的 re-prefill 比
  full 少 2.7×、比 trim 少 3.5×，TTFT P95 955.3 ms（full 1,715.2 ms / trim 1,383.2 ms）；
  1,120 次丢弃全部旋转复用、零回退。full 的 P95 由窗口滑落整窗冷预填主导——正是 AKV 的核心动机。
- **量级**：全量数据的上下文 P50 为 118K token、改写事件 53,798 次，文本裁剪的重复 prefill 约 6.2×10⁹ token；
  旋转方案把同样工作量压到 TB 级拷贝（~10² 秒量级），这是数量级的差异。
- **已修正的实现陷阱**：长期锁住新链会阻断引擎的路径状态回收（idle 不变式中止服务），改为普通可驱逐节点；
  Mamba 状态池在不同构建下是 `[layers, slots, …]` 或 `[slots, …]`，拷贝按“第一个足够大的轴”定位；
  真实尺度下竞技场被逐次新 salt 的旧链填满，`repos` 分配失败需先 LRU 驱逐再重试（对齐 `bind`）。
- **未完成**：任务质量评测（fixture 通过率/动作正确性）仍是边界；Mamba 状态保留原历史摘要，与 AKV 的
  “保留 KV 携带被丢历史”同源，但不是对紧凑序列的精确重算。

### 5.1 真实 DSH 闭环与 Triton 旋转核（本阶段的补充）

- **DSH 消息级丢弃闭环已接通**：`dsh-agentic` 在 `llm/stream` 按消息数阈值（`dropAfterMessages`，默认 32）
  做保留决策（任务锚点 + 最新轮，从不切开工具往返），发 `repos` 事件经 `/agentic/v1/events` 下发；
  中间层以消息级 token range（递增前缀 tokenize + 引擎校验兜底）计算丢弃 spans、必要时先
  `prepare` 物化源链（no_buffer 下 chat 不入树）、再执行引擎 `repos`；保留视图经 `next(derived options)`
  传给 provider。`dsh_bench.mjs --suite drop`（双轮 fixture：17 → 42 + marker）全门禁通过，
  真实视图 drop 后 391/1102 token 旋转复用。边界：丢弃边界必须落在模板干净处（user 前一条消息非
  tool——Qwen3 模板把 tool 块与其后 user 合并渲染，落在 tool 后则保留尾重渲染不同、自动冷回退）。
- **Triton 键旋转核已交付**：`kernels.py` 的 `_rotate_kv_kernel`（每行拷贝 + 原地 RoPE 对旋转，
  `AGENTIC_TRITON_ROTATE=1` 启用、非 3D 布局自动回退 eager），与 eager 逐位一致（bf16 下 max|ΔK|=0），
  24K×8 层旋转 2.5×、8K 3.7×。但**旋转拷贝本身只占 repos 墙钟的 1–3 ms**：实测 22K 链 repos 共
  29–57 ms（旋转 687 MiB），主开销在 TreeCore 插入与 RPC，Triton 化对端到端 P95 的增益有限；
  更狠的 P95 需优化插入/匹配路径。
