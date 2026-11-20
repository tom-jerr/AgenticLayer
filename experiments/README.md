# 实验手册：先原生基线，再回放与策略对照

部署按 [执行手册](../agenticlayer/RUNBOOK.md)，机制和锁定源码见 [设计文档](../agenticlayer/DESIGN.md)。[原始方案](DSH_AGENTICLAYER_SGLANG_PLAN.md) 保留编号与完整推导。本目录存实验合同，不存数据或大型结果；运行产物统一落在 `/root/autodl-tmp/agentic-layer-runs/<run_id>`。

**实机状态：** [RTX 5090 实测报告](GPU_5090_RESULTS.md)。已验证原生 32K、冷数值参考、原生 HiCache、Full+Mamba 恢复、DSH 工具闭环和固定窗口提前预填充。下文 E1–E6 是完整验收规约，未列入实机报告的条件不算通过。

真实 FreeInference 留出工作集的四臂 GPU 回放、论文指标映射、Python 绘图命令和结果见 [FreeInference 论文图表](FREEINFERENCE_PAPER.md)。

同一留出工作集上的 decode Graph / recurrent checkpoint 交叉对照，以及工具、生命周期、放置和提前准备的独立结果，见 [FreeInference GPU 消融](FREEINFERENCE_ABLATION.md)。

[固定 Graph 论文机制实验](FIXED_GRAPH_CAMPAIGN.md) 单独验证 E1–E6；所有基线、单开与组合臂都固定 `decode=full`、`prefill=disabled`，并核对实际捕获桶。

[论文式 HTML 展示](agenticlayer-paper.html)：从动机和三层技术路径讲到主实验、敏感性、留一消融和原始事件案例；图表内嵌，直接打开或打印即可阅读。

## 1. 执行顺序与门禁

| 阶段 | 最小通过证据 | 实际入口 |
|---|---|---|
| 固定环境 | 权重、模板、配置、源码、依赖及硬件记录齐全；校验 SSD 目录和自有进程。模型/依赖/源码变化另开 run。 | [运行手册 §1](../agenticlayer/RUNBOOK.md#1-复制源码隔离依赖) |
| 原生 baseline | 三个原生臂按下文启动；参数和启动日志证明 TP=1、page=1、Triton、BF16 KV、NVFP4 GEMM。真实 32767 prompt + 1 token 无 OOM。相同 1024 前缀冷/热后追加一个 token，热命中增加且数值一致。在 HiCache 臂施加容量压力，证明 GPU 驱逐后有正 host 命中。最后通过一次 DSH SDK 工具往返。 | [原生 HTTP](../sglang/python/sglang/srt/entrypoints/http_server.py)、[DSH runner](../agenticlayer/dsh-agentic/src/runner.ts) |
| FreeInference 画像和 smoke | 先全量核对计数、排除原因和抽样 hash；在固定 smoke 上核对周隔离、匿名链 LCP 和真实因果顺序，保存模型和屏障的时间线。 | [受控 batch ingress](../agenticlayer/agentic_layer/server.py)、[原生 tokenizer/batch](../sglang/python/sglang/srt/managers/tokenizer_manager.py) |
| 扩展正确性 | scheduler 握手成功；DSH、混合 KV、拷贝均通过下述门禁，失败的臂不得进入性能表。 | [DSH hooks](../agenticlayer/dsh-agentic/src/index.ts)、[引擎插件](../agenticlayer/agentic_layer/sglang_plugin.py)、[缓存控制](../agenticlayer/agentic_layer/engine_cache.py)、[Triton](../agenticlayer/agentic_layer/kernels.py) |
| 校准后测量 | 在第 1–4 周及固定 microbench 冻结成本与等待预测；再用留出周和 E1–E6。每预算档都有匹配 baseline；缺失或 fingerprint 改变时不得产配对结果。 | [Calibration / Policy](../agenticlayer/agentic_layer/policy.py) |

从项目根、用 [隔离 Python](../agenticlayer/RUNBOOK.md) 运行。所有 `--name` 唯一；每次只运行一个带 GPU 的命令。

| 已实现入口 | 用途 |
|---|---|
| `"$PYTHON" -m experiments.gpu_run --run-dir "$RUN" --arm native-radix` | 原生 smoke；另有 `reference-cold`、`native-hicache`。 |
| `"$PYTHON" -m experiments.hybrid_gpu --run-dir "$RUN" --name hybrid --triton` | 真实混合恢复；不加 `--triton` 得原生拷贝组。 |
| `"$PYTHON" -m experiments.copy_calibrate --run-dir "$RUN" --name copy` | 三档、每臂 5 次预热 + 20 次真实 D2H/H2D；分别核对 Full、Mamba 和原生 ACK。 |
| `"$PYTHON" -m experiments.cancel_gpu --run-dir "$RUN" --name cancel` | 物理迁移 fence 与 batch 单成员取消。 |
| `"$PYTHON" -m experiments.prepare_gpu --run-dir "$RUN" --name prepare --policy strong --pairs 20` | 固定输入早准备 on/off；下一策略传入同一输出的 `--calibration`。 |
| `"$PYTHON" -m experiments.trace_prepare profile --dataset /root/autodl-tmp --run-dir "$RUN"` | 画像；用同参数改 `select --split calibration` 或 `select --split heldout` 固定投影。默认每层至多 16 条。 |
| `"$PYTHON" -m experiments.trace_matrix --run-dir "$RUN" --name replay --sample "$RUN/trace-heldout-smoke.jsonl" --arms native-radix joint --smoke-uncalibrated --turns-per-stratum 1 --smoke-max-decode 512` | 真实 GPU 接线 smoke。收益对照移除 smoke 选项，按合约传冻结 `--calibration`、固定全量样本并配对 5 次。 |

## 2. 主对照与消融

| 臂 | 合同 |
|---|---|
| `native-radix` | 原生按需、radix/LRU、无 HiCache、无插件；同一 DSH fixture。 |
| `native-hicache` | 在上述配置上仅加实例私有 DRAM HiCache，作为原生按需恢复参照。 |
| `reference-cold` | 禁 radix 和 HiCache；仅用于逐 token 数值验证，不纳入性能主对照。 |
| `strong` | 主对照。必要前台优先，再独立按准入 → 恢复/放置 → 准备仲裁。 |
| `joint` | 同一执行器、样本、预测与合法性/容量/老化/准备保护。仅替换可选动作之间的仲裁；主要效应为 `joint − strong`。 |

消融：`simple`（无老化/slack 保护，不作主对照）、`parallel-only`、`lifecycle-only`、`offload-only`、`lookahead-only`、`ttl`、`task-prefix`、`joint-no-prefetch`、`joint-no-predict`。`oracle` 只在同一回放用合法的真实剩余等待替代预测，不是数学上界。结果缓存默认关，另做同 `joint` 的 off/on；同 layout 的拷贝组使用 `AGENTIC_TRITON_COPY=0`（原生）、`1`（双向 Triton）、`d2h`（仅卸载用 Triton）。原生体验演示的 stock DSH 语义单列，不把它误称为同一工具集合的 `serial` 臂。

校准按长度 `[1,1024,4096,8192,16384,32768]`、batch `[1,4,8]`，每桶 5 次 warm-up + 20 次实际采样；记录 prefill、D2H、H2D、recurrent restore；缺桶取上邻桶，不外推。按等待类型冻结剩余等待分布与驱逐风险，在线只用已完成观察更新同类型 EWMA。保留校准哈希；不能从留出负载倒填校准。

## 3. FreeInference 回放：测负载，不测任务正确性

- 输入 `/root/autodl-tmp/release/<week>/traces/<day>.jsonl.zst`，逐行解压根会话；递归展开子调用。稳定身份 `(week, session_id, child_call_path, step)`，cache 隔离 `(week,user_id)`。记录空 block、时间异常、断链、嵌套深度、LCP、append/mutation/switch 和超窗排除数。不读取或修改 `viewer/` 来拼工具信息。
- 用固定模型 tokenizer 排除 special/control，取最小 16384 个普通 ID。每个 16-token 匿名 block 按 `(week, 父链摘要, block_id)` 哈希投影为两个 ID，检测同父 2-ID 碰撞，稳定排序加 salt 解决。按块保留真实共享前缀，不把 8× 投影解释回原文。投影后 `prompt+128>32768` 排除整轮，不截断；非长尾请求按 `max(1,min(记录 completion tokens,128))` 生成，缺值按 1 并计数，`ignore_eos=true`，生成内容不串到下一请求。
- 第 1–4 周仅校准，最后四周留出。稳定 hash 排序按 `子会话 > mutation/switch > 多调用 > 普通` 互斥分层，每层 smoke 16、主测 64 个完整根轮次，不足则报缺额，不跨层补齐。记录样本 hash、排除数和各层选择率。总体延迟只谈标注的等权混合样本，不冒充生产分布。
- 每组至多 8 个不同根会话，根轮次外生间隔裁到 `[0,1000]ms`；同会话要等上一轮及当前外生到达，后继只在实际前驱提交后发出。工具屏障为 `min(2000,max(有效 latency_ms))`，缺值记 0 并计数。子调用按记录 `call_index` 顺序递归，屏障与总子耗时取 `max`，不相加。连接不足沿记录顺序。工具侧不执行匿名命令，不从时间重合推断并行。
- 两组模型都实际使用 GPU。扩展臂 `/agentic/v1/batch` 先按分支/版本/`rid` 准入，再发原生 `/generate` 的预分词 ID 数组；原生臂使用相同原生 batch 路径。保留逐项状态和 cancellation；本地 SSE 不为凑批等待。

## 4. E1–E6 与正确性证据

| 场景 | 操作与观察 | 对应源码 |
|---|---|---|
| E1 并行/提交顺序 | 两个独立读同一快照（17、25），强制后声明项先完成；核对原 `callId`、声明序提交、最终 42。用一项失败和取消证明 drain，不准提前提交后继。 | [fixture](../agenticlayer/dsh-agentic/src/fixtures.ts)、[执行 hooks](../agenticlayer/dsh-agentic/src/index.ts) |
| E2 并发边界/缓存 | 固定总工作，扫 1/2/4/8；比较 serial、合法即发、预算派发。写后校验、未知读写/重命名不重叠。相同快照/参数/权限两次读物理执行一次；换数据或权限不能命中，写入只发生一次，失败/取消不缓存。 | [fixture](../agenticlayer/dsh-agentic/src/fixtures.ts)、[LRU](../agenticlayer/dsh-agentic/src/tool-cache.ts) |
| E3 压缩窗口 | 等待 25/250/2000 ms、提前返至 25%，同摘要文本/同切换位置。当前只可记录原生同步路径的正确性；DSH 提前候选未接线，**不得填 prepare-on 成绩**。原生直连的生成式辅助摘要不是同采样对照。 | [压缩](../agenticlayer/dsh-agentic/src/compaction.ts)、[候选与 CAS](../agenticlayer/agentic_layer/versions.py) |
| E4 fork、租约、恢复 | 父 `[A‖B]` fork 后替换为 `[A′‖B]`，子继续旧视图；D2H/H2D 与父/子取消交错。copy 未 ACK 时释放 0 Bytes；最终 lease/fence 完成才下降，重复回执不重复 free。验证 all-of 子 A 先返不能放行；B 失败保留错误。 | [控制执行器](../agenticlayer/agentic_layer/engine_cache.py)、[DSH 子执行](../agenticlayer/dsh-agentic/src/index.ts) |
| E5 同 tick 竞争 | 同时制造 fork 准入、候选准备、恢复。记录合法集合、容量/恢复债务、优先级、实际耗时。若真实负载无排序冲突，只报告零增量；可用合法注入 fixture 单列证明可达，不代替实测。 | [策略](../agenticlayer/agentic_layer/policy.py)、[Coordinator](../agenticlayer/agentic_layer/server.py) |
| E6 压力/漂移 | 以实测 Full token / Mamba slot 的 100/80/60% 向下取整；不够一条完整上下文或 8 个活动状态时，各臂同档使用满足合同的最小值并披露实际比例。host 16/16/4 GB。每第四屏障缩到 25% 或拉长至 2×；50% 实测带宽令牌桶限流单列，只在相同限流下比较。 | [池统计与预留](../agenticlayer/agentic_layer/engine_cache.py)、[候选成本](../agenticlayer/agentic_layer/policy.py) |

**GPU 硬门禁：** 用不同分支真实合批，取消一个不影响另一个。按真实 Full-only 与 Full+Mamba 分开核对命中。对 512/4096/16384 token、seed 42/43/44，cold 重复 5 次建立逐位置 greedy ID 和 top-5 logprob 波动包络；同 token teacher-forced logprob 偏差不得超 `max(1e-4, 2×max_cold_abs_delta)`，稳定位置 greedy 序列必须一致；cold 不稳定位置不允许新的未观测 ID。压缩重写按新目标 cold 比，恢复按原始状态比；摘要质量另做固定证据/任务评分。

**传输硬门禁：** Full K/V、conv、SSM 做稀疏乱序和连续行的实际 HBM→host→HBM round trip；比原生逐字节、未触及行 sentinel。检查重复源、空批、尾 tile、重复目标拒绝、真实映射、正数 per-pool bytes/ACK 和混合恢复。若 native ACK 无法准确归属动作（`transfer_metrics_complete=false`），保留缺测，不做零值或替换后端。

## 5. 记录与统计

每个匹配预算的 baseline 保存自己的 `baseline.json`、manifest、锁定 hash、请求/响应和原始时间线。每条件重启、冷缓存；warm-up 用不相交的校准 namespace。5 次配对重复，循环轮换各臂运行顺序。对每轮外生 user-ready 开始计时，900 s 截止；失败/超时/取消/未完成按 `+∞` 进入 nearest-rank P50/P95/P99，同时报成功数和成功条件分位数。用单调时钟重建关键路径，重叠区间不重复相加。

报告 `joint−strong` 的 P95 差及 session-cluster 配对 bootstrap 95% CI（10000 次，seed 20261118）；CI 含 0 报无明确增益；分位数未达到则给成功分母，不做无穷值相减。另报 TTFT、re-prefill、两类命中、各级唯一字节峰值/byte-seconds、传输、后台候选完成/废弃、缓存命中/实际工具执行次数、吞吐与资源压力。未来报告输出 `summary.csv`、`paired.json`、`timeline.jsonl.zst`，分列真实闭环、固定语义任务、GPU 轨迹投影与 oracle。闭环只在自建固定工作区跑真实工具、核对可判分结果，不与等工作量系统消融混算。
