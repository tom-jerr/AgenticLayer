# 论文机制：固定 CUDA Graph 的实验协议

[论文 main.pdf](../main.pdf) · [旧消融](FREEINFERENCE_ABLATION.md) · [工程安全门禁](README.md#4-e1e6-与正确性证据)

本轮所有被比较的 GPU 引擎固定 decode CUDA Graph `full`、prefill `disabled`，捕获 decode batch `1,2,3,4,5,6,7,8`。每个引擎必须同时通过 `/server_info` 和实际捕获日志核对；缺任何一个不进入比较。相同捕获桶不等于动态有效 batch 组成相同；按原始时间线记录实际批次、前台入队、GPU 缓存命中与尾部延迟。L2 checkpoint、相同量化权重、后端、上下文、Full/Mamba/DRAM 配额匹配；只在对应机制臂改变处理方式，不把 Graph 的加速归给论文机制。

## 预注册：负载和主要判据

每一行只开该机制，与同语义、同工作量对照。低越好的主要指标按 `100×(baseline−variant)/baseline`，超过 5% 才达工程门槛；端到端的“明显”还要求配对区间排除 0。同步成本、准备作废和跨分支干扰均计入。门禁失败不采纳性能结论；不因结果不好改阈值、样本或计时边界。用机制能触发的 fixture 识别因果，同时在相同的 FreeInference 留出轨迹判断覆盖率；资料匿名化，不复现原工具或原任务正确性。

| 对应论文 | 单项开关和不变项 | 主指标 / 正确性 |
|---|---|---|
| E1，§3.4–3.6 | 已确认的独立工具/子智能体：同真实任务，串行→合法并行；未授权或有冲突仍串行 | 完整任务活跃 P95；另报工具/子分支 all-of 暴露跨度，调用及结果配对、声明顺序、失败排空、唯一副作用 |
| E2，§3.6.1 | 立即派发全部合法任务→限额派发；固定每档相同的合法任务集合与真实 CPU/IO/GPU 回流 | 活跃 P95；另报服务和回流排队。若没有争用则没有可节省等待，不把低并发本身算作胜利 |
| E3，§4.5.1/4.6.1 | 同摘要、同切换点、同尾部，必要时同步重算→等待内包括摘要生成和精确候选准备；另做早返/前文变化 | 活跃 P95；另报切换停顿、作废、双驻留。门禁需实际生成相同摘要、合法 CAS、改写不命中旧后缀、不能超窗 |
| E4，§4.3–4.6 | 在同一兼容版本历史下，无显式提前回收→失效后且引用/fence 全部完成再回收 | 终止分支的 HBM byte-seconds（低是收益）；另外报告压力下活跃 P95、存活子恢复等价、引用在途释放必须为 0 |
| §5.4.2，放置 | 配对：HBM 保留、按需 host 恢复、提前恢复、无候选冷重算；固定容量、输入及恢复/重算决策条件 | 活跃 P95（含必要传输或重算）；另外报告 HBM、host byte-seconds、实际 Full/Mamba ACK，不能将冷重算比较代替物理迁移等价性 |
| E5，§5.5–5.6 | 同一真实合法竞争集合、同容量/安全保护下，强组合→联合；还测单项和联合去掉一项，不把多个单项加速相加 | 活跃 P95，配对看单项和留一边际及差值的差值；观察同 tick 的准入/准备/恢复竞争及各自完成情况 |
| E6，§5.6 | 对相同压力档、带宽和早返，单独关预取/预测、比较 simple/strong/joint、TTL、任务前缀基线、未来信息 oracle 单列 | 活跃 P95，失败按不可达计，观察容量、压测、漂移、后台有效性与冷请求公平性；不把 oracle 当可部署优化 |

结果缓存并非 §3.5 的论文核心前提；有真实工作可独立记录。确认/安全检查是所有臂共同约束，不为追求 5% 关闭。资源型机制（回收、容量保护）用预注册资源指标和必要的端到端不退化检查；不能要求它无条件降低无压力轮次延迟。若某机制在理论上不存在可用窗口或固定 workload 必须等同一个前驱，原样报告该条件的负结果，再在预注册的不同负载分层定位边界。

```
export AGENTIC_FIXED_GRAPH=1 AGENTIC_DECODE_GRAPH=full AGENTIC_CHECKPOINT_LEVEL=2
export AGENTIC_FULL_TOKENS=278241 AGENTIC_MAMBA_SLOTS=157 AGENTIC_TRITON_COPY=0
```

运行工件写入 `/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01`，每臂捕获指纹在 `logs/*-graph.json`。如在相同语义与正确性下某一差值的配对结果仍未达工程门槛，记录瓶颈和下一次单旋钮对照；不要把不同负载的成功样本合并为“一次联合实验”。

### 已验证的配置与控制面优化

- 固定图参考的四臂都实际捕获 `1…8`，prefill 为 `disabled`，逐项验证同样 12 轮、43 请求、29 屏障、656,370 prompt / 5,100 输出 tokens。实际解码批次构成仍各不相同，见 [原始汇总](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/trace-single-figures/arms.csv) 与 [柱图、完成曲线和调度区间图](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/trace-single-figures/fixedgraph-reference)。
- 真实首四周 1,011 个等待样本上，原 `remaining_p50` 每个候选反复排序并重建 Kaplan–Meier。现在每次读入冻结校准时只构建一次，在线用两次二分。4,011 个真实阈值、4,514 个包含重复和删失事件的边界查询与旧实现逐值一致；单次查询 median 165.13 → 0.520 µs（约 318×）。这是控制面热点，不外推为轮次提升。
- `baseline` 及 `*-only` 不再隐式执行显式过期回收；`joint-minus-*` 在同一仲裁器上仅关闭指定的可选动作。所有臂保留原生必要恢复、引用安全和硬资源门禁。

### 首轮同桶、同工作量回执

| 场景 | 对照 | 观察 |
|---|---|---|
| FreeInference 留出回放（单次四臂） | 原生 Radix、原生 HiCache、`strong`、`joint` | P95 为 32.508 / 32.564 / 33.583 / 36.073 s。受控仅 `admit`，原生 `cached_tokens` 全为 0。`joint−strong` +2.490 s，12 会话重采样探索区间 `[+2.136,+4.190] s`；没有可归因于可选动作的联合收益。该批次仅 1 或 2 项，结批差的总机会低于 0.185 s，解释不了 2.49 s；实际 TTFT、prefill 与解码环境同时有变化，不据此称仲裁算法必然慢。 |
| E1 真实 SDK 工具（6 对，含首次 SDK 启动） | 同一组两个独立读，串行 vs 并行 | 读阶段 P50 1896 → 1008 ms（−46.8%），所有调用配对、顺序与结果通过。活跃 P95 13799 → 13005 ms（−5.75%），配对差区间 `[−794,+514] ms`，未达明确端到端门槛；另测启动好后按相同业务输入计时。 |
| E2 固定预算（CPU/IO × 1/2/4/8，每格 6 对） | 一次确认的同一工具集及 GPU 回流，立即 vs 从工具启动到 GPU 返回最多 2 个在途 | CPU/HBM 同预算下，4 / 8 分支的配对 P95 改善中位数 28.34% / 22.84%，区间分别 `[15.39,31.50]%` / `[9.76,41.40]%`。IO 4 / 8 分支反而 −21.46% / −7.51%（负值代表退化，区间跨 0），不可无条件套用。真实 `fdatasync`、实际 GPU 和全部 48 对工作量门禁；这不是生产 DSH 的资源预算接线。 |
| E3 付费摘要+状态准备（各 10 对，70/70 门禁通过） | 同摘要、同切换和工作量，只调时机 | 稳定 2000 ms：10/10 精确命中，整轮 P95 3220 → 2860 ms，配对中位收益 18.12%（区间 `[14.92,22.13]%`）。实际 250 ms：10/10 摘要来不及结束、无候选命中，但提前摘要仍隐藏一部分开销，P95 1776 → 1425 ms。25 ms：活跃 P95 反而 1128 → 1243 ms，未达门槛；该物理窗口最多遮住 25 ms，不能要求对 1 s 轮次无条件提升 5%。改写保留前文时 10/10 原候选作废，不拼接旧缓存。 |
| E5 两动作竞争（10 对） | 同一个合法 `admit` / `prepare` 集，两臂只替换仲裁 | `strong` 10/10 先 `admit`，`joint` 10/10 先 `prepare`。完成 P95 1856.0 → 1301.7 ms（−29.9%；配对差区间 `[−748,−340] ms`），P50 1551.3 → 1086.6 ms；两路答案均正确。不是三动作或生产覆盖率。 |
| E4 无压力过期回收（每形状 5 对） | 保持/回收相同的两条物理状态 | 释放 354.25 MiB（4K）与 1122.25 MiB（16K）；同一后续前向的原生墙钟 P50 只有约 0.3% 变化。不把池内可用空间等同进程 VRAM 或无压力时的延迟收益。 |
| E4 父改写、子仍活跃（每形状 5 对） | 确认真实 lease 后保留或显式回收原前缀 | 4K/16K 过期状态持有 byte-seconds 减少 88.43% / 98.13%。4K/16K 各 5/5 次拒绝提前回收、父改写不复用原 KV、子完整复用旧前缀；释放后重复回收请求只重放回执。 |
| 放置（每形状 5 个三元组） | 常驻 / DRAM 恢复+前向 / 冷重算 | 4K P50 155 / 164 / 294 ms，P95 281 / 187 / 358 ms；16K P50 178 / 226 / 805 ms，P95 359 / 235 / 856 ms。16K 腾出 561.125 MiB；计入 D2H 后 P50/P95 为 345/429 ms，仍低于同种子冷重算。P95 在 5 次中就是最大值，不据此称恢复比常驻快。固定 fixture greedy 门禁 10/10，不外推强冷数值门禁。 |

固定图 [两动作热路径](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-compete-joint.json) · [回收与放置物理回执](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-state.json) · [在途取消与逐项隔离](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-copy-fence.json)

[E4 共享前缀、安全回收与实际 byte-seconds](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-shared-reclaim-recheck.json)。该配对从已证实脱离活跃引用后测 600 ms，不把保护期内合法持有计为可释放空间；用原生控制动作直接重建版本边界，不代替 DSH 的完整封装对照。

[E1 SDK 原始报告](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/dsh-bench/bench-2026-11-19T16-24-36-282Z-1c0fd678-d0d1-4815-b475-6e91152244bf/report.json) · [E2 公共预算对照](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-admission-e2.json) · [E3 70 对付费摘要](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-compaction-e3.json)

同一可选工作即将失去价值时，`joint-on-time` 仅多一条后台门禁：预计完整准备或恢复到不了安全切换点则不启动，不影响必要前台动作。在同一个三动作队列测试，不能通过改工作量追求 5%。用 `--reuse-engine` 的受控筛查每臂刷新缓存并重启 ingress，复用一个已核验的真实图捕获以排除不同实例和重复 autotune；不混用初次独立重启图表。

三动作 `admit/prepare/restore` 的 [未约束自然输出对照](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-three-way.json) 整组仅 2/6 通过：不同调度有不同动态批次，部分 `joint`/`joint-minus-prediction` 子请求输出也发生变化；不能按统一数值门禁比较整组。预设的 `strong`→`joint-on-time` 单独 6/6 三路完整 token 相同，P95 2055.1 → 1809.9 ms（−11.93%，配对区间 `[−412.8,−222.5] ms`）；`joint-minus-lookahead` 单独也 6/6 相同，P95 1689.9 ms（−17.77%）。原因：预填充中位服务 684.6 ms，超过该试验的 600 ms 等待；原始 `joint` 6/6 次仍优先启动，0/6 次到切换点完成。600 ms 下关掉准备可能最好，不能据此从长窗口永久删除。

### 留出轨迹逐项：工作量一致不代表数值等价

同一固定图引擎每臂冷清缓存后，8 臂均为 12/43/29 且每个请求输入与输出数量一致。基线活跃 P95 为 33.516 s；并行、生命周期、放置、前瞻单开依次为 34.636 / 34.260 / 33.836 / 33.518 s。没有 `prepare`、`offload` 或 `restore`；生命周期首次选出 1 次过期候选，但原生控制反复以 `checkpoint has unbacked component state` 拒绝。其他臂相对基线只有 33–38/43 个请求输出序列逐 token 相同（并行 35/43）；这些时延用于诊断未触发和抖动，不满足“相同生成结果”的强门禁。匿名历史工具事件不提供可确认的独立读写集、稳定摘要目标或成果真值。

[逐臂动作和图桶 CSV](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/trace-single-figures/arms.csv) · [P95 柱图、完成折线、调度 min/median/max](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/trace-single-figures/fixedgraph-trace-single) · [回收原生拒绝回执](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/logs/fixedgraph-trace-single-1-lifecycle-only-ingress.log)

回收修复不发起强制 D2H：仅已终止且没有其他活跃版本引用的分支可授权卸掉未备份设备叶；设备端仍拒绝 lease、前缀不完整、子树、host 状态和写入/加载未完成的节点。TTL 可保留其弱策略，不自动获得弃置未备份叶权限。单项历史矩阵保留原包行为；新实现用独立 GPU 夹具测字节和引用，再重跑同一留出集。

修复后，[真实 HiCache 独立 GPU 对照](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-hicache-expired.json) 的 4K/16K 均 5/5 对等，过期 HBM byte-seconds 分别减少 98.25% / 98.00%；有 lease 和没有弃置授权的前缀均拒绝，先后原生输出不变。真实留出回放复测中，每个开启生命周期的臂只完成 1 次回收（14 个 Full slots 和 1 个 recurrent slot），其余可选动作和原生命中均为 0；P95 的跨重复离散不能归因给这次物理释放。

### 单项复测、明确适用条件

| 机制 | 固定图的实测结论 | 不启用或不外推的条件 |
|---|---|---|
| E1 预热后的真实 SDK 并行 | 12/12 对同工具参数、顺序和单次写入。读阶段 P50 1725.9 → 1007.6 ms；活跃 P95 8929.3 → 8401.0 ms（−5.92%，配对差 95% 区间 `[−788,−161] ms`）。另有 6/6 对真实子激活的正向试验，P95 8079.3 → 6215.0 ms（−23.08%，`[−2150,−1147] ms`），先验证两个子分支均终态，再提交父结果。 | 激活失败与子任务正常处理读错误不同。先前的错误对照仅 6/12 实际读出异常：模型可能不调用工具，原始失败保留，不能把文本报告当原生错误。 |
| E2 分阶段预算 | CPU 夹具只有整工作者限额达到当前置信门槛。仅限制 CPU/IO 工具阶段对 CPU 4/8 的配对中位数是 +3.06%/−11.62%；仅限制 GPU 回流为 +18.72%/+15.18%，两组区间都跨 0。 | I/O 8 分支仅限工具退化 34.69%，仅限 GPU 接近无效（+0.21%）。对类似 fdatasync 工作集保留立即派发；不把其他资源瓶颈混进 CPU 的全段限额。 |
| E6 独立提前恢复 | 相同 250 ms 屏障、相同 D2H/H2D 字节、20/20 对输出一致且 H2D 均在截止前 ACK。4K P95 509.1 → 465.7 ms（−8.52%，配对收益中位数区间 `[4.37,12.41]%`）；16K P95 510.9 → 475.8 ms（−6.88%，区间 `[6.04,12.54]%`）。只有 16K 的收益区间全高于 5%。 | 零窗口 16K 0/10 提前 ACK，P95 405.1 → 456.6 ms（退化 12.72%），不能用按需刚开始的拷贝冒充预取收益。 |
| 数值门禁 | 512/4K/16K × 3 seeds，相同热分段下迁移与不迁移完全一致（9/9，logprob 最大差 0）。 | 相对原生整段冷参考，热对照和迁移都 0/9 通过原阈值，最大差 2.2612。不能把冷/热分段差异归因于拷贝，也不能宣称跨分段数值等价。 |

[预热 SDK](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/dsh-bench/bench-2026-11-19T17-19-56-102Z-e0007cda-e494-4912-9762-cf4da3b6132c/report.json) · [子激活及负例](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/dsh-bench/bench-2026-11-19T17-15-36-156Z-ef856976-b135-4987-9323-75f13660ada8/report.json) · [限工具](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-admission-tool-recheck.json) · [限 GPU](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-admission-gpu-recheck.json) · [提前恢复](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-prefetch-e6.json) · [零窗口](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-prefetch-no-window.json) · [逐 token 热对照 / 冷参考](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-forced-migration.json)。

子任务错误的实机复验不在提示里泄露故障：先在 SDK 私有工作区把已批准 `a.txt` 改出原快照，再要求子任务调用真实读。12/12 产生与 `callId` 对应的原生 `fixture snapshot changed`，父节点在两子任务终态之后按 `good,bad` 顺序报告 `25` 和快照错误。原始报告的 5 个字面措辞失败属检查器把 “snapshot for … has changed” 当失败；新检查保留原生错误、调用对应、顺序和语义并用同批 12 个原始 SDK 通知重验。它不是强制中止 activation 的试验。[原始报告及通知目录](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/dsh-bench/bench-2026-11-20T03-25-07-123Z-8a4116ae-67a8-4cfc-9463-f5a04381aa00/report.json)

### 冻结开关的联合效果

- **已确认的并行读 × 付费前瞻**：同工作量 10/10 对通过。以两项全关为基线，活跃 P95 如下。并行后前瞻仍有 −960 ms（区间 `[−1011,−484] ms`）；P95 差值的差值 +84 ms（`[−827,+832] ms`），没有明确的相互增强或抵消。联合相对全关 −1842 ms（`[−1894,−931] ms`）。这是实际文件+GPU 受控夹具，不是 DSH 异步压缩或匿名回放的端到端收益。

| 读从属关系 | 同步付费摘要 | 等待内付费摘要与准备 |
|---|---:|---:|
| 串行 | 3581.5 ms | 2537.5 ms |
| 已确认可并行 | 2699.2 ms | **1739.2 ms（−51.44% vs 全关）** |

独立重启同样捕获配置、冻结同样四臂再做 10/10 对：[独立联合复验](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-parallel-lookahead-confirm.json) 全关 3558.5 → 两项全开 2080.8 ms（−41.53%，配对差 `[−2081,−1039] ms`）；在并行读后单开付费前瞻仍少 415 ms（`[−1019,−73] ms`）。交互项 +346 ms（`[−544,+922] ms`）仍跨 0。并行缩短了确定的重叠窗口：`serial/early` 两批各 10/10 命中，而 `parallel/early` 初测 7/10、复测 5/10，全部回退仍计费；两项全开仍是同工作量四臂中最优。

- **同 tick 三动作**：[自然输出短窗口独立复测](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-three-way-short-recheck.json) 600 ms 的 10/10 完全一致，`strong` 1878.1 ms → `joint-on-time` 1626.5 ms（−13.39%，`[−301,−239] ms`）或 `joint-minus-lookahead` 1619.4 ms（−13.77%）。没有任何提前准备命中；收益来自避免无用准备和改变动作启动时机。早返至 150 ms 时仍然 10/10 等价，却由 `strong` 1227.6 ms 胜出，闸门臂退化 9.85%；按名义等待做预测并不能保障早返场景。80% + 50% 拷贝发放额度的长窗口自然输出仅 375/1500 ms 各 6/10、5/10 完全一致，不得比较速度。
- **隔离前缀分段**：同一候选若未在截止准备好，另一臂在截止之后付完全相同的前缀准备，再走与命中臂一致的生成分段。600 ms 五臂通过 10/10；`joint-on-time` 去前瞻比完整闸门 P95 少 472 ms，但区间 `[−606,+140] ms`，独立边际不明确；去预取只少 74 ms，区间也跨 0。150、375、1500 ms 的整组仍仅 2/10、5/10、6/10 通过。逐路回执表明，剩下的输出差异都在同时竞争的 `child`，不在准备目标或恢复输入；限制分段不能消除动态批次数值变化。[短窗口](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-segmented-short.json) · [带宽受限长窗口](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-segmented-paced-long.json)。
- **FreeInference 正式留一和压力矩阵**：100% `strong` 34.911 s → `joint` 33.148 s（名义 −5.05%），80%（7 个长等待扰动）43.770 → 37.993 s，60%（7 个早返扰动，host 4 GB）35.447 → 31.595 s。80/60% 差值区间都跨 0；所有矩阵只有 `admit` 与一个候选回收，无自动迁移、提前恢复、准备和原生命中。采样 Full 使用峰值最高 0.770，Mamba 峰值最高 0.085；60% 是池容量配比，不等于达到全部饱和压力。没有同语义并发授权且输出不完全相同，所有留一差和 oracle 只能作为覆盖/瓶颈诊断，不能据此选择赢家。

[并行 × 前瞻原始四臂](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-parallel-lookahead.json) · [联合/留一/压力各图](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-figures-verified) · [逐臂数值门禁与原生调度 CSV](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-figures-verified/arms.csv)。

### 选型与独立 FreeInference 复验

剔除首批的全部 `(week, session_id)`，冻结互不重叠的第二批。为保留四个分层各 3 个入测机会，固定解码资格上限由 768 上调到 3072；包含因果前驱共 13 轮/12 会话，135 请求、119 屏障、1,795,456 prompt / 16,881 输出 tokens。固定 60% Full/Mamba 配额、4 GB host，重用一次捕获并重启 ingress。三臂全部完成，在相同工作量下仅选出 `admit`；无可选 native 动作。

| 独立留出策略 | 活跃 P95 | 相对 `strong` 的差（会话配对 95% 区间） | 等于 `strong` 的完整输出 |
|---|---:|---:|---:|
| `strong` | 85.926 s | — | 135/135 |
| `joint` | 84.586 s | −1.340 s `[−8.102,+5.439]` | 57/135 |
| `joint-on-time` | 96.306 s | +10.379 s `[+2.025,+10.379]` | 61/135 |

原生 KV 命中顺序也不同（352,256 / 487,424 / 217,088 tokens），不能把 P95 差归因给未触发的优化。原始引擎实际执行过 1–8 个请求的 Graph 批次，但同捕获配置不要求各臂运行时每桶次数相同。[冻结抽样清单](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-heldout-confirm.manifest.json) · [独立复验原始报告](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-independent-confirm.json) · [柱图、完成曲线和排队 min/median/max](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/independent-figures/fixedgraph-independent-confirm)。

逐步定位数值差异：失败三动作组的首个差异均在第 2 个生成 token 或之后。原生 4096-token 子请求单独运行 vs 同批增加一个独立请求，在 Graph 实际执行 batch 1 / 2 时，默认与 `--enable-deterministic-inference` 都是 6/6 相同 64 个生成 token。默认最大 logprob 差 `9.54e−6`，确定性开关为 0；但这组没有复制三动作时不同的预填充/解码交错，不能用它宣布联合门禁已通过。[原生逐 token 探针](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-verify-row-numerics.json)

三动作自然长窗口的原生 Top-5 回执在第一个不同 token 上，对共同候选的最大 logprob 差达 2.521；强制同一目标分段后仍达 0.513。差异不能按两个候选恰好并列消除，也不能在未隔离并发路径前归因给某次 H2D。

在原三动作热路径上实际开启确定性内核（同 `full/disabled`、同桶、同 80% 配额与拷贝发放额度），375/1500 ms 仍各只有 7/10 个完整五臂配对等价，出错仍是 `child`。早返 375 ms 时，预设的 `joint-on-time` 与仅去掉前瞻一对一 10/10 相同：P95 5451.8 → 3622.0 ms（−33.56%，配对区间 `[−1830,−1068] ms`），准备 0/10 命中；去预取和去预测的动作时机均未变，区间分别跨 0。不能把这个子对照升级成“联合相对 strong”的改善，也不能推广为长窗口应该关前瞻。[确定性三动作原始回执](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-deterministic-paced-long.json)

**只开预测的复验：** 不再把 1500 ms 标称等待用于短负载；在独立校准样本测得等待中位 375.9 ms、准备中位 670.1 ms，随后固定同一 375 ms 工作负载与 80% 容量、50% H2D 发放额度，仅切换预测。10/10 三路输出完全一致。关闭预测 10/10 先发准备、0/10 命中；开启后 10/10 先发实际恢复、准备不启动。活跃 P95 **2101.3 → 1466.4 ms（−30.21%，配对差 `[−646,−536] ms`）**。预测训练分布须匹配条件；上一组 1500 ms 校准下的 375 ms 早返没有此效果。[短等待预测单开原始回执](/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01/fixedgraph-deterministic-short-forecast.json)

**按实测选型：** 独立性已确认且长等待可预测时，优先合法并行 + 付费候选；两次固定工作量试验中分别降低 P95 51.44% / 41.53%。对 600 ms 的三动作条件，预计准备晚于安全切换点时不要启动准备；不要把该规则用于未知早返作全局加速承诺。生命周期可在真正过期后回收；缓存迁移仅在需要腾显存、同分段数值一致且预计完整 ACK 可赶上时使用；不要把纯 DMA 时长当截止前的完整可用时间（16K H2D 的设备事件/回执墙钟中位数 12.30/34.84 ms）。资源受限的 CPU 工具有证据使用整段 2 槽；I/O 保留合法即发。匿名回放及未验证的长窗口／漂移不给全开提速结论，保持 `strong` 作为保守对照，不因此修改默认策略。

## 复现命令

按 [执行手册](../agenticlayer/RUNBOOK.md) 新建一个私有 `RUN`，保留隔离 Python、SGLang 源码、编译缓存与已锁定模型。所有 GPU 命令串行运行，同一命令中的对照工作量不能修改。

```bash
BASE=/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01
PAPER=/root/autodl-tmp/agentic-layer-runs/freeinference-paper-01
PY="$RUN/env/python/bin/python"
export PYTHONPATH="$PWD/agenticlayer" PYTHONDONTWRITEBYTECODE=1
export AGENTIC_FIXED_GRAPH=1 AGENTIC_DECODE_GRAPH=full AGENTIC_CHECKPOINT_LEVEL=2
export AGENTIC_FULL_TOKENS=278241 AGENTIC_MAMBA_SLOTS=157 AGENTIC_TRITON_COPY=0
SAMPLE="$BASE/trace-heldout-smoke.jsonl"
ARGS=(--run-dir "$RUN" --sample "$SAMPLE" --calibration "$PAPER/calibration.json"
      --turns-per-stratum 3 --max-turn-decode 768)

"$PY" -m experiments.trace_matrix "${ARGS[@]}" --name fixedgraph-reference
"$PY" -m experiments.admission_gpu --run-dir "$RUN" --name fixedgraph-admission-e2 --budget 2 --repeats 6
"$PY" -m experiments.compaction_gpu --run-dir "$RUN" --name fixedgraph-compaction-e3 --pairs 10
"$PY" -m experiments.state_ablation_gpu --run-dir "$RUN" --name fixedgraph-state
"$PY" -m experiments.shared_reclaim_gpu --run-dir "$RUN" --name fixedgraph-shared-reclaim
"$PY" -m experiments.cancel_gpu --run-dir "$RUN" --name fixedgraph-copy-fence
"$PY" -m experiments.competition_gpu --run-dir "$RUN" --name fixedgraph-compete --runs 10 \
  --calibration "$BASE/prepare-joint-paired-fixed-calibration.json"
"$PY" -m experiments.joint_fixture_gpu --run-dir "$RUN" --name fixedgraph-three-way --pairs 6 \
  --arms strong joint simple joint-on-time joint-minus-lookahead joint-minus-prefetch joint-minus-prediction
"$PY" -m experiments.trace_matrix "${ARGS[@]}" --reuse-engine --name fixedgraph-trace-single \
  --arms baseline parallel-only lifecycle-only offload-only lookahead-only ttl task-prefix simple
"$PY" -m experiments.trace_matrix "${ARGS[@]}" --reuse-engine --name fixedgraph-trace-joint \
  --arms strong joint joint-on-time joint-minus-parallel joint-minus-lifecycle joint-minus-placement \
         joint-minus-lookahead joint-minus-prefetch joint-minus-prediction oracle
AGENTIC_FULL_TOKENS=166944 AGENTIC_MAMBA_SLOTS=94 AGENTIC_HOST_GB=4 \
  "$PY" -m experiments.trace_matrix "${ARGS[@]}" --reuse-engine --name fixedgraph-trace-pressure60 \
  --wait-drift early-quarter --arms strong joint joint-on-time joint-minus-prefetch joint-minus-prediction
AGENTIC_FULL_TOKENS=222592 AGENTIC_MAMBA_SLOTS=125 \
  "$PY" -m experiments.trace_matrix "${ARGS[@]}" --reuse-engine --name fixedgraph-trace-pressure80 \
  --wait-drift long-double --arms strong joint joint-on-time
"$PY" -m experiments.shared_reclaim_gpu --run-dir "$RUN" --name fixedgraph-hicache-expired --hicache
"$PY" -m experiments.state_ablation_gpu --run-dir "$RUN" --name fixedgraph-prefetch-e6 \
  --scope prefetch --wait-ms 250 --repetitions 10
"$PY" -m experiments.state_ablation_gpu --run-dir "$RUN" --name fixedgraph-prefetch-no-window \
  --scope prefetch --wait-ms 0 --lengths 16384 --repetitions 10
"$PY" -m experiments.interaction_gpu --run-dir "$RUN" --name fixedgraph-parallel-lookahead --pairs 10
"$PY" -m experiments.interaction_gpu --run-dir "$RUN" --name fixedgraph-parallel-lookahead-confirm --pairs 10
"$PY" -m experiments.joint_fixture_gpu --run-dir "$RUN" --name fixedgraph-three-way-short-recheck \
  --pairs 10 --wait-ms 600 --return-ms 150 600 --arms strong joint-on-time joint-minus-lookahead
AGENTIC_FULL_TOKENS=222592 AGENTIC_MAMBA_SLOTS=125 \
  "$PY" -m experiments.joint_fixture_gpu --run-dir "$RUN" --name fixedgraph-segmented-paced-long \
  --pairs 10 --wait-ms 1500 --return-ms 375 1500 --h2d-fraction .5 --matched-segmentation \
  --arms strong joint joint-on-time joint-on-time-minus-lookahead joint-on-time-minus-prefetch
AGENTIC_FULL_TOKENS=222592 AGENTIC_MAMBA_SLOTS=125 AGENTIC_DETERMINISTIC=1 \
  "$PY" -m experiments.joint_fixture_gpu --run-dir "$RUN" --name fixedgraph-deterministic-paced-long \
  --pairs 10 --wait-ms 1500 --return-ms 375 1500 --h2d-fraction .5 --matched-segmentation \
  --arms strong joint-on-time joint-on-time-minus-lookahead joint-on-time-minus-prefetch joint-on-time-minus-prediction
AGENTIC_FULL_TOKENS=222592 AGENTIC_MAMBA_SLOTS=125 AGENTIC_DETERMINISTIC=1 \
  "$PY" -m experiments.joint_fixture_gpu --run-dir "$RUN" --name fixedgraph-deterministic-short-forecast \
  --pairs 10 --wait-ms 375 --h2d-fraction .5 --matched-segmentation \
  --arms joint-on-time joint-on-time-minus-prediction
"$PY" -m experiments.gpu_run reference-cold --run-dir "$RUN" --name fixedgraph-reference-cold-forced
"$PY" -m experiments.forced_gpu --run-dir "$RUN" --name fixedgraph-forced-migration \
  --reference "$RUN/fixedgraph-reference-cold-forced.json"  # 当前跨分段冷容差不通过，保留非零退出
python -m experiments.plot_fixed_graph --matrix "$RUN/fixedgraph-reference.json" "$RUN/fixedgraph-trace-single.json" \
  "$RUN/fixedgraph-trace-joint.json" "$RUN/fixedgraph-trace-pressure80.json" "$RUN/fixedgraph-trace-pressure60.json" \
  --out "$RUN/fixedgraph-figures-verified"
```

独立留出复验使用现有冻结文件及同名清单，不重新抽样：

```bash
EVIDENCE=/root/autodl-tmp/agentic-layer-runs/freeinference-fixedgraph-01
AGENTIC_FULL_TOKENS=166944 AGENTIC_MAMBA_SLOTS=94 AGENTIC_HOST_GB=4 \
  "$PY" -m experiments.trace_matrix --run-dir "$RUN" --sample "$EVIDENCE/fixedgraph-heldout-confirm.jsonl" \
  --calibration "$PAPER/calibration.json" --turns-per-stratum 3 --max-turn-decode 3072 \
  --reuse-engine --name fixedgraph-independent-confirm --arms strong joint joint-on-time
python -m experiments.plot_fixed_graph --matrix "$RUN/fixedgraph-independent-confirm.json" --out "$RUN/independent-figures"
```

真实 E1 使用 [`dsh_bench.mjs`](dsh_bench.mjs) 的 `--suite e1 --repeats 12` 与 `--suite allof-pair --repeats 6`。按手册启动同一有捕获记录的 `extended` 引擎和 `strong` ingress，再传 `--mode controlled --run-dir "$RUN" --events "$INGRESS_LOG"` 及编译 runner、锁定 DSH 路径、本地 token。成对子分支验证需要编译含 `exclusive_child` 许可的当前插件。未启动受控 ingress 的直接 HTTP 不代替此项。
