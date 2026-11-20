# RTX 5090 实测：DSH × AgenticLayer × SGLang

原始运行目录：[`5090-nvfp4-01`](/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01)。部署见 [执行手册](../agenticlayer/RUNBOOK.md)，机制见 [设计](../agenticlayer/DESIGN.md)，完整对照合约见 [实验手册](README.md)。下列指标不是论文或历史机器的结果。

已通过阶段的原始文件指纹：[verified-evidence.sha256](/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01/verified-evidence.sha256)。固定窗口、双动作竞争与 E1 的可复算数字：[gpu-observed-summary.csv](/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01/gpu-observed-summary.csv)。

## 版本与基线

- RTX 5090，32,607 MiB，driver `580.95.05`；与原计划的 `580.105.08` 不同。固定同 GPU、隔离 CUDA 13.0 nvcc `V13.0.88`、Torch `2.14.1+cu130`、FlashInfer `0.7.4`。`gpu-initial.txt`、`python.lock.txt`、`source-archives.sha256`、`model.sha256` 在运行目录。
- 模型 `AxionML/Qwen3.5-9B-NVFP4@97aef92393f126bf649f310cd40861be8dad3279`；SGLang `703e763d5f3af75ea3587275d79155c28d3c5d4f`；DSH `9c2670d3bce5275e0d1610272958121af87787bb`。ModelOpt NVFP4、FlashInfer CUTLASS GEMM、Triton 全量和线性 attention、BF16 KV、FP32 recurrent、TP=1、page=1。
- `native-radix-32k.json`：实际输入 32,767 tokens 加生成 1 token，未 OOM，2,409.62 ms。相同 1,024-token 前缀：冷命中 0，热命中 1,024，所比 logprob 差 0；这一对样本 286.20 → 69.22 ms，不是本设计加速结果。
- `reference-cold-forced.json`：512 / 4,096 / 16,384 输入 × 42 / 43 / 44 种子，每格独立冷 greedy 和 teacher forcing 各 5 次；提供逐位置所比 token 的冷波动包络。
- `native-hicache-prefix.json`：在原生容量压力下，从实例私有 DRAM 命中 16,384 tokens；正命中且下一 greedy token 正确，234.92 ms。原生 Full 池记录为 278,241 token slots。

配置实测纠正：[launch_engine.sh](launch_engine.sh) 把 FlashInfer 并行编译限为 `MAX_JOBS=3`（原先编译进程被终止），为 SGLang 本地模型 warmup 显式绕过外部代理。固定 SGLang 版内部从配置扣 6，再用 exclusive 输入上限；统一设 32,774 得到实报 32,768，未启用截断。详见 [RUNBOOK §2](../agenticlayer/RUNBOOK.md#2-启动-sglang30080)。

## 物理混合 KV：原生拷贝与 Triton

三种形状各完成 Full KV 与 Mamba checkpoint 的 D2H/H2D。均有两个池的正拷贝字节计数和原生完成 ACK；前缀 CUDA 生成命中分别为 512 / 4,096 / 16,384。每种形状与相同输入的冷跑生成 8 tokens，IDs 一致，所比生成 logprob 最大绝对差为 0。准备租约未释放和 copy in-flight 均拒绝 reclaim。

物理拷贝三档重复试验：每档、每臂 5 次预热 + 20 次测量，表为 CUDA event 的 P50／P95（ms），包含 Full 和 Mamba，不是 HTTP 延迟。

| 前缀 tokens | 每方向物理 bytes | D2H 原生 | D2H Triton | H2D 原生 | H2D Triton |
|---:|---:|---:|---:|---:|---:|
| 512 | 68,288,512 | 3.621 / 4.159 | 2.550 / 2.767 | 2.099 / 2.145 | 22.397 / 23.422 |
| 4,096 | 185,729,024 | 20.352 / 23.031 | 5.716 / 6.393 | 4.447 / 4.461 | 34.894 / 36.016 |
| 16,384 | 588,382,208 | 77.362 / 80.629 | 16.748 / 18.234 | 12.284 / 12.321 | 44.337 / 45.215 |

三档均是 Triton D2H 更快、原生 H2D 更快；不能把 D2H 优势外推到前台恢复。证据：`copy-native-triton-calibration.json` 与对应 `logs/*-wire.jsonl`。首轮试次 `hybrid-{native,triton}-io` 包含首次调用开销，原生末尾的 `mha` 字段误写已用原始 `full`/`mamba` 计数重验，不计入重复结果。

据此实现并实机复测 `AGENTIC_TRITON_COPY=d2h`：仅 D2H 使用 Triton，H2D 保留原生。新配对每档每臂仍是 5 次预热 + 20 次测量。16,384-token 档，D2H P50 **76.855 → 16.954 ms**，H2D **12.308 → 12.354 ms**；逐样本两个方向的物理耗时之和，P50 **89.214 → 29.301 ms（−67.2%）**。512 和 4,096-token 的来回耗时 P50 分别为 **5.795 → 4.690 ms**、**24.715 → 10.340 ms**。这不是端到端请求加速。`hybrid-directional-io.json` 核实两池确实分别走 Triton D2H、原生 H2D，三个前缀均恢复，所比 8-token greedy IDs 一致、生成 logprob 最大差 0；`copy-directional-calibration.json` 与 `directional-evidence.sha256` 保存重复结果和指纹。

**严格冷数值门禁未过。** 另测三长度 × 三种子，前缀预热但不迁移，与迁移后按相同分段做 teacher forcing：原生拷贝、Triton 各 **9/9 同形通过**，迁移前后所比 logprob 的最大差都是 **0**，另比 greedy IDs 一致。两臂相对整段冷 prefill 均 **0/9**，最大 logprob 差 **2.2612**；迁移未引入该差，不能把它归因于 copy，更不能用同形通过代替冷门禁。`hybrid-warm-control-{native,triton}.json` 分列两个判据，`reference-cold-forced.json` 保存冷包络。保持默认原生拷贝；数值不等价的臂不进入正式策略性能结论。

最新短 batch 复测 `cancel-gpu-native-retired.json`：两条不同分支均已输出，再取消指定 `rid`；受害者收到 `canceled`，幸存者生成的全部 8 个 IDs 等于其冷参考。该证据证明同一次原生 batch HTTP 的独立取消，不声称两条处于同一个设备 batch。16,384-token D2H cancel/reclaim 检查同次也通过。
另一个实测将 16,384-token 原生 D2H 的 submit/cancel/reclaim 放在同一控制批。提交后拒绝提前 reclaim；物理 588,382,208 bytes ACK 到达后才产生 `canceled`，未 demote，源状态仍匹配 16,384 tokens，8 个生成 IDs 和所比 logprob 等于冷跑。`logs/cancel-gpu-native-wire.jsonl` 内 `canceled-transfer-verified`。同次长 256-token 合批只取消指定 `rid`、另一条实际完成 256 tokens；它与单条冷跑从零基第 13 位置起出现 greedy 分叉，**不能**把长序列数值当成已通过，也不能据此归因于取消（batch shape 不同）。

## DSH 闭环证据

- 原生 SDK 直连：`dsh-bench/bench-2026-11-19T12-07-55-644Z-48ce15db-0216-46ed-8bda-3dbaa5bc76ac/report.json` 的 smoke 通过。闭环含两次 `fixture_read`、`fixture_sum`、一次实际 marker 写入、持久结果和最终 `42`。
- 受控 SDK：`dsh-bench/bench-2026-11-19T12-06-39-003Z-c7999b1a-b507-490d-af51-99930d14e922/parallel/task.result.json` 两次读同一步、第二项先物理完成、持久结果仍按声明序，写入与最终答案正确。`logs/dsh-controlled-e1-v2-ingress.log` 有前台 `rid` 的物理前缀命中 0 / 1,246 / 1,306 / 1,353。配对 serial 当次多发了两个非法参数的工具调用；该对子不算 E1 性能证据。
- 实测修复：Cordis 在配置表达式求值前就解析插件名称；动态 `!!js` 名称没有挂载 fixture，且替换 compaction 因名称不匹配被跳过。[runner](../agenticlayer/dsh-agentic/src/runner.ts) 写私有 overlay，插入绝对模块路径并显式禁用原 compactor。读写 fixture 与委派 fixture 按套件隔离；marker 的工具契约用整数入参并生成实际换行，避免模型两次转义。

固定参数复测 E1 已通过：真实读延迟 1,000/650 ms；串行先完成 A 再启动 B，并行 B 先完成但持久提交仍是 A、B。读批时间跨度 **1,861.45 → 1,008.86 ms（−45.8%）**，两臂同工具、同参数、同 `42` 输出、同 marker。单次配对，不是整个 agent 回合的加速比例。证据 `dsh-bench/bench-2026-11-19T12-22-17-031Z-a899e8bd-e05d-49ef-856e-7eb9dd51092c/report.json`。

## 轨迹画像（不执行匿名工具）

`trace-profile.json` 校验全部 16 周、119 个压缩文件：12,002 根会话、85,295 根轮次、1,222,199 个模型请求。18,674 根轮次按固定投影上下文门禁排除。首四周冻结了 64 个校准投影轮次，每层 16；`trace-calibration-smoke.manifest.json` 记录 SHA-256 与每层分母。投影 token 并非原任务内容，不能用于任务正确率。

末四周同样冻结 64 个留出轮次，每层 16；`trace-heldout-smoke.manifest.json` 的投影 SHA-256 为 `10db5b2078fdd861256842013dff6871fc08785298b6c5628dc9fb849f828264`。

其中按确定规则每层取一个短轮次（各轮总 decode 不超过 512），`native-radix` 与受控组都完成 4/4 轮、10 个相同模型请求、6 个因果工具屏障，各生成 1,264 tokens。保存 `heldout-gpu-connection.json`、逐臂 `*-timeline.jsonl` 与样本 hash。没有为这个 smoke 开启预测，不用 4 轮的分位数判断策略收益；它不是留出主测。

## 被等待窗口隐藏的预填充：实测有效

固定输入 7,065 tokens，800 ms 相同外生等待，每策略 20 个 on/off 配对，交替顺序。配对之间清 GPU 缓存，每次用新 namespace；同一冻结的校准文件包含 5 次预热、20 次冷 prefill 及真实等待后的缓存检查。两个策略内 20/20 提前准备都完成，每个 on 请求实际复用 7,064 tokens，所有 off 为 0；80 次最终输出均严格为 `42`。这是同一受控执行器上的机制消融，不是 DSH 压缩 E3 或生产流量。

| 策略 | 前台 P50 off → on | 前台 P95 off → on | 含等待 P50 off → on | 含等待 P95 off → on | 提前组更快 |
|---|---:|---:|---:|---:|---:|
| `strong` | 522.13 → 353.97 ms | 876.69 → 577.89 ms | 1,323.12 → 1,160.13 ms | 1,677.66 → 1,384.31 ms | 19/20 |
| `joint` | 522.51 → 311.30 ms | 846.56 → 858.02 ms | 1,323.48 → 1,117.32 ms | 1,647.52 → 1,664.16 ms | 18/20 |

在固定窗口，`strong` 前台 P50 降 **32.2%**、P95 降 **34.1%**；含等待 P50 降 **12.3%**、P95 降 **17.5%**。`joint` 前台 P50 降 **40.4%**，但 P95 微升 **1.4%**。只有独立准备动作时，不能归因于跨动作仲裁，也不能由两组 P50 差推导 `joint` 优于 `strong`。

对同次实验的 20 对样本做探索性配对 bootstrap（10,000 次、seed `20261118`），前台 P95 的 on−off 95% 区间：`strong` **[−596.7, −91.0] ms**，`joint` **[−394.1, +428.1] ms**。这不覆盖跨次启动或生产流量方差。数据 `prepare-window-bootstrap.json`。

同一固定输入直连 `native-radix`（20 次、无提前准备）前台 P50/P95 为 **383.71 / 389.32 ms**，含等待 **1,184.73 / 1,190.25 ms**。受控提前组两策略在本组观测的 P50 都低于直连，但 P95 都更高；不建议把 ingress 当作无条件提速层。该原生组单独重启，未与每次受控样本交叉轮换，因此仅作同工作量上下文对照。证据 `prepare-native-window.json`。

关键修复见 [server.py](../agenticlayer/agentic_layer/server.py)：完整目标 token 用来验证 cutover，只预填 `ids[:-1]`，使 Full 与 Mamba 在原生生成真正能匹配的边界留 checkpoint。可选提案先记事件再异步做 GPU 准备；低预测样本或已到切换窗口都不阻塞事件序列。一次实测在准备尚未物理结束时收到新前台，暴露 `branch already has an active request` 503；现在显式取消旧准备并等待原生终态后再 bind 新请求。这个等待算进实测延迟。

短等待复测使用同一 800 ms 校准但强制 100 ms 提前返：6/6 配对正确，均无候选命中、无 503 或超时，其中 5 次已运行的准备收到原生取消终态。再次失败的原因是 chunked prefill 可通过 `RequestRetirer.abort()` 退出而不进入 batch-result hook；[插件](../agenticlayer/agentic_layer/sglang_plugin.py) 现在从原生实际退休后记终态，不再按调度列表猜测；正常 socket cancel 不再发重复 HTTP abort。证据 `prepare-joint-early-return-retired.json` 和同名 ingress 日志。100 ms 漂移场景证明正确回冷，不宣称提速。

原始数据：`prepare-{strong,joint}-window.json`，`logs/prepare-{strong,joint}-window-{pairs.jsonl,ingress.log}`，冻结校准 `prepare-joint-paired-fixed-calibration.json`。保留早期拒绝和 503 的失败日志 `logs/prepare-joint-paired{,-fixed}-*`。

## 同 tick 竞争：联合排序确实改变执行

注入同一个已确认依赖的两个候选：7,065-token 目标的提前准备，与另一分支的真实短 GPU 生成准入。两者在同一 `mono_ns` 的选择中均合法，一个 `selected`、一个 `not_selected`。`strong` 5/5 先准入；`joint` 5/5 先准备。两个真实模型答案都过门禁（`42` / `7`）。

| 策略 | 目标前台 P50 | 短生成 P50 | 从依赖提交到两者结束 P50 |
|---|---:|---:|---:|
| `strong` | 406.87 ms | 409.11 ms | 1,211.54 ms |
| `joint` | 313.95 ms | 374.82 ms | 1,119.46 ms |

观察到总耗时 P50 降约 **7.6%**。这只覆盖可同时执行的两类动作且每组 5 次；短分支估计在每臂单独用实机采样，未加入第三个恢复动作。不能当成完整 E5、高压力效果或生产 P95 置信区间。证据：`e5-two-action-{strong,joint}.json` 及同名 ingress 日志。

## 当前使用边界

- 固定等待、分词目标可冻结且同一版本能够切换时，优先验证 `strong` 的早准备；受控尾延迟不能用来替代与原生直连的比较。没有已冻结预测或没有足够等待时走常规前台。已实测的并行收益只用于已证实独立的工具。
- `joint` 已通过注入竞争的选择检验，未通过真实负载的普适收益验证。不默认开启双向 Triton；`d2h` 是已复测的定向实验配置，正式性能结论仍受严格冷数值门禁约束。
- 此次没有完成 E2 的并发全扫描、DSH E3 晚切换闭环、E4 全部 fork 谱系、E5 三类动作竞争、E6 显存/带宽压力矩阵或 5 次完整留出主测。这些不计为通过；报告也不把回放当成任务质量或线上提速。
