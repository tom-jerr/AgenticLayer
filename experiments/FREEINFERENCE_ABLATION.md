# FreeInference：GPU 优化与独立机制消融

[上轮真实轨迹与论文图表](FREEINFERENCE_PAPER.md) · [本轮原始产物](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01) · [锁定 RTX 5090 环境](GPU_5090_RESULTS.md)

## 实验约束

- 轨迹来自 `/root/autodl-tmp/release` 的留出周。每层选 3 个完整根轮次，固定同一份清单。每臂 12 轮、43 个 GPU 请求、29 个因果工具屏障、656,370 个投影 prompt tokens、5,100 个输出 tokens。不执行匿名工具；投影输出仅用于对比，不用于评价原任务质量。
- 引擎做 2×2：decode eager / CUDA Graph full × Mamba checkpoint L2 / L3。保持权重、预填充后端、2048-token chunk、Full 和 Mamba 容量以及请求预算相同。Graph 只捕获解码，实际日志必须出现 graph decode。L3 多保留可复用 recurrent checkpoint；Full-only 命中不当成预填充节省。
- 轮次 P50/P95 为 nearest-rank；12 轮的 P95/P99 都是最慢轮次。四格一次各测一个完整 replay，配对区间按根会话重采样，只是探索性区间，不衡量不同重启的波动。`decode phase` 是原生 prefill-finish 到 request-finish 的墙钟跨度，不是独立 kernel 时间。
- 固定长度、请求 ID 和 greedy 配置，但只要完整输出 token 不一致，就不把该对照当成满足逐 token 等价的最终配置。短的隔离数值 fixture 与完整并发轨迹单独判断；在匿名投影上出现数值漂移不等于已证明真实任务质量退化，也不等于已证明可以上线。
- 真实轨迹与具体机制 fixture 分开。工具并行和结果缓存使用真实 DSH SDK 和真实工具；提前准备使用已知且哈希不变的 chat surface；生命周期和放置看原生 allocator/ACK，不用虚拟 slot 或 DRAM 延迟。机制收益的计时边界不同，不相加。

## 同一真实留出集：两项优化彼此独立

| 同样 12 轮、43 请求 | 活跃 P50 / P95 (s) | TTFT 中位数 (ms) | Decode phase 中位数 (s) | Full+Mamba 联合命中 | 完整输出与 eager/L2 一致 |
|---|---:|---:|---:|---:|---:|
| Eager / L2 | 29.794 / 59.748 | 724.2 | 6.713 | 0/43 | 基准 |
| Eager / L3 | 24.008 / 50.989 | 116.3 | 6.231 | 24/43 | 18/43 |
| Graph / L2 | 13.063 / 31.813 | 796.7 | 1.722 | 0/43 | 25/43 |
| Graph / L3 | 9.032 / 21.698 | 187.2 | 1.275 | 24/43 | 15/43 |

- **只开 Graph（L2 不变）：**P50 −56.2%，P95 −46.8%；P95 差的会话重采样区间 `[−27.93, −23.87] s`。没有新增缓存命中，主要收益来自解码。L3 下单开 Graph 的 P95 也下降 29.29 s。
- **只加密 checkpoint（eager 不变）：**P95 −14.7%，24 个请求复用 373,414 tokens，占总投影 prompt 的 56.9%。Graph 下单开 L3 同样复用这些 tokens、P95 下降 10.12 s。原生采样 Mamba 峰值从 5.10% 升到 10.19%；未在压力下测得净效应。
- **两者组合：**P50 −69.7%，P95 −63.7%；相同工作量的整批回放 110.28 → 41.15 s。不是逐 token 等价的替换：隔离 fixture 的 512/4K/16K greedy 与所记录的生成 logprobs 四臂完全一致，32K 验活也通过，但同 L2 的 Graph vs eager 在并发回放中只有 25/43 个完整序列一致（首 token 40/43）。对照的两次 eager/L2 自然波动也有 3/43 个完整序列变化，但不足以为 Graph 或 L3 解释全部差异。默认仍是 eager/L2。
- Graph/L2 的 3 个首 token 差异发生在 [预填充提交首 token](../sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py#L466-L488) 时，不应全部归咎于 Graph 解码算子的数值；改变解码速度也会改变后续请求到达及连续批处理的组成。
- 固定 batch 把两项隔离：保持 L2、冷命名空间和批形状，batch/prompt 分别为 `1×512`、`2×4096`、`4×1024`、`4×4096`、`8×16384`，每请求生成 64 tokens。同形状日志实际进入 Graph，19/19 个请求的完整 greedy 序列、1,216 个生成位置的选中 token logprobs 全相同（最大差 `0`）；每形状测一次的请求墙钟为原来的 `1/1.35` 至 `1/6.10`。[固定 batch 回执](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/graph-fixed-batch.json) 支持在这些形状上使用 Graph/L2，不替代动态批次门禁。

[柱图](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/figures/01_engine_ablation.svg) · [完成轮次折线](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/figures/02_ablation_completion.svg) · [调度 min / median / max](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/figures/03_ablation_queue_spread.svg) · [四臂 CSV](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/figures/arms.csv) · [单因素差值和区间](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/figures/effects.csv)

## 配对烟测：checkpoint 的独立作用

| 同样 4 轮、10 请求 | 原生命中请求 | 实际复用 prompt tokens | TTFT 中位数 | 活跃 P95 |
|---|---:|---:|---:|---:|
| Eager / L2 | 0/10 | 0 | 1,123.0 ms | 34.901 s |
| Eager / L3 | 5/10 | 100,892 | 596.5 ms | 31.730 s |

两臂各 189,548 个 prompt / 1,264 个输出 tokens。命中 5 项的对照 TTFT 中位数 1,336.4 → 125.0 ms。**完整 greedy 只相同 5/10**：命中项 1/5，无命中项 4/5，不能仅凭吞吐把 L3 升为默认。[烟测汇总 CSV](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/smoke-figures/arms.csv)

## 已独立验证的机制基线

| 机制 | 配对控制与结果 | 结论边界 |
|---|---|---|
| 并行派发两项独立工具 | 同一任务、交错串行/并行顺序，4 对；真实读阶段 medians 1,698.5 → 1,010.2 ms（−40.5%）。四个并行组都观察到后声明的短读先完成。 | 8/8 都按声明顺序提交工具结果并产生正确的 `42`；计时仅为两次读覆盖的墙钟跨度，不是全轮。 |
| 工具结果缓存 | 成功配对中，串行模式重复读 305.9 → 6.2 ms、并行模式 306.7 → 5.5 ms。各模式独立比较 on/off；实际调用由 4 次降为 3 次，冷读、改数据、改权限均 miss。 | 证明缓存不依赖并行也能省一次 I/O；只量工具读阶段。两个配对来自各自完整通过的同次运行对照，并非一组四臂同时通过。 |
| 精确 surface 提前预填充 | 同 `strong`、800 ms 窗口，新跑 12 对；前台 P50 607.9 → 316.0 ms（−48.0%），含等待 P50 1,408.9 → 1,122.0 ms（−20.4%）。12/12 返回正确 `42` 且提前臂命中 7,064 tokens。之前 20 对分别为 −32.2% / −12.3%。 | 已知输入和足够等待时有作用；不代表 DSH E3 提前压缩，也不把匿名轨迹后继 prompt 当作已知。 |
| 过期生命周期回收 | 保留或回收两条互相独立的已结束分支；每形状 5 对，4K 回收 8,192 Full slots + 2 个 Mamba slots（354.25 MiB），16K 回收 32,768 + 2（1,122.25 MiB）。 | 前向输出完全一致；下一个无关前向的原生中位耗时基本不变。释放的是池内可用容量，不是进程持有的 VRAM；没有把低压回收当成延迟收益。 |
| 放置：常驻 / DRAM / 冷重算 | 每形状 5 对。16K 常驻、恢复+生成、冷重算分别 433.2 / 473.0 / 1,075.9 ms，迁移期间腾出 561.125 MiB HBM；含 D2H 的总墙钟 P50 为 585.0 ms。4K 三者为 412.3 / 445.1 / 551.2 ms，腾出 177.125 MiB。 | 放置不是无条件提速：有 HBM 则常驻最快；必须让出时，恢复比本 fixture 的冷重算快。此固定 seed 的 10/10 个三元组 8-token greedy 和选中 token logprobs 相同；不是替代更强的冷数值门禁。 |
| 两动作联合仲裁 | 同一合法 admit / prepare 竞争，5 对；makespan P50 1,211.5 → 1,119.5 ms（−7.6%）。 | 有真实竞争时可见；非三动作 E5。上轮真实轨迹只有 `admit`，`joint−strong` P95 区间跨 0。 |
| 单独更换 D2H 内核 | 4K/16K 每形状每臂 20 个设备 ACK，H2D 保持原生；双向物理耗时 P50 分别 24.715 → 10.339 ms、89.214 → 29.301 ms。 | 独立加速物理搬运；不等于整轮提速。逐 token 强数值门禁：同形状暖未迁移 vs 暖迁移 9/9，通过；对不可复用的冷路径 0/9，不能用暖路径结果代替。 |

基线原文件位于 [`5090-nvfp4-01`](/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01)：`prepare-strong-window.json`、`e5-two-action-{strong,joint}.json`、`copy-directional-calibration.json` 和冷/暖数值回执。

工具并行的 [四对 SDK 与 ingress 回执](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/dsh-bench/bench-2026-11-19T15-35-18-711Z-4838b509-83dc-4c3f-a7e8-a5c58e507b60/report.json) 保留原 `callId`。缓存实验首次使用相同对话时模型直接复述旧回答，未发出第二次读取，不能当成缓存命中；改用同 SDK 进程、相同 workspace／权限但新对话历史触发物理读。

[并行缓存 on/off 成功对照](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/dsh-bench/bench-2026-11-19T15-44-22-126Z-295674b1-fdf5-49c8-a8f3-aef00ad9a4de/report.json) 和 [串行缓存 on/off 成功对照](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/dsh-bench/bench-2026-11-19T15-48-18-333Z-23cea09c-1c9f-48ef-86e0-56a840edf116/report.json) 保留失败条件：一组有模型把引号放进文件名，另一组模型漏传 `path` 后重发，违反一次调用的门禁。无一整组通过四臂 E2，不能据此报告交互效应或完整 E2 通过；不修改工具路径校验，也不把非法调用过滤后计为成功。

[提前准备 12 对回执](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/lookahead-recheck.json) · [生命周期 allocator 回执](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/state-ablation.json)

放置首次 `offload` 超时，尚未进入计时样本。在新隔离引擎仅延长不计时 warmup 的 RPC 等待，实测首次 D2H 提交 ACK 用时 6.60 s；之后恢复 2 s 原控制期限，5+5 组全部完成，无测量回退。[放置物理 ACK 与独立条件](/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-01/state-placement-recheck.json)

## 复现

使用 [执行手册](../agenticlayer/RUNBOOK.md) 准备的私有 SSD 运行目录，保留 `src/sglang`、环境和缓存。命令逐条执行，不并发占用同一 RTX 5090。

```bash
BASE=/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01
export RUN=/root/autodl-tmp/agentic-layer-runs/freeinference-ablation-rerun
export PYTHONPATH="$PWD/agenticlayer" PYTHONDONTWRITEBYTECODE=1 AGENTIC_TRITON_COPY=0
PY="$RUN/env/python/bin/python"
"$PY" -m experiments.ablate_trace_gpu --run-dir "$RUN" --sample "$BASE/trace-heldout-smoke.jsonl" \
  --name engine-factorial --configs graph-l3 graph-l2 eager-l2 eager-l3
python -m experiments.plot_ablation --factorial "$RUN/engine-factorial.json" --out "$RUN/figures"
"$PY" -m experiments.state_ablation_gpu --run-dir "$RUN" --name state-ablation
"$PY" -m experiments.prepare_gpu --run-dir "$RUN" --name lookahead --policy strong --pairs 12 \
  --calibration "$BASE/prepare-joint-paired-fixed-calibration.json"
```

工具组使用同一个 `extended` 引擎和 `strong` 控制器，运行 `node experiments/dsh_bench.mjs --mode controlled --run-dir "$RUN" --events "$RUN/logs/$INGRESS_LOG" --suite e1 --repeats 4`，再用 `--suite e2`（不加 `--repeats`）。按 [工具闭环证据](GPU_5090_RESULTS.md) 设置本地 token 和锁定 runner / DSH 路径；E2 在串行和并行配置内各自做缓存 on/off，并保留数据版本、权限变动两个必 miss 对照。

源码：[启动参数](launch_engine.sh)、[四格驱动](ablate_trace_gpu.py)、[GPU 时间线复算](plot_ablation.py)、[真实工具驱动](dsh_bench.mjs)、[生命周期与放置驱动](state_ablation_gpu.py)、[物理控制回执](../agenticlayer/agentic_layer/engine_cache.py)。
