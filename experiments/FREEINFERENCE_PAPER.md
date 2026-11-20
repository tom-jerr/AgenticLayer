# FreeInference 真实轨迹 GPU 回放与论文图表

论文：[main.pdf](../main.pdf)。运行目录：[`freeinference-paper-01`](/root/autodl-tmp/agentic-layer-runs/freeinference-paper-01)。真实输入是 `/root/autodl-tmp/release/<week>/traces/*.jsonl.zst`，不使用历史数据集中不同 GPU 的计时来填本轮实验结果。

## 论文口径 → 实测字段

| 论文位置 | 图表/数据 | 可解释的边界 |
|---|---|---|
| §2.1、§3.6.2、§5.6.2（PDF 第 20、31、47 页） | `01_active_turn_latency`：所有已提交根轮次的 P50/P95/P99，nearest rank；失败、取消、超时、不完整均计为未完成，不把成功子集 P95 当总体 P95。`04_gpu_completion_progress`：自首个投影到达起的实际 GPU 回放累计完成数。 | 活跃轮次从外生输入就绪到轮次完成，包含前驱等待、模型和回放工具屏障；不计此前用户空闲。不把各阶段有重叠的耗时直接相加。 |
| §2.5、§3.6.1（PDF 第 22–23、30 页） | `02_weekly_workload`：全部 16 周的根会话、根轮次、模型请求、工具调用，按周折线（对数纵轴）。 | 产自全量轨迹画像，不是新 GPU 回放的观测延迟；周流量不可直接视为模型负载变化。 |
| §3.6.2、§5.4.4（PDF 第 31、46 页） | `03_scheduling_min_median_max`：真实 SGLang `queue_time`；受控模型许可 RTT、工具屏障许可 RTT；有选中动作时控制器组建候选和执行选择的 CPU 墙钟跨度。横线为 min–max，实点为 median。 | 四项不能相加。许可 RTT 含本地 HTTP；CPU 跨度不是引擎排队。原生不经过 AgenticLayer，相关栏目显示 n/a 而不是零。 |
| §4.3.3、§4.6.2（PDF 第 36、40 页） | `summary.csv` / `summary.json`：TTFT，原生 forward-entry 至 prefill-finish，prefill-finish 至 request-finish，缓存命中 token 比例、未缓存 prompt tokens、约 1 Hz 的 SGLang Full/Mamba 容量占比和主机槽位、可用时的传输计数。 | API 时间戳不是独立 GPU kernel 时间；`uncached_prompt_tokens` 是重算工作量代理，不能精确识别重复预填充；采样峰值不是唯一物理分配的精确峰值，未测 byte-seconds、真实工具服务/汇合暴露时间、摘要质量。 |
| §5.5、§5.6.1、§5.6.4（PDF 第 46–48 页） | `native-radix`、`native-hicache`、带相同校准文件的 `strong` 与 `joint`；CSV 保留各统计量的观测分母。 | 固定单卡、模型、每臂请求与生成长度。每臂一次运行不能替代多次配对；只报告本工作集，不声称生产泛化。 |

`main.pdf` 中引用的其他论文数字和假设算例都未入图。FreeInference 空间里是匿名 block，不具备原始 prompt、工具命令、权限或真实结果。锁定 tokenizer 将每个已发布的 16-token block 映射为两个普通 token，保留经投影的共享前缀；工具用记录等待和子分支因果回放，**不执行原始工具，也不测原任务正确性**。选择按固定 hash、每层 3 个完整根轮次；根候选的记录 decode 预算不超过 768 个 token，保留样本清单要求的前驱，未截断选中轮次。末四周只测试，首四周只校准；原生与受控没有外生后继强行定时投递。

`workload.json` 冻结了 12 个根轮次的 hash，来自末四周 12 个不同根会话。按输入投影每臂预期 **43 个模型请求、29 个工具屏障、656,370 个 prompt tokens、5,100 个生成 tokens**，最大 prompt 27,240 tokens。全量画像包含 119 个压缩源、12,002 根会话、85,295 根轮次、1,222,199 模型请求、1,275,252 工具调用。

配置沿用 [5090 实测](GPU_5090_RESULTS.md) 的锁定模型、SGLang、原生双向拷贝；每臂重启独立引擎。HiCache 组同为本机 DRAM。GPU 转移成本来自同机实测 4,096 / 16,384 单请求桶；首四周的有效、无子分支的工具屏障建立等待分布。没有已测的压力分层驱逐概率，不补虚拟样本，依赖该数据的可选后台动作按策略关闭。这个矩阵不是论文的 E5/E6 全校准主测，先前 [混合 KV 的严格冷数值门禁](GPU_5090_RESULTS.md#物理混合-kv原生拷贝与-triton) 也仍不通过。

## 结果与图表

RTX 5090 四臂均完成 **12/12** 根轮次，核对每臂 **43** 个相同模型请求、**29** 个相同因果工具屏障和 **5,100** 个生成 tokens。首四周校准含 **1,011** 个有效等待样本。P95 的分母未排除任何已提交轮次；这个 `n=12` 下，nearest-rank P95 和 P99 都是最慢轮次。

| 策略 | 活跃轮次 P50 (s) | P95 (s) | 原生排队 min / median / max (ms，n=43) |
|---|---:|---:|---:|
| `native-radix` | 29.986 | 60.542 | 0.506 / 0.732 / 1,347.912 |
| `native-hicache` | 30.340 | 61.090 | 0.595 / 0.815 / 1,445.739 |
| `strong` | 30.917 | 61.658 | 0.655 / 0.954 / 1,377.329 |
| `joint` | 31.082 | 62.348 | 0.635 / 0.970 / 1,598.986 |

`joint − strong` 的 P95 差为 **+0.690 s**，12 个会话配对 bootstrap（10,000 次，seed `20261118`）的探索性 95% 区间为 **[−0.795, +0.690] s**。区间跨 0，**该工作集没有测出联合策略的明确收益**。模型许可 RTT median 是 `strong` **7.09 ms**、`joint` **11.83 ms**；控制器候选和决策 CPU 墙钟 median 分别 **0.373 ms** 和 **0.378 ms**。两者不是同一计时边界，不能把许可差全部归因于决策计算。

四臂原生 `cached_tokens` 均报 **0**。受控两臂各有 **25/43** 个请求存在正的 Full-only 匹配，但物理回执中的 Full+Mamba 联合匹配均为 0。HiCache 三臂实际采到 D2H 写入约 **0.63–0.69 GB**，未见可报告的 H2D 正拷贝量。受控日志只选出 `admit`，未观察到 `prepare/offload/restore` 的联合仲裁。因此不能用这组分位数判断有可复用 Mamba 状态或三个动作竞争时的收益。

图表在 [`figures/`](/root/autodl-tmp/agentic-layer-runs/freeinference-paper-01/figures)，每图有 PNG/SVG：

- [活跃轮次 P50/P95/P99 柱状图](/root/autodl-tmp/agentic-layer-runs/freeinference-paper-01/figures/01_active_turn_latency.svg)
- [16 周负载统计折线图](/root/autodl-tmp/agentic-layer-runs/freeinference-paper-01/figures/02_weekly_workload.svg)
- [调度 min/median/max 区间图](/root/autodl-tmp/agentic-layer-runs/freeinference-paper-01/figures/03_scheduling_min_median_max.svg)
- [GPU 回放累计完成数折线图](/root/autodl-tmp/agentic-layer-runs/freeinference-paper-01/figures/04_gpu_completion_progress.svg)

[汇总 CSV](/root/autodl-tmp/agentic-layer-runs/freeinference-paper-01/figures/summary.csv) · [汇总 JSON](/root/autodl-tmp/agentic-layer-runs/freeinference-paper-01/figures/summary.json) · [`workload.json`](/root/autodl-tmp/agentic-layer-runs/freeinference-paper-01/workload.json)

## 复现命令

从项目根执行。`RUN` 必须是新的私有 SSD 路径，包含准备好的 `env/`、`src/sglang/`、`cache/`、`logs/` 与 `tmp/`，见 [执行手册](../agenticlayer/RUNBOOK.md)。同卡不要并发启动两个引擎。

```bash
BASE=/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01
export RUN=/root/autodl-tmp/agentic-layer-runs/freeinference-paper-01
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PWD/agenticlayer" AGENTIC_TRITON_COPY=0
PY="$RUN/env/python/bin/python"
SAMPLE="$BASE/trace-heldout-smoke.jsonl"

"$PY" -m experiments.trace_calibration \
  --copy "$BASE/copy-directional-calibration.json" \
  --sample "$BASE/trace-calibration-smoke.jsonl" --out "$RUN/calibration.json"
for pair in native controlled; do
  options=()
  if [[ "$pair" == native ]]; then arms=(native-radix native-hicache)
  else arms=(strong joint); options=(--calibration "$RUN/calibration.json"); fi
  "$PY" -m experiments.trace_matrix --run-dir "$RUN" --name "paper-$pair" \
    --sample "$SAMPLE" --turns-per-stratum 3 --max-turn-decode 768 \
    --arms "${arms[@]}" "${options[@]}"
done

# 绘图使用带 Matplotlib 的 Python，独立于 GPU 环境。
python -m experiments.plot_freeinference --run-dir "$RUN" \
  --matrix "$RUN/paper-native.json" --matrix "$RUN/paper-controlled.json" \
  --profile "$BASE/trace-profile.json" --out "$RUN/figures"
```

每臂的原生字段、调度轨迹和 GPU 模型输出在 `logs/paper-*-timeline.jsonl`；受控侧策略决策在同名 `*-ingress.log`。图表从上述原始文件复算，PNG 便于预览，SVG 用于论文排版。
