# AgenticLayer

面向智能体工作负载的跨层协同服务与 KV 缓存管理：DeepSeek Harness (DSH) → AgenticLayer → SGLang 三层机制，以及 RTX 5090 + Qwen3.5-9B NVFP4 上的对照实验与回执。

## 目录

| 路径 | 内容 |
|---|---|
| `agenticlayer/agentic_layer/` | 中间层：`protocol.py`（线协议）、`server.py`（ingress/版本 CAS）、`engine_client.py`、`versions.py`、`policy.py`（强组合/联合策略）、`engine_cache.py` + `sglang_plugin.py`（SGLang 插件、混合 Radix、HBM↔DRAM）、`kernels.py`（Triton copy hooks） |
| `agenticlayer/dsh-agentic/` | DSH 插件：确认调用、版本原语、并行、`delegate_all` 委派、工具结果缓存、compaction overlay |
| `agenticlayer/DESIGN.md`、`RUNBOOK.md` | 机制设计与部署/执行手册 |
| `experiments/` | 实验脚本、合同、报告，以及论文式 HTML 展示 |
| `experiments/results/` | 四个 run 的紧凑回执（summary/paired/report JSON、CSV、图表）；不含日志与回放语料 |
| `main.pdf` | 第三至五章的设计合同（生命周期、版本转移、跨层联合控制） |

## 打开论文展示

`experiments/agenticlayer-paper.html` 是自包含页面：图表内嵌 SVG，中文字体为本地 `experiments/paper-assets/paper-han.woff2`（Noto Serif SC 子集，SIL OFL，许可证随附），离线可读。

```bash
xdg-open experiments/agenticlayer-paper.html   # 或直接在浏览器打开该文件
```

章节顺序：摘要 → 动机与背景 → 技术路径 → 对比与实验设置 → 主实验 → 敏感性分析 → 消融实验 → 案例研究 → 讨论与结论 → 资料与复现。

## 实验结果

`experiments/results/<run_id>/` 保留了可复核的紧凑产物，原始 run 目录（15 GB+，含日志、venv、模型缓存、回放语料）留在实验机的 `/root/autodl-tmp/agentic-layer-runs/<run_id>`，未纳入仓库。

| run | 用途 |
|---|---|
| `5090-nvfp4-01` | 原生基线、数值参考、HiCache、工具闭环、准备窗口、Triton 迁移 |
| `freeinference-paper-01` | FreeInference 留出工作集四臂回放（论文指标映射） |
| `freeinference-ablation-01` | decode Graph / recurrent checkpoint 交叉对照与状态消融 |
| `freeinference-fixedgraph-01` | 固定 `decode=full`、`prefill=disabled` 的 E1–E6 机制验证 |

复现入口见 `experiments/README.md` 与 `agenticlayer/RUNBOOK.md`。

## 说明

- `deepseek-harness/` 与 `sglang` 是指向固定 release/SHA 的上游树，按 `RUNBOOK.md` 在 run 目录内单独取用，不随本仓库分发。
- 报告口径：配对 P95 差与 cluster bootstrap 95% CI；CI 含 0 即记为无明确增益，不把回放加速说成任务质量提升。
