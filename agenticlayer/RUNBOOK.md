# 执行手册：固定版本、本地双端口

从项目根用 Bash 执行。源码地图见 [设计文档](DESIGN.md)，运行顺序、门禁和实机报告入口见 [实验文档](../experiments/README.md)。

| 项目 | 固定配置 |
|---|---|
| 模型 | `AxionML/Qwen3.5-9B-NVFP4` snapshot `97aef92393f126bf649f310cd40861be8dad3279`，ModelOpt NVFP4 权重、BF16 推理/KV、FP32 Mamba SSM |
| SGLang | `703e763d5f3af75ea3587275d79155c28d3c5d4f`，由 [sglang/](../sglang/) 导航；Python 3.12、Torch 2.14.1、FlashInfer cu13 0.7.4、Python TreeCore |
| DSH | `dsh-v0.2.2-rc.5` / `9c2670d3bce5275e0d1610272958121af87787bb`，Node 26.13.1、pnpm 11.28.5、完整 `sdk` profile |
| 设备 | 单 RTX 5090；当前运行实测 driver `580.95.05`（原计划 `580.105.08`）；SSD 上 CUDA 13.0 nvcc `V13.0.88`，不更新宿主驱动 |

## 1. 复制源码、隔离依赖

每次使用新 `RUN`，不在项目根、SGLang 软链接目标或现有 DSH clone 上构建。假设 `uv` 和指定版本的 Node/pnpm 已可用，`CUDA_HOME` 已指向隔离的 CUDA 13.0.2 toolkit。

```bash
set -euo pipefail
export PROJECT="$(pwd -P)"
export RUN=/root/autodl-tmp/agentic-layer-runs/5090-nvfp4-01
export MODEL=/root/.cache/huggingface/hub/models--AxionML--Qwen3.5-9B-NVFP4/snapshots/97aef92393f126bf649f310cd40861be8dad3279
DSH_REV=9c2670d3bce5275e0d1610272958121af87787bb
test -L "$PROJECT/sglang"
test -f "$MODEL/model.safetensors"
mkdir -p "$(dirname "$RUN")"
mkdir "$RUN"
mkdir -p "$RUN/src/dsh" "$RUN/src/sglang" "$RUN/env" "$RUN/tmp" \
  "$RUN/cache"/{uv,triton,flashinfer,cuda,sglang}
git -C "$PROJECT/deepseek-harness" archive "$DSH_REV" > "$RUN/dsh.tar"
tar -xf "$RUN/dsh.tar" -C "$RUN/src/dsh"
cp -a "$PROJECT/sglang/." "$RUN/src/sglang/"
cp -a "$PROJECT/agenticlayer" "$RUN/src/agenticlayer"
sha256sum "$RUN/dsh.tar" \
  "$(dirname "$(readlink -f "$PROJECT/sglang")")/sglang.tar.gz" > "$RUN/source-archives.sha256"
(
  cd "$RUN/src"
  find dsh sglang agenticlayer -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$RUN/sources.sha256"
(
  cd "$MODEL"
  sha256sum config.json hf_quant_config.json model.safetensors tokenizer.json tokenizer_config.json \
    chat_template.jinja generation_config.json vocab.json
) > "$RUN/model.sha256"

export UV_CACHE_DIR="$RUN/cache/uv" TMPDIR="$RUN/tmp"
export TRITON_CACHE_DIR="$RUN/cache/triton" FLASHINFER_WORKSPACE_DIR="$RUN/cache/flashinfer"
export CUDA_CACHE_PATH="$RUN/cache/cuda"
export PATH="${CUDA_HOME:?set CUDA_HOME to the pinned toolkit}/bin:$PATH"
nvcc --version
uv venv --python 3.12 "$RUN/env/python"
export PYTHON="$RUN/env/python/bin/python"
SGLANG_BUILD_RUST_EXTS=none SETUPTOOLS_SCM_PRETEND_VERSION=0.5.23.dev0+g703e763d \
  UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=2 \
  uv pip install --python "$PYTHON" "$RUN/src/sglang/python"
uv pip install --python "$PYTHON" "$RUN/src/agenticlayer"
uv pip freeze --python "$PYTHON" > "$RUN/python.lock.txt"
```

在正式运行 manifest 中比较归档与副本的哈希、固定 SHA、依赖和模型指纹；单有校验和文件不是来源认证。固定依赖安装、`sm_120` 初始化或 HiCache 失败时保留原始日志，不用宿主 Torch 2.7、其他模型或其他注意力后端替代。

## 2. 启动 SGLang（30080）

设置 `ENGINE_ARM=native-radix|reference-cold|native-hicache|extended|extended-hicache` 后执行。一次仅启动一个引擎，确认 `30080`/`30081` 未被占用；不要结束非本 run 的进程。为每臂单独保存命令、stdout/stderr 与退出状态。

```bash
ENGINE_ARM=native-radix bash "$PROJECT/experiments/launch_engine.sh"
```

`reference-cold` 只作数值对照；HiCache 使用实例私有 DRAM，无 L3。原生臂不挂插件，不要将 `SGLANG_PLUGINS` 设为空串（会启用插件发现）。扩展臂不要开启 `--sleep-on-idle`。

`AGENTIC_FULL_TOKENS`、`AGENTIC_MAMBA_SLOTS` 和 `AGENTIC_HOST_GB` 只为受控容量对照设置；同一比较内三个限制必须一致，报告实际分配容量而不是仅报告启动参数。

默认 `AGENTIC_DECODE_GRAPH=disabled`、`AGENTIC_CHECKPOINT_LEVEL=2`。`AGENTIC_DECODE_GRAPH=full` 只捕获 batch 1–8 的解码，预填充仍用原后端；`AGENTIC_CHECKPOINT_LEVEL=3` 减少混合前缀的重算、增加 Mamba 状态占用。启用任一项后保留逐请求 token 门禁，不能把 [固定 batch 完全一致](../experiments/FREEINFERENCE_ABLATION.md) 外推为动态批次逐 token 等价。

按论文做同桶比较时同时设置 `AGENTIC_FIXED_GRAPH=1 AGENTIC_DECODE_GRAPH=full`；Python 驱动拒绝不一致的 `/server_info` 或实际捕获日志。不要将这个标志用于原先的 eager / graph 交叉对照。

逻辑预算仍是 32768 tokens。固定版 [`tp_worker.py`](../sglang/python/sglang/srt/managers/tp_worker.py) 对配置上限扣 6 tokens，scheduler 把输入上限当 exclusive；直接设 `--context-length 32768` 在实机拒绝了 32767+1。所有臂统一用 32774 补足该保留量，并验证 `/server_info` 的 `max_req_input_len=32768` 及真实 32767+1。Ingress 与回放仍显式限制总 token 数 ≤32768；不打开自动截断。

`AGENTIC_TRITON_COPY=0`（默认）双向原生拷贝，`1` 双向 Triton，`d2h` 仅 HBM→DRAM 用 Triton、DRAM→HBM 用原生。[传输 hook](agentic_layer/kernels.py) 不改变 attention 后端；方向在启动时冻结。映射、bitwise 或实模型混合恢复失败时，所选 Triton 方向失败，不算成原生结果。保存 `/health_generate`、`/server_info`、`/v1/models` 和启动日志；HTTP 健康不等于模型配置已核对或插件已接线。

## 3. 启动受控 ingress（仅扩展组，30081）

用 [server.py](agentic_layer/server.py) 启动，且 Engine/Ingress 使用同一个绝对 socket 路径。以下启动方式同时打开 `agentic_layer` 的 INFO 决策及回执日志；可用 `tee` 写入本 run。Bearer 与受控 DSH 共享，不写入日志或结果文件。

```bash
export AGENTIC_ENGINE_SOCKET="$RUN/engine.sock"
export AGENTIC_LOCAL_TOKEN
AGENTIC_LOCAL_TOKEN="${AGENTIC_LOCAL_TOKEN:-$("$PYTHON" -c 'import secrets; print(secrets.token_hex(32))')}"
export AGENTIC_TOKENIZER_HASH
AGENTIC_TOKENIZER_HASH="$(cd "$MODEL" && sha256sum tokenizer.json tokenizer_config.json chat_template.jinja | sha256sum | cut -d' ' -f1)"
export AGENTIC_LAYOUT_HASH
AGENTIC_LAYOUT_HASH="$(printf '%s\n' 'tp=1;page=1;attn=triton;kv=bf16;ssm=fp32;mamba=no_buffer;positions=model-config' \
  "$(sha256sum < "$MODEL/config.json")" | sha256sum | cut -d' ' -f1)"
export AGENTIC_POLICY="${AGENTIC_POLICY:-strong}"
"$PYTHON" -c 'import logging; logging.basicConfig(level=logging.INFO); from agentic_layer.server import main; main()'
```

哈希必须与该 run 的实际模板文件、位置/池布局和 dtype 一致；配置变化另开 run。`AGENTIC_BACKEND` 默认 `http://127.0.0.1:30080`。主对照只改 `AGENTIC_POLICY=strong|joint`；正式配对给两者传入同一冻结的 `AGENTIC_CALIBRATION`。schema 见 [policy.py](agentic_layer/policy.py)。没有校准文件时不启动可选预测；这不构成联合策略对比。

同一仲裁器的留一消融可在 ingress 设置 `AGENTIC_DISABLE_MECHANISMS`（空格分隔：`parallel lifecycle placement lookahead prefetch prediction`）；关掉的是可选机制，不关闭必要前台、权限与引用校验。回放矩阵用对应 `joint-minus-*` 臂，避免残留环境变量污染其他条件。

在另一终端用同一个 token 核验 `curl -fsS -H "Authorization: Bearer $AGENTIC_LOCAL_TOKEN" http://127.0.0.1:30081/health`。必须返回 engine `epoch` 和 `bind/prepare/lease/release/completion_paging` 能力；L2 臂还应有 `offload/restore`。HTTP 健康后 2 s 内没有 scheduler 握手则失败。丢失控制后关闭该 run 的自有进程再重启，不把旧 epoch socket 或请求当成新运行。

## 4. 构建 DSH overlay 并做真实 SDK 冒烟

从已解包的 DSH 固定版构建。插件编译在 SSD 的 AgenticLayer 副本中，通过链接解析同一个 release 的依赖，而不是为插件安装另一套 DSH。

```bash
export DSH="$RUN/src/dsh"
export PLUGIN="$RUN/src/agenticlayer/dsh-agentic"
(cd "$DSH" && pnpm install --frozen-lockfile && \
  DSH_CLIENT_COMMIT_HASH=9c2670d3bce5275e0d1610272958121af87787bb pnpm run build)
mkdir -p "$PLUGIN/node_modules/@deepseek-ai" "$PLUGIN/node_modules/@types"
while read -r name path; do
  ln -s "$DSH/$path" "$PLUGIN/node_modules/@deepseek-ai/$name"
done <<'PEERS'
cordis vendor/cordis
dsh-agent packages/core/agent
dsh-session packages/core/session
dsh-tools packages/core/tools
dsh-compaction packages/compaction/compaction
dsh-compaction-basic packages/compaction/compaction-basic
dsh-llm packages/llm/llm
dsh-llm-pi-ai packages/llm/llm-pi-ai
dsh-sdk-client packages/sdk/client
dsh-subagent packages/subagent/subagent
PEERS
ln -s "$DSH/node_modules/@types/node" "$PLUGIN/node_modules/@types/node"
(cd "$DSH" && pnpm exec tsc -p "$PLUGIN/tsconfig.json")
```

构建产物必须有 `lib/index.js`、`lib/compaction.js`、`lib/runner.js`，CLI 入口是 `apps/cli/lib/bin.js`。如果 registry 备选路径被使用，必须校验同版制品的 registry integrity，并满足 [runner.ts](dsh-agentic/src/runner.ts) 的 runtime `realpath` 一致性检查；不能用裸 HTTP runner 代替 DSH。

此例运行一次真实 SDK 工具往返。原生阶段用 `DSH_MODE=native`（直连 30080）；扩展阶段同一任务用 `DSH_MODE=controlled`，与 ingress 共享 token。每次使用新的 fixture 工作区和 `dshHome`。

```bash
node experiments/dsh_bench.mjs --run-dir "$RUN" --suite smoke --mode "${DSH_MODE:-native}"
```

门禁同时检查原生 `tool/call` 和持久 `tool/result`、声明顺序、最终答案 `42` 与 marker 内容 `42` 加实际换行。SDK 工具调用和 SGLang 回执必须按实际 `rid` 关联。受控 E1/E2 用 `--mode controlled --suite e1|e2 --events <ingress 日志>`，先按 §3 启动并在当前环境导出 Bearer。
