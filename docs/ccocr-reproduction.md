# IJDAR 2026：公开 CC-OCR 复现

目标论文：[Hallucination-aware Evaluation and Confidence-guided Selective Routing for Hybrid VLM-OCR](https://doi.org/10.1007/s10032-026-00620-7)。作者复现包尚未取得。本页只描述外部 CC-OCR 的数据、模型与运行入口，五系统结果见 [当前结果](ccocr-results.md)；不把缺失的 SHC 实验或现有合成 A/B 基线算作复现结果。

## 冻结协议

- 900 张原图：TotalText 300、InverseText 500、Hieragent 100；全部为外部 test，不拆训练集、不拟合 scaler、不调阈值。ConfRoute 本身无需训练；学习型路由需要缺失的 SHC 开发集，暂不训练。
- 数据来自 [CC-OCR 官方项目](https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/fb6187ebe5374d5682ac0bc05f31e3639ab6fb15/Benchmarks/CC-OCR) 指向的公开数据，使用 [wulipc/CC-OCR](https://huggingface.co/datasets/wulipc/CC-OCR/tree/c64517e92179991d509776064174776700cdd5a2) 的 TSV。版本、授权说明、逐文件 SHA-256 见 `configs/ccocr_public.json`。原图许可仍受原始来源条款约束。
- 已提取 manifest SHA-256：`22149779ff2f61d41a12c322a16e2ae90802693c5406d8639810f91f8cc59ad9`。下载器拒绝覆盖不同清单，逐图保存内容哈希与原始参考文本。
- 外部 ConfRoute：仅对非空识别文本的置信度取算术均值，四舍五入到四位小数；无有效文本为 null，比较时视为 0；严格小于 0.90 才选 VLM。不混用 SHC 的最小框置信度规则。
- CER：折叠/去除首尾空白，保留大小写和标点；逐图编辑距离除以 `max(1, GT字符数)`，截断到 1，再作 macro 平均。这里不执行旧基线的 NFC 归一化。
- Eval-Trans：遵循固定官方提交的英文词袋匹配 F1，包括重复词计数、大小写与标点处理；保留上游 [MIT 许可](licenses/CC-OCR-MIT.txt)。报告 pooled、各子集和等权 source macro，避免混淆汇总口径。

## 模型和运行差异

模型修订、许可证、资源计划在 `configs/ccocr_models.json`，下载后保存每个文件的哈希与校验状态。下载必须显式加 `--allow-download`；推理仅从完整校验的本地目录读取，关闭模型联网检查及 Hugging Face 自动下载。

- PaddleOCR 3.4.0，PP-OCRv5 server det/rec；关闭文档方向分类、图像矫正和行方向模型。作者未公开精确权重哈希，官方权重只是冻结的复现候选。
- Qwen3-VL-8B-Instruct 官方 BF16 权重，greedy、batch=1、seed=42。使用补充材料 P0 文本提取提示；`max_new_tokens=2048` 为本项目冻结选择，作者值未公开；记录触顶情况。
- 论文列出的 Transformers 4.46 早于 Qwen3-VL 支持。本项目使用 Transformers 4.57.6、直接 Torch/SDPA 推理，不宣称重建了作者的 vLLM 环境。Paddle CPU 使用 SDK 默认的 MKLDNN 优化后端和识别 batch=6；作者未给出对应 CPU 设置。每实例 `cpu_threads=1`，避免与 Torch 混用时的 MKL 动态库错误。原图尺寸及检测参数保持 SDK 默认值。
- 独立真实模型镜像基于本机缓存的 NVIDIA PyTorch 25.12（解析摘要 `sha256:b5541f591b87e68d3883040bfdbf89707bd6f80e2e38b95411c9ff3e1cb63eeb`），Python 3.12、Torch `2.10.0a0+b4e4ee8`；依赖使用 `requirements.ccocr.lock` 的精确版本和哈希。实际镜像 ID 通过 `CCOCR_IMAGE_ID` 写入运行元数据。基础 tag 若发生变化须重新核对摘要。
- 默认真实实验仍为 CPU，8 核、64 GiB 内存；不向容器暴露 GPU、Docker socket、SSH 或宿主凭据。Qwen 权重约 17.5 GB。GPU 需先获得 CPU-only 约束的明确例外并重新检查资源；不能干扰 GPU 2 上既有训练。
- 延迟包括单张预处理、推理及解码，不包括模型加载；无 warmup，因此保留首张开销。OCR 可用 `--workers 2` 并行采集，每批最多 8 张后重建进程以释放原生推理缓存；资源竞争及反复冷启动会影响耗时。先采集全部 OCR/VLM，再反事实回放 ConfRoute 的调用及串行耗时，并非在线调度实测。CPU/不同 GPU 的时间不可直接对照论文 RTX 4090。

## 命令

所有命令在项目根目录执行；数据、权重、逐图预测、运行输出均由 Git 忽略。

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose run --rm -T dev python -m scripts.prepare_ccocr
docker compose run --rm -T dev python -m scripts.download_ccocr_models --allow-download --weight-mirror modelscope
docker compose -f compose.yaml -f compose.ccocr.yaml build dev
export CCOCR_IMAGE_ID=$(docker image inspect degradation-aware-routing-ocr:ccocr --format '{{.Id}}')
docker compose -f compose.yaml -f compose.ccocr.yaml run --rm -T -e CCOCR_IMAGE_ID dev python -m scripts.collect_ccocr --mode ocr --per-subset 1 --output outputs/ccocr-ocr-pilot
docker compose -f compose.yaml -f compose.ccocr.yaml run --rm -T -e CCOCR_IMAGE_ID dev python -m scripts.collect_ccocr --mode ocr --workers 2 --output outputs/ccocr-ocr-mkldnn-full
docker compose run --rm -T dev python -m src.ccocr --observations outputs/ccocr-ocr-mkldnn-full/observations.jsonl --output outputs/ccocr-ocr-mkldnn-full-eval
```

完整模型下载校验后可在 CPU 用 `--mode both` 采集两种模型。中断使用完全相同命令并加 `--resume`；配置、模型、数据、依赖或 Git 提交变化时拒绝混合续跑。`--per-subset` 只做运行检查，评估报告会标记非完整 900 张，不能用它声称论文结果。

2026-10-24 用户已明确授权本次公开 CC-OCR 实验仅使用宿主 GPU 1；授权写入 `AGENTS.md` 和独立的 `configs/ccocr_models_gpu.json`。原 CPU 登记文件保持不变。`compose.ccocr-gpu.yaml` 只分配宿主 GPU 1，容器内为 `cuda:0`；与 `compose.ccocr.yaml` 叠加使用。授权绑定物理 GPU UUID，加载模型前检查仅一张可见卡、UUID 一致、无其他计算进程、至少 40 GiB 空闲显存；不满足就退出，不停止其他任务。

授权后的自动流水线：等全部权重完成哈希校验及完整 OCR 观测就绪，先跑三个子集各 1 张 GPU pilot，再跑全量 Qwen 和 ConfRoute 回放及评估；任何阶段失败即停止。等待最长 6 小时，等待时不加载模型。复用原 CPU OCR 的文本、置信度和耗时，并按 ID、图像哈希、GT 和原图组逐项校验，不重复运行 OCR。CPU/GPU 混合耗时不可直接对照论文 GPU 基线。

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
export CCOCR_IMAGE_ID=$(docker image inspect degradation-aware-routing-ocr:ccocr --format '{{.Id}}')
docker compose -f compose.yaml -f compose.ccocr.yaml -f compose.ccocr-gpu.yaml run --rm -T -e CCOCR_IMAGE_ID dev python -m scripts.run_ccocr_gpu
```

日志为本地 `outputs/ccocr-gpu-workflow.log`，观测和评估目录分别为 `outputs/ccocr-gpu-{pilot,full}/`、`outputs/ccocr-gpu-{pilot,full}-eval/`。重复执行拒绝覆盖旧结果；中断的采集可用 `scripts.collect_ccocr` 同一配置加 `--resume`，完成后再评估。

不得为接近论文分数而在这 900 张上选择权重、提示或阈值。作者包缺失意味着不能保证严格复现，必须报告差异。

## CPU 就绪检查

已有结果不要重新运行覆盖。需要重新检查输入时，以下命令核验冻结清单、900 张原图、OCR 缓存、模型文件和离线处理器，且不初始化 CUDA；`ready: true` 不代表 GPU 推理通过。

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
export CCOCR_IMAGE_ID=$(docker image inspect degradation-aware-routing-ocr:ccocr --format '{{.Id}}')
docker compose -f compose.yaml -f compose.ccocr.yaml run --rm -T -e CCOCR_IMAGE_ID dev python -m scripts.preflight_ccocr --output outputs/ccocr-ready-recheck.json
```

## 补充模型与 GPU Paddle

InternVL / DeepSeek 使用 `configs/ccocr_additional_gpu.json` 与独立的 `compose.internvl.yaml` / `compose.deepseek.yaml`。采集入口 `scripts.collect_additional --model internvl|deepseek --per-subset 1 --output 新目录`；pilot 验证后，`--per-subset 0` 才是 900 张全量。必须使用对应模型镜像，并叠加 `compose.ccocr-gpu.yaml`，在 GPU 1 空闲时顺序执行。

GPU Paddle 使用 `configs/ccocr_paddle_gpu.json` 与 `compose.paddle-gpu.yaml`；入口 `scripts.collect_paddle_gpu`，同样先 pilot 再全量，复用完整 Qwen 观测并逐图验证身份。采集期间不修改其登记或源码，不并行争用 GPU。

五系统完成后的汇总命令（输出文件须不存在）：

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose run --rm -T dev python -m scripts.compare_external \
  --qwen outputs/ccocr-gpu-full/observations.jsonl \
  --internvl outputs/ccocr-internvl-full/observations.jsonl \
  --deepseek outputs/ccocr-deepseek-full/observations.jsonl \
  --output outputs/ccocr-five-system-report.json
```

此命令使用历史 CPU Paddle 配对缓存；当前 GPU 主对照须将 `--qwen` 改为 `outputs/ccocr-paddle-gpu-full/observations.jsonl`，并指定新的输出文件。两套冻结报告均已生成，勿覆盖。旧运行诊断见 [历史协议](archive/ccocr-protocol-through-2026-10-28.md)。
