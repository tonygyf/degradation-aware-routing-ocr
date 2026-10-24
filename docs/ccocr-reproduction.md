# IJDAR 2026：公开 CC-OCR 复现

目标论文：[Hallucination-aware Evaluation and Confidence-guided Selective Routing for Hybrid VLM-OCR](https://doi.org/10.1007/s10032-026-00620-7)。作者复现包尚不可得，用户选择先推进公开 CC-OCR。这里实现外部评估 PaddleOCR、Qwen3-VL-8B、ConfRoute(0.90)；不把缺失的 SHC 实验或现有合成 A/B 基线算作复现结果。

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
- 论文列出的 Transformers 4.46 早于 Qwen3-VL 支持。本项目使用 Transformers 4.57.6、直接 Torch/SDPA 推理，不宣称重建了作者的 vLLM 环境。Paddle CPU 禁用 MKLDNN，并显式设置每个实例 `cpu_threads=1`，避免与 Torch 混用时的 MKL 动态库错误。
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
docker compose -f compose.yaml -f compose.ccocr.yaml run --rm -T -e CCOCR_IMAGE_ID dev python -m scripts.collect_ccocr --mode ocr --workers 2 --output outputs/ccocr-ocr-full-bounded
docker compose run --rm -T dev python -m src.ccocr --observations outputs/ccocr-ocr-full/observations.jsonl --output outputs/ccocr-ocr-full-eval
```

完整模型下载校验后可在 CPU 用 `--mode both` 采集两种模型。中断使用完全相同命令并加 `--resume`；配置、模型、数据、依赖或 Git 提交变化时拒绝混合续跑。`--per-subset` 只做运行检查，评估报告会标记非完整 900 张，不能用它声称论文结果。

`compose.ccocr-gpu.yaml` 是待授权的具体配置：只分配宿主 GPU 1，容器内为 `cuda:0`；与 `compose.ccocr.yaml` 叠加使用。登记文件 `gpu_authorized=false` 时，即使误带 GPU 配置，采集脚本仍拒绝 GPU 推理。不能仅修改布尔值来推断用户同意。

论文表 13 对照值仅作为目标：

| 方法 | Eval-Trans | CER | 空输出率 | VLM 调用率 |
| --- | ---: | ---: | ---: | ---: |
| PaddleOCR | 0.3386 | 0.7015 | 0.0711 | 0 |
| Qwen3-VL-8B | 0.7680 | 0.4536 | 0 | 1 |
| ConfRoute(0.90) | 0.7044 | 0.5018 | 0 | 0.6833 |

不得为接近目标而在这 900 张上选择权重、提示或阈值。作者包缺失意味着无法保证 1:1 或分数接近；必须报告真实差异。

## 2026-10-24 运行验证

900 张数据及 Paddle det/rec 权重均已完成哈希校验。真实 CPU pilot 在三个子集各取第一张，串行与 3 个进程采集的文本、置信度完全相同；3 张仅验证流程，不能据此判断论文分数是否复现。Qwen 运行库导入通过，权重尚在显式断点下载。新增下载续传、模型身份与哈希、GPU 授权前置检查、空输出中断续跑及指标边界测试。

8 进程全量试跑在完成 60 张后触发本项目 64 GiB 容器内存上限，完整日志和已采集观测保留本地。改为最多 2 个进程、每批 8 张重建。MKLDNN 打开时首张文本相同，但不混用该探测的观测；正式采集仍使用关闭设置。ModelScope 官方模型镜像可显式选用以避开 HF 下载慢的问题，最终仍必须匹配已冻结的 HF 文件 SHA-256；镜像 master 发生变化会被拒绝。Eval-Trans 已对照固定上游源码，逐图计算完全相同，汇总也保留官方分母 `+1e-9`。
