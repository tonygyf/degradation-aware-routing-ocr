> 历史记录，保留当时结论；当前进度见 [复现状态](../paper-reproduction-status.md)。

# 全论文复现状态与验收边界（2026-10-28）

**结论：整篇论文尚未复现完成，未通过科研指标验收。**
当前已完成 CC-OCR 的 PaddleOCR、Qwen 与 ConfRoute 三系统 900 张真实结果；环境就绪、模型下载、CPU 初始化、pytest/冒烟通过均不能替代模型指标验收。

目标：[IJDAR 论文正文](https://link.springer.com/article/10.1007/s10032-026-00620-7)。

## 全文实验范围与本地证据

| 实验范围 | 原文要求 | 当前状态 | 验收结论 |
|---|---|---|---|
| 内部文档级 Baseline | 579 文档的 Support/Dev/Cal/Test，Test 116 张；模型与 ConfRoute 比较 | 未取得作者对应角色清单和完整实验产物 | 未复现、未验收 |
| SHC 主实验 | 5,550 样本，Test 3,897；双参考、critical span、失败类型；七主系统和三诊断系统 | 缺原作者样本映射、裁剪/字段重建信息、双参考和关键片段；无对应真实评估产物 | 未复现、未验收 |
| 提示、风险覆盖与学习路由 | 指定开发角色、特征、提示、校准程序和统计分析 | 通用 A/B/MLP 与风险原型不等同于论文实现，只有软件检查 | 未复现、未验收 |
| 外部 CC-OCR | 900 张，PaddleOCR/Qwen/DeepSeek-OCR-2/InternVL3.5-8B/ConfRoute | Paddle、Qwen 与 ConfRoute 全量结果已完成；DeepSeek/InternVL 权重续传中，尚无真实结果 | 部分执行、未验收 |
| 补充统计与复现审计 | 重复诊断、配对/聚类 bootstrap、重叠审计等 | 仅有部分独立数据审计；原文完整统计流程未完成 | 未复现、未验收 |

作者官网正文仍保留匿名仓库占位和发布说明；本项目目前没有可用于严格重建 SHC 与内部 Baseline 的作者复现包。
这只描述当前已取得的材料，不断言其他渠道不存在发布。下载六个原始公开数据集，也不能自动替代作者特定的裁剪、角色划分与人工标注。

## 唯一已有真实结果：CC-OCR PaddleOCR

本地证据：`outputs/ccocr-ocr-mkldnn-full-eval/metrics.json`，900 张完整观测。

| 指标 | 论文表 13 | 本地结果 | 差异（本地减论文） |
|---|---:|---:|---:|
| Eval-Trans，越高越好 | 0.3386 | 0.3223317554 | -0.0162682446 |
| clipped macro-CER，越低越好 | 70.15% | 70.92674804% | +0.77674804 个百分点 |
| 空输出率 | 7.11% | 6.88888889% | -0.22111111 个百分点 |

空输出更少不自动意味着转录更准确。CPU 平均约 6.008 秒/张，不能直接与论文 GPU 延迟判等。
没有事先约定的数值等价容差，也没有完整比较结果与置信区间，因此不能凭“看起来接近”判定过关。

## 执行优先级纠正

先把论文实验矩阵、数据/模型身份和缺失作者材料逐项对齐；补齐可执行的指标与真实基线，再报告差异和无法验证项。
复现状态仅使用“未执行 / 部分执行 / 已有结果待验收 / 验收通过 / 缺材料无法验证”，不以测试数量代替指标状态。
严格复现缺失时，独立重建实验必须单独命名，不能填入原论文复现表冒充同一实验。

退化风险原型、新 Total-Text 开发集和其参考文本初审属于另行登记的扩展准备，保留现有成果，暂不继续扩展算法或声称科研收益。
新算法分支仍需从明确验收的基线提交起步；当前尚无这样的科研冻结点。

## 2026-10-28：按论文补齐的软件实现

已增加独立实现 `src/evaluation/shc.py`：空白归一化、逐样本截断的双参考 macro-CER、critical-span flag、S6 空图输出判定、语义占比及各失败类型汇总。关键片段必须来自预先标注，不能从预测反推；拒绝纯标点片段。没有 critical span 时用 JSON null。

`src/routing/confroute.py` 要求显式指定 `image_mean`（内部 Baseline／CC-OCR）或 `box_min`（SHC 单独缓存），历史阈值默认 0.90，严格小于时升级。SHC 无框置信度为 0，不回退到文档级 scalar。

独立观测交换格式每行一个系统的一个样本，字段为 `id`、`failure_type`、`reference_visible`、`reference_full`、`critical_span`、`prediction`。这不是声称掌握了作者文件格式。评估命令：

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose run --rm dev python -m scripts.evaluate_shc \
  outputs/shc/observations.jsonl --output outputs/shc/metrics.json
```

输入需由真实、已审计的标注和模型观测准备；此路径仅为使用示例，目前无完整 SHC 真实观测。输出保留输入 SHA256，并明确 `author_dataset_identity_verified=false`，防止将公式软件检查误写为原论文数据复现完成。

继续按论文题名、作者与复现脚本名检索，尚未取得可核验的作者仓库。仍缺内部角色清单、SHC 映射与人工标注、完整模型结果、统计分析；科研验收状态保持未通过。

本次 Docker CPU 验证：43 tests passed；`ruff check .`、`ruff format --check .`、`python -m scripts.smoke` 均通过。日志：`outputs/paper-core-checks-20261028.log`。新增 SHC 冒烟为合成数据，仅验证软件行为。

## GPU 实验与统计实现（2026-10-28）

用户授权使用一张空闲 GPU 继续实验。现场检查 GPU 1 无计算进程、仅 14 MiB 显存占用后，启动独立容器 `hybrid-vlm-ocr-gpu1-20261028`，限制 8 CPU / 64 GiB RAM，仅映射原授权 GPU 1 UUID。三张真实 pilot 已完成并评估，全量 900 张采集已启动；完成状态以 `outputs/ccocr-gpu-full/observations.jsonl` 和评估产物为准，不能把运行中计为完成。日志为 `outputs/ccocr-gpu-workflow-20261028.log`。

新增 `scripts.analyze_ccocr`：按补充表 25 对 ConfRoute−Qwen、ConfRoute−Paddle 的 Eval-Trans、CER、平均延迟、调用率执行 10,000 次按子集配对 bootstrap（seed 42），报告百分位 95% CI；同时报告连续三块 2–20 token n-gram 重复诊断及生成触顶数。pilot 的真实观测已成功执行该入口，统计仅用于流程验证。

新增 SHC `--cluster-bootstrap`：需要真实审计的 `family_id` 与 `source_image`，两种方案分别全局均匀重采样簇并纳入所有后代，计算单模型 operational flag rate。与 CC-OCR 分层配对方案不同。缺作者标注时仅测试算法，不生成冒充 SHC 的真实结果。

独立实现使用 NumPy default_rng/PCG64 和百分位区间；作者 RNG 实现未取得，不能保证同 seed 逐次抽样或输出字节一致。重复诊断按空白切词、保留大小写标点；作者 tokenizer 源码未取得，该解释明确记录在结果中。

本轮检查：48 tests passed，Ruff、格式和冒烟均通过。日志 `outputs/paper-statistics-checks-20261028.log`。全量统计输出预定为 `outputs/ccocr-gpu-full-statistics.json`；CPU 后处理容器等待完整观测，最长一小时，未获得完整文件则报错退出。

## SHC 路由回放及补充模型准备（2026-10-28）

`src/evaluation/shc_routing.py` / `scripts.replay_shc` 增加 Paddle、Qwen、ConfRoute 与 NoText+ConfRoute 的同样本回放。后者只检查 OCR 归一化输出是否为空，不读取 S6 标签决定是否跳过 VLM，因此保留论文指出的非 S6 误过滤现象。阈值固定 0.90、信号为独立框置信度最小值；SHC 平均延迟按补充材料 `mean(OCR) + call_rate * mean(VLM)` 估计，明确不是在线实测。输入沿用双参考字段，增加 `role=Test`、`ocr`（text/latency_ms/confidence_boxes）、`vlm`（text/latency_ms）。作者真实样本仍未取得，只有软件验证。

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose run --rm dev python -m scripts.replay_shc \
  --observations outputs/shc/paired.jsonl --output outputs/shc/routing.json
```

补充模型登记在 `configs/ccocr_additional_models.json`：官方 InternVL3.5-8B 与 DeepSeek-OCR-2，冻结 revision、Apache-2.0 许可、数据授权、磁盘和推理资源估计；当前仅显式下载权重和准备代码审查，尚未就绪或推理。它们是原论文比较对象，不是新算法扩展。

下载容器 `hybrid-vlm-ocr-model-download-20261028`，日志 `outputs/additional-model-download-20261028.log`。首次下载发现 DeepSeek 模型卡图片经镜像返回无效范围，已修正下载器只对 safetensors/pdiparams 使用权重镜像；正在运行的旧下载进程不会热替换该逻辑，须确认其终止后再续传，不能并行写同一权重文件。

固定 revision 的 10 个自定义 Python 源文件已单独下载并核验 Git blob/SHA256，保存在 `outputs/model-code-review/`，尚未执行。初审发现 DeepSeek `infer(eval_mode=True)` 固定 max_new_tokens=8192/no_repeat_ngram_size=35，非 eval 分支为 20；InternVL 有自身 chat 模板与系统提示。这些默认行为和依赖必须显式记录、核对论文后再接入，不能假定等同于当前 Qwen 适配器。源码下载和关键路径检查不表示已完成兼容性或安全审查。

本轮验证：51 tests passed；Ruff、格式、冒烟全部通过。日志 `outputs/shc-routing-checks-20261028.log`。本轮未形成新的全量科研指标。

## 独立模型运行环境与置信度差异（2026-10-28）

已构建 `compose.internvl.yaml` / `Dockerfile.internvl`：原冻结 ccocr 镜像加哈希锁定的 timm 1.0.20；已构建 `compose.deepseek.yaml` / `Dockerfile.deepseek`：独立安装官方 requirements 指定的 Transformers 4.46.3、tokenizers 0.20.3，以及 addict 2.4.0。没有修改正在运行的 Qwen 镜像/容器。镜像 ID、离线导入结果见 `outputs/additional-runtime-preflight-20261028.json`。两种模型类均能离线导入，不加载权重、不初始化 CUDA；这不是 GPU 推理通过。

`src/recognition/internvl_preprocess.py` 对齐固定官方模型 README 的动态切块、thumbnail、448 尺寸、ImageNet 归一化。200 组尺寸与 3 张真实 pilot 图像逐值交叉核验通过，记录 `outputs/internvl-preprocessing-crosscheck.json`；作者具体图像预处理配置仍未公开，不能从官方代码一致推导出论文实验完全一致。

**已确定的实际差异：当前 900 张 Paddle 缓存产生 667 次升级，调用率 74.1111%，高于论文 68.33% 约 5.78 个百分点。** 见 `outputs/frozen-confroute-budget-audit.json`。这表明置信度分布未对齐，需继续审计权重身份、预处理、推理后端与作者样本身份；保留冻结阈值 0.90，不用外部 Test 调阈值追分。

本轮验证 52 tests passed，Ruff、格式和冒烟通过；日志 `outputs/additional-runtime-checks-20261028.log`。权重下载仍进行中，两个补充模型尚未真实推理，所有整体科研验收状态保持未通过。

## 全量结果更新

Qwen 及 ConfRoute 的 900 张结果、10,000 次 bootstrap 已完成并交叉核验观测 SHA256。完整数值及局限见 [三系统全量结果](../ccocr-results.md)。调用率与延迟结论尚未对齐，未通过整体验收。

补充模型采集入口 `scripts.collect_additional` 已接通，具有权重/源码/数据校验、单卡空闲检查、逐图持久化与断点续跑；结果按真实模型名评估。56 项测试通过，Ruff/格式/冒烟通过。补充模型权重的首次下载已确认因连接中断退出，现用修复后的下载器续传，容器 `hybrid-vlm-ocr-model-download-retry-20261028`，日志 `outputs/additional-model-download-retry-20261028.log`。

## 补充模型与 GPU Paddle 进展（2026-10-28）

InternVL 与 DeepSeek 官方候选权重已完成固定 revision 文件哈希校验。InternVL 已在 GPU 1 完成三个子集各一张的真实 pilot，观测及指标位于 `outputs/ccocr-internvl-pilot/`；随后启动 900 张采集，容器 `hybrid-vlm-internvl-full-20261028`，日志 `outputs/internvl-full-20261028.log`。实际进度和终止状态以容器与日志为准。三个 pilot 样本不用于判定分数复现，作者的预处理及生成长度仍未验证。

为审计 CPU OCR 与论文 GPU OCR 之间的调用率和延迟差异，新增 `scripts.collect_paddle_gpu` 和 `configs/ccocr_paddle_gpu.json`。它使用相同冻结的官方候选检测/识别权重与关闭方向分类等设置，验证 GPU 空闲和 UUID，逐图采集 GPU Paddle，并按样本 ID 复用已经完整采集的 GPU Qwen 观测，允许中断续跑。运行应等当前 InternVL 容器结束后顺序进行，先三个子集各 1 张 pilot，复核输出后再跑全量；不能依照 Test 得分更改设置或阈值。

Paddle GPU 3.2.2 的 CUDA 12.9 官方 wheel 已取得，ZIP CRC 和包元信息通过，保存观察到的 SHA256；供应方未提供可核验的文件 SHA256，因此此哈希是首次下载的内容身份，不是发行方校验值。CUDA 依赖的版本和发行方哈希来自各包 PyPI 元信息。首次镜像构建中 cuDNN 下载反复截断失败，已改为独立容器分段续传、逐文件验证 SHA256；任务 `hybrid-vlm-paddle-deps-retry-20261028`，日志 `outputs/paddle-deps-download-retry.log`。依赖齐备后才重新构建 `compose.paddle-gpu.yaml` 镜像并进行 CPU 导入及 GPU pilot。

58 项测试通过，Ruff、格式与冒烟通过，日志 `outputs/paddle-gpu-collector-checks.log`。测试是软件证据，不能当成真实 GPU Paddle 指标。DeepSeek 尚待 GPU 1 空闲后单独 pilot，全论文依然未完成。

SHC 独立指标实现增加正文表 8 的归一化后相同/不同双参考分组汇总；59 项测试及 Ruff、格式、冒烟通过，日志 `outputs/shc-reference-overlap-checks.log`。源分组路由差异与空白框敏感性检查见 [CC-OCR 报告](../ccocr-results.md)。真实 SHC 标注仍缺。

## 五系统结果汇总入口

`scripts.compare_external` 在 Qwen、InternVL、DeepSeek 全部 900 张完成后，将 PaddleOCR、Qwen、ConfRoute 和另外两个模型的 pooled / 来源分组指标及重复诊断汇总。入口先验证固定 manifest 的 SHA256、精确 ID 集合、图像 hash、参考文本、原图组、数据集及模型身份；不完整数据会报错，不能生成正式汇总。生成后的报告还需与论文值逐项对照并保留配置差异。

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose run --rm dev python -m scripts.compare_external \
  --qwen outputs/ccocr-gpu-full/observations.jsonl \
  --internvl outputs/ccocr-internvl-full/observations.jsonl \
  --deepseek outputs/ccocr-deepseek-full/observations.jsonl \
  --output outputs/ccocr-five-system-report.json
```

上述路径在对应采集完成前不可用；目前 DeepSeek 尚未启动。该命令使用 Qwen 观测中保存的 CPU Paddle；如之后取得 GPU Paddle 全量观测，应另用其配对缓存独立重算，不覆盖现有报告。61 项测试、Ruff、格式与冒烟均通过，日志 `outputs/five-system-comparison-checks.log`。

## 用户交办：实验结束后整理

本轮复现实验结束后整理 README、复现状态、实验协议和历史记录：合并重复说明，标明历史快照并修正过时链接，给结果和运行命令建立清晰入口。先核对脚本引用与实验元数据，保留冻结数据、模型来源、逐图观测、日志和可追溯性；不移动当前运行所依赖的路径，不擅自删除证据。整理后验证文档链接与剩余命令。

## 内部 Baseline 配对统计入口

`src/evaluation/baseline.py` 与 `scripts.evaluate_baseline` 补充单图截断 macro-CER、空输出率、调用率、估计串行延迟、Qwen−Paddle 配对 bootstrap 95% 区间和双侧符号翻转检验。默认 bootstrap 10,000 次、seed 42；符号翻转采用本项目独立实现的 10,000 次、seed 42、plus-one p 值校正。论文未给出可核验的作者 RNG 或符号翻转细节，因此它是计算实现，不声称字节级等价。尚未实现作者的 bootstrap p 值约定。

入口需要真实 `split=test` 的配对 OCR/VLM 观测及参考文本；它记录输入 SHA256，116 条数量匹配只表示数量一致，报告显式标记作者数据身份未验证。`data/fixtures` 和合成样本不能作为科研结果。当前缺失作者角色清单及 116 张真实观测，仍没有内部 Baseline 结果。

本轮 63 项测试、Ruff、格式和冒烟通过；日志 `outputs/baseline-statistics-checks.log`。

## 内部原始数据准备

公开 SROIE 第三方转换镜像 `jsdnrs/ICDAR2019-SROIE` 已在 `configs/baseline_sources.json` 登记固定 revision、两个 parquet 的发布 SHA256、来源、资源和许可声明，使用显式下载入口 `scripts.download_baseline_sources --allow-download`。镜像维护者声明 CC-BY-4.0，并说明有原数据修订及 14 条额外 test 标注；原始比赛官网本次访问失败，原授权条款尚未独立核验。原始材料下载任务 `hybrid-vlm-sroie-source-20261028`，日志 `outputs/sroie-source-download.log`。

该镜像不是作者 579 张的角色清单，也不提供 SHC 双参考或关键片段；不建立论文 Test、训练/验证划分，不用于模型训练。原作者身份、重叠和转录规则仍需对照。该下载入口的 64 项测试和其他规定检查均通过，日志 `outputs/baseline-sources-checks.log`。

Paddle GPU 依赖分段下载现已全部通过发布 SHA256 核验，正在重建 `compose.paddle-gpu.yaml`，仍需 CPU 导入检查和 GPU pilot；正在运行的 InternVL 作业不受影响。

SROIE 两个镜像档案已下载并重新核验发布 SHA256；仍未解析成论文角色，也未确定作者样本身份。Paddle GPU 镜像已构建，使用 CUDA 驱动 stub 的无 GPU 导入检查通过：Paddle 3.2.2、PaddleOCR 3.4.0、CUDA 12.9；记录 `outputs/paddle-gpu-cpu-preflight.json`。stub 不执行 GPU 计算，尚不能证明真实 GPU 推理可用。

SROIE 镜像审计：987 条（镜像 train 626 / test 361），无重复 ID、无空图片、无区域文本与框数量不一致；图片字节重复的额外记录共 16 条，其中 8 个图片哈希跨镜像 train/test。论文提及的原始编码问题样本 `X51006619503` 在镜像 test 中，而论文将它列为内部 Dev 排除项。镜像角色与论文角色不能等同，且镜像若用于后续开发须先按原图组处理重复。与 CC-OCR 暂无完全相同的图片字节，但不同编码和近似图片尚未排除，不能据此声称视觉无重叠。审计 `outputs/sroie-reference-audit-v2.json`，代码与输入哈希均已记录；65 项测试和规定检查通过，日志 `outputs/sroie-audit-checks.log`。

进一步检查相同图片的标注：16 组中每组的区域文字列表、框列表均不完全相同；其中 8 组跨镜像 train/test。逐词文本多重集（保留大小写、标点及重复）也均不相同，说明差异不只是列表顺序。抽查一组图片和两组标注的前四个区域与图片相符；尚不能据此判定哪份完整标注更可靠。按原图去重时需保留映射并复核参考文本，不能仅随意选择一个 ID。明细保存在 `outputs/sroie-duplicate-groups-detailed.json`，汇总 `outputs/sroie-duplicate-word-audit.json`；检查不构成作者角色或完整参考文本重建。

已安排单卡顺序执行：监视当前 InternVL 容器终止，检查全量观测与指标均存在后，运行 DeepSeek 三图 pilot；仅在 pilot 正常产出且 GPU 1 再次通过空闲/UUID 检查时启动 DeepSeek 900 张。控制进程与日志位置见 `outputs/deepseek-scheduler.json`；任一阶段失败即停止，不重复覆盖现有结果。该安排不表示 DeepSeek 已通过真实推理。

DeepSeek 已通过模型文件和源码哈希核验、离线 `AutoConfig`/`AutoTokenizer` 以及 meta 设备结构实例化，未加载权重或初始化 CUDA；记录 `outputs/deepseek-meta-preflight.json`。CPU 预检采用 eager，仅验证接口与结构，实际单卡 pilot 采用登记的 FlashAttention 2，仍需真实推理验证。

DeepSeek 的现有加载路径不传 `device_map`，无需本镜像缺失的 accelerate。进一步在无 GPU 容器完成离线 `from_pretrained` 检查：2,707 个参数张量均在 CPU，missing/unexpected/mismatched/error 均为 0，见 `outputs/deepseek-loader-preflight-v2.json`。运行库提示 checkpoint model_type 与自定义类名称不一致，仍须真实 GPU pilot 验证；CPU 加载结果不证明 FlashAttention 或生成输出正确。

2026-10-29：InternVL 全量完成并核验，数值见 [外部结果报告](../ccocr-results.md)。旧顺序启动控制进程现已缺失，未留下 DeepSeek 产物；确认 GPU 1 空闲后已独立启动 DeepSeek 三图 pilot，并完成三子集观测与评估，随后启动 900 张全量容器 `hybrid-vlm-deepseek-full-20261029`。日志 `outputs/deepseek-full-20261029.log`；目前全量运行中，不能记为完成。原生接口的 attention-mask/pad-token 警告保留在日志，作者相关实现未取得，未静默改写原生接口。
