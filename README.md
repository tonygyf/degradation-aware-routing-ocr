# Degradation-Aware Region Routing OCR

低光照区域级 OCR/VLM 选择性路由。**第一阶段交付：可运行的 CPU 离线基线；样例预测为人工构造，不能作为模型效果或论文证据。** 没有下载 PaddleOCR/Qwen 权重，没有进行真实 GPU 推理。

主开发目标：Ubuntu 实验室服务器 `gyf@10.8.176.211:/home/gyf/degradation-aware-routing-ocr`。Windows 同名交付目录是可独立运行的副本。两个目录没有自动同步，实验记录分别保存。

## 快速运行

在项目根目录，使用 Python 3.10–3.12。所有命令只作用于本项目虚拟环境。

```bash
uv sync --frozen --group dev
uv run --frozen python -m pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen python -m scripts.smoke
uv run --frozen python -m src.run --config configs/baseline_a.json
uv run --frozen python -m src.run --config configs/baseline_b.json
uv run --frozen python -m src.run --config configs/baseline_b_mlp.json
```

服务器已备好独立 `.venv` 和 `.tools/uv`，无需重新安装即可：

```bash
cd /home/gyf/degradation-aware-routing-ocr
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m scripts.smoke
```

Windows 可直接用 `.venv\Scripts\python.exe -m scripts.smoke`。如没有 uv：`python -m venv .venv`，然后使用该环境的 `python -m pip install --require-hashes -r requirements.lock`，在根目录执行模块。服务器系统 Python 缺少 ensurepip，已使用本项目 `.tools/uv` 建立隔离环境，无需 sudo 或改动科研容器。

## 目录

```text
configs/                    A/B/MLP 配置、真实模型登记模板
src/
  datasets/                 缓存格式、内容哈希与组隔离校验
  detection/                区域多边形数据结构
  recognition/              缓存、PaddleOCR 3、Qwen-VL HTTP 接口
  routing/                  固定阈值、LogReg/MLP、C/D 接口
  evaluation/               CER、代理风险、调用/成本/延迟统计
  run.py                    训练路由器并评估锁定划分
  cache.py                  显式真实推理缓存采集入口
scripts/                    fixture、smoke、doctor、分组划分
tests/                      单元、隔离、预算、接口与端到端测试
data/fixtures/              60 个自制 SVG 与 mock 观测（32/12/16）
data/raw/ data/cache/        私有真实数据，git 忽略
models/                     本地模型路径，git 忽略
docs/                       协议、环境、Skills 核验、验收及路线
outputs/                    每次独立实验日志、配置、指标、预测
uv.lock requirements.lock   精确版本与分发文件哈希
```

## 基线定义与实验追踪

- A：OCR 框置信度算术均值 `< 0.7` 调用 VLM；无框的置信度均值定义为 0。边界等于阈值不调用。
- B：`StandardScaler + LogisticRegression`；另有单隐层 8 单元 MLP 配置。特征仅用 OCR 均值、最小值、标准差、框数、文本长度；不读取 GT、VLM 输出或文件名。
- B 标签只来自 train：`CER(VLM) < CER(OCR)`，平局归为不调用；没有改善样例/非改善样例之一时明确报错。Scaler 仅在 train 拟合。概率阈值固定为 0.5，未经验证集调优。
- C：`DegradationRouter.score(ocr, degradation)`；D：`RegionBudgetRouter.select(regions, budget_units)`。当前只有接口，未实现区域选取或低光照特征算法。
- 每次结果位于 `outputs/runs/<UTC时间>-<配置名>-<随机标识>/`。保存 config、metadata、router 参数、metrics、predictions 和 run.log。失败写日志并抛错；不自动重试，不伪造成功结果。
- 固定 seed=42、单线程、train/val/test ID 与 manifest/config SHA-256。metadata 记录依赖、平台、Git 提交和工作区是否修改。不同平台的浮点拟合允许微小差异。
- 样例划分已冻结；真实数据先按**原始图像**定义 group_id，所有亮度变体/裁剪必须同组。`scripts.split_manifest` 使用 seed+组 ID 的 SHA-256 分桶，近似 70/15/15，划分后提交清单哈希。
- A/B 按 ID 固定顺序处理，调用上限和单位预算双重限制；不足时保留 OCR，记录 budget_denied。这是顺序预算上限，不是 D 的最优区域分配。

指标口径、数据格式和真实推理前置条件见 [实验协议](docs/protocol.md)；环境与已有任务见 [环境记录](docs/environment.md)；安装证据见 [Skills 来源](docs/skills.md)；实测验收见 [交付验证](docs/validation.md)。
