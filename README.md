# Degradation-Aware Region Routing OCR

低光照区域级 OCR/VLM 选择性路由。现有 A/B 样例为人工构造，仅作软件检查；另推进 IJDAR 2026 论文的公开 CC-OCR 900 张外部实验，数据与官方 Paddle 权重已下载并校验，Qwen 权重显式下载中。真实实验入口、冻结协议及复现差异见 [CC-OCR 复现](docs/ccocr-reproduction.md)。

后续开发、测试和实验统一使用本项目独立的 Docker 环境（Python 3.12、CPU），不再依赖宿主机虚拟环境。

## 快速运行

需要 Docker Engine 和 Docker Compose v2。在项目根目录执行；Linux 下先设置文件所有者：

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose build dev
docker compose run --rm dev python -m pytest -q
docker compose run --rm dev ruff check .
docker compose run --rm dev ruff format --check .
docker compose run --rm dev python -m scripts.smoke
docker compose run --rm dev python -m src.run --config configs/baseline_a.json
docker compose run --rm dev python -m src.run --config configs/baseline_b.json
docker compose run --rm dev python -m src.run --config configs/baseline_b_mlp.json
```

进入交互开发 Shell：

```bash
docker compose run --rm dev bash
```

源码挂载到 `/workspace`，修改即时生效，输出保留在宿主机 `outputs/`。依赖改变后重新构建镜像；不要在宿主机安装项目包。Windows Docker Desktop 可跳过 `export`。CPU、内存限制及隔离说明见 [Docker 环境](docs/environment.md)。

## 目录

```text
configs/                    A/B/MLP 配置、真实模型登记模板
Dockerfile compose.yaml     独立 CPU 开发环境、资源限制与源码挂载
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

指标口径、数据格式和真实推理前置条件见 [实验协议](docs/protocol.md)；开发方式见 [Docker 环境](docs/environment.md)；实测验收见 [交付验证](docs/validation.md)。
