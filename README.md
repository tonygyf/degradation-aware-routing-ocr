# Hybrid VLM-OCR 论文复现

复现《Hallucination-aware Evaluation and Confidence-guided Selective Routing for Hybrid VLM-OCR》（IJDAR，2026）。**整篇论文尚未复现完成，也未通过指标验收。先完成论文基线，再研究新路由策略。**

CC-OCR 只是论文的外部验证。五个系统及 GPU Paddle 对照均已有 900 张真实观测；调用预算、部分模型和延迟分布仍未对齐论文。内部 Baseline 与 SHC 缺作者样本映射、角色划分和标注，不能用其他公开数据替代后宣称完成。

- [当前复现状态与缺口](docs/paper-reproduction-status.md)
- [真实结果与论文差异](docs/ccocr-results.md)
- [论文软件实现与运行入口](docs/paper-implementation.md)
- [作者材料索取清单与邮件草稿](docs/author-material-request.md)
- [文档导航](docs/README.md)

## 开发与检查

统一使用 Docker。Linux 先设置文件所有者；依赖变化后重新构建：

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose build dev
docker compose run --rm dev python -m pytest -q
docker compose run --rm dev ruff check .
docker compose run --rm dev ruff format --check .
docker compose run --rm dev python -m scripts.smoke
```

这些检查使用合成 fixtures 和缓存，只证明软件行为，不代表论文指标复现。默认开发环境为 CPU；真实 GPU 实验必须遵守 [AGENTS.md](AGENTS.md) 和已登记的设备授权，并重新检查资源。运行方法见 [Docker 环境](docs/environment.md) 与 [CC-OCR 协议](docs/ccocr-reproduction.md)。

## 目录

| 路径 | 用途 |
|---|---|
| `src/evaluation/`、`src/routing/` | 论文指标、ConfRoute 与保留的策略原型 |
| `src/recognition/` | PaddleOCR、Qwen、InternVL、DeepSeek 接口 |
| `scripts/` | 数据审计、显式下载、推理采集、评估和冒烟入口 |
| `configs/` | 冻结模型、数据来源与实验配置 |
| `tests/`、`data/fixtures/` | 软件检查与自制样例 |
| `docs/`、`docs/archive/` | 当前说明与历史记录 |
| `data/raw/`、`data/cache/`、`models/`、`outputs/` | 本地数据、权重、预测、结果及日志，Git 忽略 |
| `Dockerfile*`、`compose*.yaml`、`requirements*.lock` | 独立运行环境及依赖锁 |

原 A/B/MLP 与退化风险原型保留供后续研究，定义见 [原型协议](docs/protocol.md)；目前不作为论文科研基线，也不继续扩展算法。新策略应从经过验收并冻结的复现提交创建分支，当前尚无这一冻结点。
