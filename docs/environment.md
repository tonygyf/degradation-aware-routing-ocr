# Docker 开发环境

2026-10-24（Asia/Shanghai）起，开发、测试和实验统一在本项目 Docker 环境中执行。宿主机只需要 Git、Docker Engine 和 Compose v2，不安装项目 Python 依赖。原 Windows/Ubuntu 虚拟环境记录为历史验证，不再是开发入口。

## 使用

Linux 在项目根目录执行：

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose build dev
docker compose run --rm dev bash
```

容器内执行 `python -m pytest -q`、`ruff check .`、`ruff format --check .`、`python -m scripts.smoke`。也可以像 README 中一样直接用 Compose 执行单条命令。Windows Docker Desktop 可跳过 UID/GID 设置；默认 1000:1000。源码挂载到 `/workspace`，输出写入宿主项目，退出即删除临时容器。不要使用全局 `docker system prune` 或操作其他项目的容器。

## 依赖与资源

- 基础镜像为 `python:3.12-slim-bookworm`，在 Dockerfile 中固定 SHA-256 摘要。Git 用于记录实验提交版本。
- `requirements.lock` 固定依赖版本和分发哈希；构建使用 `pip --require-hashes --only-binary=:all:`，禁用 pip 缓存，不安装模型运行时或权重。`uv.lock` 保留为原始依赖锁记录。
- 新增依赖需同时更新 `pyproject.toml` 与锁文件，重新构建并运行四项基线检查；所有依赖维护也在容器中进行。
- Compose 限制 2 个 CPU、4 GiB 内存，数值计算单线程。不申请 GPU，`CUDA_VISIBLE_DEVICES` 为空，`NVIDIA_VISIBLE_DEVICES=void`。没有复用既有科研容器，也没有停止训练进程。
- 仅挂载项目目录，不挂载 Docker socket、SSH 目录、宿主凭据或其他科研目录。Git 推送在宿主机通过既有 SSH 密钥执行。
- `.dockerignore` 排除私有数据、模型、输出、密钥、Git 历史和宿主缓存。Compose 开发挂载仍可访问项目内的本地数据，因此不要将此开发镜像当作隔离不可信代码的沙箱。
- 第一阶段测试和 smoke 只使用自制 fixtures 与 mock 观测，不构成科学证据。真实推理仍须先登记模型版本、许可证、数据授权及资源需求，并重新检查资源。

## 清理

Docker 检查通过后可删除本项目旧 `.venv/`、`.tools/`、`.pytest_cache/`、`.ruff_cache/`、`__pycache__/` 和 `*.egg-info/`。容器禁写 Python 字节码，pytest/ruff 禁用项目缓存。实验输出保留本地且不进入 Git；不要删除真实数据或模型来清理软件缓存。

## 论文真实模型环境

默认 `compose.yaml` 仍是 2 CPU / 4 GiB 的开发环境。真实模型使用独立镜像和覆盖配置，均为 8 CPU / 64 GiB；不修改已有训练容器。

| 覆盖配置 | 用途 |
|---|---|
| `compose.ccocr.yaml` | Paddle CPU / Qwen 真实运行环境 |
| `compose.internvl.yaml` | InternVL 原生接口及独立依赖 |
| `compose.deepseek.yaml` | DeepSeek 原生接口及 Transformers 4.46.3 独立依赖 |
| `compose.paddle-gpu.yaml` | Paddle GPU 3.2.2 / CUDA 12.9 独立依赖 |
| `compose.ccocr-gpu.yaml` | 仅映射已授权宿主 GPU 1，需与对应模型配置叠加 |

GPU 作业只在单卡空闲、UUID 一致且资源检查通过时顺序运行。模型、数据和源码身份以 `configs/ccocr*_gpu.json`、固定 revision 及逐次输出 metadata 为准。镜像构建与导入成功不代表真实推理通过，当前执行状态见 [复现状态](paper-reproduction-status.md)。

文档整理只移动历史 Markdown 和环境快照；源码、Compose、依赖锁与实验目录保留原路径，避免破坏运行和续跑校验。
