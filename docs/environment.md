# 环境检查记录（2026-10-24，Asia/Shanghai）

Windows 起始目录不是 Git 仓库，无已有项目源码；没有发现适用的上层 AGENTS.md。Python 3.10.11、Git、uv 0.12.13 可用。RTX 4060 Laptop 8 GB，在首次检查约使用 864 MiB，未用于本项目。系统环境有 NumPy/sklearn/Torch，缺 PaddleOCR/Paddle/Transformers/pytest/ruff；本项目创建独立 `.venv`，未修改系统包。

已有 SSH 配置将 `10.8.176.211` 映射到授权用户 gyf，使用既有 host key 和 BatchMode 成功连接。目标 Ubuntu 主机 x4rx6000d-MS03-CE0-000，系统 Python 3.12.3，没有 pip/ensurepip，四块 NVIDIA RTX 6000D（约 85,651 MiB/块），251 GiB 内存，home 所在盘约 2.7 TB 可用。

首次检查 GPU 2 上有 Python PID 4080287，约 24,300 MiB 显存、84% GPU 利用率；进程 cwd 为 /workspace，解释器为容器 /opt/conda/bin/python3.11，有活跃 pt_data_worker。其他卡的瞬时空闲不构成资源授权；本次一律 CPU。

运行中的科研容器包括 extremeocr-research-1、seeread-v5-gpu3-20261022、seeread-v5-run-20261020b、seeread-v5-det-20261020、seeread-v5-20261020 和 seeread-research-20261019。**没有进入容器安装包、停止进程、改变 GPU 设置或重启服务。**

只读检查了 /home/gyf/seeread_project 与 /home/gyf/DarkText-private 的目录、Git 状态；都已有大量未提交修改，因此没有复用或改写。新项目独立放在 /home/gyf/degradation-aware-routing-ocr，CPU 单线程运行测试。没有读取 .env、私钥或令牌内容。

服务器项目专用 `.tools/uv` 来源为 PyPI uv==0.12.13 Linux wheel，下载时核对 PyPI SHA-256；不是向系统安装软件。隔离环境由 uv 建立，只安装 requirements.lock/uv.lock 指定的轻量 CPU 依赖。最终环境快照见 docs/environment-server.json、docs/environment-windows.json。
