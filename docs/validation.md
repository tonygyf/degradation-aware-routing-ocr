# 第一阶段交付验收

验收日期：2026-10-24（Asia/Shanghai）。目标为独立 Ubuntu 项目，Windows 为运行与交付副本。

| 检查 | Windows Python 3.10.11 | Ubuntu Python 3.12.3 |
|---|---|---|
| `uv sync --frozen --group dev` | 成功，包括 editable 构建 | 成功，包括 editable 构建 |
| `uv pip check` | 兼容 | 兼容 |
| `ruff check .` | 通过 | 通过 |
| `ruff format --check .` | 通过（22 个 Python 文件） | 通过（22 个 Python 文件） |
| `python -m pytest -q` | 19 passed | 19 passed |
| `python -m scripts.smoke` | A/B/MLP 全部完成 | A/B/MLP 全部完成 |

测试覆盖 CER/Unicode/空参考、阈值边界/无检测、分组与内容跨划分泄漏、重复 ID/非有限置信度、训练集约束、GT/VLM 不进入路由特征、固定种子重复拟合、调用与成本预算/惰性 VLM、幻觉代理口径、真实入口审核与显式开关、图像哈希门禁、缓存采集流程、Paddle3/Qwen 的 mock API 契约及独立运行目录。

Paddle/Qwen 的测试使用伪响应，不是实际模型推理。loopback Qwen 请求禁用环境代理与 HTTP 重定向；没有向外部 API 发送图像。

## 人工样例的管线结果

下表只验证软件行为。16 个 test 样例、210 个参考字符，Windows 与 Ubuntu 的下列数值一致；没有据此选择“最佳”方案。

| 策略 | CER | VLM 选择次数/率 | 平均缓存模拟延迟 | 声明成本单位 |
|---|---:|---:|---:|---:|
| A 固定阈值 | 0.123810 | 12 / 75% | 102 ms | 12 |
| B LogReg | 0.114286 | 14 / 87.5% | 117 ms | 14 |
| B MLP | 0.142857 | 8 / 50% | 72 ms | 8 |

这是已知 mock 预测的回放。延迟为人工设置，不能用于硬件性能比较；VLM“调用”表示访问选定缓存。实际 routing_ms、wall time 各次不同，保存在实验 metrics.json。

实际输出：服务器 `outputs/smoke/`；Windows `outputs/smoke/` 及服务器结果副本 `outputs/server-smoke/`。每次独立目录保留配置、split ID、输入哈希、Git 状态、router 参数、日志与预测。

## 环境保护与 Skills

主科研 GPU 2 的 Python PID 4080287 在验证后仍存在，显存占用 24,300 MiB；本项目没有创建 CUDA 推理进程。原有容器和两个科研项目未修改，既有 Git 改动保留。所有包只装在新项目 `.venv`。

两个 Skill 已在服务器 `/home/gyf/.codex/skills/{ponytail,grill-me}/` 安装，SHA-256 与 Windows 现有文件和固定版本来源一致，见 skills.md。只安装 Markdown，没有运行第三方脚本或插件 hooks。

## 明确未完成

- 没有下载权重、启动 Qwen 服务或跑真实 Paddle/GPU 推理；真实模型环境与完整依赖锁尚待基于实际平台验证。
- 没有导入既有实验室数据；授权、源图分组、模型 checkpoint revision 和资源预留必须先登记。
- C/D 只有接口；尚未实现退化特征、区域分配和区域输出合并，也没有科研有效性结论。
- 真实失败重试/恢复、token 级预算、线上服务延迟、近重复人工审计与可信幻觉标注留给后续实验。

下一步先执行 protocol.md 的小规模真实接口验证，再按 roadmap.md 建立同预算对比和 C/D 消融。
