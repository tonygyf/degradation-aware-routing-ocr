# CC-OCR 当前结果与论文分差（2026-10-31）

**仅外部验证；整篇论文尚未复现或通过验收。** 内部 Baseline/SHC 缺作者角色划分和双参考标注，尚无可比较的真实分数。

900 张冻结 Test：TotalText 300 / InverseText 500 / HierAgent 100。以下以 GPU Paddle 对照为主，Qwen、InternVL、DeepSeek 使用已冻结的 GPU 观测；论文数值取 [正文与补充表](https://link.springer.com/article/10.1007/s10032-026-00620-7)。

## 识别分数

差值均为本地减论文；Eval-Trans 越高越好，CER 越低越好。Eval-Trans 差用原始 0–1 单位，CER 差用百分点。

| 系统 | Eval-Trans 本地 / 论文 | 差值 | CER 本地 / 论文 | 差值（百分点） |
|---|---:|---:|---:|---:|
| PaddleOCR GPU | 0.3225 / 0.3386 | -0.0161 | 70.94% / 70.15% | +0.79 |
| Qwen GPU | 0.7604 / 0.7680 | -0.0076 | 45.63% / 45.36% | +0.27 |
| ConfRoute GPU 回放 | 0.6946 / 0.7044 | -0.0098 | 49.72% / 50.18% | -0.46 |
| DeepSeek GPU | 0.4672 / 0.4696 | -0.0024 | 61.70% / 61.13% | +0.57 |
| InternVL GPU | 0.5265 / 0.4571 | +0.0694 | 71.46% / 72.28% | -0.82 |

多数总体识别分数接近，但没有预设等价容差，不能事后判为“过关”。InternVL 在 HierAgent 上 Eval-Trans 为 0.6913，论文为 0.3361，明显不一致；总体分数更高也不能证明严格复现。

## 路由预算与延迟

| 指标 | 本地 GPU 对照 | 论文 | 差值 |
|---|---:|---:|---:|
| ConfRoute VLM 调用率 | 73.67%（663/900） | 68.33% | +5.34 个百分点 |
| 相对 always-Qwen 的平均延迟节省 | 19.96% | 32.28% | −12.32 个百分点 |
| ConfRoute 平均延迟 | 2.509 s | 1.028 s | +1.481 s |
| Qwen 平均延迟 | 3.135 s | 1.518 s | +1.617 s |
| ConfRoute p95 延迟 | 9.299 s | 2.834 s | +6.465 s |

固定阈值 0.90，不在 Test 上调参。按来源调用率为 TotalText 56.00%、InverseText 89.20%、HierAgent 49.00%；论文分别为 53.33%、87.00%、20.00%。CPU→GPU 仅改变 4 张路由决定，不能解释全部预算差异。

本地 RTX 6000D 与论文 RTX 4090 不同；模型分别运行、无 warmup、排除加载时间，ConfRoute 使用逐图串行反事实回放。因此上述延迟不是同硬件在线端到端复现。

10,000 次按来源配对 bootstrap（seed 42）：ConfRoute−Qwen 的 Eval-Trans 为 −0.06585，95% CI [−0.07958, −0.05260]；CER 为 +4.09 个百分点，CI [3.00, 5.23]；平均延迟为 −0.626 s，CI [−0.895, −0.391]。调用率 CI [71.11%, 76.22%]。

## 证据与历史对照

真实预测、权重和日志仅保留本地，未提交 Git；当前汇总来自以下冻结文件：

- 五系统主汇总：`outputs/ccocr-five-system-gpu-ocr-report.json`，SHA256 `bd493d83b8628177f03b93d03e279bccefbbdbce045354ed93badd753e3342db`。
- GPU 配对观测：`outputs/ccocr-paddle-gpu-full/observations.jsonl`，SHA256 `d676a96636bf69e5a0a8d7fe65d2b261104acefca20cbc35295b39687d55c99e`。
- 配对统计：`outputs/ccocr-paddle-gpu-full-statistics.json`；模型审计：`outputs/{paddle-gpu,internvl,deepseek}-result-audit.json`。
- 冻结清单 SHA256：`22149779ff2f61d41a12c322a16e2ae90802693c5406d8639810f91f8cc59ad9`。

旧 CPU Paddle 对照下，ConfRoute 为 Eval-Trans 0.6971、CER 49.68%、调用率 74.11%、平均 8.429 s；不与 GPU 主对照混合。完整历史数值、来源级识别分数、重复诊断、生成长度及运行警告见 [归档结果](archive/ccocr-results-through-2026-10-29.md)。未知作者权重、样本身份、适配器与生成参数仍见 [复现状态](paper-reproduction-status.md)。
