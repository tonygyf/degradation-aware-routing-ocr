# 历史 CC-OCR 结果与论文对照（更新于 2026-10-29）

历史快照；当前 GPU 主对照及分差见 [当前结果](../ccocr-results.md)。

900 张（TotalText 300 / InverseText 500 / Hieragent 100）全部完成，观测与主评估、统计报告的 SHA256 一致。本页保留使用 CPU Paddle 配对缓存的结果，InternVL 与 DeepSeek 均已完成 900 张；GPU Paddle 对照已完成，见后文。整篇论文仍未复现或验收。

| 系统 | Eval-Trans 本地 / 论文 | CER 本地 / 论文 | 空输出率 本地 / 论文 | VLM 调用率 本地 / 论文 |
|---|---:|---:|---:|---:|
| paddleocr | 0.3223 / 0.3386 | 70.93% / 70.15% | 6.89% / 7.11% | 0.00% / 0.00% |
| qwen | 0.7604 / 0.7680 | 45.63% / 45.36% | 0.00% / 0.00% | 100.00% / 100.00% |
| confroute | 0.6971 / 0.7044 | 49.68% / 50.18% | 0.00% / 0.00% | 74.11% / 68.33% |
| deepseek | 0.4672 / 0.4696 | 61.70% / 61.13% | 0.00% / 0.00% | 100.00% / 100.00% |
| internvl | 0.5265 / 0.4571 | 71.46% / 72.28% | 0.00% / 0.00% | 100.00% / 100.00% |

ConfRoute−Qwen 的 Eval-Trans 差为 -0.0633，95% 配对分层 bootstrap 区间 [-0.0768, -0.0502]；论文差值为 −0.0637。CER 增加 4.05 个百分点，区间 [2.95, 5.19]。

本地升级 667/900，74.11%，10,000 次按来源分层 bootstrap（seed 42）的 95% 区间为 [71.44%, 76.67%]，论文点估计为 68.33%。补充统计 `outputs/ccocr-source-macro-latency-and-calls.json` 同时记录了未加权来源平均延迟；这些时间包含 CPU OCR，不代表论文 GPU 同硬件结果。保留历史阈值 0.90，不能在这批 Test 上重新选阈值以追近论文调用率。

Qwen 42/900 张达到本项目冻结的 2048 token 上限。重复诊断：Paddle 5、Qwen 52、ConfRoute 46；论文分别为 7、57、48。未删截输出后重评分。生成上限和 tokenizer 源码未获作者确认。

延迟：本地 Paddle CPU 平均 6.008 s，Qwen GPU 平均 3.135 s，混合 ConfRoute 回放平均 8.429 s。本地回放没有复现论文加速结论，不能把 CPU 与 GPU 混合耗时解释成论文同硬件对比；GPU Paddle 对照现已完成，见后文；CPU/GPU 结果与统计继续分开保存。

严格一致性仍不能证明：作者权重哈希、完整适配器、生成参数和样本清单未取得；当前官方候选版本、独立实现与环境差异都已登记。指标接近是实测现象，不等于通过预设等价检验。

证据：`outputs/ccocr-gpu-full/observations.jsonl`、`outputs/ccocr-gpu-full-eval/metrics.json`、`outputs/ccocr-gpu-full-statistics.json`。
观测 SHA256：`3fcc24a74920e72479e592378517bc3ababb43cb773f0978532016b32a4273aa`。

论文来源：[正文与附录入口](https://link.springer.com/article/10.1007/s10032-026-00620-7)。

## 调用率差异定位

保持阈值 0.90 和已冻结 OCR 观测，按来源对照补充表 23：

| 来源 | 本地调用率 | 论文调用率 | 差异 |
|---|---:|---:|---:|
| TotalText | 56.33% | 53.33% | +3.00 个百分点 |
| InverseText | 89.80% | 87.00% | +2.80 个百分点 |
| HierAgent | 49.00% | 20.00% | +29.00 个百分点 |

HierAgent 对总体差异贡献明显，但这只能定位现象，不能据此断定原因。OCR 中有 5 个非空但仅由空白构成的识别框；作为一次口径敏感性检查，将其从平均置信度排除时，900 张的路由决定没有任何变化。正式结果继续使用当前按原文实现的冻结口径。审计记录 `outputs/confidence-subset-audit.json`。

HierAgent 的 Paddle Eval-Trans 本地 0.6466、论文 0.7471，CER 本地 60.55%、论文 60.88%；这两个指标的差异方向不同，说明仅比较 CER 不足以核验该来源。已核查当前 HierAgent 100 张图片字节哈希均不重复、参考文本均非空，但由于作者冻结样本清单和原始输出缺失，不能据此证明样本或解析器一致，更不能断言差异成因。

## InternVL 全量结果更新（2026-10-29）

InternVL 官方候选已完成 900 张，图片/参考身份和输入 SHA256 已核验。Eval-Trans 0.5265（论文 0.4571）、CER 71.46%（论文 72.28%）、空输出 0、重复诊断 40/900（论文 63/900）。

| 来源 | 本地 Eval-Trans / 论文 | 本地 CER / 论文 |
|---|---:|---:|
| TotalText | 0.6304 / 0.5816 | 64.67% / 61.98% |
| InverseText | 0.4312 / 0.4066 | 77.14% / 76.93% |
| HierAgent | 0.6913 / 0.3361 | 63.40% / 79.97% |

较大差异仍出现在 HierAgent。当前使用固定官方候选、原生 chat、448 像素、最多 12 个动态切块加 thumbnail、greedy、2048 token 上限，均在真实推理前登记；作者图像预处理、生成长度和精确权重身份仍缺。不能由 pooled CER 接近认定完全一致，也不能通过 Test 改配置追近论文。完整指标 `outputs/ccocr-internvl-full/metrics.json`，审计 `outputs/internvl-result-audit.json`。

## DeepSeek 全量结果（2026-10-29）

900 张已采集完成，冻结样本身份、模型标签、登记与采集源码哈希通过核验。Eval-Trans **0.4672**（论文 0.4696）、CER **61.70%**（论文 61.13%）、空输出 0。数值较接近，但未预设等价容差，不能据此认定严格复现或整篇论文验收通过。

| 来源 | 本地 Eval-Trans / 论文 | 本地 CER / 论文 |
|---|---:|---:|
| TotalText | 0.6188 / 0.6207 | 46.14% / 45.19% |
| InverseText | 0.3559 / 0.3595 | 69.33% / 69.12% |
| HierAgent | 0.5685 / 0.5667 | 70.25% / 68.99% |

本地平均 1.836 s/张（论文 0.896 s），中位数 0.498 s、p95 3.732 s；存在超过 200 秒、输出万余字符的长样本。原生 eval 接口固定 `max_new_tokens=8192`，日志中的 attention-mask/pad-token 与总长度提示保留原样；作者对应参数未知，不在 Test 上改长度追分。硬件、运行时与包装器差异仍需报告。

证据：`outputs/ccocr-deepseek-full/{observations.jsonl,metadata.json,metrics.json}`、`outputs/deepseek-result-audit.json`，日志 `outputs/deepseek-full-20261029.log`。观测 SHA256：`bd108095d9cdafe1f54377cc9ccfc9f409ddab7760c81bbd89ef2fba4fb68eae`。

## GPU Paddle 对照（2026-10-29）

使用相同冻结的检测/识别权重和设置，在 GPU 1（NVIDIA RTX 6000D）采集 900 张；Qwen 逐样本复用相同 GPU 缓存。模型分别运行、无 warmup、排除模型加载；ConfRoute 是串行反事实回放，不是在线端到端测量，硬件也与论文 RTX 4090 不同。

| 系统 | 本地 Eval-Trans / 论文 | 本地 CER / 论文 | 本地调用率 / 论文 | 本地平均秒数 / 论文 |
|---|---:|---:|---:|---:|
| PaddleOCR GPU | 0.3225 / 0.3386 | 70.94% / 70.15% | 0 / 0 | 0.089 / 0.130 |
| Qwen GPU | 0.7604 / 0.7680 | 45.63% / 45.36% | 100% / 100% | 3.135 / 1.518 |
| ConfRoute GPU 回放 | 0.6946 / 0.7044 | 49.72% / 50.18% | 73.67% / 68.33% | 2.509 / 1.028 |

GPU 对照 10,000 次按来源配对 bootstrap：ConfRoute−Qwen 的 Eval-Trans 为 −0.06585，95% 区间 [−0.07958, −0.05260]；CER 增加 4.09 个百分点，区间 [3.00, 5.23]；平均延迟减少 0.626 s，区间 [0.391, 0.895] s。

本地 ConfRoute 相对 always-Qwen 平均延迟减少 **19.96%**，论文为 32.28%；本地 p95 为 9.299 s，论文 2.834 s，延迟分布也未对齐。调用率 663/900，95% 分层 bootstrap 区间 [71.11%, 76.22%]。

CPU/GPU OCR 文本不同共 92 张，但路由仅 4 张改变（667→663 次）；HierAgent 仍调用 49%，论文 20%。在当前冻结候选下，后端差异不足以解释调用预算偏差，作者精确权重、适配器与样本身份仍需材料核验。不在 Test 上更换候选模型或参数追分。

观测及配对审计：`outputs/ccocr-paddle-gpu-full/`、`outputs/paddle-gpu-result-audit.json`；主指标：`outputs/ccocr-paddle-gpu-full-eval/metrics.json`。GPU 配对观测 SHA256：`d676a96636bf69e5a0a8d7fe65d2b261104acefca20cbc35295b39687d55c99e`。统计 `outputs/ccocr-paddle-gpu-full-statistics.json`、五系统 GPU OCR 汇总 `outputs/ccocr-five-system-gpu-ocr-report.json` 均已完整生成，输入哈希、评估源码和各报告主指标一致性检查通过。

CPU Paddle 五系统汇总已完成：`outputs/ccocr-five-system-report.json`。重复诊断：Paddle 5、Qwen 52、ConfRoute 46、DeepSeek 81、InternVL 40；论文分别为 7、57、48、72、63。重复诊断不是官方 Eval-Trans 或 SHC flag，分词解释尚无作者源码确认。
