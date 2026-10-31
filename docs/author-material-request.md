# 作者材料索取清单与邮件草稿

更新：2026-10-29（Asia/Shanghai）。用户已确认：取得作者原包后继续严格复现；不另建替代数据集。本文仅供用户审阅和发送，**本助手未代发邮件，用户是否自行发送未在项目登记**。

论文 DOI：[10.1007/s10032-026-00620-7](https://doi.org/10.1007/s10032-026-00620-7)。通讯作者 **Jinhong Yang，jinhong@inje.ac.kr**，已于上述日期对照 [Springer 官方通讯作者信息](https://link.springer.com/article/10.1007/s10032-026-00620-7) 核验。网页第 3.5 节仍为匿名仓库占位，Data Availability 描述接受后公开；本项目尚未取得可核验作者包，不断言其他渠道不存在发布。

## 材料清单

先索取完整仓库或版本固定的归档包；若暂不能整体提供，请优先提供以下 P0 项。文件名和格式可沿用作者原结构，不要求转换成本项目格式。

| 优先级 | 材料 | 需要核对的内容 |
|---|---|---|
| P0 | 复现包、固定版本及入口 | 仓库 URL/commit 或归档 SHA256、许可/使用条件、README、依赖锁、全部配置和复现表格的主入口；包括论文点名的 `scripts/audit_author_required_items.py` |
| P0 | 内部 Baseline 清单及 GT 组装 | 579 张逐图来源 ID、原始数据版本、文件名/哈希、Support 29 / Dev 347 / Cal 87 / Test 116 的固定角色；全图转录及阅读顺序、忽略区域、编码和排除规则，尤其 `X51006619503` 的处理 |
| P0 | SHC 样本重建映射 | 5,550 样本与 933 个 canonical source 的对应关系；原始语料/版本、源图 ID、`family_id`、裁剪坐标或字段标识、重建脚本和预处理规则 |
| P0 | SHC 标注及角色 | 每样本 visible/full transcript、预先指定的 critical span（含未指定情况）、S1–S6/L1–L2/R1–R2 类型；Exemplar 546 / Prompt_dev 546 / Tau_dev 561 / Test 3,897 的固定清单 |
| P0 | 冻结模型与推理实现 | 每个系统的 checkpoint/revision/hash；Paddle 检测/识别型号及权重；原提示、chat template、图像缩放/切块、生成长度与停止规则、seed、精度、解析/清理逻辑；实际可运行环境及本地补丁 |
| P0 | 作者 CC-OCR 清单和包装器 | 冻结 900 张的来源 ID、图像哈希、GT 解析和官方 Eval-Trans 封装；scalar 置信度、四位舍入、空框和路由处理的原代码，用于核对外部差异 |
| P1 | 原始/解析输出及计时缓存 | Baseline 各角色、SHC 各角色和 CC-OCR 的逐模型观测；`rec_texts`/`rec_scores`、独立 `confidence_boxes`、原始与解析文本、逐图耗时及冷启动/warmup/同步的口径 |
| P1 | 学习路由、校准与统计脚本 | 密度与置信度特征定义、零值处理、缩放、L2 强度/求解器/类别权重、Tau_dev 决策阈值；完整 Cal 候选与 λ 分析；bootstrap/permutation 的抽样、分层/聚类、RNG、p 值约定 |
| P1 | 人工审计与源图重叠材料 | 400 样本双标及仲裁记录；Qwen 100 auto-positive/100 auto-negative 人工审核样本、抽样池及判定，DeepSeek positive 审计；canonical source 规则、角色/跨资源重叠矩阵及过滤清单 |

P0 用于恢复同一实验身份和运行路径；P1 用于分离推理、解析、聚合差异并重算论文附加统计。若某项不能提供，请作者注明缺失或未归档的部分。

不要求重新分发原始图像或模型权重：来源 ID、版本、哈希、裁剪/字段映射和重建脚本即可支持按原条款获取与核验。也不索取标注人员的个人信息。上述结构来自本项目的复现需求整理，不声称是作者已经公开的文件格式。

## 英文邮件草稿

收件人：`jinhong@inje.ac.kr`。发送前填写署名占位符，按实际情况决定是否保留机构信息。

Subject: Request for the reproduction package — IJDAR 2026, DOI 10.1007/s10032-026-00620-7

Dear Jinhong Yang,

I am independently reproducing your IJDAR 2026 study (DOI: 10.1007/s10032-026-00620-7). I would like to reproduce the original experiments using the same samples, annotations, and frozen implementation. The published article still displays an anonymized repository placeholder, and I have not been able to locate a verifiable public release.

Could you please share the repository URL and exact release/commit, or a versioned reproduction archive? The most important materials for unblocking the reproduction are:

1. The internal Baseline manifests and full-page reference construction rules, including the fixed Support/Dev/Cal/Test assignments and exclusions.
2. SHC annotations and role manifests, with visible/full references, critical spans, failure types, canonical source/family IDs, crop coordinates or field identifiers, and reconstruction scripts.
3. Exact model revisions and inference configurations, prompts, preprocessing, generation limits, output parsers, confidence calculation, and the working dependency environment.
4. The frozen CC-OCR subset manifest and evaluation/routing scripts. If available, cached per-model outputs, confidence records, timings, and the learned-router/statistical audit scripts would also help us isolate implementation differences.

We have completed our independently implemented 900-image external runs and a GPU PaddleOCR control. At the frozen threshold of 0.90, our GPU replay calls Qwen on 73.67% of samples, compared with the reported 68.33%. We would like to verify the original sample identities, checkpoints, and wrapper before making any interpretation; we have not retuned the threshold on the external test set.

Source identifiers and reconstruction metadata are sufficient where image redistribution is restricted. If a complete release is not yet available, a partial package containing the manifests, annotations, and frozen configuration would be very helpful. Please also let us know the applicable use conditions and whether any listed material is unavailable.

Thank you for your time and assistance.

Best regards,
[Your name]
[Affiliation, if applicable]
[Contact email]

## 收到材料后的执行顺序

1. 登记作者包的来源、版本、哈希及使用条件，与现有候选和冻结结果分开保存；核对角色数量、逐样本身份、双参考、源图映射和原文重叠审计。
2. 先在 Docker CPU 环境重放作者缓存及评分/统计脚本，核对原表，区分标注、解析、指标与真实推理的偏差。暂缺缓存时先核验重建后的输入及配置，不用 mock 分数替代。
3. 对齐作者实际运行环境和模型身份，再用小规模 Support/pilot 验证真实接口；需要 GPU 时重新核对已授权 GPU 1 的资源。后续按内部 Baseline、SHC 主实验与附加分析、外部验证逐项推进。
4. 保留历史阈值及原角色，不在 Test 追分；报告仍无法一致的环境与结果。完整验收后再冻结科研基线、创建新策略分支。

当前结果和差异见 [CC-OCR 报告](ccocr-results.md)，全论文执行状态见 [复现状态](paper-reproduction-status.md)。取得原包前，严格复现仍受阻。
