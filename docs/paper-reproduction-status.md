# 全论文复现状态（2026-10-31）

**整篇论文尚未复现完成，未通过科研指标验收。执行状态：按公开内容继续补齐代码，作者材料缺失仍限制真实验证。** 当前任务是先复现原论文，暂停新路由策略开发。CC-OCR 是外部验证，不代表全部论文。

目标：[IJDAR 正文与附录入口](https://link.springer.com/article/10.1007/s10032-026-00620-7)。原始过程记录见 [历史日志](archive/reproduction-log-through-2026-10-29.md)。

## 实验矩阵

| 实验 | 论文要求 | 当前证据 | 状态 |
|---|---|---|---|
| 内部文档 Baseline | 579 文档，Support 29 / Dev 347 / Cal 87 / Test 116 | 指标、配对统计、Dev 双信号前沿、Cal 候选及 λ 独立回放已实现；缺角色清单及真实观测 | 缺材料，未复现 |
| SHC 主实验 | 5,550 样本、933 源图，Test 3,897；双参考、critical span、S1–S6 | 双参考/flag、阈值与回退敏感性、风险—覆盖、聚类 bootstrap 可运行；缺原始样本映射和标注 | 缺材料，未复现 |
| 提示、风险覆盖与学习路由 | 原文开发角色、特征、校准和各系统比较 | 已增加配对提示比较和论文 L2 学习器；未知设置显式登记，缺真实结果；旧 A/B 不替代论文系统 | 软件检查通过，真实验证待材料 |
| CC-OCR 外部验证 | 900 张，五系统 | 五系统及 GPU Paddle 对照均完成 900 张；两套五系统汇总、GPU 对照 bootstrap 与配对审计均完成 | 已有结果待验收 |
| 统计与审计 | 重复诊断、配对/聚类统计、来源重叠等 | CC-OCR 两套五系统汇总、GPU 配对 bootstrap 和分组审计完成；主实验统计仍缺真实数据 | 部分完成 |

## 已有真实结果

五系统各 900 张的 **GPU Paddle 主对照、论文分差、调用率、延迟与证据路径** 统一见 [CC-OCR 结果](ccocr-results.md)。CPU Paddle 配对仅作为历史对照保存，不与 GPU 主结果混用。

关键缺口：ConfRoute 调用率 73.67%（论文 68.33%），节省延迟 19.96%（论文 32.28%）；InternVL 的 HierAgent Eval-Trans 差异明显。没有预设等价容差，尚未通过科研指标验收。阈值维持冻结的 0.90，不在外部 Test 上追分。

所有本项目全量采集已完成并退出；此处记录的是已完成实验，不保证设备当前空闲。后续真实推理必须重新检查设备与资源，按既有授权隔离运行。

## 数据与实现边界

CC-OCR 900 张原图、参考与清单已冻结并逐图校验，300 TotalText / 500 InverseText / 100 HierAgent，全部为外部 test。清单 SHA256：`22149779ff2f61d41a12c322a16e2ae90802693c5406d8639810f91f8cc59ad9`。

作者复现包尚未取得；2026-10-29 通过 GitHub 连接器复查题名、ConfRoute、hybrid-route、hybrid-vlm-ocr、论文点名的 audit_author_required_items.py 及 SHC/Tau_dev 字段，未找到可核验作者包；两个宽泛候选的 README 均属其他项目。Springer 正文第 3.5 节仍为匿名链接占位，Data Availability 仍写待接受后发布。记录 `outputs/reproduction-sources/author-package-recheck-20261029.json`；这不证明其他渠道不存在发布。缺内部角色清单、SHC 裁剪/字段映射、可见/完整双参考、关键片段、失败标签及精确模型/适配器配置。原始公开数据不能自动还原这些人工处理。原文角色存在源图重叠，亦不能声称严格源图独立。

公开 SROIE 镜像已下载并核验发布哈希、完成来源审计，但不是作者实验划分：987 条中有 16 组图片字节重复，其中 8 组跨镜像 train/test；相同图片的标注也不同。论文 Dev 排除样本在镜像 test 中。与 CC-OCR 无相同编码字节不等于视觉无重叠。审计见 `outputs/sroie-reference-audit-v2.json`、`outputs/sroie-duplicate-word-audit.json`；不用于模型训练或替代论文 Test。

2026-10-31 已增加论文 L2 学习路由器，仅 Tau_dev/train 拟合；requested 冻结阈值取补充表的 0.459。优化器/缩放采用显式未核验配置，密度特征必须提供定义来源，不能保证概率尺度与作者一致；旧 A/B 原型不作为该实验的替代实现。

本地 `outputs/reproduction-sources/article.html` 实为网站 Client Challenge，不是正文；2026-10-29 已通过官方正文与 PDF 重新核对。验证记录 `outputs/reproduction-sources/article-source-validation.json` 保留该事实，直接 PDF 下载亦未取得有效 PDF。

论文口径独立实现了空白归一化、截断 macro-CER、官方 Eval-Trans、两种 ConfRoute 置信度、SHC operational flags 等；Eval-Trans 与固定上游源码 1,000 组随机文本交叉校验误差为 0。缺作者完整源码，不能声称代码逐项完全一致。Qwen/InternVL 的 2048 token 上限、官方候选版本、Torch 推理后端及本地硬件等差异均保留记录，不在 Test 上调整追分。

## 继续原论文复现所需的材料

用户于 2026-10-29 确认：整理作者材料索取清单，取得原包后继续严格复现；不另行重建替代数据。完整优先级清单、经核验的通讯作者联系方式、未发送的英文邮件草稿及取得材料后的执行顺序见 [作者材料索取](author-material-request.md)。

本地材料核查覆盖 Git 忽略目录：516 个 JSON、137 个 JSONL，共 16,240 条 JSONL 记录，未发现作者 SHC 双参考标注或开发角色结构。记录 `outputs/reproduction-materials-audit-20261029.json` 保留审计时的清点结果；材料到位前，整篇论文仍受阻，现有数据、观测和配置保持冻结。

## 2026-10-31 代码补齐

用户确认作者尚未反馈后，继续依据公开描述补齐实现，不另造作者标签或更改冻结 Test。新增 Dev/Cal 敏感性、SHC 多回退/阈值、风险—覆盖与 ranking、L2 学习路由、缓存提示比较、源图重叠及人工 flag 计数重加权。完整覆盖、输入格式、命令和未核验参数见 [论文软件实现](paper-implementation.md)。这些功能已通过软件检查，尚无对应真实 SHC 结果。

## 内部 Cal 回放入口

固定候选、输入格式、Docker 命令和显式未知参数统一见 [论文软件实现](paper-implementation.md)。目前缺作者真实 Cal 观测，不将 CC-OCR 用作 Cal，也不生成科研 Cal 分数。

## 验收与文档

软件检查最新为 95 项测试及 Ruff、格式、smoke 通过，见 [软件验证](validation.md)。这不替代科研验收。新策略分支须从经过验收的代码与结果冻结点创建，目前尚无该点，见 [后续路线](roadmap.md)。

文档已按当前状态、结果、协议、历史归档分开整理；真实数据、权重、预测、日志和运行路径保留原位。入口见 [文档导航](README.md)。
