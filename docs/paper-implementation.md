# 原论文的软件实现与运行入口

更新：2026-10-31。用户要求在作者暂未回复时，继续按公开正文和补充材料尽可能补齐代码。本页区分已知公式、独立实现约定和仍需作者数据的验证；不表示整篇论文已复现。

依据：[IJDAR 正文与补充材料入口](https://link.springer.com/article/10.1007/s10032-026-00620-7)。本地补充文本保存在 `outputs/reproduction-sources/supplement.txt`，不是作者代码包。

## 实现覆盖

| 论文内容 | 本地实现 | 当前验证 |
|---|---|---|
| clipped macro-CER、Eval-Trans、历史 ConfRoute | `ccocr.py`、`confroute.py`、原采集脚本 | 已有 CC-OCR 五系统及 GPU Paddle 的真实冻结结果 |
| 内部 Baseline 与 Cal 四候选 | `baseline.py`、`evaluate_baseline`、`replay_calibration` | 软件检查通过；缺作者 579 文档角色与真实观测 |
| Dev scalar/box-min 前沿及 Cal λ | `replay_calibration --role Dev` / `--lambda-sensitivity` | 0.05 步长全网格；Dev 不选择阈值，λ 仅用 Cal，历史 0.90 保持不变 |
| SHC 双参考、flag、类型与参考重叠分组 | `shc.py`、`evaluate_shc` | 软件检查通过；缺原作者双参考/关键片段 |
| SHC 阈值敏感性、空文本过滤、InternVL 回退 | `shc_routing.py`、`replay_shc` | 0.85/0.90/0.95、真实 always 端点、空文本 precision/recall；InternVL 单独字段，不重命名为 Qwen |
| 风险—覆盖与两种 ranking 目标 | `shc_analysis.py`、`analyze_shc` | 全集替换 CER、随机对照、AUROC/AP；route-benefit 与 OCR operational-flag 目标分开 |
| L2 学习路由器 | `shc_learned.py`、`configs/paper_shc_router.json` | 六组特征接口；仅 Tau_dev/train 拟合；未知密度特征要求定义来源；无 Test 阈值拟合 |
| 提示与失败类型比较 | `shc_prompts.py`、`compare_shc_prompts` | 固定 P0/P2/P3/P2+P3/P0+P3；检查逐样本身份；只在 Prompt_dev 报告 CER 最优，Test 仅探索性比较 |
| canonical-source 重叠与过滤敏感性 | `paper_sources.py`、`audit_paper_sources` | Baseline×SHC 4×4 源图矩阵、SHC 跨角色父图、开发集保留 ID；不改原角色或 Test |
| 分层人工 flag 审计重加权 | `shc_audit.py` | 补充材料公布的 Qwen 汇总计数可复算 precision/recall/修正率；没有重做人工标注 |

## 运行方式

下面均为接口示例，作者 SHC/Baseline 输入尚未到位；这些路径不表示本地已有真实标注。所有命令在项目根目录通过 Docker 运行。输出文件须不存在。

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
# 默认 Cal 四候选；可选仅 Cal 的 λ 敏感性
docker compose run --rm -T dev python -m scripts.replay_calibration --observations outputs/author/cal.jsonl --lambda-sensitivity --output outputs/author/cal-replay.json
# Dev 两种置信度的描述性前沿，不执行选阈值
docker compose run --rm -T dev python -m scripts.replay_calibration --role Dev --observations outputs/author/dev.jsonl --output outputs/author/dev-frontier.json
# SHC 固定敏感性与独立 InternVL 回退
docker compose run --rm -T dev python -m scripts.replay_shc --observations outputs/author/shc-test-paired.jsonl --sensitivity --include-internvl --output outputs/author/shc-routing.json
# 风险—覆盖、两类 ranking，以及可选的 Tau_dev 学习器
docker compose run --rm -T dev python -m scripts.analyze_shc --observations outputs/author/shc-test-paired.jsonl --train outputs/author/shc-tau-train.jsonl --router-config configs/paper_shc_router.json --output outputs/author/shc-analysis.json
# 缓存提示比较：Prompt_dev 或探索性 Test，不能混在一次运行
docker compose run --rm -T dev python -m scripts.compare_shc_prompts --observations outputs/author/shc-prompts.jsonl --output outputs/author/shc-prompt-report.json
# 只读原角色及源图元数据，不重建或重划数据集
docker compose run --rm -T dev python -m scripts.audit_paper_sources --baseline outputs/author/baseline-source-map.jsonl --shc outputs/author/shc-source-map.jsonl --output outputs/author/source-overlap.json
```

仅做 ranking/风险—覆盖时省略 `--train` 与 `--router-config`。CLI 均要求 `synthetic=false`，合成样例通过单元测试和 `scripts.smoke` 检查，不生成正式科研报告。

## 独立观测格式

采用本项目交换格式，不能将其当作作者已公开的文件结构：

- Baseline Cal/Dev：通用 manifest，`role=Cal/Dev`、`split=val`、`reference` 和配对 `ocr`/`vlm`；box-min 前沿额外需要 `ocr.confidence_boxes`。
- SHC 配对：`id`、`role`、`reference_visible`、`reference_full`、`failure_type`、`critical_span`、`ocr`、`vlm`。`vlm` 约定为 Qwen；可选 `internvl` 含独立的 `text/latency_ms`。每个 OCR 需要独立 box-cache 的 `confidence_boxes[].confidence`，不回退到 scalar。
- 学习路由：训练额外要求 `role=Tau_dev, split=train`，最终输入 `role=Test`；禁止相同样本 ID 进入训练和 Test。源图复用按原论文报告，不假称独立。`source_corpus/source_image` 用于核查父图交集。
- 能确定的特征由 box-cache 计算：min/mean/geometric mean、框数、空文本指示。密度与每框字符数在原文中缺少精确定义，必须通过 `ocr.paper_features` 提供，附非空 `ocr.paper_feature_provenance`；缺失时报错。请求的三特征组需要 `text_density`；full 组还需 `character_density/characters_per_box`。
- 提示缓存：SHC 标注字段加 `model`、`prompt`、`role`、`prediction`、`latency_ms`。所有模型/提示必须有相同 ID 和标注，每个模型含 P0；丢失或重复样本均拒绝。Test 最优提示不被选择，也不替换 P0 主比较。
- 源图元数据：每行 `id,role,source_corpus,source_image,synthetic`。必须提供父图字段；不从裁剪后的 `image_id` 猜测源图。Windows/Unix 路径、扩展名和大小写被规范化，已支持的语料别名写入报告。

## 尚未核验的实现约定

以下约定是为了让独立实现可运行而明确登记，不等于作者参数，不能据此保证复现分数：

| 项目 | 当前约定 |
|---|---|
| L2 优化器/缩放 | C=1、StandardScaler、lbfgs、max_iter=1000、class_weight=None、seed=42；版本、系数、scaler、训练 ID 和配置记录进报告 |
| 学习决策阈值 | requested/route-benefit 使用补充表 15 公布的 0.459，作为固定阈值；作者优化参数未知，因此概率尺度不保证一致。阈值必须属于对应 target，不能在 Test 上重选 |
| 风险—覆盖 | Paddle 保留数为 floor(coverage×N)，高置信度优先，并列按 ID；随机对照默认 1000 次、seed 42、PCG64、总体标准差。作者取整/并列/随机试验次数未提供 |
| ranking | AUPRC 用 sklearn average precision；单类别返回 null，不填造 AUROC。作者具体 PR 面积算法未取得 |
| λ | CER 小数 + λ×串行秒数；并列依次取少调用、低阈值；这是显式独立约定，不恢复历史选阈值过程 |
| 端点计时 | Dev 网格串行计时含 OCR；另报告 VLM-only 耗时，避免混淆 always-VLM 与先 OCR 后全升级。原始计时缓存仍需核验 |
| 空框与几何均值 | 没有框时相关 cheap confidence 为 0；有 0 分数时几何均值为 0。SHC 最小值与 scalar 平均置信度继续分开 |

## 验证与剩余范围

完整 Docker 检查：95 项测试、Ruff、格式和 CPU 冒烟通过，日志 `outputs/project-cleanup-checks-20261031.log`。冒烟包含上述分析组合与 JSON 序列化；所有样例均明确为合成软件检查。

公布汇总计数的独立算术复算：auto-positive/negative 池为 559/3338，正池审计 TP/FP=89/11，负池审计 FN/TN=1/99；得到 89.0%、93.7%、12.77%、13.62%，与补充表舍入值一致。记录 `outputs/published-flag-audit-replay-20261031.json`。这是公布计数的重计算，未取得原人工标签，不是新的独立人工审计。

仍未完成的科研部分包括作者数据身份和真实内部/SHC 各系统结果、模型与环境精确对齐、人工双标重做及完整敏感性复算。字符相似度/WER 的作者实现约定、密度特征定义等仍缺；不以相近名称的指标或默认提取器代替后宣称等价。取得作者包后的严格核验继续按 [材料清单](author-material-request.md) 推进，当前真实外部结果见 [CC-OCR 报告](ccocr-results.md)。
