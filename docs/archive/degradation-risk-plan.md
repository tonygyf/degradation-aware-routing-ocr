> 历史记录，保留当时结论；当前进度见 [复现状态](../paper-reproduction-status.md)。

# 退化与升级恶化风险路由：第一阶段

2026-10-27 执行优先级纠正：先回到[全论文基线复现与验收](../paper-reproduction-status.md)。本页记录扩展准备，不计作论文复现结果；原型保留，暂不继续扩展算法。

用户于 2026-10-27 确认：固定 PaddleOCR＋Qwen，优先做退化与风险感知策略；
风险首先指调用 Qwen 后 CER 变差；独立开发数据使用公开数据的官方训练划分并先做重叠审计。
这不是已验证的新算法或真实幻觉检测器。

## 已有 CPU 原型

`src/routing/risk.py` 提供图像统计和双预测头：

- 输入：已有五个 OCR 特征，加亮度均值、灰度标准差、梯度 RMS。图像统计不等于有监督的退化强度标签。
- 收益监督：`g = clipped_CER(OCR) - clipped_CER(Qwen)`，采用论文的空白归一化、逐图截断口径。
- 风险监督：`r = 1[g < 0]`；平局不是恶化。
- train 上分别拟合 `StandardScaler + Ridge` 和 `StandardScaler + LogisticRegression`。原型固定 Ridge alpha=1，不声称已选出最佳模型。
- 离线批次效用为 `predicted_gain - risk_weight * predicted_regression_probability`，正效用样例按排名至多调用 K 次。权重必须后续在独立 val 选择并冻结；当前没有用 CC-OCR 选择参数。
- 决策接口只接收 OCR 和图像统计，不接收 GT 或 Qwen 输出。图像 ID 只用于稳定打破平局。
- 输出概率未校准，不能声称有形式化风险保证。调用预算是上限，正效用不足时实际调用更少；同预算比较必须同时报告实际调用数。

`python -m scripts.smoke` 包含合成缓存和 mock 图像统计的原型检查；它不是科学实验。
未审核的全图参考文本会在原型训练入口被拒绝。

## 独立开发数据准备

候选为 [Total-Text 官方训练集](https://github.com/cs-chan/Total-Text-Dataset/tree/59538b936fc7a55025987960270267d8d081e72e/Dataset)，
配套官方 refined training groundtruth。固定来源和归档观测 SHA-256 在 `configs/totaltext_development.json`。
首次下载没有官方公布的校验和；记录的是官方分发源首次完整下载的观测哈希，后续按此拒绝变化。
许可依据为 [Dataset Research Use Terms](https://github.com/cs-chan/Total-Text-Dataset/blob/59538b936fc7a55025987960270267d8d081e72e/DATASET_TERMS.md)：
独立非商业学术研究；底层图像权利另行适用，不将软件 BSD 许可混用于图像。数据和标注不提交 Git。

已在 CPU 容器中执行 `scripts.prepare_totaltext_development`：

- 官方训练图 1,255 张，与冻结的 900 张外部测试图核对来源 ID、文件哈希、解码像素哈希及 64-bit dHash。
- 发现三对跨开发/外部测试的 dHash-zero 图像。逐对目视复核显示相同源照片、不同编码，已隔离对应三张开发图。
- 开发集内部 dHash 距离不超过 4 的图像连通分组，避免疑似近重复跨 train/val。
- 保留 1,252 张候选，按源组和 seed=42 固定哈希作 80/20 候选划分，实际为 train 1,007、val 245。
- 当前 `training_ready=false`：官方逐词标注顺序尚未作为全图阅读顺序审核，也没有这批图的真实 OCR/Qwen 观测。
- dHash 仅作保守筛查，不能保证发现所有裁剪、变换或同场景重复；源图分组仍需保留审计边界。

本地产物：`data/raw/totaltext-development/prepared/{audit.json,overlap-review.json,development.jsonl}`。
另准备了 20 张参考文本审核候选（12 train/8 val、按组确定性抽样），位于 `reference-review-pilot.jsonl`，目前均未审核。
900 张 CC-OCR 的清单、GT 和原图保持冻结。

CPU 重跑审计需指定新的输出目录，依赖已有完整归档：

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose -f compose.yaml -f compose.ccocr.yaml run --rm -T dev python -m scripts.prepare_totaltext_development --output outputs/totaltext-audit-repeat
```

## 进入真实策略实验前的工作

1. 确定并审核第一批全图转录参考，保留逐词原标注和忽略区域信息，冻结所用样本及阅读顺序规则。
2. 在独立登记和允许的资源范围内采集这批开发图的真实 Paddle/Qwen 反事实观测；现有 GPU 例外只覆盖 CC-OCR，不自动扩展至新开发数据。
3. 完成 CC-OCR 三系统真实基线与必要统计分析，再保存可复查的基线提交/结果哈希；当前准备及原型代码尚未形成这一科研冻结点。
4. 从冻结提交进入 `feat/degradation-risk-router`。用 train 拟合，val 选风险权重与调用预算，最终再评估冻结外部 test。
5. 比较 confidence-only、OCR-only learned、加入图像统计、去掉风险项四组；同调用预算下报告 CER、调用后恶化率、空输出率、实际调用及计算开销。算法创新性与真实收益尚待证据。
