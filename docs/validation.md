# 软件验证

最新完整检查：Docker CPU 环境 **95 tests passed**，`ruff check .`、`ruff format --check .`、`python -m scripts.smoke` 均通过。本次合并论文分析入口的重复 JSONL 校验与报告写入；空输入、未明确标记真实数据的输入和合成输入均被拒绝。报告以独占创建方式写入，SHC 单系统评估也不再覆盖旧报告。新增检查覆盖空输入、Unicode 与已有证据保护。日志：`outputs/project-cleanup-checks-20261031.log`。

检查覆盖指标、路由边界、数据身份与划分、冻结模型校验、外部系统配对、原型和采集入口。合成样例与 mock 延迟只验证软件，科研结果及验收结论见 [复现状态](paper-reproduction-status.md) 和 [真实结果](ccocr-results.md)。

复查命令见 [项目 README](../README.md)。基线代码变更需完成全部四项检查；文档整理检查本地链接和归档完整性。

2026-10-24 至 25 的跨平台、Docker 迁移及早期准备记录见 [历史验收](archive/validation-2026-10-24-to-25.md)。历史虚拟环境命令不再作为开发入口。

长度差快速判定与独立编辑距离实现的 1,000 组随机文本检查通过；另在五系统各 900 张真实观测上重算，全部指标及重复诊断与原报告完全一致。证据 `outputs/cer-length-bound-crosscheck.json` 保存两版评估器和原报告哈希，原代码保留在 `outputs/reproduction-sources/ccocr-evaluation-before-length-bound.py`。这证明该优化保持现有结果，不证明作者配置一致。

内部 Cal 固定候选回放的软件检查已通过：0.80/0.85/0.90/0.95、置信度舍入与边界、无框、逐图串行延迟、无 Test 调参及合成输入拒绝。没有作者真实 Cal 数据，未生成或宣称科研 Cal 分数。

2026-10-31：新增论文分析模块的边界、训练隔离、目标/阈值一致性、Test 不选提示、样本身份配对与源图过滤测试通过。CPU 冒烟覆盖组合及 JSON 序列化，使用合成数据；公布人工计数的复算另标为算术验证，不算新人工实验。详见 [实现范围](paper-implementation.md)。
