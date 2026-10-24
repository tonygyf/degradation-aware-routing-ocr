# 实验协议与真实推理准备

## 指标（A/B 共用）

文本只做 NFC 和 CRLF→LF，不去空格、不忽略大小写。CER 为 Levenshtein 编辑数总和 / max(参考字符总数,1)，允许超过 1；单条空参考使用分母 1，空白图像上的新增文字仍计入编辑数。中文按 Unicode code point 而非词或 grapheme cluster。插入数来自最优编辑路径，平局优先替换/匹配，再删除，再插入。

`vlm_call_rate = 实际选择调用数 / 图像数`。cached 运行中的调用表示选择读取对应缓存，不是网络调用。`insertion_proxy` 是插入数 / 参考字符数；`blank_nonempty_rate` 是空参考图像中非空预测的比例；`vlm_regression_rate` 是已调用样例中最终 CER 比原 OCR 更差的比例；`ocr_vlm_disagreement_rate` 是已调用样例中两者文本不一致比例。分母为零时返回 0，并同时保留 blank_samples/vlm_calls。

这些是**风险代理**：插入可能是正常纠错，不一致也可能是改善，不能当作真实幻觉率。真正的视觉无依据文字需要人工标注及可见性审查。

缓存延迟为 `OCR_ms + selected * VLM_ms`，串行执行；单独保存路由耗时均值和含路由训练的整次 wall time。p95 用 nearest rank。mock 延迟 12/120 ms 是手填 fixture；不声称代表 GPU 性能。真实缓存采集耗时包含单次请求/推理，不含模型加载；不等于线上队列、冷启动和端到端吞吐。真实 replay 需把配置 latency_kind 改为缓存的 `measured_serial_inference_ms_excludes_model_load`，不匹配将报错。

成本是显式 `cost_per_call` 的**声明单位**，不是货币、真实 token 或 FLOPs。预算在调用前判断，不超过 `max_calls` 和 `max_cost_units`；没有隐藏重试。真实 Qwen 返回的 usage 保存在缓存，但 token 成本预算和按区域估算成本属于下一阶段。

## 数据与防泄漏

一行一个 JSON，必填 id、group_id、image、image_sha256、split、reference、license。离线缓存另含 ocr={text,confidences,latency_ms}、可选 vlm={text,latency_ms,usage} 和 latency_kind。B 的 train 必须有 VLM 观测；未选择调用的测试条目可以不含 VLM。

同一源图及其退化、裁剪、多尺度变体必须共享 group_id。加载器阻止重复 ID、跨 split 同组和跨 split 相同图像哈希；哈希不能发现近重复或错误的 group_id，需要数据管理员先审计。测试 GT 仅在决策后计算指标；B 特征接口只能接收 OCR 字典。构造训练效益标签用到训练 GT/VLM 是监督学习所需，不用于测试决策。验证集用于未来阈值选择，测试集只做冻结方案的最终报告。

自制 60 个 SVG 只验证管线，其中含高置信度错误、低置信度正确、无检测和 VLM 造字等失败模式；预测与延迟是人工设计的。不要用它选择科研模型、宣称优于 A，或证明低光照泛化。SVG 不是当前真实接口的 PNG/JPEG/WebP 输入。

## 真实模型登记与采集（尚未执行）

1. 将 `configs/models.example.json` 复制为独立登记文件，填写真实数据授权/来源/划分、模型 ID、不可变 revision/文件哈希、模型许可、服务版本、资源与设备预留。完成审核后才设置 reviewed 和 data_authorized。模板没有预填虚构模型 revision。
2. OCR 接口目标为 `paddleocr==3.2.0`、CPU `paddlepaddle==3.2.0`，只接收已有本地 detector/recognizer 目录，关闭方向/矫正辅助模型。**适配器只做了伪响应契约测试，尚未验证真实版本兼容性**。真实依赖未纳入已验证的轻量环境；应另建 `.venv-real` 并生成该平台完整哈希锁文件，不在现有科研环境安装。
3. Qwen-VL 使用已有 OpenAI-compatible loopback 服务，如 SSH 转发后的 `http://127.0.0.1:8000/v1/chat/completions`；填写服务实际 model ID 和 checkpoint revision。代码不启动服务、不下载权重、不自动选择 GPU。请求 temperature=0，但 GPU/服务实现仍可能非确定。服务端应另记录 seed、图像分辨率、max_tokens=1024、提示词、推理引擎参数和 warmup。
4. 在另一次只读检查确认无资源冲突后，由资源所有者分配服务 GPU。现有科研容器无论 GPU 是否瞬时空闲均不复用。OCR CPU 首次验证建议 2–4 CPU、4–8 GB RAM；Qwen 显存需求必须按实际模型/量化/分辨率测量，当前没有已测值，也没有预留 GPU。
5. 用少量已授权 PNG/JPEG/WebP 写 raw 清单并实际计算文件 SHA-256，再冻结分组划分。原图/GT/缓存可能含敏感文本，只保存在私有目录；不要上传 Git。

```bash
python -m scripts.split_manifest --input data/raw/unsplit.jsonl --output data/raw/split42.jsonl
python -m src.cache --manifest data/raw/split42.jsonl --registry configs/models.reviewed.json --output data/cache/pilot-001 --allow-real
```

`src.cache` 是**全量反事实采集**：对每张图调用 OCR 和 VLM，用于训练标签与离线策略比较，实际采集调用率为 100%，不能把 replay 中的选择率当作采集时节省的开销。已有缓存不重复采集；输出目录必须不存在，失败保留 partial.jsonl 和登记信息，不自动恢复/覆盖。真实 adapter 异常直接终止该次运行，预算指标只在完整离线评估时生成。

将 A/B 配置复制后指向真实 observations.jsonl，更新 latency_kind，先在 val 上预定阈值/预算曲线，冻结后才评估 test。私有实验输出含 GT 和预测，分享前脱敏。
