> 历史安装核验记录；当前用法见 [Skills](../skills.md)。

# Skills 来源、审查与安装

本次核验的是与用户 Windows 已安装文件**逐字节 SHA-256 相同**的公开仓库版本；不将同名搜索结果当作用户要安装的版本。

| Skill | 来源与固定 commit | SKILL.md SHA-256 |
|---|---|---|
| Ponytail | https://github.com/DietrichGebert/ponytail/tree/09c434ab73ca43761c5d8b0c5d341926c2fba710/skills/ponytail | 9badd22a32e19cb884e46dfa8d3aae82bb81dfa5484105b8b4f0262df823a509 |
| GrillMe (`grill-me`) | https://github.com/RobMitt/grill-me-skill/tree/31d61d68fc406f8cc2a944b4b10e6f23877720f1 | 4c4db26d8b01994cfa0ac755e105033c07f30dac4f6e8b84ca0b1d8c701eeab1 |

Ponytail 的 INSTALL.md 支持只安装 skills/ponytail/SKILL.md；完整插件另有 hooks，本次**没有安装完整插件或运行其 hooks/scripts**。选定子目录只有 SKILL.md，MIT。GrillMe 选定仓库只有 SKILL.md 和 README.md，是与本地文件一致的分发来源；没有声称它是最初作者，仓库没有提供独立 LICENSE，后续再分发需核查许可。

已阅读全部选定指令：Ponytail 约束最小实现与必要测试；GrillMe 是设计访谈流程，包含 Claude 的 AskUserQuestion 工具名称。安装内容无代码执行器、联网脚本、令牌访问或下载命令。Markdown 仍是助手指令，不能保证所有未来用途绝对安全；用户指令优先。本任务只核验/安装 GrillMe，不启动访谈。

先使用本地预置 skill-installer 脚本按 commit 下载到 `work/verified-skills`，随后列出全部内容并与已安装 Windows 文件核对哈希。向服务器只复制审查过的 Markdown，创建前确认目标目录不存在，不覆盖任何已有 Skill。没有执行第三方安装脚本、npx 安装器或全局插件钩子。

```text
Windows（原已存在，核验后保留）
C:/Users/15634/.codex/skills/ponytail/SKILL.md
C:/Users/15634/.codex/skills/grill-me/SKILL.md

Ubuntu（本次新安装）
/home/gyf/.codex/skills/ponytail/SKILL.md
/home/gyf/.codex/skills/grill-me/SKILL.md
```

服务器 Skill 在下一轮/新会话发现后可用；不会改变已经运行的科研任务。若该服务器上的旧会话未刷新 Skill 列表，开启新会话即可。

复现暂存命令（目标存在时 installer 拒绝覆盖）：

```text
python <Codex预置skill-installer>/scripts/install-skill-from-github.py --repo DietrichGebert/ponytail --ref 09c434ab73ca43761c5d8b0c5d341926c2fba710 --path skills/ponytail --dest <暂存目录> --method download
python <Codex预置skill-installer>/scripts/install-skill-from-github.py --repo RobMitt/grill-me-skill --ref 31d61d68fc406f8cc2a944b4b10e6f23877720f1 --path . --name grill-me --dest <暂存目录> --method download
```
