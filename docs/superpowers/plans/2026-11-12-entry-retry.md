# Entry recovery retry implementation plan

> Use superpowers:executing-plans inline; fresh reviewer before integration.

**Goal:** Technical pre-submit rejections must not restart the market quiet window; expose actionable bounded diagnostics.
**Architecture:** Keep existing controller/ledger and two independent clocks. Typed local validation errors allow safe messages; unknown external exceptions expose type only.
**Tech Stack:** Existing Python/Freqtrade 2026.10, SQLite, native Telegram. No dependencies.
**Spec:** docs/superpowers/specs/2026-11-12-entry-retry-design.md

## Constraints and review focus

- Only definitively rejected pre-submit attempts may retry after60s. Unknown and terminal states never gain permission.
- New shock and continuity gaps still enforce900s; all quote/trend/SL/TTL/risk checks remain in the final boundary.
- Preserve quote timestamps, accounting, original entry/deadline and old durable metadata; no live database migration.
- Diagnostics must survive restart, clear current error on fresh successful evaluation, and redact unexpected exception text.

## Task: Retry policy and diagnostics

- [x] Run existing pause tests in pinned offline runtime as baseline.
- [x] Reproduce rejection extending quiet window; test both directions, 60s boundary, repeated failures, new shock, continuity, unknown and terminal behavior.
- [x] Add safe rejection/data reasons and query regressions; first observe failure, then minimal controller/formatter changes.
- [x] Run hooks, full offline suite, independent review and resolve findings.
- [x] Commit/push, snapshot, deploy per prior authorization; verify actual source/image and fresh healthy loop. Do not send test trades.

## 用户追加纠正

原生 RUNNING 与开单后通知应恢复上游原格式；自定义样式仅用于行情、持仓、异常。移除前版原生开平/撤单/强平排版覆写及STATUS分支，保留公共行情文本转义与运行WARNING/EXCEPTION；新增上游实际文案回归。与恢复规则一起交付，不中止原任务。

## Ledger

- Baseline pause37项通过（17.340s）。新增回归首先复现 quiet_after 被移动、guard隐藏错误、query缺少诊断；21项中5失败1缺字段错误，其余为测试导入意外重复了旧测试，已改为模块导入。

- 新增重试回归6项通过（5.976s），通知回归4项通过（2.949s）；后者直接检查上游STATUS/ENTRY/ENTRY_FILL/EXIT/EXIT_FILL/ENTRY_CANCEL原文，异常与行情仍使用既定自定义通道。
- Ruling: 保留旧计划历史quiet_after，不猜测迁移；重启后的连续观察要求仍保证900秒有效覆盖。同一拒绝尝试重复核对不顺延retry_after。

- 完整离线453项通过（85.275s）；hooks/Black/flake8/diff检查通过。独立审查未发现阻断项，另独立复跑新增重试与通知10项通过。
- 候选镜像 e0ca8a2c53a7695a13cf683d988f64ce474c072e463b017dda6811b166ea8273，RPC按实际策略装载顺序验证成功；STATUS与三个原生交易formatter均回到 _telegram_base。
- 发布前只读核实：ARX原计划已于14:24:32北京时间自行恢复，订单212402555、限价0.201、497 ARX，最新本地回执open/filled0。不是由测试提交。
- 已备份切换前源码和在线数据库/状态至忽略目录 .assistant-archive/entry-retry-20261112T063245Z。部署不迁移数据库，不修改现有计划计时。

## 发布结果

- 代码12dd823已提交并推送main，全部提交hooks通过。06:33:55 UTC切换候选镜像；06:34:06进入RUNNING，新进程交易心跳持续更新，重启次数0、启动错误0，独立守护故障0。
- 实际容器三个入场模块与RPC Telegram四项哈希匹配候选源码；现有ARX订单212402555在切换后仍为open/filled0，本地回执与计划持续更新，新增完整检查诊断已落库。
- 回滚镜像tag rollback-before-entry-retry-20261112T063245Z；切换时再保存在线数据库与计划快照。只回滚源码/镜像，不覆盖后续实盘数据库。未改写已有计划历史计时，未执行测试交易。
- 审查非阻断文档措辞已改为“助手自定义文本经Markdown发送时整体转义”，原生通知沿用上游。
