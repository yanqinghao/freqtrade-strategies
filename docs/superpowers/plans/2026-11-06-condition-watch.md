# Condition Watch Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 一次性交付五种条件监控与 Telegram 引导操作。
**Architecture:** `condition_watch.py` 负责规则、持久状态与公开行情后台；`watch_panel.py` 负责原生 Telegram 向导。复用 AlertQueue 的数据库实现状态与通知原子提交。
**Tech Stack:** Python / SQLite / python-telegram-bot / pinned Freqtrade 2026.10。
**Spec:** `docs/superpowers/specs/2026-11-06-condition-watch-design.md`

## Global Constraints
- 无新依赖，无自动交易；仅支持 BTC/ETH/SOL 与15m/1h/4h/1d。
- 10秒后台调度，状态持久保存，行情过期不推断；通知5分钟过期且低于关键持仓通知优先级。
- 用户要求直接完成；在当前干净工作区原地实施，不另设审批流程。

## Review Focus
- 创建时间在K线中间、触区发生在之前的影线不得误触发（任务1）。
- 突破与回踩不得用同根OHLC臆测先后（任务1）。
- 取消与后台完成竞争，不得复活或继续发旧信号（任务1）。
- 两种面板的ForceReply不能互相吞掉；旧按钮不能重放（任务2）。
- 调度接线、镜像内导入路径必须真实覆盖（任务3）。

## Task 1: 条件引擎
- [x] 增加 `tests/test_condition_watch.py`，验证五种多空、时序、无效输入、期限、去重与事务；先运行失败。
- [x] 实现 `WatchStore(queue).create(spec, owner, request_id, now)`、`list(owner)`、`cancel(id, owner)`、`apply(snapshot, result, message, now)`；纯函数 `evaluate(watch, rows, quote, now)`。
- [x] 实现 `ConditionMonitor`、分币与周期缓存、异常隔离和数据状态。
- [x] 运行条件测试及既有通知/行情测试。

## Task 2: Telegram向导
- [x] 增加 `tests/test_watch_panel.py`，先验证缺失入口，随后覆盖完整五模式向导、错误修正、返回、取消、复制、身份隔离、重复确认。
- [x] `watch_panel.py` 实现 `WatchPanelMixin`，注册 `/watch`、回调和共享回复路由，复用订单面板token/发送基础能力。
- [x] 更新 DESIGN.md 与帮助入口；运行 Telegram 回归。

## Task 3: 接线与交付
- [x] 策略初始化/启动/循环/清理接入后台，Docker包含面板模块。
- [x] README记录规则、操作与时效限制。
- [x] 全套离线回归、UI静态审计、候选镜像构建；独立最终代码审查，修复重要发现。

## Ledger
- 初始工作区干净；用户要求全部实施，保持当前工作区，不生成新交易监控实例或执行测试交易。

- Task 1 complete: 新规则测试先因模块缺失失败，随后通过；含五模式双向、前置顺序、窗口与行情时间、取消竞争、重启与事务回滚。
- Task 2 complete: 向导测试先因入口缺失失败，直接改单项测试先失败后通过；覆盖所有模式创建、复制、取消、身份、无关回复、坏价格、过期按钮与重复确认。
- Task 3 complete: 候选镜像 `freqtrade-strategies-freqtrade:condition-watch-candidate` 构建通过；按真实策略临时导入路径与镜像内Telegram模块完成注册/启动/清理验证。
- Final review: 独立review_condition_watch审查发现两项Important，无Critical。修复前均用离线测试复现，修复后通过：跨收盘请求按实际quote时间排序；每条通知发送前重新核对当前版本与有效期，跳过取消/替换的通知。
- Final validation: 候选镜像断网、工作区只读运行 `python -m unittest discover -s /workspace/tests`，348 tests，53.664s，OK。日志 `/tmp/condition-final-tests.log`。`git diff --check` 通过。
- UI validation: 原生Telegram模拟传输交互测试通过；未向真实TG发送测试消息。UI静态审计按项目策略源码范围、DESIGN.md所有权映射运行，0 findings；`/tmp/condition-ui-audit.json`。首次默认审计误扫`.venv`依赖的JS，改为显式项目源码范围，没有修改第三方代码。
- Packaging ruling: Docker构建发现白名单遗漏新面板，已补入`.dockerignore`并重新构建通过。没有增加运行依赖或服务。
- Release status: 尚未重启生产服务；只读检查为实盘、容器运行中、本地1笔未平仓记录。候选已准备完毕，上线重启时机待用户确认。未提交或推送代码，未执行交易。
- Release follow-up: 用户澄清唯一未结束记录为挂单；交易所只读回执确认 ETH 空头限价2278、数量0.131、filled=0、status=open，ETH实际仓位为空。退出配置cancel_open_orders_on_exit=false。
- 已建立本地私有备份 `.assistant-archive/20261106T062737Z-before-condition-watch`（SQLite在线备份、deploy配置及HEAD代码快照），保留rollback-before-condition-watch镜像。
- 自动审批拒绝 `docker compose up -d --no-deps --no-build freqtrade`：用户仅澄清挂单状态，未明确批准此时重启实盘机器人。没有执行重启，latest预发布标签已恢复为原运行镜像。候选镜像继续保留，等待明确重启授权；不得绕过此拒绝。
- Release complete: 用户明确回复“允许”后，于2026-11-06 06:29 UTC执行更新并重启。运行镜像已为 `sha256:3f97c007a0c559d73eff51af79d41ec147a2659cf0a2370787010a9e91b88eec`，RUNNING且交易心跳恢复；运行镜像内/watch命令与底部键盘验证通过，后台已创建condition_watches表，未代用户创建任何监控。
- 更新后交易所只读复核：原ETH挂单仍open，价格2278、数量0.131、filled=0、remaining=0.131，ETH实际仓位为空。保留原订单，没有撤单或重复下单。启动日志无ERROR/Traceback/Condition异常，健康快照及交易循环新鲜。
- Git收尾：用户指出未提交改动后，核对17个文件均属于本功能；提交前再次离线运行全套回归，348 tests、49.885s、OK（`/tmp/condition-commit-tests.log`），`git diff --check`通过。本地提交包含实现、集成、测试、文档与部署验证记录，不包含运行配置、凭据或数据库。
- 提交钩子收尾：应用仓库Black/字符串格式，修正长行、未使用导入、测试变量名，并将按钮绑定lambda替换为等效functools.partial。修正后再次全量回归348 tests、55.047s、OK（`/tmp/condition-commit-final-tests.log`）；末次字符串格式调整经AST比较无语义变化。此次仅代码仓库收尾，不重启运行服务。
