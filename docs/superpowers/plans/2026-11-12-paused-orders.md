# Paused order panel plan

Goal: Include paused initial entry intents in orders and provide safe cancel from detail.
Spec: docs/superpowers/specs/2026-11-12-paused-orders-design.md
Execute inline with TDD and one fresh review, following prior authorized commit/push/deploy workflow.

- [x] Reproduce missing plan-only rows; test filters/paging/dedup/PnL and durable cancellation.
- [x] Add plan snapshots/stable revision checks and native detail/confirmation callbacks.
- [x] Check cancel-before-resume race, unknown receipts, button ownership and double clicks.
- [x] Update README/DESIGN, run project hooks/full pinned offline suite, independent review.
- [x] Snapshot, commit/push/deploy; verify actual runtime hashes and fresh heartbeat without live test trades.

## 验证与审查记录

- 基线72e226e，独立副本/tmp/paused-orders-coimvlts/repo。新增6项首先暴露缺少列表行/管理入口；实现后6项通过（4.845s）。
- 自查回归先证明已观察但未记账的成交没有使确认失效；将observed_quantities纳入稳定版本后修正。首轮完整462项通过（71.120s）。
- 独立审查P2：SQLite删除零成交交易后会复用trade.id，单凭旧trade_id错误合并同币种新仓。新增同/反向ID复用回归先失败，现关联须币种/方向一致且有入场order_id或当前req标签证据；不凭数值trade_id。

- 最终新增8项通过（5.431s），完整463项通过（85.205s）；项目hooks/Black/flake8/diff检查通过。独立复核P2修复通过，另运行相关37项通过。
- 候选镜像cea9a8ed1c202745a243edc6e922eb47ec46cf290c97826d08bd225f178d3658按真实策略/RPC装载顺序验证成功，原生STATUS格式保留。
- 发布前源码与在线数据库/计划/健康快照在忽略目录 .assistant-archive/paused-orders-20261112T073523Z。未发送测试交易或创建实盘测试计划。

## 发布结果

- 代码2558aaa已推送main；07:37:37 UTC切换候选镜像，启动后产生新RUNNING交易心跳，重启次数0、启动错误0、守护故障0。
- 实际容器RPC order_panel.py与order_management.py两项哈希匹配候选源码。
- 切换时再次做在线数据库与计划快照。回滚镜像tag rollback-before-paused-orders-20261112T073523Z；回滚仅还原源码和镜像，不覆盖后续实盘数据库。
- 未发送实盘测试指令、未取消任何用户计划或修改实际订单；按钮取消的持久性、未知回执、成交竞态和重复点击均由离线回归验证。
