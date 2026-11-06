---
product: Telegram manual trading assistant
surface: native Telegram bot
locale: zh-CN
omitted:
  - colors: Telegram client owns theme and contrast.
  - typography: Telegram client owns fonts and scaling.
  - spacing: Telegram client owns button sizing and message layout.
  - motion: Telegram client owns transitions and loading spinner.
---

# Telegram 订单管理设计

面向交易者的操作界面，沿用已有中文命令和 Telegram 原生按钮。
以订单编号、币种、方向和状态为首行，价格和数量保留有效精度。正文每行一项；按钮每行最多三个，中文短动词标识操作。不得通过颜色或 emoji 单独表达状态。

/orders和 /status table 进入同一管理列表；列表分页、挂单与持仓筛选、详情返回和刷新一致。空列表说明当前无订单；刷新失败标记数据未确认，不显示可执行交易按钮。

详情字段修改用 ForceReply 收一个数值；软止损和延伸条件分步录入并原子保存。显示输入中的字段及已填值，允许取消。只有关联提示消息的本人回复被接收。过期或状态变化必须刷新后重试。

快速全平、撤单、挂单改价显示对象、数量及操作含义后二次确认。处理中及时清除按钮回调等待状态；未知结果明确要求核实，不能显示成交成功或自动重发。未完成 TP 可以调整，已完成阶段和生效保护不可回退。

运行时所有文案和键盘由 user_data/strategies/order_panel.py 生成；业务约束由 order_management.py 与现有策略/RPC 共同校验。主机平台控制布局，不引入网页、CSS 或替代 Telegram 的设计系统。离线渲染/交互测试验证文本、键盘、错误恢复和授权绑定；实盘不执行测试交易。

订单列表正文显示每笔已记账余仓的浮盈亏 USDT 金额与收益率，金额保留四位小数、比例保留两位，均带正负号；详情采用相同口径。底部汇总所有页和筛选外的持仓，明确标注范围。金额和比例沿用原生手续费计算，已实现分段止盈不混入当前浮盈亏。未成交挂单不显示虚假的零收益；任何未知价格或待记账成交必须标记不可用，总额只显示已知小计并注明汇总不完整。显示查询时间并保留刷新按钮，短订单按钮与正文编号对应。

## 条件监控

`/watch` 与订单管理使用相同的中文原生按钮、六条分页、详情返回和刷新方式。五种模式统一向导：选模式、币种、方向、周期，再逐项录入价格；每步显示已填条件，输入错误保留草稿，支持上一步和取消。确认页显示自然语言规则，支持直接修改单个字段。只有点击“确认创建”才保存监控，重复确认不新增；复制必须重新确认。

输入和按钮的身份/消息绑定由 `order_panel.py` 的共享面板工具负责；`watch_panel.py` 持有监控向导和文案，`condition_watch.py` 持有价格规则与生命周期。两种面板共用回复入口，依据提示消息分流，不接受无关回复。未完成草稿10分钟过期，重启后重新创建；已创建监控持久保存。创建、取消、结果与行情不可用均使用文字状态。监控确认只代表自定义条件满足，不能显示“已开单”或声称趋势成立。

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Select/Listbox | Telegram InlineKeyboard、共享面板发送与token工具 | DESIGN.md | Telegram原生按钮 | tests/test_watch_panel.py、tests/test_order_panel.py |
| Form | ForceReply、消息与本人绑定、共享回复路由 | DESIGN.md | 新建、修改草稿 | tests/test_watch_panel.py |
| Date | UTC时间文本、明确行情健康与确认窗口 | README.md | 只读时间、有效期按钮 | tests/test_condition_watch.py |
| Toast | AlertQueue，条件状态和提醒同事务；5分钟过期 | README.md | Telegram消息 | tests/test_condition_watch.py、tests/test_alert_delivery.py |
| CRUD | WatchStore，取消、复制、重启恢复与去重 | README.md | 确认创建、取消、复制 | tests/test_condition_watch.py、tests/test_watch_panel.py |
