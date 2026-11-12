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

底部快捷命令与消息上的操作按钮分开管理。打开订单/监控面板时，以静默消息附带持久快捷键，再展示操作卡片；ForceReply期间允许切换到文字输入，完成、取消或退出输入后立即恢复快捷键，不依赖下一次定时推送。连续价格输入不反复恢复，普通按钮切换不额外刷恢复消息。共享输入/恢复逻辑由 `order_panel.py` 负责，发送到原会话和话题；回复仍绑定原提示消息。Telegram客户端拥有最终键盘显示行为。

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| Select/Listbox | Telegram InlineKeyboard、共享面板发送与token工具 | DESIGN.md | Telegram原生按钮 | tests/test_watch_panel.py、tests/test_order_panel.py |
| Form | ForceReply、消息与本人绑定、共享回复路由 | DESIGN.md | 新建、修改草稿 | tests/test_watch_panel.py |
| Date | UTC时间文本、明确行情健康与确认窗口 | README.md | 只读时间、有效期按钮 | tests/test_condition_watch.py |
| Toast | AlertQueue，条件状态和提醒同事务；5分钟过期 | README.md | Telegram消息 | tests/test_condition_watch.py、tests/test_alert_delivery.py |
| CRUD | WatchStore，取消、复制、重启恢复与去重 | README.md | 确认创建、取消、复制 | tests/test_condition_watch.py、tests/test_watch_panel.py |

## 自动通知

所有自动推送共用 `user_data/strategies/notification_format.py` 的纯文本卡片，不引入表格、HTML、装饰分隔线或 emoji 状态编码。先显示事件、对象和状态，其后是结论/建议，再按短段列关键数字，最后给下一步与一个明确标注的北京时间。历史触发时刻、K线收盘/确认时刻属于证据，可单独保留并标明含义；不追加重复的 UTC 状态时间。一个数字一项，同类短字段可并列。

覆盖持仓、行情、异动及后续、极端放量、条件监控、入场计划/回执、止损保护、退出和独立守护。原生开平仓也采用中文相同结构；提交与成交回执、分批退出与整笔退出、待确认与已核验分别表述。原生 Markdown 发送时整体转义，队列与守护直接发送纯文本，币种下划线及错误文本不能破坏版式。

持仓先列剩余仓位的浮盈亏和建议；走势分周期、支撑阻力分行，原保护与下一有效TP单列。延伸目标沿用实际执行目标。未知报价、待记账、过期核验和保护不足显式展示，不以零替代。详细原计划仍可在 /overview、/order 查看。通知不改变执行、阈值、队列优先级、去重或重试规则。

| Capability | Canonical owner | Source of truth | Allowed variants | Verification |
|---|---|---|---|---|
| 自动推送卡片 | notification_format.py | DESIGN.md | 原生Markdown转义、队列/守护纯文本 | test_notification_format.py、各通知语义回归、test_alert_delivery.py |
