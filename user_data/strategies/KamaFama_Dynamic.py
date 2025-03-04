# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from functools import reduce
from pandas import DataFrame, Series

# --------------------------------
import talib.abstract as ta
import pandas_ta as pta
import pandas as pd  # noqa
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IntParameter,
)
import logging
import os
import json

logger = logging.getLogger(__name__)

pd.options.mode.chained_assignment = None  # default='warn'

# ------- Strategie by Mastaaa1987


def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
    of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
    Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
    of its recent trading range.
    The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe['high'].rolling(center=False, window=period).max()
    lowest_low = dataframe['low'].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe['close']) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
    )

    return WR * -100


class KamaFama_Dynamic(IStrategy):
    INTERFACE_VERSION = 2
    can_short = True

    @property
    def protections(self):
        return [
            {
                'method': 'LowProfitPairs',
                'lookback_period_candles': 60,
                'trade_limit': 1,
                'stop_duration_candles': 60,
                'required_profit': -0.05,
            },
            {'method': 'CooldownPeriod', 'stop_duration_candles': 5},
        ]

    minimal_roi = {'0': 0.087, '372': 0.058, '861': 0.029, '2221': 0}
    cc_long = {}
    cc_short = {}

    # 策略模式状态跟踪
    pair_strategy_mode = {}

    # 停止开单记录
    stop_entry_records = {}

    # 回测模式下跟踪每个交易对的当前蜡烛时间
    current_candle_date = {}

    # Stoploss:
    stoploss = -0.345

    # Sell Params
    sell_fastx = IntParameter(50, 100, default=84, space='sell', optimize=True)

    # 需要添加的新参数 - 为做空策略专门优化
    buy_fastx_short = IntParameter(0, 50, default=16, space='buy', optimize=True)

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True

    use_custom_stoploss = True

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99,
    }

    # Optional order time in force.
    order_time_in_force = {'entry': 'gtc', 'exit': 'gtc'}

    # Optimal timeframe for the strategy
    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 400

    plot_config = {
        'main_plot': {
            'mama': {'color': '#d0da3e'},
            'fama': {'color': '#da3eb8'},
            'kama': {'color': '#3edad8'},
        },
        'subplots': {
            'fastk': {'fastk': {'color': '#da3e3e'}},
            'cond': {'change': {'color': '#da3e3e'}},
        },
    }

    def __init__(self, config: dict):
        """
        初始化策略，加载外部策略模式配置
        """
        super().__init__(config)

        # 尝试从外部JSON文件加载策略模式配置
        self.load_strategy_mode_config()

        # 加载停止开单记录
        self.load_stop_entry_records()

        # 存储上次检查止损的时间
        self.last_stoploss_check_time = datetime.now()

        # 输出当前策略模式配置(仅用于调试)
        if self.config.get('runmode', None) in ('live', 'dry_run'):
            pairs_count = len(self.pair_strategy_mode)
            long_count = sum(1 for mode in self.pair_strategy_mode.values() if mode == 'long')
            short_count = sum(1 for mode in self.pair_strategy_mode.values() if mode == 'short')
            if getattr(self, 'dp', None) and hasattr(self.dp, 'send_msg'):
                self.dp.send_msg(
                    f"已加载策略模式配置: 共 {pairs_count} 个交易对 (多头: {long_count}, 空头: {short_count})"
                )
                logger.info(
                    f"已加载策略模式配置: 共 {pairs_count} 个交易对 (多头: {long_count}, 空头: {short_count})"
                )
            else:
                logger.info(
                    f"已加载策略模式配置: 共 {pairs_count} 个交易对 (多头: {long_count}, 空头: {short_count})"
                )

    def load_strategy_mode_config(self):
        """
        从外部JSON文件加载策略模式配置
        """
        import os
        import json

        # 获取策略切换状态文件路径
        state_file = 'user_data/strategy_state.json'

        # 检查文件是否存在
        if not os.path.exists(state_file):
            if getattr(self, 'dp', None) and hasattr(self.dp, 'send_msg'):
                self.dp.send_msg(f"警告: 策略模式配置文件 {state_file} 不存在，将使用默认多头策略")
                logger.info(f"警告: 策略模式配置文件 {state_file} 不存在，将使用默认多头策略")
            else:
                logger.info(f"警告: 策略模式配置文件 {state_file} 不存在，将使用默认多头策略")
            return

        try:
            # 读取JSON文件
            with open(state_file, 'r') as f:
                state_data = json.load(f)

            # 获取策略模式配置
            if 'pair_strategy_mode' in state_data:
                self.pair_strategy_mode = state_data['pair_strategy_mode']
            if getattr(self, 'dp', None) and hasattr(self.dp, 'send_msg'):
                self.dp.send_msg(f"成功加载策略模式文件: {self.pair_strategy_mode}")
                logger.info(f"成功加载策略模式文件: {self.pair_strategy_mode}")
            else:
                logger.info(f"成功加载策略模式文件: {self.pair_strategy_mode}")

        except Exception as e:
            if getattr(self, 'dp', None) and hasattr(self.dp, 'send_msg'):
                self.dp.send_msg(f"加载策略模式配置时出错: {e}")
                logger.info(f"加载策略模式配置时出错: {e}")
            else:
                logger.info(f"加载策略模式配置时出错: {e}")

    def load_stop_entry_records(self):
        """
        加载停止开单记录
        """
        # 在回测模式下不需要持久化，只在内存中记录
        if self.config.get('runmode', None) not in ('live', 'dry_run'):
            return

        records_file = 'user_data/stop_entry_records.json'

        # 如果文件不存在，初始化为空记录
        if not os.path.exists(records_file):
            logger.info(f"停止开单记录文件 {records_file} 不存在，将初始化为空记录")
            return

        try:
            with open(records_file, 'r') as f:
                records_data = json.load(f)

            # 转换字符串时间为datetime对象
            for pair, record in records_data.items():
                if 'end_time' in record:
                    record['end_time'] = datetime.fromisoformat(record['end_time'])

            self.stop_entry_records = records_data

            # 清理过期记录
            self._clean_expired_records()

            # 记录当前状态
            active_records = sum(
                1
                for r in self.stop_entry_records.values()
                if r.get('end_time', datetime.now()) > datetime.now()
            )
            logger.info(f"已加载停止开单记录: 共 {len(self.stop_entry_records)} 条记录，其中 {active_records} 条有效")

        except Exception as e:
            logger.error(f"加载停止开单记录时出错: {e}")

    def save_stop_entry_records(self):
        """
        保存停止开单记录到文件
        """
        # 在回测模式下不需要持久化
        if self.config.get('runmode', None) not in ('live', 'dry_run'):
            return

        records_file = 'user_data/stop_entry_records.json'

        try:
            # 深拷贝记录以避免修改原始数据
            records_to_save = {}
            for pair, record in self.stop_entry_records.items():
                records_to_save[pair] = record.copy()
                # 将datetime转换为ISO格式字符串以便JSON序列化
                if 'end_time' in records_to_save[pair] and isinstance(
                    records_to_save[pair]['end_time'], datetime
                ):
                    records_to_save[pair]['end_time'] = records_to_save[pair][
                        'end_time'
                    ].isoformat()

            # 保存到文件
            with open(records_file, 'w') as f:
                json.dump(records_to_save, f, indent=4)

            logger.info(f"已保存停止开单记录到 {records_file}")

        except Exception as e:
            logger.error(f"保存停止开单记录时出错: {e}")

    def _clean_expired_records(self):
        """
        清理过期的停止开单记录
        """
        now = datetime.now()
        expired_pairs = []

        for pair, record in self.stop_entry_records.items():
            if 'end_time' in record and record['end_time'] < now:
                expired_pairs.append(pair)

        for pair in expired_pairs:
            del self.stop_entry_records[pair]

        if expired_pairs:
            logger.info(f"已清理 {len(expired_pairs)} 条过期的停止开单记录")
            self.save_stop_entry_records()

    def record_stop_entry(self, pair, direction, duration_hours=24, current_time=None):
        """
        记录停止开单信息

        参数:
        pair: 交易对
        direction: 开单方向 ('long' 或 'short')
        duration_hours: 停止开单持续时间（小时）
        current_time: 当前时间，回测模式下使用
        """
        # 确定当前时间和结束时间
        if current_time is None:
            current_time = datetime.now()

        end_time = current_time + timedelta(hours=duration_hours)

        self.stop_entry_records[pair] = {
            'direction': direction,
            'end_time': end_time,
            'recorded_at': current_time,
        }

        logger.info(f"已记录 {pair} {direction} 方向停止开单，从 {current_time} 到 {end_time}")

        # 在非回测模式下保存到文件
        if self.config.get('runmode', None) in ('live', 'dry_run'):
            self.save_stop_entry_records()

    def is_entry_allowed(self, pair, direction):
        """
        检查是否允许开单

        参数:
        pair: 交易对
        direction: 开单方向 ('long' 或 'short')

        返回:
        bool: 如果允许开单返回True，否则返回False
        """
        # 先清理过期记录
        self._clean_expired_records()

        # 检查是否有该交易对的停止开单记录
        if pair in self.stop_entry_records:
            record = self.stop_entry_records[pair]

            # 确定记录是否过期（回测和实盘使用不同的判断逻辑）
            if self.config.get('runmode', None) in ('live', 'dry_run'):
                # 实盘模式：检查end_time是否超过当前时间
                is_expired = record.get('end_time', datetime.now()) <= datetime.now()
            else:
                # 回测模式：检查candle的时间是否已经超过禁止期
                # 通过freqtrade提供的date_utc或当前dataframe的date判断
                current_date = self.current_candle_date.get(pair, datetime.now())
                is_expired = record.get('end_time', current_date) <= current_date

            # 如果记录的方向与当前方向相同且尚未过期
            if record['direction'] == direction and not is_expired:
                logger.debug(f"{pair} {direction} 方向开单被禁止，直到 {record['end_time']}")
                return False

        return True

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        # 常规止损逻辑
        if current_profit >= 0.05:
            return -0.002

        # 这里不要加其他处理逻辑，因为这个函数会被频繁调用
        # 止损后的处理逻辑应该放在 exit_positions 里处理
        return None

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 获取当前pair
        pair = metadata['pair']

        # 更新当前candle的时间（用于回测模式下的时间判断）
        if len(dataframe) > 0:
            self.current_candle_date[pair] = dataframe.iloc[-1]['date']

        # PCT CHANGE
        dataframe['change'] = 100 / dataframe['open'] * dataframe['close'] - 100

        # MAMA, FAMA, KAMA
        dataframe['hl2'] = (dataframe['high'] + dataframe['low']) / 2
        dataframe['mama'], dataframe['fama'] = ta.MAMA(dataframe['hl2'], 0.25, 0.025)
        dataframe['mama_diff'] = (dataframe['mama'] - dataframe['fama']) / dataframe['hl2']
        dataframe['kama'] = ta.KAMA(dataframe['close'], 84)

        # CTI
        dataframe['cti'] = pta.cti(dataframe['close'], length=20)

        # profit sell indicators
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']

        # RSI
        dataframe['rsi_84'] = ta.RSI(dataframe, timeperiod=84)
        dataframe['rsi_112'] = ta.RSI(dataframe, timeperiod=112)

        # Williams %R
        dataframe['r_14'] = williams_r(dataframe, period=14)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据当前交易对的策略模式决定使用哪种入场逻辑，并检查是否允许开单
        """
        pair = metadata['pair']

        # 默认设置
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        dataframe.loc[:, 'enter_tag'] = ''

        # 如果数据库中不存在该对的策略模式，使用默认策略（多头）
        strategy_mode = self.pair_strategy_mode.get(pair, 'long')

        # 获取原始信号
        if strategy_mode == 'long':
            dataframe = self._populate_long_entry(dataframe, metadata)
        else:
            dataframe = self._populate_short_entry(dataframe, metadata)

        # 在回测和实盘中都要检查是否允许开单
        # 检查是否在止损后的禁止开单期
        long_allowed = self.is_entry_allowed(pair, 'long')
        short_allowed = self.is_entry_allowed(pair, 'short')

        # 如果不允许相应方向的开单，将信号置为0
        if not long_allowed:
            dataframe.loc[dataframe['enter_long'] == 1, 'enter_tag'] += '_blocked'
            dataframe.loc[:, 'enter_long'] = 0

        if not short_allowed:
            dataframe.loc[dataframe['enter_short'] == 1, 'enter_tag'] += '_blocked'
            dataframe.loc[:, 'enter_short'] = 0

        return dataframe

    def _populate_long_entry(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        buy = (
            (dataframe['kama'] > dataframe['fama'])
            & (dataframe['fama'] > dataframe['mama'] * 0.981)
            & (dataframe['r_14'] < -61.3)
            & (dataframe['mama_diff'] < -0.025)
            & (dataframe['cti'] < -0.715)
            & (dataframe['close'].rolling(48).max() >= dataframe['close'] * 1.05)
            & (dataframe['close'].rolling(288).max() >= dataframe['close'] * 1.125)
            & (dataframe['rsi_84'] < 60)
            & (dataframe['rsi_112'] < 60)
        )
        conditions.append(buy)
        dataframe.loc[buy, 'enter_tag'] += 'buy'

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_long'] = 1

        return dataframe

    def _populate_short_entry(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于KamaFama_2策略的反向做空逻辑
        """
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        # 做空入场条件 - 与做多逻辑相反
        short = (
            (dataframe['kama'] < dataframe['fama'])  # KAMA低于FAMA - 趋势可能反转向下
            & (dataframe['fama'] < dataframe['mama'] * 1.019)  # FAMA低于MAMA一定比例
            & (dataframe['r_14'] > -38.7)  # Williams %R处于高位 - 可能超买
            & (dataframe['mama_diff'] > 0.025)  # MAMA与FAMA差异为正且足够大
            & (dataframe['cti'] > 0.715)  # CTI处于高位 - 表明可能超买
            & (dataframe['close'].rolling(48).min() <= dataframe['close'] * 0.95)  # 近期最低点比当前低5%
            & (
                dataframe['close'].rolling(288).min() <= dataframe['close'] * 0.875
            )  # 长期最低点比当前低12.5%
            & (dataframe['rsi_84'] > 40)  # RSI处于中高位置
            & (dataframe['rsi_112'] > 40)  # 长期RSI也处于中高位置
        )
        conditions.append(short)
        dataframe.loc[short, 'enter_tag'] += 'short'

        if conditions:
            dataframe.loc[reduce(lambda x, y: x | y, conditions), 'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: 'Trade',
        current_time: 'datetime',
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """根据交易类型选择合适的退出逻辑"""

        # 检查是否触发止损
        is_stoploss = False
        if hasattr(trade, 'exit_reason') and trade.exit_reason:
            is_stoploss = 'stoploss' in trade.exit_reason.lower()
        elif current_profit <= self.stoploss:
            # 在没有明确exit_reason的情况下，根据利润判断是否是止损
            is_stoploss = True

        # 如果是止损出场，记录停止开单
        if is_stoploss:
            direction = 'short' if trade.is_short else 'long'
            # 在回测模式下，使用当前K线时间；在实盘模式下使用系统时间
            if self.config.get('runmode', None) in ('live', 'dry_run'):
                self.record_stop_entry(pair, direction, duration_hours=24)
            else:
                self.record_stop_entry(
                    pair, direction, duration_hours=24, current_time=current_time
                )
            logger.info(f"检测到止损退出: {pair} {direction}方向将在24小时内禁止开单")

        # 根据trade的类型判断是多头还是空头
        if trade.is_short:
            return self._custom_exit_short(
                pair, trade, current_time, current_rate, current_profit, **kwargs
            )
        else:
            return self._custom_exit_long(
                pair, trade, current_time, current_rate, current_profit, **kwargs
            )

    def _custom_exit_short(
        self,
        pair: str,
        trade: 'Trade',
        current_time: 'datetime',
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """空头退出逻辑"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if self.config['runmode'].value in ('live', 'dry_run'):
            state = self.cc_short
            pc = state.get(
                trade.id,
                {
                    'date': current_candle['date'],
                    'open': current_candle['close'],
                    'high': current_candle['close'],
                    'low': current_candle['close'],
                    'close': current_rate,
                    'volume': 0,
                },
            )
            # 更新candle状态逻辑...与原代码类似
            if current_candle['date'] != pc['date']:
                pc['date'] = current_candle['date']
                pc['high'] = current_candle['close']
                pc['low'] = current_candle['close']
                pc['open'] = current_candle['close']
                pc['close'] = current_rate
            if current_rate > pc['high']:
                pc['high'] = current_rate
            if current_rate < pc['low']:
                pc['low'] = current_rate
            if current_rate != pc['close']:
                pc['close'] = current_rate

            state[trade.id] = pc

        if current_profit > 0:  # 对于做空，利润为正表示价格下跌
            if self.config['runmode'].value in ('live', 'dry_run'):
                if current_time > pc['date'] + timedelta(minutes=9) + timedelta(seconds=55):
                    df = dataframe.copy()
                    df = df._append(pc, ignore_index=True)
                    stoch_fast = ta.STOCHF(df, 5, 3, 0, 3, 0)
                    df['fastk'] = stoch_fast['fastk']
                    cc = df.iloc[-1].squeeze()
                    # 对于做空，在fastk值较低时平仓
                    if cc['fastk'] < self.buy_fastx_short.value:
                        return 'fastk_profit_buy_to_cover'
            else:
                if current_candle['fastk'] < self.buy_fastx_short.value:
                    return 'fastk_profit_buy_to_cover'

        # 如果交易对的策略模式已切换为多头但仍有空头仓位，加速平仓
        if self.pair_strategy_mode.get(pair, 'long') == 'long' and current_profit > 0.01:
            return 'strategy_switched_to_long'

        return None

    def _custom_exit_long(
        self,
        pair: str,
        trade: 'Trade',
        current_time: 'datetime',
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """多头退出逻辑"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        if self.config['runmode'].value in ('live', 'dry_run'):
            state = self.cc_long
            pc = state.get(
                trade.id,
                {
                    'date': current_candle['date'],
                    'open': current_candle['close'],
                    'high': current_candle['close'],
                    'low': current_candle['close'],
                    'close': current_rate,
                    'volume': 0,
                },
            )
            if current_candle['date'] != pc['date']:
                pc['date'] = current_candle['date']
                pc['high'] = current_candle['close']
                pc['low'] = current_candle['close']
                pc['open'] = current_candle['close']
                pc['close'] = current_rate
            if current_rate > pc['high']:
                pc['high'] = current_rate
            if current_rate < pc['low']:
                pc['low'] = current_rate
            if current_rate != pc['close']:
                pc['close'] = current_rate

            state[trade.id] = pc

        if current_profit > 0:
            if self.config['runmode'].value in ('live', 'dry_run'):
                if current_time > pc['date'] + timedelta(minutes=9) + timedelta(seconds=55):
                    df = dataframe.copy()
                    df = df._append(pc, ignore_index=True)
                    stoch_fast = ta.STOCHF(df, 5, 3, 0, 3, 0)
                    df['fastk'] = stoch_fast['fastk']
                    cc = df.iloc[-1].squeeze()
                    if cc['fastk'] > self.sell_fastx.value:
                        return 'fastk_profit_sell_2'
            else:
                if current_candle['fastk'] > self.sell_fastx.value:
                    return 'fastk_profit_sell'

        # 如果交易对的策略模式已切换为空头但仍有多头仓位，加速平仓
        if self.pair_strategy_mode.get(pair, 'long') == 'short' and current_profit > 0.01:
            return 'strategy_switched_to_short'

        return None

    def check_for_stoploss_trades(self):
        """
        检查最近的止损交易，并记录停止开单信息
        实盘/模拟盘模式专用方法
        """
        # 只在live和dry_run模式下执行
        if self.config.get('runmode', None) not in ('live', 'dry_run'):
            return

        try:
            # 获取最近关闭的交易
            closed_trades = Trade.get_trades([Trade.is_open.is_(False)]).all()

            # 检查是否有止损交易
            for trade in closed_trades:
                # 只检查最近30分钟内关闭的交易
                if trade.close_date and (datetime.now() - trade.close_date).total_seconds() <= 1800:
                    # 检查是否是止损
                    if trade.sell_reason and 'stoploss' in trade.sell_reason.lower():
                        pair = trade.pair
                        direction = 'short' if trade.is_short else 'long'

                        # 记录止损后24小时内禁止相同方向开单
                        self.record_stop_entry(pair, direction, duration_hours=24)

                        if getattr(self, 'dp', None) and hasattr(self.dp, 'send_msg'):
                            self.dp.send_msg(f"{pair} 触发止损，{direction}方向将在24小时内禁止开单")
                        logger.info(f"{pair} 触发止损，{direction}方向将在24小时内禁止开单")
        except Exception as e:
            logger.error(f"检查止损交易时出错: {e}")

    def bot_loop_start(self, **kwargs) -> None:
        """
        在每个机器人循环开始时调用，用于检查止损交易
        """
        self.check_for_stoploss_trades()
        # 每30分钟检查一次止损交易
        current_time = datetime.now()
        if (current_time - self.last_stoploss_check_time).total_seconds() >= 1800:  # 30分钟
            self.check_for_stoploss_trades()
            self.last_stoploss_check_time = current_time
