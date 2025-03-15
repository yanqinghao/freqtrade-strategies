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
import itertools
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

    # 固定点位监控
    coin_monitoring = {}

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

        # 存储上次检查止损的时间
        self.last_stoploss_check_time = datetime.now()

        # 输出当前策略模式配置(仅用于调试)
        if self.config.get('runmode', None) in ('live', 'dry_run'):
            pairs_count = len(self.pair_strategy_mode)
            long_count = sum(1 for mode in self.pair_strategy_mode.values() if mode == 'long')
            short_count = sum(1 for mode in self.pair_strategy_mode.values() if mode == 'short')
            monitoring_count = len(self.coin_monitoring)

            if getattr(self, 'dp', None) and hasattr(self.dp, 'send_msg'):
                self.dp.send_msg(
                    f"已加载策略模式配置: 共 {pairs_count} 个交易对 (多头: {long_count}, 空头: {short_count})"
                )
                if monitoring_count > 0:
                    self.dp.send_msg(f"已加载固定点位监控: 共 {monitoring_count} 个交易对")
                logger.info(
                    f"已加载策略模式配置: 共 {pairs_count} 个交易对 (多头: {long_count}, 空头: {short_count})"
                )
                if monitoring_count > 0:
                    logger.info(f"已加载固定点位监控: 共 {monitoring_count} 个交易对")
            else:
                logger.info(
                    f"已加载策略模式配置: 共 {pairs_count} 个交易对 (多头: {long_count}, 空头: {short_count})"
                )
                if monitoring_count > 0:
                    logger.info(f"已加载固定点位监控: 共 {monitoring_count} 个交易对")

    def load_strategy_mode_config(self):
        """
        从外部JSON文件加载策略模式配置
        """

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

            # 获取固定点位监控配置
            if 'coin_monitoring' in state_data:
                self.coin_monitoring = state_data['coin_monitoring']

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

        # 检查是否有固定点位监控
        if (
            pair in self.coin_monitoring
            and self.coin_monitoring.get(pair)
            and list(itertools.chain(*[i['entry_points'] for i in self.coin_monitoring[pair]]))
        ):
            # 处理固定点位监控逻辑
            dataframe = self._populate_fixed_entry(dataframe, metadata)
        else:
            # 如果数据库中不存在该对的策略模式，使用默认策略（多头）
            strategy_mode = self.pair_strategy_mode.get(pair, 'long')

            # 获取原始信号
            if strategy_mode == 'long':
                dataframe = self._populate_long_entry(dataframe, metadata)
            else:
                dataframe = self._populate_short_entry(dataframe, metadata)

        return dataframe

    def _populate_fixed_entry(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        基于固定点位监控生成入场信号
        """
        pair = metadata['pair']
        monitoring_config = self.coin_monitoring.get(pair, [])

        # 没有监控配置，直接返回原始dataframe
        if not monitoring_config:
            return dataframe

        # 遍历每个监控配置
        for config in monitoring_config:
            direction = config.get('direction', 'long')
            entry_points = config.get('entry_points', [])

            if not entry_points:
                continue

            # 计算当前价格是否在入场点附近
            for entry_point in entry_points:
                # 对于多头
                if direction == 'long':
                    # 如果当前价格在入场点附近（上下0.5%范围内）
                    entry_condition = (dataframe['close'] >= entry_point * 0.995) & (
                        dataframe['close'] <= entry_point * 1.005
                    )
                    dataframe.loc[entry_condition, 'enter_long'] = 1
                    dataframe.loc[entry_condition, 'enter_tag'] = f'fixed_long_entry_{entry_point}'

                # 对于空头
                elif direction == 'short':
                    # 如果当前价格在入场点附近（上下0.5%范围内）
                    entry_condition = (dataframe['close'] >= entry_point * 0.995) & (
                        dataframe['close'] <= entry_point * 1.005
                    )
                    dataframe.loc[entry_condition, 'enter_short'] = 1
                    dataframe.loc[entry_condition, 'enter_tag'] = f'fixed_short_entry_{entry_point}'

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
        direction = 'short' if trade.is_short else 'long'
        # 检查是否是固定点位监控的交易对
        if (
            pair in self.coin_monitoring
            and self.coin_monitoring.get(pair)
            and list(
                itertools.chain(
                    *[
                        i['exit_points']
                        for i in self.coin_monitoring[pair]
                        if i['direction'] == direction
                    ]
                )
            )
        ):
            # 找到对应方向的监控配置
            for config in self.coin_monitoring[pair]:
                if config.get('direction') == direction:
                    exit_points = config.get('exit_points', [])

                    # 如果有退出点位配置
                    if exit_points:
                        # 对于多头和空头使用不同的退出判断
                        if direction == 'short':
                            # 对于空头，当价格低于退出点时退出
                            for exit_point in exit_points:
                                if current_rate <= exit_point:
                                    logger.info(
                                        f"固定点位监控交易 {pair} 触发退出: 当前价格 {current_rate} <= 退出价格 {exit_point}"
                                    )
                                    return f"fixed_short_exit_{exit_point}"
                        else:
                            # 对于多头，当价格高于退出点时退出
                            for exit_point in exit_points:
                                if current_rate >= exit_point:
                                    logger.info(
                                        f"固定点位监控交易 {pair} 触发退出: 当前价格 {current_rate} >= 退出价格 {exit_point}"
                                    )
                                    return f"fixed_long_exit_{exit_point}"
                    break
        else:
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
                        return 'fastk_profit_buy_to_cover_2'
            else:
                if current_candle['fastk'] < self.buy_fastx_short.value:
                    return 'fastk_profit_buy_to_cover'
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
        return None
