from freqtrade.strategy import IStrategy, IntParameter, Trade
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from datetime import timezone, datetime


class LeverageSupertrend(IStrategy):
    INTERFACE_VERSION: int = 3

    can_short = True

    buy_params = {
        'buy_m1': 5,  # 长期趋势，低敏感度
        'buy_m2': 3,  # 中期趋势，中等敏感度
        'buy_m3': 1,  # 短期趋势，高敏感度
        'buy_p1': 24,  # 24小时
        'buy_p2': 8,  # 8小时
        'buy_p3': 4,  # 4小时
        'buy_rsi_period': 14,
        'buy_rsi_period_long': 100,
    }

    sell_params = {
        'sell_m1': 5,
        'sell_m2': 3,
        'sell_m3': 1,
        'sell_p1': 24,
        'sell_p2': 8,
        'sell_p3': 4,
        'sell_rsi_period': 14,
        'sell_rsi_period_long': 100,
    }

    minimal_roi = {
        '0': 0.06,  # 初始目标降低到6%
        '20': 0.04,  # 20分钟后降到4%
        '40': 0.025,  # 40分钟后降到2.5%
        '60': 0.02,  # 60分钟后降到2%
        '90': 0.015,  # 90分钟后降到1.5%
        '120': 0.01,  # 2小时后降到1%
        '240': 0.005,  # 4小时后降到0.5%
        '480': 0,  # 8小时后可以不盈利退出
    }

    stoploss = -0.24

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.4
    trailing_only_offset_is_reached = False

    timeframe = '1h'

    startup_candle_count = 18

    # Supertrend parameters
    buy_m1 = IntParameter(1, 7, default=1)
    buy_m2 = IntParameter(1, 7, default=3)
    buy_m3 = IntParameter(1, 7, default=4)
    buy_p1 = IntParameter(7, 21, default=14)
    buy_p2 = IntParameter(7, 21, default=10)
    buy_p3 = IntParameter(7, 21, default=10)

    sell_m1 = IntParameter(1, 7, default=1)
    sell_m2 = IntParameter(1, 7, default=3)
    sell_m3 = IntParameter(1, 7, default=4)
    sell_p1 = IntParameter(7, 21, default=14)
    sell_p2 = IntParameter(7, 21, default=10)
    sell_p3 = IntParameter(7, 21, default=10)

    # RSI parameters
    buy_rsi_period = IntParameter(10, 20, default=14)
    buy_rsi_period_long = IntParameter(50, 200, default=100)  # 长周期RSI

    sell_rsi_period = IntParameter(10, 20, default=14)
    sell_rsi_period_long = IntParameter(50, 200, default=100)  # 长周期RSI

    def get_rsi_thresholds(self, rsi_long: float, rsi_long_prev: float) -> tuple:
        """
        根据长周期RSI和其变化趋势动态调整RSI阈值
        """
        # 计算RSI的变化率来检测趋势转换
        rsi_change = rsi_long - rsi_long_prev

        # 强势趋势判断
        if rsi_long > 70:  # 强势牛市
            return 85, 40
        elif rsi_long < 30:  # 强势熊市
            return 60, 15
        # 趋势转换判断
        elif rsi_change > 3:  # RSI快速上升
            return 80, 35  # 适当放宽上涨阈值
        elif rsi_change < -3:  # RSI快速下降
            return 65, 20  # 适当放宽下跌阈值
        # 普通趋势判断
        elif rsi_long > 50:  # 普通牛市
            return 75, 35
        else:  # 普通熊市
            return 65, 25

    def leverage(
        self,
        pair: str,
        current_time: 'datetime',
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs
    ) -> float:
        """
        根据趋势强度动态调整杠杆
        """
        # dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        # last_candle = dataframe.iloc[-1].squeeze()

        # rsi_long = last_candle['rsi_long']

        # # 根据趋势强度和方向调整杠杆
        # if side == "long":
        #     if rsi_long > 70:  # 强势牛市
        #         return min(3, max_leverage)
        #     elif rsi_long > 50:  # 普通牛市
        #         return min(2, max_leverage)
        #     else:  # 熊市
        #         return 2
        # else:  # short
        #     if rsi_long < 30:  # 强势熊市
        #         return min(3, max_leverage)
        #     elif rsi_long < 50:  # 普通熊市
        #         return min(2, max_leverage)
        #     else:  # 牛市
        #         return 2
        return 2

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['TR'] = ta.TRANGE(dataframe)
        dataframe['ATR'] = ta.SMA(dataframe['TR'], 14)  # 使用14周期的ATR

        # 计算ATR百分比
        dataframe['ATR_pct'] = dataframe['ATR'] / dataframe['close'] * 100

        # 添加RSI指标
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.buy_rsi_period.value)
        dataframe['rsi_long'] = ta.RSI(dataframe, timeperiod=self.buy_rsi_period_long.value)

        # 只计算实际需要用到的supertrend
        # 买入信号的三条线
        dataframe['supertrend_1_buy'] = self.supertrend(
            dataframe, self.buy_m1.value, self.buy_p1.value
        )['STX']

        dataframe['supertrend_2_buy'] = self.supertrend(
            dataframe, self.buy_m2.value, self.buy_p2.value
        )['STX']

        dataframe['supertrend_3_buy'] = self.supertrend(
            dataframe, self.buy_m3.value, self.buy_p3.value
        )['STX']

        # 卖出信号的三条线
        dataframe['supertrend_1_sell'] = self.supertrend(
            dataframe, self.sell_m1.value, self.sell_p1.value
        )['STX']

        dataframe['supertrend_2_sell'] = self.supertrend(
            dataframe, self.sell_m2.value, self.sell_p2.value
        )['STX']

        dataframe['supertrend_3_sell'] = self.supertrend(
            dataframe, self.sell_m3.value, self.sell_p3.value
        )['STX']

        # RSI相关计算
        dataframe['rsi_long_prev'] = dataframe['rsi_long'].shift(1)
        dataframe[['rsi_upper', 'rsi_lower']] = dataframe.apply(
            lambda x: self.get_rsi_thresholds(x['rsi_long'], x['rsi_long_prev']),
            axis=1,
            result_type='expand',
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        volatility_condition = dataframe['ATR'] / dataframe['close'] * 100 < 3.0  # ATR不超过3%
        dataframe.loc[
            (
                (dataframe['supertrend_1_buy'] == 'up')
                & (dataframe['supertrend_2_buy'] == 'up')
                & (dataframe['supertrend_3_buy'] == 'up')
                & (dataframe['rsi'] < dataframe['rsi_upper'])
                & (dataframe['volume'] > 0)
                & (volatility_condition)  # 添加波动率过滤
                & (dataframe['close'] > dataframe['close'].shift(1))
                & (dataframe['volume'] > dataframe['volume'].shift(1))  # 价格在下跌  # 放量下跌
            ),
            'enter_long',
        ] = 1

        dataframe.loc[
            (
                (dataframe['supertrend_1_sell'] == 'down')
                & (dataframe['supertrend_2_sell'] == 'down')
                & (dataframe['supertrend_3_sell'] == 'down')
                & (dataframe['rsi'] > dataframe['rsi_lower'])
                & (dataframe['volume'] > 0)
                & (dataframe['ATR'] / dataframe['close'] * 100 < 4.0)
                & (dataframe['close'] < dataframe['close'].shift(1))
                & (dataframe['volume'] > dataframe['volume'].shift(1))  # 价格在下跌  # 放量下跌
            ),
            'enter_short',
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['supertrend_1_buy'] == 'down')
                | (dataframe['rsi'] >= dataframe['rsi_upper'])  # 使用第一条趋势线
            ),
            'exit_long',
        ] = 1

        dataframe.loc[
            (
                (dataframe['supertrend_1_sell'] == 'up')
                | (dataframe['rsi'] <= dataframe['rsi_lower'])
                | (
                    (dataframe['close'] < dataframe['close'].shift(2))
                    & (dataframe['volume'] > dataframe['volume'].shift(1) * 1.5)  # 连续上涨  # 放量
                )
            ),
            'exit_short',
        ] = 1

        return dataframe

    def supertrend(self, dataframe: DataFrame, multiplier, period):
        df = dataframe.copy()

        df['TR'] = ta.TRANGE(df)
        df['ATR'] = ta.SMA(df['TR'], period)

        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)

        df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
        df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(period, len(df)):
            df['final_ub'].iat[i] = (
                df['basic_ub'].iat[i]
                if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1]
                or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1]
                else df['final_ub'].iat[i - 1]
            )
            df['final_lb'].iat[i] = (
                df['basic_lb'].iat[i]
                if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1]
                or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1]
                else df['final_lb'].iat[i - 1]
            )

        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = (
                df['final_ub'].iat[i]
                if df[st].iat[i - 1] == df['final_ub'].iat[i - 1]
                and df['close'].iat[i] <= df['final_ub'].iat[i]
                else df['final_lb'].iat[i]
                if df[st].iat[i - 1] == df['final_ub'].iat[i - 1]
                and df['close'].iat[i] > df['final_ub'].iat[i]
                else df['final_lb'].iat[i]
                if df[st].iat[i - 1] == df['final_lb'].iat[i - 1]
                and df['close'].iat[i] >= df['final_lb'].iat[i]
                else df['final_ub'].iat[i]
                if df[st].iat[i - 1] == df['final_lb'].iat[i - 1]
                and df['close'].iat[i] < df['final_lb'].iat[i]
                else 0.00
            )

        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down', 'up'), np.NaN)

        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={'ST': df[st], 'STX': df[stx]})

    def custom_exit(
        self,
        pair: str,
        trade: 'Trade',
        current_time: 'datetime',
        current_rate: float,
        current_profit: float,
        **kwargs
    ) -> str:
        """
        自定义退出逻辑:
        1. 持仓时间超过8小时
        2. 收益在-10%到10%之间
        则强制止损
        """
        # 确保 current_time 有时区信息
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        # 确保 trade.open_date 有时区信息
        if trade.open_date.tzinfo is None:
            open_date = trade.open_date.replace(tzinfo=timezone.utc)
        else:
            open_date = trade.open_date
        # 计算持仓时间（分钟）- 使用 total_seconds() 转换为分钟
        hold_time_minutes = (current_time - open_date).total_seconds() / 60

        # 如果持仓超过16小时(960分钟) 且 收益在-0.1到0.1之间
        if hold_time_minutes > 960 and current_profit < -0.15:

            return 'force_exit_time_profit_range'

        return None  # 不满足条件则不退出
