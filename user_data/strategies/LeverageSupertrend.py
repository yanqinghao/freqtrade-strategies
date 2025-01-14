import logging
from numpy.lib import math
from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from typing import Dict


class LeverageSupertrend(IStrategy):
    INTERFACE_VERSION: int = 3
    
    can_short = True
    position_adjustment_enable = True

    buy_params = {
        "buy_m1": 4,
        "buy_m2": 7,
        "buy_m3": 1,
        "buy_p1": 8,
        "buy_p2": 9,
        "buy_p3": 8,
        "buy_rsi_period": 14,
        "buy_rsi_period_long": 100  # 长周期RSI
    }

    sell_params = {
        "sell_m1": 1,
        "sell_m2": 3,
        "sell_m3": 6,
        "sell_p1": 16,
        "sell_p2": 18,
        "sell_p3": 18,
        "sell_rsi_period": 14,
        "sell_rsi_period_long": 100  # 长周期RSI
    }

    minimal_roi = {"0": 0.1, "30": 0.75, "60": 0.05, "120": 0.025}
    
    stoploss = -0.24

    trailing_stop = True
    trailing_stop_positive = 0.03
    trailing_stop_positive_offset = 0.6
    trailing_only_offset_is_reached = False

    timeframe = "1h"

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

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                proposed_leverage: float, max_leverage: float, side: str,
                **kwargs) -> float:
        """
        根据趋势强度动态调整杠杆
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        rsi_long = last_candle['rsi_long']
        
        # 根据趋势强度和方向调整杠杆
        if side == "long":
            if rsi_long > 70:  # 强势牛市
                return min(3, max_leverage)
            elif rsi_long > 50:  # 普通牛市
                return min(2, max_leverage)
            else:  # 熊市
                return 2
        else:  # short
            if rsi_long < 30:  # 强势熊市
                return min(3, max_leverage)
            elif rsi_long < 50:  # 普通熊市
                return min(2, max_leverage)
            else:  # 牛市
                return 2

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 添加短周期和长周期RSI指标
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=self.buy_rsi_period.value)
        dataframe['rsi_long'] = ta.RSI(dataframe, timeperiod=self.buy_rsi_period_long.value)
        
        # Supertrend indicators
        for multiplier in self.buy_m1.range:
            for period in self.buy_p1.range:
                dataframe[f'supertrend_1_buy_{multiplier}_{period}'] = self.supertrend(dataframe, multiplier, period)['STX']

        for multiplier in self.buy_m2.range:
            for period in self.buy_p2.range:
                dataframe[f'supertrend_2_buy_{multiplier}_{period}'] = self.supertrend(dataframe, multiplier, period)['STX']

        for multiplier in self.buy_m3.range:
            for period in self.buy_p3.range:
                dataframe[f'supertrend_3_buy_{multiplier}_{period}'] = self.supertrend(dataframe, multiplier, period)['STX']

        for multiplier in self.sell_m1.range:
            for period in self.sell_p1.range:
                dataframe[f'supertrend_1_sell_{multiplier}_{period}'] = self.supertrend(dataframe, multiplier, period)['STX']

        for multiplier in self.sell_m2.range:
            for period in self.sell_p2.range:
                dataframe[f'supertrend_2_sell_{multiplier}_{period}'] = self.supertrend(dataframe, multiplier, period)['STX']

        for multiplier in self.sell_m3.range:
            for period in self.sell_p3.range:
                dataframe[f'supertrend_3_sell_{multiplier}_{period}'] = self.supertrend(dataframe, multiplier, period)['STX']

        # 计算RSI变化并添加动态RSI阈值
        dataframe['rsi_long_prev'] = dataframe['rsi_long'].shift(1)
        dataframe[['rsi_upper', 'rsi_lower']] = dataframe.apply(
            lambda x: self.get_rsi_thresholds(x['rsi_long'], x['rsi_long_prev']), 
            axis=1, 
            result_type='expand'
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
               (dataframe[f'supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}'] == 'up') &
               (dataframe[f'supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}'] == 'up') &
               (dataframe[f'supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}'] == 'up') &
               (dataframe['rsi'] < dataframe['rsi_upper']) &
               (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
               (dataframe[f'supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}'] == 'down') &
               (dataframe[f'supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}'] == 'down') &
               (dataframe[f'supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}'] == 'down') &
               (dataframe['rsi'] > dataframe['rsi_lower']) &
               (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe[f'supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}'] == 'down') |
                (dataframe['rsi'] >= dataframe['rsi_upper'])  # 使用动态RSI上限退出多头
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                (dataframe[f'supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}'] == 'up') |
                (dataframe['rsi'] <= dataframe['rsi_lower'])  # 使用动态RSI下限退出空头
            ),
            'exit_short'] = 1

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
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] >  df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                            df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] <  df['final_lb'].iat[i] else 0.00
                            
        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down',  'up'), np.NaN)

        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={
            'ST' : df[st],
            'STX' : df[stx]
        })