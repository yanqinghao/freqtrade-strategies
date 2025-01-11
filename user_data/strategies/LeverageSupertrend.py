import logging
from numpy.lib import math
from freqtrade.strategy import IStrategy, IntParameter
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
from typing import Dict


class LeverageSupertrend(IStrategy):
    INTERFACE_VERSION: int = 3
    
    can_short = True  # 启用做空
    position_adjustment_enable = True  # 启用仓位调整

    buy_params = {
        "buy_m1": 4,
        "buy_m2": 7,
        "buy_m3": 1,
        "buy_p1": 8,
        "buy_p2": 9,
        "buy_p3": 8,
    }

    sell_params = {
        "sell_m1": 1,
        "sell_m2": 3,
        "sell_m3": 6,
        "sell_p1": 16,
        "sell_p2": 18,
        "sell_p3": 18,
    }

    minimal_roi = {"0": 0.1, "30": 0.75, "60": 0.05, "120": 0.025}
    
    stoploss = -0.265

    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    timeframe = "1h"

    startup_candle_count = 18

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

    def leverage(self, pair: str, current_time: 'datetime', current_rate: float,
                proposed_leverage: float, max_leverage: float, side: str,
                **kwargs) -> float:
        """
        自定义杠杆计算函数
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # 获取三个不同时间周期的趋势信号
        signals = []
        if side == "long":
            signals = [
                last_candle[f'supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}'] == 'up',
                last_candle[f'supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}'] == 'up',
                last_candle[f'supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}'] == 'up'
            ]
        else:  # short
            signals = [
                last_candle[f'supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}'] == 'down',
                last_candle[f'supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}'] == 'down',
                last_candle[f'supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}'] == 'down'
            ]

        # 根据趋势信号的一致性确定杠杆
        true_count = sum(1 for x in signals if x)
        
        # 如果三个信号都一致，使用3倍杠杆
        if true_count == 3:
            return min(3.0, max_leverage)
        # 如果两个信号一致，使用2倍杠杆
        elif true_count == 2:
            return min(2.0, max_leverage)
        # 如果只有一个信号，使用1倍杠杆
        else:
            return 1.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
               (dataframe[f'supertrend_1_buy_{self.buy_m1.value}_{self.buy_p1.value}'] == 'up') &
               (dataframe[f'supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}'] == 'up') &
               (dataframe[f'supertrend_3_buy_{self.buy_m3.value}_{self.buy_p3.value}'] == 'up') &
               (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        dataframe.loc[
            (
               (dataframe[f'supertrend_1_sell_{self.sell_m1.value}_{self.sell_p1.value}'] == 'down') &
               (dataframe[f'supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}'] == 'down') &
               (dataframe[f'supertrend_3_sell_{self.sell_m3.value}_{self.sell_p3.value}'] == 'down') &
               (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                dataframe[f'supertrend_2_sell_{self.sell_m2.value}_{self.sell_p2.value}'] == 'down'
            ),
            'exit_long'] = 1

        dataframe.loc[
            (
                dataframe[f'supertrend_2_buy_{self.buy_m2.value}_{self.buy_p2.value}'] == 'up'
            ),
            'exit_short'] = 1

        return dataframe

    def supertrend(self, dataframe: DataFrame, multiplier, period):
        df = dataframe.copy()

        df['TR'] = ta.TRANGE(df)
        df['ATR'] = ta.SMA(df['TR'], period)

        st = 'ST_' + str(period) + '_' + str(multiplier)
        stx = 'STX_' + str(period) + '_' + str(multiplier)

        # Compute basic upper and lower bands
        df['basic_ub'] = (df['high'] + df['low']) / 2 + multiplier * df['ATR']
        df['basic_lb'] = (df['high'] + df['low']) / 2 - multiplier * df['ATR']

        # Compute final upper and lower bands
        df['final_ub'] = 0.00
        df['final_lb'] = 0.00
        for i in range(period, len(df)):
            df['final_ub'].iat[i] = df['basic_ub'].iat[i] if df['basic_ub'].iat[i] < df['final_ub'].iat[i - 1] or df['close'].iat[i - 1] > df['final_ub'].iat[i - 1] else df['final_ub'].iat[i - 1]
            df['final_lb'].iat[i] = df['basic_lb'].iat[i] if df['basic_lb'].iat[i] > df['final_lb'].iat[i - 1] or df['close'].iat[i - 1] < df['final_lb'].iat[i - 1] else df['final_lb'].iat[i - 1]

        # Set the Supertrend value
        df[st] = 0.00
        for i in range(period, len(df)):
            df[st].iat[i] = df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] <= df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_ub'].iat[i - 1] and df['close'].iat[i] >  df['final_ub'].iat[i] else \
                            df['final_lb'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] >= df['final_lb'].iat[i] else \
                            df['final_ub'].iat[i] if df[st].iat[i - 1] == df['final_lb'].iat[i - 1] and df['close'].iat[i] <  df['final_lb'].iat[i] else 0.00
                            
        # Mark the trend direction up/down
        df[stx] = np.where((df[st] > 0.00), np.where((df['close'] < df[st]), 'down',  'up'), np.NaN)

        # Remove basic and final bands from the columns
        df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

        df.fillna(0, inplace=True)

        return DataFrame(index=df.index, data={
            'ST' : df[st],
            'STX' : df[stx]
        })