
# --- Do not remove these libs ---
from freqtrade.strategy import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame
# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa


class Strategy002_with_short(IStrategy):
    # 添加做空功能
    can_short = True  # 启用做空
    
    # 其他基础配置保持不变
    INTERFACE_VERSION: int = 3
    minimal_roi = {
        "60":  0.01,
        "30":  0.03,
        "20":  0.04,
        "0":  0.05
    }
    stoploss = -0.10
    timeframe = '5m'
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 保留原有指标
        stoch = ta.STOCH(dataframe)
        dataframe['slowk'] = stoch['slowk']
        dataframe['slowd'] = stoch['slowd']  # 添加slowd用于判断
        
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        
        # Inverse Fisher transform on RSI
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)
        
        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_upperband'] = bollinger['upper']  # 添加上轨用于做空
        dataframe['bb_middleband'] = bollinger['mid']
        
        # SAR Parabol
        dataframe['sar'] = ta.SAR(dataframe)
        
        # 看多形态
        dataframe['CDLHAMMER'] = ta.CDLHAMMER(dataframe)
        dataframe['CDLDRAGONFLYDOJI'] = ta.CDLDRAGONFLYDOJI(dataframe)  # 蜻蜓十字
        dataframe['CDLINVERTEDHAMMER'] = ta.CDLINVERTEDHAMMER(dataframe)  # 倒锤子
        dataframe['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(dataframe)  # 晨星
        
        # 看空形态
        dataframe['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(dataframe)
        dataframe['CDLHANGINGMAN'] = ta.CDLHANGINGMAN(dataframe)  # 上吊线
        dataframe['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(dataframe)  # 暮星
        dataframe['CDLENGULFING'] = ta.CDLENGULFING(dataframe)  # 吞没形态
        
        # 添加MACD用于趋势确认
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 做多条件
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) &  # RSI超卖
                (dataframe['slowk'] < 20) &  # Stoch超卖
                (dataframe['bb_lowerband'] > dataframe['close']) &  # 价格在布林带下轨之下
                (
                    (dataframe['CDLHAMMER'] > 0) |  # 任何锤子线形态
                    (dataframe['CDLDRAGONFLYDOJI'] > 0) |  # 蜻蜓十字
                    (dataframe['CDLINVERTEDHAMMER'] > 0) |  # 倒锤子
                    (dataframe['CDLMORNINGSTAR'] > 0)  # 晨星
                )
            ),
            'enter_long'] = 1

        # 做空条件
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) &  # RSI超买
                (dataframe['slowk'] > 80) &  # Stoch超买
                (dataframe['bb_upperband'] < dataframe['close']) &  # 价格在布林带上轨之上
                (
                    (dataframe['CDLSHOOTINGSTAR'] > 0) |  # 流星线
                    (dataframe['CDLHANGINGMAN'] > 0) |  # 上吊线
                    (dataframe['CDLEVENINGSTAR'] > 0) |  # 暮星
                    (dataframe['CDLENGULFING'] < 0)  # 看空吞没
                )
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 多头出场条件
        dataframe.loc[
            (
                (dataframe['sar'] > dataframe['close']) &  # SAR翻空
                (dataframe['fisher_rsi'] > 0.3)
            ),
            'exit_long'] = 1

        # 空头出场条件
        dataframe.loc[
            (
                (dataframe['sar'] < dataframe['close']) &  # SAR翻多
                (dataframe['fisher_rsi'] < -0.3)
            ),
            'exit_short'] = 1

        return dataframe