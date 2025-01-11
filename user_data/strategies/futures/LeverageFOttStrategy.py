import logging
from numpy.lib import math
from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib

class LeverageFOttStrategy(IStrategy):
    can_short = True
    INTERFACE_VERSION: int = 3
    
    # ROI表:
    minimal_roi = {"0": 0.1, "30": 0.75, "60": 0.05, "120": 0.025}

    # 止损:
    stoploss = -0.265

    # 追踪止损:
    trailing_stop = True
    trailing_stop_positive = 0.05
    trailing_stop_positive_offset = 0.1
    trailing_only_offset_is_reached = False

    # 添加杠杆设置
    leverage_optimization = True  # 是否启用杠杆优化
    max_leverage = 3  # 最大杠杆倍数
    base_leverage = 2  # 基础杠杆倍数
    leverage_step = 1  # 杠杆调整步长
    
    timeframe = "1h"
    startup_candle_count = 18

    def leverage_adjustment(self, dataframe: DataFrame, current_rate: float, **kwargs) -> float:
        """
        根据市场情况动态调整杠杆
        """
        if not self.leverage_optimization:
            return self.base_leverage

        # 获取当前K线的ADX值
        current_adx = dataframe['adx'].iat[-1]
        
        # 根据ADX值调整杠杆
        if current_adx < 20:  # 趋势较弱
            return max(1, self.base_leverage - self.leverage_step)
        elif current_adx > 40:  # 趋势较强
            return min(self.max_leverage, self.base_leverage + self.leverage_step)
        
        return self.base_leverage

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ott"] = self.ott(dataframe)["OTT"]
        dataframe["var"] = self.ott(dataframe)["VAR"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        
        # 添加杠杆相关指标
        if self.leverage_optimization:
            dataframe['leverage'] = dataframe.apply(
                lambda row: self.leverage_adjustment(
                    dataframe.loc[:row.name], row['close']
                ),
                axis=1
            )
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (qtpylib.crossed_above(dataframe["var"], dataframe["ott"])),
            "enter_long",
        ] = 1

        dataframe.loc[
            (qtpylib.crossed_below(dataframe["var"], dataframe["ott"])),
            "enter_short",
        ] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (dataframe["adx"] > 60),
            "exit_long",
        ] = 1

        dataframe.loc[
            (dataframe["adx"] > 60),
            "exit_short",
        ] = 1

        return dataframe

    def ott(self, dataframe: DataFrame):
        df = dataframe.copy()

        pds = 2
        percent = 1.4
        alpha = 2 / (pds + 1)

        df["ud1"] = np.where(
            df["close"] > df["close"].shift(1), (df["close"] - df["close"].shift()), 0
        )
        df["dd1"] = np.where(
            df["close"] < df["close"].shift(1), (df["close"].shift() - df["close"]), 0
        )
        df["UD"] = df["ud1"].rolling(9).sum()
        df["DD"] = df["dd1"].rolling(9).sum()
        df["CMO"] = ((df["UD"] - df["DD"]) / (df["UD"] + df["DD"])).fillna(0).abs()

        df["Var"] = 0.0
        for i in range(pds, len(df)):
            df["Var"].iat[i] = (alpha * df["CMO"].iat[i] * df["close"].iat[i]) + (
                1 - alpha * df["CMO"].iat[i]
            ) * df["Var"].iat[i - 1]

        df["fark"] = df["Var"] * percent * 0.01
        df["newlongstop"] = df["Var"] - df["fark"]
        df["newshortstop"] = df["Var"] + df["fark"]
        df["longstop"] = 0.0
        df["shortstop"] = 999999999999999999

        for i in df["UD"]:
            def maxlongstop():
                df.loc[(df["newlongstop"] > df["longstop"].shift(1)), "longstop"] = df[
                    "newlongstop"
                ]
                df.loc[(df["longstop"].shift(1) > df["newlongstop"]), "longstop"] = df[
                    "longstop"
                ].shift(1)
                return df["longstop"]

            def minshortstop():
                df.loc[
                    (df["newshortstop"] < df["shortstop"].shift(1)), "shortstop"
                ] = df["newshortstop"]
                df.loc[
                    (df["shortstop"].shift(1) < df["newshortstop"]), "shortstop"
                ] = df["shortstop"].shift(1)
                return df["shortstop"]

            df["longstop"] = np.where(
                ((df["Var"] > df["longstop"].shift(1))),
                maxlongstop(),
                df["newlongstop"],
            )

            df["shortstop"] = np.where(
                ((df["Var"] < df["shortstop"].shift(1))),
                minshortstop(),
                df["newshortstop"],
            )

        df["xlongstop"] = np.where(
            (
                (df["Var"].shift(1) > df["longstop"].shift(1))
                & (df["Var"] < df["longstop"].shift(1))
            ),
            1,
            0,
        )

        df["xshortstop"] = np.where(
            (
                (df["Var"].shift(1) < df["shortstop"].shift(1))
                & (df["Var"] > df["shortstop"].shift(1))
            ),
            1,
            0,
        )

        df["trend"] = 0
        df["dir"] = 0
        for i in df["UD"]:
            df["trend"] = np.where(
                ((df["xshortstop"] == 1)),
                1,
                (np.where((df["xlongstop"] == 1), -1, df["trend"].shift(1))),
            )

            df["dir"] = np.where(
                ((df["xshortstop"] == 1)),
                1,
                (np.where((df["xlongstop"] == 1), -1, df["dir"].shift(1).fillna(1))),
            )

        df["MT"] = np.where(df["dir"] == 1, df["longstop"], df["shortstop"])
        df["OTT"] = np.where(
            df["Var"] > df["MT"],
            (df["MT"] * (200 + percent) / 200),
            (df["MT"] * (200 - percent) / 200),
        )
        df["OTT"] = df["OTT"].shift(2)

        return DataFrame(index=df.index, data={"OTT": df["OTT"], "VAR": df["Var"]})