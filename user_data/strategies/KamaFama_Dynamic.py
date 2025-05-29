# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from functools import reduce
from pandas import DataFrame, Series
import numpy as np

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
import time
import ccxt

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
    position_adjustment_enable = True

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

    minimal_roi = {'0': 0.087, '372': 0.068, '861': 0.045}
    cc_long = {}
    cc_short = {}

    # 策略模式状态跟踪
    pair_strategy_mode = {}

    price_range_thresholds = {}

    # 固定点位监控
    coin_monitoring = {}

    # 回测模式下跟踪每个交易对的当前蜡烛时间
    current_candle_date = {}

    # Stoploss:
    stoploss = -1

    # Sell Params
    sell_fastx = IntParameter(50, 100, default=84, space='sell', optimize=True)

    # 需要添加的新参数 - 为做空策略专门优化
    buy_fastx_short = IntParameter(0, 50, default=16, space='buy', optimize=True)

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.05
    trailing_only_offset_is_reached = True
    max_entry_position_adjustment = 5

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
        从外部JSON文件加载策略模式配置并处理策略转换情况：
        - 删除不再推荐做空的交易对的空头逻辑
        - 确保策略模式与监控配置保持一致
        - 默认使用多头策略
        """
        if self.config.get('runmode', None) in ('live', 'dry_run'):
            self.state_file = 'user_data/strategy_state_production.json'
        else:
            self.state_file = 'user_data/strategy_state.json'

        if not os.path.exists(self.state_file):
            logger.info(f"警告: 策略模式配置文件 {self.state_file} 不存在，将使用默认多头策略")
            return

        try:
            with open(self.state_file, 'r') as f:
                state_data = json.load(f)

            # 加载策略模式配置
            if 'pair_strategy_mode' in state_data:
                self.pair_strategy_mode = state_data['pair_strategy_mode']

            # 加载价格范围阈值
            if 'price_range_thresholds' in state_data:
                self.price_range_thresholds = state_data['price_range_thresholds']

            # 加载并处理固定点位监控配置
            if 'coin_monitoring' in state_data:
                self.coin_monitoring = state_data['coin_monitoring']
                updated_configs = False

                # 处理每个交易对的监控配置
                for pair in list(self.coin_monitoring.keys()):
                    # 获取该交易对的当前策略模式(默认为'long')
                    current_mode = self.pair_strategy_mode.get(pair, 'long')

                    # 删除与当前策略模式不匹配的监控配置
                    has_matching_config = False
                    valid_configs = []

                    for config in self.coin_monitoring[pair]:
                        direction = config.get('direction', 'long')
                        auto = config.get('auto', True)
                        if not auto:
                            logger.info(f"交易对 {pair}({direction}) 已关闭自动更新，")
                            valid_configs.append({**config, 'auto_initialized': True})
                            has_matching_config = True
                            continue

                        # 关键逻辑：如果当前模式是long，删除所有short配置
                        if current_mode == 'long' and direction == 'short':
                            logger.info(f"交易对 {pair} 不再推荐做空，移除空头监控配置")
                            updated_configs = True
                            continue

                        # 如果当前模式是short，仅保留short配置
                        if current_mode == 'short' and direction == 'short':
                            valid_configs.append({**config, 'auto_initialized': False})
                            has_matching_config = True

                        # 多头配置总是被保留
                        if direction == 'long' and current_mode == 'long':
                            valid_configs.append({**config, 'auto_initialized': False})
                            has_matching_config = True

                        if direction == 'long' and current_mode == 'short':
                            logger.info(f"交易对 {pair} 不再推荐做多，移除多头监控配置")
                            updated_configs = True
                            continue

                    # 如果没有与当前策略模式匹配的配置，添加一个默认配置
                    if not has_matching_config:
                        logger.info(f"交易对 {pair} 没有与策略模式 '{current_mode}' 匹配的监控配置，添加默认配置")
                        valid_configs.append(
                            {
                                'direction': current_mode,
                                'auto': True,
                                'entry_points': [],
                                'exit_points': [],
                                'auto_initialized': False,
                            }
                        )
                        updated_configs = True

                    # 更新或删除交易对的监控配置
                    if valid_configs:
                        # 如果原始配置和过滤后的配置数量不同，记录日志
                        if len(valid_configs) != len(self.coin_monitoring[pair]):
                            logger.info(f"交易对 {pair} 监控配置已更新，移除了不匹配的策略配置")
                            updated_configs = True

                        self.coin_monitoring[pair] = valid_configs
                    else:
                        # 如果没有有效配置，删除该交易对的监控
                        logger.info(f"交易对 {pair} 没有有效监控配置，从监控列表中移除")
                        del self.coin_monitoring[pair]
                        updated_configs = True

                # 如果有配置更新，保存到策略状态文件
                if updated_configs and self.config.get('runmode', None) in ('live', 'dry_run'):
                    self.update_strategy_state_file()
                    logger.info('监控配置已更新并保存到策略状态文件')

            logger.info(f"成功加载策略模式文件: {self.pair_strategy_mode}")

            # 统计并输出配置信息
            if self.coin_monitoring:
                pairs_count = len(self.coin_monitoring)
                long_configs = sum(
                    1
                    for pair in self.coin_monitoring
                    for config in self.coin_monitoring[pair]
                    if config.get('direction') == 'long'
                )
                short_configs = sum(
                    1
                    for pair in self.coin_monitoring
                    for config in self.coin_monitoring[pair]
                    if config.get('direction') == 'short'
                )
                logger.info(
                    f"固定点位监控配置: {pairs_count}个交易对 (多头: {long_configs}, 空头: {short_configs})"
                )

        except Exception as e:
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

    # def calculate_coin_points(self, pair: str, direction: str):
    #     df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
    #     if df is None or df.empty:
    #         logger.warning(f"无法获取 {pair} 的5m数据，跳过自动设置")
    #         return

    #     # 取最近288根K线（相当于24小时的5m数据）
    #     candles_to_use = 288  # 24小时 × 12根/小时
    #     if len(df) < candles_to_use:
    #         logger.warning(f"{pair} 数据不足 {candles_to_use} 根K线，仅有 {len(df)} 根，跳过自动设置")
    #         return

    #     recent_df = df.tail(candles_to_use)  # 取最后288根K线

    #     # 计算最近288根K线的最高价和最低价
    #     recent_high = recent_df['high'].max()
    #     recent_low = recent_df['low'].min()
    #     # price_range = recent_high - recent_low

    #     config = {}
    #     if direction == 'long':
    #         config['entry_points'] = [recent_low * 1.005]  # 略高于最低价
    #         config['exit_points'] = [
    #             recent_low * 1.005 * 1.02,  # 第一目标
    #             recent_low * 1.005 * 1.04,  # 第二目标
    #             recent_low * 1.005 * 1.06,  # 接近最高价
    #         ]
    #         config['stop_loss'] = recent_low * 0.95  # 略低于最低价

    #     elif direction == 'short':
    #         config['entry_points'] = [recent_high * 0.995]  # 略低于最高价
    #         config['exit_points'] = [
    #             recent_high * 0.995 * 0.98,  # 第一目标
    #             recent_high * 0.995 * 0.96,  # 第二目标
    #             recent_high * 0.995 * 0.94,  # 接近最低价
    #         ]
    #         config['stop_loss'] = recent_low * 1.05

    #     return config

    def find_support_resistance_levels(self, dataframe, n_levels=3):
        """
        识别主要支撑位和阻力位

        参数:
            dataframe: 价格数据框架
            n_levels: 返回的支撑/阻力位数量

        返回:
            支撑位和阻力位列表
        """
        # 计算价格变动的标准差，用于判断显著价格水平
        # price_std = dataframe['close'].pct_change().std()

        # 创建价格区间，将连续价格分组
        price_range = dataframe['high'].max() - dataframe['low'].min()
        n_bins = 100  # 将价格范围分成100个区间
        bin_size = price_range / n_bins

        # 创建价格区间直方图
        price_bins = [[] for _ in range(n_bins)]
        volume_bins = [0] * n_bins

        # 填充价格和交易量数据到区间
        min_price = dataframe['low'].min()
        for i, row in dataframe.iterrows():
            # 考虑每根K线的高低点范围
            low_bin = int((row['low'] - min_price) / bin_size)
            high_bin = int((row['high'] - min_price) / bin_size)

            # 确保区间索引在有效范围内
            low_bin = max(0, min(low_bin, n_bins - 1))
            high_bin = max(0, min(high_bin, n_bins - 1))

            # 将价格点添加到相应区间
            for b in range(low_bin, high_bin + 1):
                price_bins[b].append(row['close'])
                volume_bins[b] += row['volume']

        # 分析每个区间的"停留时间"和交易量
        resistance_scores = []
        support_scores = []

        for i in range(n_bins):
            if len(price_bins[i]) > 0:
                bin_price = min_price + (i + 0.5) * bin_size
                # 计算该价格区间的得分 (基于价格点数量和交易量)
                score = len(price_bins[i]) * (
                    1 + volume_bins[i] / max(volume_bins) if max(volume_bins) > 0 else 1
                )

                # 分析价格在该区间的行为来判断是支撑还是阻力
                price_before = [
                    p
                    for j in range(max(0, i - 3), i)
                    for p in price_bins[j]
                    if len(price_bins[j]) > 0
                ]
                price_after = [
                    p
                    for j in range(i + 1, min(n_bins, i + 4))
                    for p in price_bins[j]
                    if len(price_bins[j]) > 0
                ]

                # 如果价格通常从该水平向上反弹，则可能是支撑位
                if (
                    price_before
                    and price_after
                    and np.mean(price_before) > bin_price
                    and np.mean(price_after) > bin_price
                ):
                    support_scores.append((bin_price, score))
                # 如果价格通常从该水平向下反转，则可能是阻力位
                elif (
                    price_before
                    and price_after
                    and np.mean(price_before) < bin_price
                    and np.mean(price_after) < bin_price
                ):
                    resistance_scores.append((bin_price, score))

        # 根据得分排序
        support_scores.sort(key=lambda x: x[1], reverse=True)
        resistance_scores.sort(key=lambda x: x[1], reverse=True)

        # 返回得分最高的n_levels个支撑位和阻力位
        supports = [price for price, _ in support_scores[:n_levels]]
        resistances = [price for price, _ in resistance_scores[:n_levels]]

        return supports, resistances

    def zigzag_points(self, dataframe, deviation=5):
        """
        使用ZigZag算法找出重要的转折点

        参数:
            dataframe: 价格数据框架
            deviation: 最小偏差百分比

        返回:
            高点和低点列表
        """
        highs = []
        lows = []

        # 转换为百分比
        dev = deviation / 100

        # 初始化
        last_high = dataframe['high'].iloc[0]
        last_low = dataframe['low'].iloc[0]
        high_idx = 0
        low_idx = 0
        trend = None  # None for initial, 1 for up, -1 for down

        for i in range(1, len(dataframe)):
            curr_high = dataframe['high'].iloc[i]
            curr_low = dataframe['low'].iloc[i]

            # 初始趋势判断
            if trend is None:
                if curr_high > last_high:
                    trend = 1  # 上升趋势
                elif curr_low < last_low:
                    trend = -1  # 下降趋势

            # 上升趋势中
            if trend == 1:
                # 如果找到更高点，更新最高点
                if curr_high > last_high:
                    last_high = curr_high
                    high_idx = i
                # 如果下降超过偏差，记录高点并转为下降趋势
                elif curr_low < last_high * (1 - dev):
                    highs.append((high_idx, last_high))
                    last_low = curr_low
                    low_idx = i
                    trend = -1

            # 下降趋势中
            elif trend == -1:
                # 如果找到更低点，更新最低点
                if curr_low < last_low:
                    last_low = curr_low
                    low_idx = i
                # 如果上升超过偏差，记录低点并转为上升趋势
                elif curr_high > last_low * (1 + dev):
                    lows.append((low_idx, last_low))
                    last_high = curr_high
                    high_idx = i
                    trend = 1

        # 添加最后一个点
        if trend == 1:
            highs.append((high_idx, last_high))
        else:
            lows.append((low_idx, last_low))

        # 提取价格值
        high_points = [price for _, price in highs]
        low_points = [price for _, price in lows]

        return high_points, low_points

    def get_ohlcv_history(self, pair: str, timeframe: str = '1h', limit: int = 150):
        """
        使用CCXT直接从交易所获取历史K线数据

        参数:
            pair: 交易对名称，直接使用传入的格式
            timeframe: 时间周期，例如 '1h', '4h', '1d'
            limit: 获取的K线数量

        返回:
            pandas DataFrame包含OHLCV数据，如果失败则返回None
        """
        try:
            # 获取交易所名称
            exchange_name = self.config['exchange']['name'].lower()

            # 创建交易所实例
            exchange_class = getattr(ccxt, exchange_name)

            # 获取API凭证
            api_config = {
                'apiKey': self.config['exchange'].get('key', ''),
                'secret': self.config['exchange'].get('secret', ''),
                'enableRateLimit': True,
            }

            # 过滤空值
            api_config = {k: v for k, v in api_config.items() if v}

            # 实例化交易所
            exchange = exchange_class(api_config)

            # 设置市场类型 (针对Binance等交易所)
            if hasattr(exchange, 'options'):
                if exchange_name == 'binance':
                    exchange.options['defaultType'] = 'spot'

            logger.info(f"使用CCXT获取 {pair} {timeframe} 数据")

            # 加载市场
            exchange.load_markets()

            # 获取OHLCV数据 - 不使用since参数，让交易所返回最近的数据
            ohlcv = exchange.fetch_ohlcv(symbol=pair, timeframe=timeframe, limit=limit)

            if ohlcv and len(ohlcv) > 0:
                # 转换为DataFrame
                df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

                # 转换时间戳为日期时间
                df['date'] = pd.to_datetime(df['date'], unit='ms')

                logger.info(f"成功获取 {pair} {timeframe} 数据，共 {len(df)} 条")
                return df
            else:
                logger.error(f"未获取到 {pair} 的 {timeframe} 数据")
                return None

        except Exception as e:
            logger.error(f"获取历史数据时出错: {str(e)}")
            return None

    def calculate_coin_points(self, pair: str, direction: str):
        # 获取1小时K线数据用于支撑/阻力位识别
        df_1h = self.get_ohlcv_history(pair, timeframe='1h', limit=150)

        if df_1h is None or df_1h.empty:
            logger.warning(f"无法获取 {pair} 的1h数据，跳过计算支撑/阻力位")
            return None

        # 获取5分钟数据用于更精确的进场点
        df_5m = self.get_ohlcv_history(pair, timeframe='5m', limit=350)

        if df_5m is None or df_5m.empty:
            df_5m, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            if df_5m is None or df_5m.empty:
                logger.warning(f"无法获取 {pair} 的5m数据，跳过自动设置")
                return None

        # 1. 从1小时图表识别主要支撑位和阻力位
        supports_1h, resistances_1h = self.find_support_resistance_levels(
            df_1h, n_levels=5
        )  # 增加到5个点位以获取更多候选位置

        # 2. 使用ZigZag过滤1小时图表上的噪音，找到重要转折点
        highs_1h, lows_1h = self.zigzag_points(df_1h, deviation=5)

        # 3. 分析近期5分钟数据的波动情况
        recent_5m = df_5m.tail(288)  # 最近24小时
        volatility = recent_5m['close'].pct_change().std() * 100  # 波动率百分比

        # 当前价格
        current_price = df_5m.iloc[-1]['close']

        # 设置基于预期利润的过滤参数
        expected_profit_pct = 4.0  # 预期利润百分比
        max_distance_pct = expected_profit_pct * 1.5  # 允许的最大距离百分比，稍大于预期利润

        logger.info(
            f"{pair} 当前价格: {current_price}, 24小时波动率: {volatility:.2f}%, 最大允许距离: {max_distance_pct:.2f}%"
        )

        # 4. 整合不同时间周期的结果并基于预期利润过滤点位
        if direction == 'long':
            # 对于多头，我们关注支撑位
            valid_supports = []

            # 添加1小时图表的支撑位
            for support in supports_1h:
                # 计算支撑位与当前价格的距离百分比
                distance_pct = (current_price - support) / current_price * 100
                if 0 < distance_pct <= max_distance_pct:
                    valid_supports.append(support)
                    logger.info(f"有效支撑位: {support}, 距离当前价格: {distance_pct:.2f}%")

            # 添加ZigZag低点
            for low in lows_1h:
                distance_pct = (current_price - low) / current_price * 100
                if 0 < distance_pct <= max_distance_pct:
                    valid_supports.append(low)
                    logger.info(f"有效ZigZag低点: {low}, 距离当前价格: {distance_pct:.2f}%")

            # 如果没有找到有效支撑位，则基于预期利润计算入场点
            if not valid_supports:
                valid_supports = [df_5m['low'].min()]
                logger.info(f"{pair}没有找到合适距离内的支撑位，使用当日最低: {valid_supports[0]}")

            # 根据接近当前价格的程度排序
            valid_supports.sort(key=lambda x: abs(current_price - x))

            # 选择最接近但低于当前价格的支撑位作为入场点
            entry_point = min(valid_supports) * 1.005  # 略高于支撑位

            # 根据波动率和预期利润动态调整止盈点位
            # 如果波动率低，使用更接近预期利润的目标位
            # 如果波动率高，允许更大的利润目标
            volatility_factor = min(max(volatility / 10, 0.8), 1.5)  # 将波动率影响控制在0.8-1.5之间

            tp1_pct = expected_profit_pct * 0.4 * volatility_factor  # 第一目标位
            tp2_pct = expected_profit_pct * 0.8 * volatility_factor  # 第二目标位
            tp3_pct = expected_profit_pct * 1.2 * volatility_factor  # 第三目标位

            logger.info(
                f"波动率因子: {volatility_factor:.2f}, TP1: {tp1_pct:.2f}%, TP2: {tp2_pct:.2f}%, TP3: {tp3_pct:.2f}%"
            )

            config = {
                'entry_points': [entry_point],
                'exit_points': [
                    entry_point * (1 + tp1_pct / 100),  # 第一目标
                    entry_point * (1 + tp2_pct / 100),  # 第二目标
                    entry_point * (1 + tp3_pct / 100),  # 第三目标
                ],
                'stop_loss': entry_point * (1 - expected_profit_pct * 0.4 / 100),  # 止损位，预期利润的40%
            }

        elif direction == 'short':
            # 对于空头，我们关注阻力位
            valid_resistances = []

            # 添加1小时图表的阻力位
            for resistance in resistances_1h:
                # 计算阻力位与当前价格的距离百分比
                distance_pct = (resistance - current_price) / current_price * 100
                if 0 < distance_pct <= max_distance_pct:
                    valid_resistances.append(resistance)
                    logger.info(f"有效阻力位: {resistance}, 距离当前价格: {distance_pct:.2f}%")

            # 添加ZigZag高点
            for high in highs_1h:
                distance_pct = (high - current_price) / current_price * 100
                if 0 < distance_pct <= max_distance_pct:
                    valid_resistances.append(high)
                    logger.info(f"有效ZigZag高点: {high}, 距离当前价格: {distance_pct:.2f}%")

            # 如果没有找到有效阻力位，则基于预期利润计算入场点
            if not valid_resistances:
                valid_resistances = [df_5m['high'].max()]
                logger.info(f"{pair}没有找到合适距离内的阻力位，使用当日最高: {valid_resistances[0]}")

            # 根据接近当前价格的程度排序
            valid_resistances.sort(key=lambda x: abs(current_price - x))

            # 选择最接近但高于当前价格的阻力位作为入场点
            entry_point = max(valid_resistances) * 0.995  # 略低于阻力位

            # 同样基于波动率和预期利润动态调整止盈点位
            volatility_factor = min(max(volatility / 10, 0.8), 1.5)

            tp1_pct = expected_profit_pct * 0.4 * volatility_factor
            tp2_pct = expected_profit_pct * 0.8 * volatility_factor
            tp3_pct = expected_profit_pct * 1.2 * volatility_factor

            logger.info(
                f"波动率因子: {volatility_factor:.2f}, TP1: {tp1_pct:.2f}%, TP2: {tp2_pct:.2f}%, TP3: {tp3_pct:.2f}%"
            )

            config = {
                'entry_points': [entry_point],
                'exit_points': [
                    entry_point * (1 - tp1_pct / 100),  # 第一目标
                    entry_point * (1 - tp2_pct / 100),  # 第二目标
                    entry_point * (1 - tp3_pct / 100),  # 第三目标
                ],
                'stop_loss': entry_point * (1 + expected_profit_pct * 0.4 / 100),  # 止损位，预期利润的40%
            }

        # 修复空头模式可能存在的错误
        if direction == 'short' and 'valid_supports' in locals() and 'entry_point' not in locals():
            logger.warning(f"{pair} 空头模式中错误使用了支撑位，重新计算...")
            entry_point = valid_resistances[0] * 0.995  # 使用阻力位重新计算

        return config

    def reload_coin_monitoring(self, pair: str):
        # 处理coin_monitoring的auto设置（仅在live或dry_run模式下）
        if (
            self.config.get('runmode', None) in ('live', 'dry_run')
            and pair in self.coin_monitoring
            and hasattr(self, 'dp')
        ):
            if pair in self.pair_strategy_mode:
                strategies = [
                    config
                    for config in self.coin_monitoring[pair]
                    if (config['direction'] == self.pair_strategy_mode[pair])
                    or (not config.get('auto', False))
                ]
                if self.pair_strategy_mode[pair] not in [
                    strategy['direction'] for strategy in strategies
                ]:
                    self.coin_monitoring[pair] = [
                        {
                            'direction': self.pair_strategy_mode[pair],
                            'auto': True,
                            'entry_points': [],
                            'exit_points': [],
                            'auto_initialized': False,  # 确保新配置未初始化
                        }
                    ]
            has_data = False
            configs_to_update = []  # 记录需要更新的配置

            for config in self.coin_monitoring[pair]:
                if config.get('auto', False) and not config.get(
                    'auto_initialized', False
                ):  # 检查auto且未初始化
                    # 计算点位配置
                    direction = config.get('direction', 'long')
                    point_config = self.calculate_coin_points(pair, direction)

                    if point_config:
                        # 更新配置
                        config['entry_points'] = point_config['entry_points']
                        config['exit_points'] = point_config['exit_points']
                        config['stop_loss'] = point_config.get('stop_loss', None)
                        config['auto_initialized'] = True  # 标记为已初始化
                        configs_to_update.append((direction, config))
                        has_data = True

            if has_data:
                with open(self.state_file, 'r') as f:
                    strategy_state = json.load(f)
                strategy_state['coin_monitoring'] = self.coin_monitoring
                with open(self.state_file, 'w') as f:
                    json.dump(strategy_state, f, indent=4)

                # 发送通知消息
                for direction, config in configs_to_update:
                    entry_point = config['entry_points'][0] if config['entry_points'] else 'N/A'
                    exit_points = (
                        ','.join([str(i) for i in config['exit_points']])
                        if config['exit_points']
                        else 'N/A'
                    )

                    logger.info(
                        f"自动设置 {pair} ({direction}) 使用多时间周期分析: "
                        f"entry_points={entry_point}, "
                        f"exit_points={exit_points}"
                    )
                    if hasattr(self, 'dp') and hasattr(self.dp, 'send_msg'):
                        self.dp.send_msg(
                            f"自动设置 {pair} ({direction}) 使用多时间周期分析: "
                            f"entry_points={entry_point}, "
                            f"exit_points={exit_points}"
                        )

                logger.info(f"成功更新 {pair} 的监控配置并保存到文件")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # 获取当前pair
        pair = metadata['pair']

        # 更新当前candle的时间（用于回测模式下的时间判断）
        if len(dataframe) > 0:
            self.current_candle_date[pair] = dataframe.iloc[-1]['date']

        self.reload_coin_monitoring(pair)

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

    def check_active_trades(
        self, pair: str, current_price: float, threshold_percent: float = 10
    ) -> bool:
        """
        Check if there is an active trade for the given pair and if the current price
        is within the specified threshold percentage of the opening price.

        Args:
            pair: The trading pair to check
            current_price: Current market price
            threshold_percent: Percentage threshold for price range check (default: 10%)

        Returns:
            bool: True if an active trade exists with current price within threshold% of entry price,
                False otherwise
        """
        # Skip the check in backtesting mode
        if self.config.get('runmode', None) not in ('live', 'dry_run'):
            return False

        # Get active trades for this pair
        active_trades = Trade.get_trades_proxy(is_open=True, pair=pair)

        # No active trades for this pair
        if not active_trades:
            return False

        # Check if current price is within 10% of any active trade's opening price
        for trade in active_trades:
            open_rate = trade.open_rate
            price_diff_percent = abs((current_price - open_rate) / open_rate * 100)

            # If price is within threshold% of opening price, skip this pair
            if price_diff_percent <= threshold_percent:
                logger.info(
                    f"Skipping {pair}: Current price {current_price} is within {threshold_percent}% of opening price {open_rate}"
                )
                return True

        return True

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        根据当前交易对的策略模式决定使用哪种入场逻辑，并检查是否允许开单
        """
        pair = metadata['pair']

        # 默认设置
        dataframe.loc[:, 'enter_long'] = 0
        dataframe.loc[:, 'enter_short'] = 0
        dataframe.loc[:, 'enter_tag'] = ''

        # 增加活跃交易检查，如果在回测模式下，则不进行检查
        if self.config.get('runmode', None) in ('live', 'dry_run'):

            # 检查是否有活跃交易
            active_trades = Trade.get_trades_proxy(is_open=True, pair=pair)

            # 如果有活跃交易，并且pair在监控配置中，需要关闭自动计算
            if active_trades and pair in self.coin_monitoring:
                for trade in active_trades:
                    direction = 'short' if trade.is_short else 'long'
                    current_time = datetime.now(trade.open_date_utc.tzinfo)
                    # 如果刚开仓成功（5分钟内的交易），关闭自动计算
                    if (current_time - trade.open_date_utc).total_seconds() < 300:  # 5分钟内
                        self.disable_auto_calculation(pair, direction)

            # 根据最后一行数据的收盘价进行检查
            if len(dataframe) > 0:
                current_price = dataframe.iloc[-1]['close']

                # 如果已有活跃交易且价格在开仓价格设定范围内，直接返回原始dataframe（跳过信号生成）
                # 可以根据不同交易对的波动性调整阈值
                price_range_threshold = self.price_range_thresholds.get(pair, 7)  # 默认设为7%
                if self.check_active_trades(pair, current_price, price_range_threshold):
                    return dataframe

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

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        """
        Called right before executing a trade exit order.
        This method checks if the exit is due to ROI and re-enables auto calculation if needed.
        """
        direction = 'short' if trade.is_short else 'long'

        # Check if this is a ROI exit
        if exit_reason.upper().startswith('ROI'):
            logger.info(f"{pair}: Exit triggered by ROI - re-enabling auto calculation")
            self.enable_auto_calculation(pair, direction)

        # Always confirm the exit
        return True

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
                    entry_condition = dataframe['close'] <= entry_point
                    dataframe.loc[entry_condition, 'enter_long'] = 1
                    dataframe.loc[entry_condition, 'enter_tag'] = f'fixed_long_entry_{entry_point}'

                # 对于空头
                elif direction == 'short':
                    # 如果当前价格在入场点附近（上下0.5%范围内）
                    entry_condition = dataframe['close'] >= entry_point
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

    # 1. 补仓后重新计算止盈点位的函数
    def recalculate_exit_points_after_dca(self, trade, direction):
        pair = trade.pair

        # 查找该交易对的监控配置
        if pair in self.coin_monitoring:
            for config in self.coin_monitoring[pair]:
                if config.get('direction') == direction:
                    # 保存原始入场点位
                    entry_points = config.get('entry_points', [])
                    if not entry_points:
                        continue

                    # 获取当前价格作为新的成本价
                    cost_price = trade.open_rate

                    # 根据方向重新计算止盈点位
                    if direction == 'long':
                        config['exit_points'] = [
                            cost_price * 1.02,  # 第一目标 +2%
                            cost_price * 1.04,  # 第二目标 +4%
                            cost_price * 1.06,  # 第三目标 +6%
                        ]
                    elif direction == 'short':
                        config['exit_points'] = [
                            cost_price * 0.98,  # 第一目标 -2%
                            cost_price * 0.96,  # 第二目标 -4%
                            cost_price * 0.94,  # 第三目标 -6%
                        ]

                    # 关闭自动计算
                    config['auto'] = False

                    # 更新持久化文件
                    self.update_strategy_state_file()

                    # 记录日志
                    logger.info(f"补仓后重新计算 {pair} 的止盈点位: {config['exit_points']}")
                    if hasattr(self, 'dp') and hasattr(self.dp, 'send_msg'):
                        self.dp.send_msg(f"补仓后重新计算 {pair} 的止盈点位: {config['exit_points']}")

                    break

    # 2. 开单后关闭自动计算
    def disable_auto_calculation(self, pair, direction):
        if pair in self.coin_monitoring:
            for config in self.coin_monitoring[pair]:
                if config.get('direction') == direction and config.get('auto', False):
                    config['auto'] = False
                    logger.info(f"已关闭 {pair} 的自动计算功能")
                    self.update_strategy_state_file()
                    break

    # 2. 开单后关闭自动计算
    def enable_auto_calculation(self, pair, direction):
        if pair in self.coin_monitoring:
            for config in self.coin_monitoring[pair]:
                if config.get('direction') == direction:
                    config['auto'] = True
                    config['auto_initialized'] = False
                    logger.info(f"已开启 {pair} 的自动计算功能")
                    self.reload_coin_monitoring(pair)
                    self.update_strategy_state_file()
                    break

    # 3. 更新持久化文件的函数
    def update_strategy_state_file(self):
        try:
            file_path = self.state_file
            # 简单的文件锁机制
            lock_file = f"{file_path}.lock"

            # 尝试创建锁文件
            try:
                with open(lock_file, 'x') as f:
                    pass
            except FileExistsError:
                # 如果锁文件已存在，等待然后重试
                time.sleep(0.1)
                return self.update_strategy_state_file()

            try:
                # 读取当前内容
                with open(file_path, 'r') as f:
                    strategy_state = json.load(f)

                # 更新内容
                strategy_state['coin_monitoring'] = self.coin_monitoring
                strategy_state['pair_strategy_mode'] = self.pair_strategy_mode
                strategy_state['price_range_thresholds'] = self.price_range_thresholds

                # 写入更新后的内容
                with open(file_path, 'w') as f:
                    json.dump(strategy_state, f, indent=4)

            finally:
                # 无论如何都要删除锁文件
                os.remove(lock_file)

        except Exception as e:
            logger.error(f"更新策略状态文件失败: {e}")

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        """
        使用持久化存储实现多级退出策略

        根据退出点位实现分批退出，并动态调整止损位
        - 如果只有1个点位：直接全部退出
        - 如果有2个点位：每次退出50%仓位
        - 如果有3个或更多点位：第一点位退出30%，第二点位退出剩余的50%，第三点位全部退出
        """

        pair = trade.pair
        direction = 'short' if trade.is_short else 'long'

        # 初次遇到交易时，保存初始stake金额
        if trade.get_custom_data('initial_stake') is None:
            trade.set_custom_data('initial_stake', trade.stake_amount)

        # 先检查是否需要补仓
        if current_profit < 0:

            # 检查补仓冷却期
            last_dca_time = trade.get_custom_data('last_dca_time')
            if last_dca_time is not None:
                cooldown_minutes = 60  # 30分钟冷却期 (可根据交易对波动性调整)
                last_dca_datetime = datetime.fromtimestamp(
                    last_dca_time, trade.open_date_utc.tzinfo
                )
                if current_time < last_dca_datetime + timedelta(minutes=cooldown_minutes):
                    logger.info(f"{pair}: 补仓冷却期未结束，上次补仓时间: {last_dca_datetime}")
                    return None

            # 检查当前波动性 - 为当前交易对获取适合的数据
            dataframe, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)

            if dataframe is None or len(dataframe) < 20:  # 需要至少20根K线
                return None

            # 计算最近20根K线的波动性 (标准差)
            recent_df = dataframe.tail(20)
            volatility = recent_df['close'].pct_change().std() * 100  # 转为百分比

            # 设置波动性阈值：波动性太大时不补仓
            max_volatility_threshold = 3.0  # 3%，可根据交易对特性调整
            if volatility > max_volatility_threshold:
                logger.info(
                    f"{pair}: 当前波动性 ({volatility:.2f}%) 高于阈值 ({max_volatility_threshold}%)，暂不补仓"
                )
                return None

            # 计算亏损百分比（确保为正数）
            loss_percentage = abs(current_profit)

            # 检查当前仓位金额是否已超过最大限制
            if trade.stake_amount >= 400:
                logger.info(f"{pair}: 当前仓位金额 {trade.stake_amount} 已超过最大限制 400，不再补仓")
                return None
            else:
                # 方法1: 基于亏损百分比的补仓策略 (原有逻辑)
                dca_amount_1 = 0
                dca_tag_1 = ''

                # 获取当前和前几个周期的RSI值用于判断趋势变化
                current_candle = dataframe.iloc[-1].squeeze()
                current_rsi_84 = current_candle['rsi_84']
                previous_rsi_84 = dataframe.iloc[-2]['rsi_84']  # 前一个周期的RSI
                available_length = len(dataframe)

                # 亏损20%以上，补仓50%
                if loss_percentage >= 0.20:
                    dca_amount_1 = trade.stake_amount * 0.5
                    dca_tag_1 = f"{direction}_dca_loss_20pct"
                # 亏损15%以上，补仓40%
                elif loss_percentage >= 0.125 and (
                    (direction == 'long' and current_rsi_84 > previous_rsi_84)
                    or (  # 多头RSI开始上升
                        direction == 'short' and current_rsi_84 < previous_rsi_84
                    )  # 空头RSI开始下降
                ):
                    dca_amount_1 = trade.stake_amount * 0.4
                    dca_tag_1 = f"{direction}_dca_loss_15pct"
                # 亏损10%以上，补仓30%
                elif loss_percentage >= 0.075 and (
                    (
                        direction == 'long'
                        and current_rsi_84 > previous_rsi_84
                        and current_rsi_84 > 30
                    )
                    or (
                        direction == 'short'
                        and current_rsi_84 < previous_rsi_84
                        and current_rsi_84 < 70
                    )
                ):
                    dca_amount_1 = trade.stake_amount * 0.3
                    dca_tag_1 = f"{direction}_dca_loss_10pct"

                # 方法2: 基于技术指标的补仓策略 (新增逻辑)
                dca_amount_2 = 0
                dca_tag_2 = ''

                # 检查当前K线是否满足入场条件
                if direction == 'long':
                    # 使用与多头入场相同的技术指标条件
                    if available_length >= 288:
                        indicator_condition = (
                            (current_candle['kama'] > current_candle['fama'])
                            & (current_candle['fama'] > current_candle['mama'] * 0.981)
                            & (current_candle['r_14'] < -61.3)
                            & (current_candle['mama_diff'] < -0.025)
                            & (current_candle['cti'] < -0.715)
                            & (
                                dataframe['close'].rolling(48).max().iloc[-1]
                                >= current_candle['close'] * 1.05
                            )
                            & (
                                dataframe['close'].rolling(288).max().iloc[-1]
                                >= current_candle['close'] * 1.125
                            )
                            & (current_candle['rsi_84'] < 60)
                            & (current_candle['rsi_112'] < 60)
                        )
                        if indicator_condition:
                            # 指标条件满足，根据亏损程度决定补仓比例
                            if loss_percentage >= 0.15:
                                dca_amount_2 = trade.stake_amount * 0.5  # 大亏损时补仓更多
                                dca_tag_2 = f"{direction}_dca_indicator_high_loss"
                            elif loss_percentage >= 0.075:
                                dca_amount_2 = trade.stake_amount * 0.3  # 中等亏损
                                dca_tag_2 = f"{direction}_dca_indicator_med_loss"
                            elif loss_percentage >= 0.03:
                                dca_amount_2 = trade.stake_amount * 0.2  # 小额亏损
                                dca_tag_2 = f"{direction}_dca_indicator_small_loss"

                else:  # short
                    # 使用与空头入场相同的技术指标条件
                    if available_length >= 288:
                        indicator_condition = (
                            (current_candle['kama'] < current_candle['fama'])
                            & (current_candle['fama'] < current_candle['mama'] * 1.019)
                            & (current_candle['r_14'] > -38.7)
                            & (current_candle['mama_diff'] > 0.025)
                            & (current_candle['cti'] > 0.715)
                            & (
                                dataframe['close'].rolling(48).min().iloc[-1]
                                <= current_candle['close'] * 0.95
                            )
                            & (
                                dataframe['close'].rolling(288).min().iloc[-1]
                                <= current_candle['close'] * 0.875
                            )
                            & (current_candle['rsi_84'] > 40)
                            & (current_candle['rsi_112'] > 40)
                        )
                        if indicator_condition:
                            # 指标条件满足，根据亏损程度决定补仓比例
                            if loss_percentage >= 0.15:
                                dca_amount_2 = trade.stake_amount * 0.5  # 大亏损时补仓更多
                                dca_tag_2 = f"{direction}_dca_indicator_high_loss"
                            elif loss_percentage >= 0.075:
                                dca_amount_2 = trade.stake_amount * 0.3  # 中等亏损
                                dca_tag_2 = f"{direction}_dca_indicator_med_loss"
                            elif loss_percentage >= 0.03:
                                dca_amount_2 = trade.stake_amount * 0.2  # 小额亏损
                                dca_tag_2 = f"{direction}_dca_indicator_small_loss"

                # 选择两种方法中补仓金额较大的那个
                if dca_amount_1 >= dca_amount_2:
                    dca_amount = dca_amount_1
                    dca_tag = dca_tag_1
                else:
                    dca_amount = dca_amount_2
                    dca_tag = dca_tag_2

                # 如果需要补仓
                if dca_amount > 0:
                    # 确保补仓后总金额不超过400
                    if trade.stake_amount + dca_amount > 400:
                        dca_amount = 400 - trade.stake_amount

                    # 确保补仓金额在min_stake和max_stake之间
                    if min_stake and dca_amount < min_stake:
                        # 如果最小补仓金额会导致总金额超过400，则不补仓
                        if trade.stake_amount + min_stake > 400:
                            logger.info(f"{pair}: 最小补仓金额 {min_stake} 会导致总金额超过 400，不补仓")
                            dca_amount = 0
                        else:
                            dca_amount = min_stake

                    if dca_amount > max_stake:
                        dca_amount = max_stake

                    # 如果计算出有效的补仓金额，则执行补仓
                    if dca_amount > 0:
                        logger.info(
                            f"{pair} 触发补仓: 亏损 {loss_percentage:.2%}, "
                            f"当前仓位 {trade.stake_amount}, 补仓金额 {dca_amount}, "
                            f"触发原因: {dca_tag}"
                        )
                        # 记录本次补仓信息
                        last_dca_time = current_time.timestamp()
                        trade.set_custom_data('last_dca_time', last_dca_time)

                        # 在补仓之前做好准备更新initial_stake
                        trade.set_custom_data('last_stake_amount', trade.stake_amount)
                        trade.set_custom_data('pending_dca_amount', dca_amount)

                        return dca_amount, dca_tag

        # 检查并更新补仓后的initial_stake
        last_stake_amount = trade.get_custom_data('last_stake_amount')
        pending_dca_amount = trade.get_custom_data('pending_dca_amount')

        if (
            last_stake_amount != 0
            and pending_dca_amount != 0
            and last_stake_amount is not None
            and pending_dca_amount is not None
        ):
            # 如果确认补仓已经执行（stake_amount已增加）
            if trade.stake_amount > last_stake_amount:
                # 更新initial_stake为当前的总stake金额
                trade.set_custom_data('initial_stake', trade.stake_amount)
                # 清除临时变量
                trade.set_custom_data('last_stake_amount', 0)
                trade.set_custom_data('pending_dca_amount', 0)
                logger.info(f"{pair}: 补仓后更新initial_stake为 {trade.stake_amount}")

                # 新增: 补仓成功后，重新计算止盈点位
                direction = 'short' if trade.is_short else 'long'
                self.recalculate_exit_points_after_dca(trade, direction)

                # 重置退出阶段，从第一阶段开始计算
                trade.set_custom_data('exit_stage', 0)

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

                    # 确保有足够的退出点位
                    if not exit_points or len(exit_points) < 1:
                        break

                    # 对退出点位进行排序：多头从小到大，空头从大到小
                    if direction == 'long':
                        sorted_exit_points = sorted(exit_points)
                    else:  # short
                        sorted_exit_points = sorted(exit_points, reverse=True)

                    # 获取成本价
                    cost_price = trade.open_rate

                    # 使用持久化存储获取退出阶段
                    exit_stage = trade.get_custom_data('exit_stage', default=0)

                    # 初次遇到交易时，保存初始stake金额
                    if exit_stage == 0 and trade.get_custom_data('initial_stake') is None:
                        trade.set_custom_data('initial_stake', trade.stake_amount)

                    initial_stake = trade.get_custom_data(
                        'initial_stake', default=trade.stake_amount
                    )

                    # 根据退出点位数量确定退出逻辑
                    exit_points_count = len(sorted_exit_points)

                    # 只有1个点位的情况 - 全部退出
                    if exit_points_count == 1:
                        # 检查是否达到退出点位
                        if (direction == 'long' and current_rate >= sorted_exit_points[0]) or (
                            direction == 'short' and current_rate <= sorted_exit_points[0]
                        ):
                            logger.info(f"触发唯一退出点位 {pair}: 当前价格 {current_rate} - 全部退出")
                            self.enable_auto_calculation(pair, direction)
                            return (
                                -trade.stake_amount,
                                f"{direction}_single_tp_{sorted_exit_points[0]}",
                            )

                    # 2个点位的情况 - 每次退出50%
                    elif exit_points_count == 2:
                        if exit_stage == 0:
                            # 第一个点位：出50%
                            if (direction == 'long' and current_rate >= sorted_exit_points[0]) or (
                                direction == 'short' and current_rate <= sorted_exit_points[0]
                            ):
                                trade.set_custom_data('exit_stage', 1)
                                self._adjust_stoploss(trade, cost_price)

                                # 退出50%仓位
                                logger.info(f"触发第一级退出点位 {pair}: 当前价格 {current_rate} - 出售50%仓位")
                                return (
                                    -(initial_stake * 0.5),
                                    f"{direction}_tp1_of2_{sorted_exit_points[0]}",
                                )

                        elif exit_stage == 1:
                            # 第二个点位：出剩余50%
                            if (direction == 'long' and current_rate >= sorted_exit_points[1]) or (
                                direction == 'short' and current_rate <= sorted_exit_points[1]
                            ):
                                trade.set_custom_data('exit_stage', 2)
                                logger.info(f"触发第二级退出点位 {pair}: 当前价格 {current_rate} - 出售剩余全部仓位")
                                self.enable_auto_calculation(pair, direction)
                                return (
                                    -trade.stake_amount,
                                    f"{direction}_tp2_of2_{sorted_exit_points[1]}",
                                )

                            # 处理回撤情况
                            elif (direction == 'long' and current_rate <= cost_price) or (
                                direction == 'short' and current_rate >= cost_price
                            ):
                                # 从第一阶段回撤到成本价，清仓
                                logger.info(f"{pair} 从第一点位回撤至成本价 {cost_price}，清仓")
                                self.enable_auto_calculation(pair, direction)
                                return -trade.stake_amount, f'{direction}_tp1_pullback_cost'

                    # 3个或更多点位的情况 - 原有的30%/50%/全部逻辑
                    else:
                        # 多头策略
                        if direction == 'long':
                            # 第一个点位：出30%，止损调整到成本价
                            if exit_stage == 0 and current_rate >= sorted_exit_points[0]:
                                trade.set_custom_data('exit_stage', 1)
                                self._adjust_stoploss(trade, cost_price)
                                logger.info(f"触发第一级退出点位 {pair}: 当前价格 {current_rate} - 出售30%仓位")
                                return -(initial_stake * 0.3), f"long_tp1_{sorted_exit_points[0]}"

                            # 第二个点位：出50%剩余仓位，止损调整到第一个点位
                            elif exit_stage == 1 and current_rate >= sorted_exit_points[1]:
                                trade.set_custom_data('exit_stage', 2)
                                self._adjust_stoploss(trade, sorted_exit_points[0])
                                remaining_stake = trade.stake_amount
                                logger.info(f"触发第二级退出点位 {pair}: 当前价格 {current_rate} - 出售剩余仓位的50%")
                                return -(remaining_stake * 0.5), f"long_tp2_{sorted_exit_points[1]}"

                            elif exit_stage == 2 and current_rate >= sorted_exit_points[2]:
                                logger.info(f"触发第三级退出点位 {pair}: 当前价格 {current_rate} - 出售剩余全部仓位")
                                self.enable_auto_calculation(pair, direction)
                                return -trade.stake_amount, f"long_tp3_{sorted_exit_points[2]}"

                            # 处理回撤情况 - 多头
                            elif exit_stage == 1 and current_rate <= cost_price:
                                # 第一阶段回撤到成本价，清仓
                                logger.info(f"{pair} 从第一点位回撤至成本价 {cost_price}，清仓")
                                self.enable_auto_calculation(pair, direction)
                                return -trade.stake_amount, 'long_tp1_pullback_cost'

                            elif exit_stage == 2 and current_rate <= sorted_exit_points[0]:
                                # 第二阶段回撤到第一点位，清仓
                                logger.info(f"{pair} 从第二点位回撤至第一点位 {sorted_exit_points[0]}，清仓")
                                self.enable_auto_calculation(pair, direction)
                                return -trade.stake_amount, 'long_tp2_pullback_tp1'

                        # 空头策略
                        else:  # short
                            # 第一个点位：出30%，止损调整到成本价
                            if exit_stage == 0 and current_rate <= sorted_exit_points[0]:
                                trade.set_custom_data('exit_stage', 1)
                                self._adjust_stoploss(trade, cost_price)
                                logger.info(f"触发第一级退出点位 {pair}: 当前价格 {current_rate} - 出售30%仓位")
                                return -(initial_stake * 0.3), f"short_tp1_{sorted_exit_points[0]}"

                            # 第二个点位：出50%剩余仓位，止损调整到第一个点位
                            elif exit_stage == 1 and current_rate <= sorted_exit_points[1]:
                                trade.set_custom_data('exit_stage', 2)
                                self._adjust_stoploss(trade, sorted_exit_points[0])
                                remaining_stake = trade.stake_amount
                                logger.info(f"触发第二级退出点位 {pair}: 当前价格 {current_rate} - 出售剩余仓位的50%")
                                return (
                                    -(remaining_stake * 0.5),
                                    f"short_tp2_{sorted_exit_points[1]}",
                                )

                            elif exit_stage == 2 and current_rate <= sorted_exit_points[2]:
                                logger.info(f"触发第三级退出点位 {pair}: 当前价格 {current_rate} - 出售剩余全部仓位")
                                self.enable_auto_calculation(pair, direction)
                                return -trade.stake_amount, f"short_tp3_{sorted_exit_points[2]}"

                            # 处理回撤情况 - 空头
                            elif exit_stage == 1 and current_rate >= cost_price:
                                # 第一阶段回撤到成本价，清仓
                                logger.info(f"{pair} 从第一点位回撤至成本价 {cost_price}，清仓")
                                self.enable_auto_calculation(pair, direction)
                                return -trade.stake_amount, 'short_tp1_pullback_cost'

                            elif exit_stage == 2 and current_rate >= sorted_exit_points[0]:
                                # 第二阶段回撤到第一点位，清仓
                                logger.info(f"{pair} 从第二点位回撤至第一点位 {sorted_exit_points[0]}，清仓")
                                self.enable_auto_calculation(pair, direction)
                                return -trade.stake_amount, 'short_tp2_pullback_tp1'

        return None

    def _adjust_stoploss(self, trade: Trade, new_stoploss_price: float):
        """调整止损价格的辅助函数"""
        if self.config['runmode'].value in ('live', 'dry_run'):
            # 计算相对止损比例
            if trade.is_short:
                # 对于空头，止损价格高于入场价时为负值
                stoploss_percent = (new_stoploss_price / trade.open_rate) - 1
            else:
                # 对于多头，止损价格低于入场价时为负值
                stoploss_percent = (new_stoploss_price / trade.open_rate) - 1

            # 使用Trade的API更新止损
            trade.adjust_stop_loss(trade.open_rate, stoploss_percent)
            logger.info(f"已调整 {trade.pair} 的止损到 {new_stoploss_price} (相对: {stoploss_percent:.2%})")

    def custom_exit(
        self,
        pair: str,
        trade: 'Trade',
        current_time: 'datetime',
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        """
        基于持久化存储处理最终退出点位
        """
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

                    # 确保有足够的退出点位
                    if not exit_points:
                        break

                    # 对退出点位进行排序：多头从小到大，空头从大到小
                    if direction == 'long':
                        sorted_exit_points = sorted(exit_points)
                    else:  # short
                        sorted_exit_points = sorted(exit_points, reverse=True)

                    # 获取当前退出阶段
                    exit_stage = trade.get_custom_data('exit_stage', default=0)

                    # 根据退出点位数量确定退出逻辑
                    exit_points_count = len(sorted_exit_points)

                    # 只处理3个或更多点位的第三级退出
                    if exit_points_count >= 3 and exit_stage == 2:
                        if (direction == 'long' and current_rate >= sorted_exit_points[2]) or (
                            direction == 'short' and current_rate <= sorted_exit_points[2]
                        ):
                            # 第三个点位：全部退出
                            trade.set_custom_data('exit_stage', 3)
                            logger.info(f"触发第三级退出点位 {pair}: 当前价格 {current_rate} - 出售所有剩余仓位")
                            self.enable_auto_calculation(pair, direction)
                            return f"{direction}_tp3_{sorted_exit_points[2]}"
        else:
            # 如果不满足固定点位监控的条件，使用原有退出逻辑
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
