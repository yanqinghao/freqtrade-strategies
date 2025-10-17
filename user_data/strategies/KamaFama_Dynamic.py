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

    minimal_roi = {'0': 0.5, '372': 0.3, '861': 0.2}
    cc_long = {}
    cc_short = {}

    # 策略模式状态跟踪
    pair_strategy_mode = {}

    price_range_thresholds = {}

    # 固定点位监控
    coin_monitoring = {}
    manual_open = {}

    # For price monitoring notifications
    monitoring_notification_sent = {}

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
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.10
    trailing_only_offset_is_reached = True
    max_entry_position_adjustment = 5

    use_custom_stoploss = False

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
        self.should_not_init = os.environ.get('SHOULD_NOT_INIT') == 'true'
        os.environ['SHOULD_NOT_INIT'] = 'true'

        # 尝试从外部JSON文件加载策略模式配置
        self.load_strategy_mode_config()

        # --- 新增：价格监控节流配置与状态 ---
        # 通过环境变量覆盖，单位：分钟；例如导出 MONITOR_CHECK_MINUTES=60 表示每小时检查一次
        self.monitor_check_minutes = int(os.environ.get('MONITOR_CHECK_MINUTES', '60'))
        # interval<=0 表示不节流（每根5m都检查）
        self._last_monitor_check = {}  # {pair: datetime or pandas.Timestamp}

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

            if 'manual_open' in state_data:
                self.manual_open = state_data['manual_open']

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
                            valid_configs.append(
                                {
                                    **config,
                                    'auto_initialized': True if self.should_not_init else False,
                                }
                            )
                            has_matching_config = True

                        # 多头配置总是被保留
                        if direction == 'long' and current_mode == 'long':
                            valid_configs.append(
                                {
                                    **config,
                                    'auto_initialized': True if self.should_not_init else False,
                                }
                            )
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
                                'auto_initialized': True if self.should_not_init else False,
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

    # def custom_stoploss(
    #     self,
    #     pair: str,
    #     trade: Trade,
    #     current_time: datetime,
    #     current_rate: float,
    #     current_profit: float,
    #     **kwargs,
    # ) -> float:
    #     # 常规止损逻辑
    #     if current_profit >= 0.05:
    #         return -0.002

    #     # 这里不要加其他处理逻辑，因为这个函数会被频繁调用
    #     # 止损后的处理逻辑应该放在 exit_positions 里处理
    #     return None

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

    def _managed_open_count(self) -> int:
        """只统计由策略托管的 open trades（不含手动/固定点位/外部导入）"""
        manual_prefixes = 'manual_'
        count = 0
        for t in Trade.get_trades_proxy(is_open=True):
            tag = (t.enter_tag or '').lower()
            if not tag.startswith(manual_prefixes):
                count += 1
        return count

    def _normalize_pair(self, p: str) -> str:
        # 兼容 futures 形态：BTC/USDT:USDT -> BTC/USDT
        return p.split(':')[0].upper()

    def _has_exchange_position_for_pair(self, pair: str) -> bool:
        """
        仅在 futures 模式有意义；现货返回 False。
        """
        try:
            # 获取交易所名称
            exchange_name = self.config['exchange']['name'].lower()

            # 创建交易所实例
            exchange_class = getattr(ccxt, exchange_name)

            # 获取API凭证
            api_config = {
                'apiKey': self.config['original_config']['exchange'].get('key', ''),
                'secret': self.config['original_config']['exchange'].get('secret', ''),
                'enableRateLimit': True,
                'options': {
                    # 对 Binance：
                    # 'spot' | 'margin' | 'future' | 'delivery' | 'option'
                    'defaultType': 'future',  # USDT-M 永续/期货
                    'defaultMarket': 'future',
                },
            }

            # 过滤空值
            api_config = {k: v for k, v in api_config.items() if v}

            # 实例化交易所
            exchange = exchange_class(api_config)
            # Freqtrade Exchange 层：从交易所取仓位（期货）
            positions = exchange.fetch_positions([pair]) or []
        except Exception as e:
            logger.warning(f"[confirm_trade_entry] fetch_positions error: {e}")
            return False

        # 解析 size/positionAmt/contracts 等常见字段，>0 视为有仓
        for pos in positions:
            size = None
            # 直接字段
            for k in ('contracts', 'positionAmt', 'size', 'qty'):
                if k in pos and pos[k] is not None:
                    try:
                        size = float(pos[k])
                        break
                    except Exception:
                        pass
            # 有些交易所只在 info 里
            if size is None:
                info = pos.get('info') or {}
                for k in ('positionAmt', 'size', 'qty'):
                    if k in info and info[k] is not None:
                        try:
                            size = float(info[k])
                            break
                        except Exception:
                            pass
            if size is not None and abs(size) > 0:
                return True
        return False

    def _has_managed_open_trade_for_pair(self, pair: str) -> bool:
        manual_prefix = 'manual_'
        for t in Trade.get_trades_proxy(pair=pair, is_open=True):
            tag = (t.enter_tag or '').lower()
            if not tag.startswith(manual_prefix):
                return True
        return False

    def _not_managed_open_count(self) -> int:
        open_trades = Trade.get_trades_proxy(is_open=True)
        keep_pairs = {
            t.pair for t in open_trades if 'fixed' in (getattr(t, 'enter_tag', '') or '')
        }
        return len(keep_pairs)

    def _buy_short_open_count(self) -> int:
        open_trades = Trade.get_trades_proxy(is_open=True)
        keep_pairs = {
            t.pair
            for t in open_trades
            if 'buy' in (getattr(t, 'enter_tag', '') or '')
            or 'short' in (getattr(t, 'enter_tag', '') or '')
        }
        return len(keep_pairs)

    def confirm_trade_entry(
        self, pair, order_type, amount, rate, time_in_force, current_time, entry_tag, **kwargs
    ) -> bool:
        if self._buy_short_open_count() == 0 and ('buy' in entry_tag or 'short' in entry_tag):
            return True
        
        if self._not_managed_open_count() >= 2 and 'fixed_' in entry_tag:
            if 'fixed_long_entry_' in entry_tag or 'fixed_short_entry_' in entry_tag:
                for i in self.coin_monitoring[pair]:
                    i['auto'] = True
                    i['auto_initialized'] = False
                    self.reload_coin_monitoring(pair)
                    self.update_strategy_state_file()
            return False
        
        if self._has_exchange_position_for_pair(pair) and not self._has_managed_open_trade_for_pair(
            pair
        ):
            if hasattr(self, 'dp') and hasattr(self.dp, 'send_msg'):
                self.dp.send_msg(
                    f"[confirm_trade_entry] Skip {pair}: exchange has unmanaged position."
                )
            logger.warning(f"[confirm_trade_entry] Skip {pair}: exchange has unmanaged position.")
            return False

        if 'manual_' in entry_tag:
            logger.warning(f"[confirm_trade_entry] 强制下单触发: {pair}")
            return True

        if self._managed_open_count() >= 3:
            return False

        return True

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

            # logger.info(f"使用CCXT获取 {pair} {timeframe} 数据")

            # 加载市场
            exchange.load_markets()

            # 获取OHLCV数据 - 不使用since参数，让交易所返回最近的数据
            ohlcv = exchange.fetch_ohlcv(symbol=pair, timeframe=timeframe, limit=limit)

            if ohlcv and len(ohlcv) > 0:
                # 转换为DataFrame
                df = pd.DataFrame(ohlcv, columns=['date', 'open', 'high', 'low', 'close', 'volume'])

                # 转换时间戳为日期时间
                df['date'] = pd.to_datetime(df['date'], unit='ms')

                # logger.info(f"成功获取 {pair} {timeframe} 数据，共 {len(df)} 条")
                return df
            else:
                logger.error(f"未获取到 {pair} 的 {timeframe} 数据")
                return None

        except Exception as e:
            logger.error(f"获取历史数据时出错: {str(e)}")
            return None

    def _should_run_monitoring(self, pair: str, now_ts) -> bool:
        """
        只有当距离上次检查 >= monitor_check_minutes 才返回 True。
        now_ts: 回测用 dataframe 的最后一根蜡烛时间（pd.Timestamp），
                实盘/模拟用 datetime.now()。
        """
        try:
            interval_min = getattr(self, 'monitor_check_minutes', 30)
            if interval_min <= 0:
                return True  # 关闭节流

            last = self._last_monitor_check.get(pair)
            if last is None:
                self._last_monitor_check[pair] = now_ts
                return True

            # 统一成 pandas.Timestamp 来做差，兼容 datetime / Timestamp
            now_ts_pd = pd.Timestamp(now_ts)
            last_ts_pd = pd.Timestamp(last)
            if now_ts_pd - last_ts_pd >= pd.Timedelta(minutes=interval_min):
                self._last_monitor_check[pair] = now_ts
                return True

            return False
        except Exception:
            # 出错时不要卡住主流程，保守返回 True
            self._last_monitor_check[pair] = now_ts
            return True

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

    def _check_exit_points_deviation(self, pair: str, config: dict) -> bool:
        """
        检查退出点位偏差是否大于2%
        """
        try:
            exit_points = config.get('exit_points', [])
            if not exit_points:
                return False

            # 获取当前价格
            df_5m, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
            if df_5m is None or df_5m.empty:
                return False

            current_price = df_5m.iloc[-1]['close']
            first_exit_point = exit_points[0]
            direction = config.get('direction', 'long')

            # 根据方向计算偏差
            if direction == 'long':
                # 多头：当前价格应该低于退出点位
                if current_price >= first_exit_point:
                    return False  # 价格已经超过退出点位，不需要检查偏差
            else:  # short
                # 空头：当前价格应该高于退出点位
                if current_price <= first_exit_point:
                    return False  # 价格已经低于退出点位，不需要检查偏差

            deviation_pct = abs(current_price - first_exit_point) / first_exit_point * 100

            if deviation_pct > 2.0:
                logger.info(
                    f"{pair} 偏差检查 - 当前价格 {current_price} 与退出点位 {first_exit_point} 偏差 {deviation_pct:.2f}% > 2%"
                )
                return True

            return False

        except Exception as e:
            logger.error(f"检查 {pair} 退出点位偏差时出错: {e}")
            return False

    def reload_coin_monitoring(self, pair: str, check_deviation: bool = False):
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
                should_recalculate = False

                # 检查是否需要重新计算
                if config.get('auto', False):
                    # 如果未初始化，需要重新计算
                    if not config.get('auto_initialized', False):
                        should_recalculate = True
                    # 如果启用偏差检查，检查偏差是否大于2%
                    elif check_deviation:
                        should_recalculate = self._check_exit_points_deviation(pair, config)

                if should_recalculate:
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

    def _prune_manual_open_orphans(self):
        """
        清理 manual_open 中没有对应“手动未平仓交易”的残留配置。
        逻辑：取所有 is_open 的 trade 中 enter_tag 含 'manual' 的 pair，作为保留集合；
            manual_open 里不在该集合的直接删除。
        """
        try:
            # 回测不清理，避免影响回测复现
            if self.config.get('runmode', None) not in ('live', 'dry_run'):
                return
            # 没有手动配置就不用做了
            if not getattr(self, 'manual_open', None):
                return

            # 收集所有“手动且未平仓”的交易对
            open_trades = Trade.get_trades_proxy(is_open=True)
            keep_pairs = {
                t.pair
                for t in open_trades
                if (getattr(t, 'enter_tag', '') or '').find('manual') != -1
            }

            # 删掉 manual_open 里那些没有手动未平仓单的交易对
            to_delete = [p for p in list(self.manual_open.keys()) if p not in keep_pairs]
            if not to_delete:
                return

            for p in to_delete:
                # 如需顺便恢复自动点位计算，可用下面两行（若 manual_open 里带方向）：
                # direction = (self.manual_open.get(p, {}).get('direction') or 'long').lower()
                # self.enable_auto_calculation(p, 'short' if direction == 'short' else 'long')
                del self.manual_open[p]

            self.update_strategy_state_file()

            msg = f"🧹 清理失效 manual 配置: {', '.join(to_delete)}"
            logger.info(msg)
            if hasattr(self, 'dp') and hasattr(self.dp, 'send_msg'):
                self.dp.send_msg(msg)

        except Exception as e:
            logger.warning(f"_prune_manual_open_orphans failed: {e}")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self._prune_manual_open_orphans()

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

        # New price monitoring logic
        # New price monitoring logic（增加节流）
        if self.config.get('runmode', None) in ('live', 'dry_run'):
            now_ts = datetime.now()
        else:
            # 回测用当前蜡烛时间，确保回测可重复
            now_ts = dataframe.iloc[-1]['date'] if len(dataframe) else None

        if now_ts is not None and self._should_run_monitoring(pair, now_ts):
            self.check_price_monitoring(dataframe, pair)

        return dataframe

    def check_price_monitoring(self, dataframe: DataFrame, pair: str):
        """
        三路并行通知（一次性拼接发送）：
        1) 形态反转：1h 形态(Setup) + 4h 背景 + 确认(ATR)   <-- 不再要求靠近监控位
        2) 监控位接近/穿越：5m 价格 vs entry_points（ATR 距离）
        3) 强趋势直通车：4h 趋势强时，直接“越过监控位 + ATR(1h) 余量”确认
        """
        # ===== 参数 =====
        USE_LAST_CLOSED_CANDLE = True

        # —— 形态线参数（1h）
        COOLDOWN_BARS_REVERSAL_1H = 6  # 形态确认冷静期（单位：1h K根）
        SETUP_EXPIRE_BARS_1H = 2  # Setup 有效期（单位：1h K根）
        CONFIRM_ATR_MULT = 0.25  # 确认阈值：收盘超出 anchor ± ATR*mult
        MIN_BODY_ATR_MULT = 0.15  # 最小实体：0.15 * ATR(1h)

        # —— 趋势直通车（4h）
        TREND_BYPASS = True
        ADX_TREND = 22  # 4h ADX 阈值
        BYPASS_LEVEL_ATR_MULT = 0.25  # 越级确认所需 ATR(1h) 余量倍数
        COOLDOWN_BARS_TREND_4H = 2  # 趋势直通车确认冷静期（单位：4h K根）

        # —— 4h 斜率归一化（百分比）
        # SLOPE_MODE = 'pct'  # 固定使用百分比斜率
        SLOPE_LOOKBACK_4H = 2  # t vs t-2（≈8小时）
        SLOPE_PCT_MIN = 0.004  # 0.4%：多头>+0.004；空头<-0.004

        # —— 监控位（5m）
        APPROACH_ATR_MULT = 0.35  # 接近判定：距离≤ ATR * 0.35（5m）

        if self.config.get('runmode', None) not in ('live', 'dry_run'):
            return
        if pair not in self.coin_monitoring and not getattr(self, 'manual_open', None):
            return

        # ========= 状态 =========
        if not hasattr(self, 'monitoring_notification_sent'):
            self.monitoring_notification_sent = (
                {}
            )  # {pair: {direction: {price_point: {'approaching','crossed'}}}}
        if not hasattr(self, 'reversal_setups'):
            self.reversal_setups = {}  # {pair: {'long': {...}, 'short': {...}}}
        if not hasattr(self, 'reversal_setup_notified'):
            self.reversal_setup_notified = {}  # {pair: {'long': ts, 'short': ts}}
        if not hasattr(self, 'reversal_confirm_sent'):
            self.reversal_confirm_sent = {}  # {pair: {'long': ts, 'short': ts}}
        if not hasattr(self, 'trend_notification_sent'):
            self.trend_notification_sent = {}  # {pair: {'long': ts, 'short': ts}}
        # —— 新增：manual_open 状态
        if not hasattr(self, 'manual_entry_notification_sent'):
            # {pair: {'approaching': bool, 'crossed': bool}}
            self.manual_entry_notification_sent = {}

        # ========= 工具 =========
        def _cooldown_ok(last_ts, current_ts, bars, tf_minutes=60):
            if last_ts is None:
                return True
            try:
                return (current_ts - last_ts).total_seconds() >= bars * tf_minutes * 60
            except Exception:
                return True

        def _emit(msg: str, bag: list[str]):
            if msg and isinstance(msg, str):
                bag.append(msg)

        # ========= 监控位集合 =========
        monitoring_configs = self.coin_monitoring.get(pair, [])
        levels_long, levels_short = [], []
        for cfg in monitoring_configs:
            pts = cfg.get('entry_points', []) or []
            if cfg.get('direction', 'long') == 'long':
                levels_long.extend(pts)
            else:
                levels_short.extend(pts)

        batch_msgs: list[str] = []

        # ========= ② 监控位接近/穿越（5m ATR）=========
        active_trades = Trade.get_trades_proxy(is_open=True, pair=pair)
        current_price = float(dataframe['close'].iloc[-1])

        try:
            atr_5m_series = ta.ATR(dataframe, timeperiod=14)
            atr_5m = (
                float(atr_5m_series.iloc[-1]) if len(atr_5m_series) == len(dataframe) else np.nan
            )
        except Exception:
            atr_5m = np.nan

        if not active_trades:
            for cfg in monitoring_configs:
                direction = cfg.get('direction', 'long')
                eps = cfg.get('entry_points', []) or []
                for price_point in eps:
                    state = (
                        self.monitoring_notification_sent.setdefault(pair, {})
                        .setdefault(direction, {})
                        .setdefault(price_point, {'approaching': False, 'crossed': False})
                    )

                    if direction == 'long':
                        is_approaching = (
                            (current_price > price_point)
                            and (not np.isnan(atr_5m))
                            and ((current_price - price_point) <= APPROACH_ATR_MULT * atr_5m)
                        )
                        has_crossed = current_price < price_point
                        is_away = (not np.isnan(atr_5m)) and (
                            current_price > price_point + APPROACH_ATR_MULT * atr_5m
                        )

                        if is_approaching and not state['approaching']:
                            _emit(
                                f"🔔 LONG approaching {pair} | Price: {current_price:.6f} | Point: {price_point:.6f}",
                                batch_msgs,
                            )
                            state['approaching'] = True
                            state['crossed'] = False

                        if has_crossed and not state['crossed']:
                            _emit(
                                f"✅ LONG crossed {pair} | Price: {current_price:.6f} | Point: {price_point:.6f}",
                                batch_msgs,
                            )
                            state['crossed'] = True
                            state['approaching'] = True

                        if is_away and (state['approaching'] or state['crossed']):
                            state['approaching'] = False
                            state['crossed'] = False

                    else:  # short
                        is_approaching = (
                            (current_price < price_point)
                            and (not np.isnan(atr_5m))
                            and ((price_point - current_price) <= APPROACH_ATR_MULT * atr_5m)
                        )
                        has_crossed = current_price > price_point
                        is_away = (not np.isnan(atr_5m)) and (
                            current_price < price_point - APPROACH_ATR_MULT * atr_5m
                        )

                        if is_approaching and not state['approaching']:
                            _emit(
                                f"🔔 SHORT approaching {pair} | Price: {current_price:.6f} | Point: {price_point:.6f}",
                                batch_msgs,
                            )
                            state['approaching'] = True
                            state['crossed'] = False

                        if has_crossed and not state['crossed']:
                            _emit(
                                f"✅ SHORT crossed {pair} | Price: {current_price:.6f} | Point: {price_point:.6f}",
                                batch_msgs,
                            )
                            state['crossed'] = True
                            state['approaching'] = True

                        if is_away and (state['approaching'] or state['crossed']):
                            state['approaching'] = False
                            state['crossed'] = False

        # ========= （简化）④ manual_open：仅比较 entry_price（用于未开单前的提醒） =========
        if not active_trades:
            mo = (getattr(self, 'manual_open', {}) or {}).get(pair)
            if mo and (mo.get('entry_price') is not None):
                direction = str(mo.get('direction', 'long')).lower()
                try:
                    entry_price = float(mo.get('entry_price'))
                except Exception:
                    entry_price = float('nan')

                if not np.isnan(entry_price):
                    st = self.manual_entry_notification_sent.setdefault(
                        pair, {'approaching': False, 'crossed': False}
                    )

                    if direction == 'long':
                        # 习惯：做多通常在上方接近买入位时提醒；触达=价格 <= entry
                        is_approaching = (
                            (current_price > entry_price)
                            and (not np.isnan(atr_5m))
                            and ((current_price - entry_price) <= APPROACH_ATR_MULT * atr_5m)
                        )
                        has_crossed = current_price <= entry_price
                        is_away = (not np.isnan(atr_5m)) and (
                            current_price > entry_price + APPROACH_ATR_MULT * atr_5m
                        )
                    else:  # short
                        # 做空通常在下方接近入场位；触达=价格 >= entry
                        is_approaching = (
                            (current_price < entry_price)
                            and (not np.isnan(atr_5m))
                            and ((entry_price - current_price) <= APPROACH_ATR_MULT * atr_5m)
                        )
                        has_crossed = current_price >= entry_price
                        is_away = (not np.isnan(atr_5m)) and (
                            current_price < entry_price - APPROACH_ATR_MULT * atr_5m
                        )

                    # —— 接近
                    if is_approaching and not st['approaching']:
                        _emit(
                            f"🔔 MANUAL ENTRY approaching {pair} | Dir:{direction.upper()} | "
                            f"Price: {current_price:.6f} | Entry: {entry_price:.6f}",
                            batch_msgs,
                        )
                        st['approaching'] = True
                        st['crossed'] = False

                    # —— 触达 / 穿越
                    if has_crossed and not st['crossed']:
                        _emit(
                            f"✅ MANUAL ENTRY hit {pair} | Dir:{direction.upper()} | "
                            f"Price: {current_price:.6f} | Entry: {entry_price:.6f}",
                            batch_msgs,
                        )
                        st['crossed'] = True
                        st['approaching'] = True  # 命中后视作已接近

                    # —— 离开接近区（且未命中）→ 复位
                    if is_away and (st['approaching'] and not st['crossed']):
                        st['approaching'] = False

        # ========= ① 形态反转：1h + 4h 背景 + 确认 =========
        df1h = self.get_ohlcv_history(pair, timeframe='1h', limit=300)
        df4h = self.get_ohlcv_history(pair, timeframe='4h', limit=300)
        if df1h is None or len(df1h) < 5:
            if batch_msgs and hasattr(self, 'dp') and hasattr(self.dp, 'send_msg'):
                self.dp.send_msg('\n'.join(batch_msgs))
            return

        # 1h ATR
        try:
            atr_1h_series = ta.ATR(df1h, timeperiod=14)
        except Exception:
            atr_1h_series = pd.Series(index=df1h.index, dtype=float)

        idx1h = -2 if USE_LAST_CLOSED_CANDLE else -1
        r1h = df1h.iloc[idx1h]
        atr1h = float(atr_1h_series.iloc[idx1h]) if len(atr_1h_series) >= len(df1h) else np.nan
        ts_1h = r1h['date'] if 'date' in r1h else df1h.index[idx1h]

        # 4h 背景过滤（EMA20/EMA50）
        bull_4h_ok, bear_4h_ok = True, True
        if df4h is not None and len(df4h) >= 60:
            idx4h = -2 if USE_LAST_CLOSED_CANDLE else -1
            ema20_4h = df4h['close'].ewm(span=20, adjust=False).mean()
            ema50_4h = df4h['close'].ewm(span=50, adjust=False).mean()
            r4h = df4h.iloc[idx4h]
            e20_4h = float(ema20_4h.iloc[idx4h])
            e50_4h = float(ema50_4h.iloc[idx4h])
            bull_4h_ok = (r4h['close'] >= e20_4h) or (e20_4h >= e50_4h)
            bear_4h_ok = (r4h['close'] <= e20_4h) or (e20_4h <= e50_4h)
        else:
            idx4h = -1
            ema20_4h = ema50_4h = None

        # —— 形态识别（1h）
        try:
            eng = int(ta.CDLENGULFING(df1h).iloc[idx1h])  # +100/-100
            hammer = int(ta.CDLHAMMER(df1h).iloc[idx1h])  # +100
            invham = int(ta.CDLINVERTEDHAMMER(df1h).iloc[idx1h])  # +100
            shooting = int(ta.CDLSHOOTINGSTAR(df1h).iloc[idx1h])  # -100
            try:
                morning_series = ta.CDLMORNINGSTAR(df1h, penetration=0.3)
            except TypeError:
                morning_series = ta.CDLMORNINGSTAR(df1h)
            morning = int(morning_series.iloc[idx1h])  # +100
            try:
                evening_series = ta.CDLEVENINGSTAR(df1h, penetration=0.3)
            except TypeError:
                evening_series = ta.CDLEVENINGSTAR(df1h)
            evening = int(evening_series.iloc[idx1h])  # -100
        except Exception:
            eng = hammer = invham = shooting = morning = evening = 0

        body = abs(float(r1h['close']) - float(r1h['open']))
        quality_ok = (not np.isnan(atr1h)) and (body >= MIN_BODY_ATR_MULT * atr1h)

        bull_1h = quality_ok and ((eng > 0) or (hammer > 0) or (invham > 0) or (morning > 0))
        bear_1h = quality_ok and ((eng < 0) or (shooting < 0) or (evening < 0))

        label_parts = []
        if eng > 0:
            label_parts.append('BullEngulf')
        if hammer > 0:
            label_parts.append('Hammer')
        if invham > 0:
            label_parts.append('InvHammer')
        if morning > 0:
            label_parts.append('MorningStar')
        if eng < 0:
            label_parts.append('BearEngulf')
        if shooting < 0:
            label_parts.append('ShootingStar')
        if evening < 0:
            label_parts.append('EveningStar')
        label_1h = '+'.join(label_parts) if label_parts else 'None'

        # —— Setup 登记（1h）
        pair_setups = self.reversal_setups.setdefault(pair, {})
        if bull_1h and bull_4h_ok:
            pair_setups['long'] = {
                'ts': ts_1h,
                'anchor': float(r1h['high']),
                'atr': float(atr1h),
                'label': label_1h,
            }
        if bear_1h and bear_4h_ok:
            pair_setups['short'] = {
                'ts': ts_1h,
                'anchor': float(r1h['low']),
                'atr': float(atr1h),
                'label': label_1h,
            }

        def _rev_line(direction: str, setup: dict, confirmed: bool) -> str:
            icon = '📈' if direction == 'long' else '📉'
            label = setup.get('label', 'N/A')
            anchor = float(setup.get('anchor', 0.0))
            atr0 = float(setup.get('atr', 0.0)) if setup.get('atr', None) is not None else 0.0
            thr = CONFIRM_ATR_MULT * atr0
            base = (
                f"{icon} Reversal {direction.upper()} Setup {pair} | TF:1h | Pattern: {label} | "
                f"Anchor: {anchor:.6f} | ATR14(1h): {atr0:.6f} | 有效期: {SETUP_EXPIRE_BARS_1H}x1h"
            )
            if confirmed:
                cond = (
                    f"Close(1h) > Anchor+{thr:.6f}"
                    if direction == 'long'
                    else f"Close(1h) < Anchor-{thr:.6f}"
                )
                base += f" | ✅ 确认突破: {cond}"
            return base

        def _confirm(direction_key: str) -> bool:
            setup = pair_setups.get(direction_key)
            if not setup:
                return False

            # 找 setup 的 1h 索引
            setup_idx = None
            if 'date' in df1h.columns:
                for i in range(len(df1h)):
                    if df1h.iloc[i]['date'] == setup['ts']:
                        setup_idx = i
                        break
            if setup_idx is None:
                try:
                    setup_idx = df1h.index.get_loc(setup['ts'])
                except Exception:
                    return False

            end_idx = min(len(df1h) - 1, setup_idx + SETUP_EXPIRE_BARS_1H)
            if end_idx <= setup_idx:
                return False

            idx_chk = -2 if USE_LAST_CLOSED_CANDLE else -1
            real_chk = len(df1h) + idx_chk if idx_chk < 0 else idx_chk
            if real_chk <= setup_idx:
                return False

            close_chk = float(df1h.iloc[idx_chk]['close'])
            anchor = float(setup['anchor'])
            atr0 = setup.get('atr', np.nan)
            thr = CONFIRM_ATR_MULT * (atr0 if not np.isnan(atr0) else 0.0)

            return (
                close_chk > (anchor + thr)
                if direction_key == 'long'
                else close_chk < (anchor - thr)
            )

        def _maybe_emit_reversal(direction_key: str):
            setup = pair_setups.get(direction_key)
            if not setup:
                return
            notified_ts = self.reversal_setup_notified.setdefault(pair, {}).get(direction_key)
            last_confirm_ts = self.reversal_confirm_sent.setdefault(pair, {}).get(direction_key)
            confirmed_now = _confirm(direction_key)

            if notified_ts != setup['ts']:
                _emit(_rev_line(direction_key, setup, confirmed_now), batch_msgs)
                self.reversal_setup_notified.setdefault(pair, {})[direction_key] = setup['ts']
                if confirmed_now and _cooldown_ok(last_confirm_ts, ts_1h, 0, tf_minutes=60):
                    self.reversal_confirm_sent[pair][direction_key] = ts_1h
                    self.reversal_setups.get(pair, {}).pop(direction_key, None)
                return

            if confirmed_now and _cooldown_ok(
                last_confirm_ts, ts_1h, COOLDOWN_BARS_REVERSAL_1H, tf_minutes=60
            ):
                _emit(_rev_line(direction_key, setup, True), batch_msgs)
                self.reversal_confirm_sent[pair][direction_key] = ts_1h
                self.reversal_setups.get(pair, {}).pop(direction_key, None)
                return

            if not confirmed_now:
                setup_idx = None
                if 'date' in df1h.columns:
                    for i in range(len(df1h)):
                        if df1h.iloc[i]['date'] == setup['ts']:
                            setup_idx = i
                            break
                if setup_idx is None:
                    try:
                        setup_idx = df1h.index.get_loc(setup['ts'])
                    except Exception:
                        setup_idx = None
                if setup_idx is not None and (len(df1h) - 1 - setup_idx) > SETUP_EXPIRE_BARS_1H:
                    self.reversal_setups[pair].pop(direction_key, None)

        _maybe_emit_reversal('long')
        _maybe_emit_reversal('short')

        # ========= ③ 强趋势直通车（4h 趋势 + 1h ATR 余量；斜率已归一化）=========
        try:
            adx4h_series = (
                ta.ADX(df4h, timeperiod=14) if (df4h is not None and len(df4h) >= 20) else None
            )
            adx4h = float(adx4h_series.iloc[idx4h]) if adx4h_series is not None else np.nan
        except Exception:
            adx4h = np.nan

        # 4h EMA20 百分比斜率（无量纲）
        norm_slope_4h = 0.0
        try:
            if (
                df4h is not None
                and ema20_4h is not None
                and len(df4h) > abs(idx4h - SLOPE_LOOKBACK_4H)
            ):
                ema_now = float(ema20_4h.iloc[idx4h])
                ema_prev = float(ema20_4h.iloc[idx4h - SLOPE_LOOKBACK_4H])
                if ema_prev > 0:
                    norm_slope_4h = (ema_now - ema_prev) / ema_prev
        except Exception:
            norm_slope_4h = 0.0

        slope_up_ok = norm_slope_4h > SLOPE_PCT_MIN
        slope_dn_ok = norm_slope_4h < -SLOPE_PCT_MIN

        trend_long_ok = (
            bull_4h_ok and (not np.isnan(adx4h)) and (adx4h >= ADX_TREND) and slope_up_ok
        )
        trend_short_ok = (
            bear_4h_ok and (not np.isnan(adx4h)) and (adx4h >= ADX_TREND) and slope_dn_ok
        )

        def _trend_bypass(direction: str) -> bool:
            if not TREND_BYPASS or np.isnan(atr1h):
                return False
            idx4h_local = -2 if USE_LAST_CLOSED_CANDLE else -1
            ts4h = (
                df4h.iloc[idx4h_local]['date']
                if ('date' in df4h.columns)
                else df4h.index[idx4h_local]
            )
            close_1h_chk = float(df1h.iloc[idx1h]['close'])
            thr = BYPASS_LEVEL_ATR_MULT * atr1h

            last_ts = self.trend_notification_sent.setdefault(pair, {}).get(direction)

            if (
                direction == 'long'
                and trend_long_ok
                and _cooldown_ok(last_ts, ts4h, COOLDOWN_BARS_TREND_4H, tf_minutes=240)
            ):
                for lv in levels_long:
                    if close_1h_chk > (lv + thr):
                        self.trend_notification_sent.setdefault(pair, {})['long'] = ts4h
                        _emit(
                            f"🚀 Trend LONG Confirmed {pair} | TF:4h | Break: {lv:.6f} (+{thr:.6f}) | "
                            f"ADX(4h): {adx4h:.1f} | NormSlope(4h,pct,{SLOPE_LOOKBACK_4H}): {norm_slope_4h:.6f} | "
                            f"Close(1h): {close_1h_chk:.6f}",
                            batch_msgs,
                        )
                        logger.info(
                            f"[TREND] LONG {pair} | break {lv} thr {thr} | slope {norm_slope_4h:.6f}"
                        )
                        return True
                return False

            if (
                direction == 'short'
                and trend_short_ok
                and _cooldown_ok(last_ts, ts4h, COOLDOWN_BARS_TREND_4H, tf_minutes=240)
            ):
                for lv in levels_short:
                    if close_1h_chk < (lv - thr):
                        self.trend_notification_sent.setdefault(pair, {})['short'] = ts4h
                        _emit(
                            f"📉 Trend SHORT Confirmed {pair} | TF:4h | Break: {lv:.6f} (-{thr:.6f}) | "
                            f"ADX(4h): {adx4h:.1f} | NormSlope(4h,pct,{SLOPE_LOOKBACK_4H}): {norm_slope_4h:.6f} | "
                            f"Close(1h): {close_1h_chk:.6f}",
                            batch_msgs,
                        )
                        logger.info(
                            f"[TREND] SHORT {pair} | break {lv} thr {thr} | slope {norm_slope_4h:.6f}"
                        )
                        return True
                return False

            return False

        _ = _trend_bypass('long')
        _ = _trend_bypass('short')

        # ========= 统一发送 =========
        if batch_msgs and hasattr(self, 'dp') and hasattr(self.dp, 'send_msg'):
            self.dp.send_msg('\n'.join(batch_msgs))

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
                # logger.info(
                #     f"Skipping {pair}: Current price {current_price} is within {threshold_percent}% of opening price {open_rate}"
                # )
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
                    if 'manual' in trade.enter_tag:
                        continue
                    direction = 'short' if trade.is_short else 'long'
                    current_time = datetime.now(trade.open_date_utc.tzinfo)
                    # 如果刚开仓成功（5分钟内的交易），关闭自动计算
                    if (current_time - trade.open_date_utc).total_seconds() < 60 * 60:  # 5分钟内
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
            # 在进入交易前检查偏差并重新计算点位（仅对未持有交易的交易对）
            self.reload_coin_monitoring(pair, check_deviation=True)

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
        # Manual trade cleanup
        if trade.enter_tag and 'manual' in trade.enter_tag:
            # 'amount' 是卖出的标的数量；用 trade.amount 判定是否整仓
            try:
                full_close = (
                    trade.amount is not None
                    and amount is not None
                    and abs(float(trade.amount) - float(amount))
                    <= max(1e-12, float(trade.amount) * 1e-3)  # 0.1% 容差
                )
            except Exception:
                full_close = False

            if full_close and pair in self.manual_open:
                logger.info(
                    f"Manual trade for {pair} fully closing. Cleaning up manual monitoring."
                )
                del self.manual_open[pair]
                self.update_strategy_state_file()

        direction = 'short' if trade.is_short else 'long'

        # Check if this is a ROI exit
        if exit_reason.upper().startswith('ROI'):
            logger.info(f"{pair}: Exit triggered by ROI - re-enabling auto calculation")

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

    def recalculate_all_auto_monitoring_pairs(self):
        """
        重新计算所有自动点位监控的交易对
        仅对没有活跃交易的交易对进行重新计算，避免影响正在进行的交易
        """
        if not hasattr(self, 'coin_monitoring') or not self.coin_monitoring:
            return

        updated_pairs = []

        for pair in self.coin_monitoring:
            # 检查是否有活跃交易，如果有则跳过
            if self.config.get('runmode', None) in ('live', 'dry_run'):
                from freqtrade.persistence import Trade

                active_trades = Trade.get_trades_proxy(is_open=True, pair=pair)
                if active_trades:
                    logger.info(f"跳过 {pair}：存在活跃交易，不重新计算点位")
                    continue  # 有活跃交易，跳过这个交易对

            for config in self.coin_monitoring[pair]:
                if config.get('auto', False):
                    # 重置自动初始化状态，强制重新计算
                    config['auto_initialized'] = False
                    updated_pairs.append(pair)

        # 重新加载所有需要更新的交易对
        for pair in set(updated_pairs):  # 使用set去重
            self.reload_coin_monitoring(pair)

        if updated_pairs:
            logger.info(f"已重新计算无活跃交易的自动监控交易对: {set(updated_pairs)}")
            if hasattr(self, 'dp') and hasattr(self.dp, 'send_msg'):
                self.dp.send_msg(f"已重新计算无活跃交易的自动监控交易对: {set(updated_pairs)}")

            self.update_strategy_state_file()

    # 3. 更新持久化文件的函数
    def _datetime_to_timestamp(self, dt: datetime) -> float:
        """
        将datetime对象转换为时间戳，兼容不同版本的Python

        Args:
            dt: datetime对象

        Returns:
            float: 时间戳
        """
        try:
            if hasattr(dt, 'timestamp'):
                return dt.timestamp()
            else:
                # 兼容性处理：如果没有timestamp方法，手动转换
                import calendar

                return calendar.timegm(dt.timetuple())
        except Exception as e:
            logger.error(f"转换datetime到时间戳时出错: {e}, datetime: {dt}")
            # 返回当前时间戳作为fallback
            return datetime.now().timestamp()

    def _timestamp_to_datetime(self, timestamp: float, timezone=None) -> datetime:
        """
        将时间戳转换为datetime对象

        Args:
            timestamp: 时间戳
            timezone: 时区信息，默认为None

        Returns:
            datetime: datetime对象
        """
        try:
            return datetime.fromtimestamp(timestamp, timezone)
        except Exception as e:
            logger.error(f"转换时间戳到datetime时出错: {e}, timestamp: {timestamp}")
            # 返回当前时间作为fallback
            return datetime.now(timezone) if timezone else datetime.now()

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
                strategy_state['manual_open'] = self.manual_open

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

        # --- 未真正开单（首单未完全成交）则直接跳过 ---
        # 1) open_rate 为空 => 首单还没成交完成
        if trade.open_rate is None:
            return None

        # 2) 保险起见，再加几条稳妥的保护（可选）
        if not trade.is_open:
            return None
        if trade.amount is None or trade.amount <= 0:
            return None
        # 如果还有挂着的订单（首单/补仓/减仓）也不要再下新单（可选）
        if getattr(trade, 'open_order_id', None):
            return None

        pair = trade.pair
        direction = 'short' if trade.is_short else 'long'

        # 初次遇到交易时，保存初始stake金额
        if trade.get_custom_data(
            'initial_stake'
        ) is None or trade.stake_amount != trade.get_custom_data('initial_stake'):
            trade.set_custom_data('initial_stake', trade.stake_amount)

        if trade.enter_tag and 'manual' in trade.enter_tag:
            manual_cfg = self.manual_open.get(trade.pair, {})
            sl_price = manual_cfg.get('stop_loss')
            if sl_price:
                if trade.is_short:
                    # 空头：若价格 >= SL 价，直接全部买回平仓
                    if current_rate >= sl_price:
                        return -trade.stake_amount, 'manual_sl_hit'
                else:
                    # 多头：若价格 <= SL 价，直接全部卖出
                    if current_rate <= sl_price:
                        return -trade.stake_amount, 'manual_sl_hit'

        # 先检查是否需要补仓
        if current_profit < 0:

            # 优先检查是否为止盈回撤情况（亏损小于0.5%且已经触发过减仓）
            exit_stage = trade.get_custom_data('exit_stage', default=0)
            if exit_stage >= 1 and current_profit >= -0.005:  # 已减仓且亏损小于0.5%
                # ===== ① manual_open 优先：手动单的回撤清仓 =====
                if trade.enter_tag and 'manual' in trade.enter_tag:
                    manual_cfg = self.manual_open.get(pair)
                    if manual_cfg:
                        manual_exit_points = manual_cfg.get('exit_points', []) or []
                        if manual_exit_points:
                            # 排序：多头升序，空头降序
                            if direction == 'long':
                                m_sorted_exit_points = sorted(manual_exit_points)
                            else:
                                m_sorted_exit_points = sorted(manual_exit_points, reverse=True)

                            cost_price = trade.open_rate
                            m_count = len(m_sorted_exit_points)

                            should_exit = False
                            exit_tag = ''

                            # 两点位：从 TP1 回撤到成本清仓
                            if m_count == 2 and exit_stage == 1:
                                if (direction == 'long' and current_rate <= cost_price) or (
                                    direction == 'short' and current_rate >= cost_price
                                ):
                                    should_exit = True
                                    exit_tag = f'manual_{direction}_tp1_pullback_cost'

                            # 三点位及以上
                            elif m_count >= 3:
                                if direction == 'long':
                                    # TP1 后回撤到成本
                                    if exit_stage == 1 and current_rate <= cost_price:
                                        should_exit = True
                                        exit_tag = 'manual_long_tp1_pullback_cost'
                                    # TP2 后回撤到 TP1
                                    elif (
                                        exit_stage == 2 and current_rate <= m_sorted_exit_points[0]
                                    ):
                                        should_exit = True
                                        exit_tag = 'manual_long_tp2_pullback_tp1'
                                else:  # short
                                    # TP1 后回撤到成本
                                    if exit_stage == 1 and current_rate >= cost_price:
                                        should_exit = True
                                        exit_tag = 'manual_short_tp1_pullback_cost'
                                    # TP2 后回撤到 TP1
                                    elif (
                                        exit_stage == 2 and current_rate >= m_sorted_exit_points[0]
                                    ):
                                        should_exit = True
                                        exit_tag = 'manual_short_tp2_pullback_tp1'

                            # 触发“手动单回撤清仓”
                            if should_exit:
                                # ✨ 全仓退出前清理手动监控 & 恢复自动计算
                                if hasattr(self, '_manual_cleanup_after_full_close'):
                                    self._manual_cleanup_after_full_close(pair, direction, exit_tag)
                                else:
                                    # 若你未添加该工具函数，可临时回退为以下3步：
                                    if pair in self.manual_open:
                                        del self.manual_open[pair]
                                        self.update_strategy_state_file()

                                return -trade.stake_amount, exit_tag

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
                            if exit_points and len(exit_points) >= 1:
                                # 对退出点位进行排序
                                if direction == 'long':
                                    sorted_exit_points = sorted(exit_points)
                                else:  # short
                                    sorted_exit_points = sorted(exit_points, reverse=True)

                                cost_price = trade.open_rate
                                exit_points_count = len(sorted_exit_points)

                                logger.info(
                                    f"{pair} 检测到止盈回撤: exit_stage={exit_stage}, current_profit={current_profit:.3f}, current_rate={current_rate}, cost_price={cost_price}"
                                )

                                # 检查回撤清仓条件
                                should_exit = False
                                exit_tag = ''

                                # 2个点位的回撤处理
                                if exit_points_count == 2 and exit_stage == 1:
                                    if (direction == 'long' and current_rate <= cost_price) or (
                                        direction == 'short' and current_rate >= cost_price
                                    ):
                                        should_exit = True
                                        exit_tag = f'{direction}_tp1_pullback_cost'

                                # 3个或更多点位的回撤处理
                                elif exit_points_count >= 3:
                                    if direction == 'long':
                                        if exit_stage == 1 and current_rate <= cost_price:
                                            should_exit = True
                                            exit_tag = 'long_tp1_pullback_cost'
                                        elif (
                                            exit_stage == 2
                                            and current_rate <= sorted_exit_points[0]
                                        ):
                                            should_exit = True
                                            exit_tag = 'long_tp2_pullback_tp1'
                                    else:  # short
                                        if exit_stage == 1 and current_rate >= cost_price:
                                            should_exit = True
                                            exit_tag = 'short_tp1_pullback_cost'
                                        elif (
                                            exit_stage == 2
                                            and current_rate >= sorted_exit_points[0]
                                        ):
                                            should_exit = True
                                            exit_tag = 'short_tp2_pullback_tp1'

                                # 执行回撤清仓
                                if should_exit:
                                    logger.info(f"{pair} 触发止盈回撤清仓: {exit_tag}")
                                    self.enable_auto_calculation(pair, direction)
                                    self.recalculate_all_auto_monitoring_pairs()
                                    return -trade.stake_amount, exit_tag

                                break

            # 检查补仓冷却期
            last_dca_time = trade.get_custom_data('last_dca_time')

            # 如果没有last_dca_time，使用开仓时间作为参考时间
            if last_dca_time is None:
                last_dca_time = self._datetime_to_timestamp(trade.open_date_utc)
                # logger.info(
                #     f"{pair}: 首次检查补仓冷却期，使用开仓时间: {trade.open_date_utc} (时间戳: {last_dca_time})"
                # )

            if last_dca_time is not None:
                cooldown_minutes = 60 * 24 * 7  # 30分钟冷却期 (可根据交易对波动性调整)

                # 确保last_dca_time是时间戳格式
                if isinstance(last_dca_time, datetime):
                    last_dca_timestamp = self._datetime_to_timestamp(last_dca_time)
                    logger.warning(
                        f"{pair}: last_dca_time是datetime格式，已转换为时间戳: {last_dca_timestamp}"
                    )
                else:
                    last_dca_timestamp = last_dca_time

                # 验证时间戳的合理性
                try:
                    # 转换为datetime对象进行比较
                    last_dca_datetime = self._timestamp_to_datetime(
                        last_dca_timestamp, trade.open_date_utc.tzinfo
                    )

                    if current_time < last_dca_datetime + timedelta(minutes=cooldown_minutes):
                        time_remaining = (
                            last_dca_datetime + timedelta(minutes=cooldown_minutes) - current_time
                        ).total_seconds() / 60
                        if int(time_remaining) % 60 == 0:
                            logger.info(
                                f"{pair}: 补仓冷却期未结束，上次补仓/开仓时间: {last_dca_datetime}, 剩余冷却时间: {time_remaining:.1f}分钟"
                            )
                        return None

                except Exception as e:
                    logger.error(f"{pair}: 处理补仓冷却期时出错: {e}, 跳过冷却期检查")
                    # 出错时不阻止补仓，但记录错误

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
                            # logger.info(f"{pair}: 最小补仓金额 {min_stake} 会导致总金额超过 400，不补仓")
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
                        last_dca_time = self._datetime_to_timestamp(current_time)
                        trade.set_custom_data('last_dca_time', last_dca_time)
                        logger.info(f"{pair}: 记录补仓时间戳: {last_dca_time} (对应时间: {current_time})")

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
        # Manual trade exit logic
        if trade.enter_tag and 'manual' in trade.enter_tag:
            manual_config = self.manual_open.get(pair)
            if manual_config:
                exit_points = manual_config.get('exit_points', [])
                # The logic from here is an adaptation of the coin_monitoring logic below
                if not exit_points or len(exit_points) < 1:
                    return None

                if direction == 'long':
                    sorted_exit_points = sorted(exit_points)
                else:  # short
                    sorted_exit_points = sorted(exit_points, reverse=True)

                cost_price = trade.open_rate
                exit_stage = trade.get_custom_data('exit_stage', default=0)

                if exit_stage == 0 and trade.get_custom_data('initial_stake') is None:
                    trade.set_custom_data('initial_stake', trade.stake_amount)
                initial_stake = trade.get_custom_data('initial_stake', default=trade.stake_amount)
                exit_points_count = len(sorted_exit_points)

                if exit_points_count == 1:
                    if (direction == 'long' and current_rate >= sorted_exit_points[0]) or (
                        direction == 'short' and current_rate <= sorted_exit_points[0]
                    ):
                        logger.info(f"Manual trade: Triggering single exit point for {pair}")
                        return -trade.stake_amount, f"manual_{direction}_single_tp"

                elif exit_points_count == 2:
                    if exit_stage == 0 and (
                        (direction == 'long' and current_rate >= sorted_exit_points[0])
                        or (direction == 'short' and current_rate <= sorted_exit_points[0])
                    ):
                        trade.set_custom_data('exit_stage', 1)
                        self._adjust_stoploss(trade, cost_price)
                        logger.info(f"Manual trade: Triggering TP1 of 2 for {pair}")
                        return -(initial_stake * 0.5), f"manual_{direction}_tp1_of_2"

                    elif exit_stage == 1:
                        if (direction == 'long' and current_rate >= sorted_exit_points[1]) or (
                            direction == 'short' and current_rate <= sorted_exit_points[1]
                        ):
                            trade.set_custom_data('exit_stage', 2)
                            logger.info(f"Manual trade: Triggering TP2 of 2 for {pair}")
                            return -trade.stake_amount, f"manual_{direction}_tp2_of_2"
                        # Pullback logic for 2 exit points
                        elif (
                            (direction == 'long' and current_rate <= cost_price)
                            or (direction == 'short' and current_rate >= cost_price)
                        ) and (current_profit >= -0.005):
                            logger.info(
                                f"Manual trade for {pair} pulling back to cost price. Exiting position."
                            )
                            return -trade.stake_amount, f'manual_{direction}_tp1_pullback_cost'

                else:  # 3 or more points
                    if exit_stage == 0 and (
                        (direction == 'long' and current_rate >= sorted_exit_points[0])
                        or (direction == 'short' and current_rate <= sorted_exit_points[0])
                    ):
                        trade.set_custom_data('exit_stage', 1)
                        self._adjust_stoploss(trade, cost_price)
                        logger.info(
                            f"Manual trade: Triggering TP1 of {exit_points_count} for {pair}"
                        )
                        return -(initial_stake * 0.3), f"manual_{direction}_tp1"

                    elif exit_stage == 1:
                        if (direction == 'long' and current_rate >= sorted_exit_points[1]) or (
                            direction == 'short' and current_rate <= sorted_exit_points[1]
                        ):
                            trade.set_custom_data('exit_stage', 2)
                            self._adjust_stoploss(trade, sorted_exit_points[0])
                            remaining_stake = trade.stake_amount
                            logger.info(
                                f"Manual trade: Triggering TP2 of {exit_points_count} for {pair}"
                            )
                            return -(remaining_stake * 0.5), f"manual_{direction}_tp2"
                        # Pullback from TP1 to cost
                        elif (
                            direction == 'long'
                            and current_rate <= cost_price
                            and current_profit >= -0.005
                        ) or (
                            direction == 'short'
                            and current_rate >= cost_price
                            and current_profit >= -0.005
                        ):
                            logger.info(
                                f"Manual trade for {pair} pulling back to cost price from TP1. Exiting position."
                            )
                            return -trade.stake_amount, f'manual_{direction}_tp1_pullback_cost'

                    elif exit_stage == 2:
                        if (direction == 'long' and current_rate >= sorted_exit_points[2]) or (
                            direction == 'short' and current_rate <= sorted_exit_points[2]
                        ):
                            logger.info(
                                f"Manual trade: Triggering TP3 of {exit_points_count} for {pair}"
                            )
                            return -trade.stake_amount, f"manual_{direction}_tp3"
                        # Pullback from TP2 to TP1
                        elif (direction == 'long' and current_rate <= sorted_exit_points[0]) or (
                            direction == 'short' and current_rate >= sorted_exit_points[0]
                        ):
                            logger.info(
                                f"Manual trade for {pair} pulling back to TP1 price. Exiting position."
                            )
                            return -trade.stake_amount, f'manual_{direction}_tp2_pullback_tp1'

            # If it's a manual trade, we've handled it or decided not to act.
            # Don't fall through to coin_monitoring logic.
            return None

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
                            # 重新计算所有自动点位监控的交易对
                            self.recalculate_all_auto_monitoring_pairs()
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
                                # 重新计算所有自动点位监控的交易对
                                self.recalculate_all_auto_monitoring_pairs()
                                return (
                                    -trade.stake_amount,
                                    f"{direction}_tp2_of2_{sorted_exit_points[1]}",
                                )

                            # 处理回撤情况
                            elif (
                                (direction == 'long' and current_rate <= cost_price)
                                or (direction == 'short' and current_rate >= cost_price)
                            ) and (current_profit >= -0.005):
                                # 从第一阶段回撤到成本价，清仓
                                logger.info(f"{pair} 从第一点位回撤至成本价 {cost_price}，清仓")
                                self.enable_auto_calculation(pair, direction)
                                # 重新计算所有自动点位监控的交易对
                                self.recalculate_all_auto_monitoring_pairs()
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
                                # 重新计算所有自动点位监控的交易对
                                self.recalculate_all_auto_monitoring_pairs()
                                return -trade.stake_amount, f"long_tp3_{sorted_exit_points[2]}"

                            # 处理回撤情况 - 多头
                            elif (
                                exit_stage == 1
                                and current_rate <= cost_price
                                and current_profit >= -0.005
                            ):
                                # 第一阶段回撤到成本价，清仓
                                logger.info(f"{pair} 从第一点位回撤至成本价 {cost_price}，清仓")
                                self.enable_auto_calculation(pair, direction)
                                # 重新计算所有自动点位监控的交易对
                                self.recalculate_all_auto_monitoring_pairs()
                                return -trade.stake_amount, 'long_tp1_pullback_cost'

                            elif exit_stage == 2 and current_rate <= sorted_exit_points[0]:
                                # 第二阶段回撤到第一点位，清仓
                                logger.info(f"{pair} 从第二点位回撤至第一点位 {sorted_exit_points[0]}，清仓")
                                self.enable_auto_calculation(pair, direction)
                                # 重新计算所有自动点位监控的交易对
                                self.recalculate_all_auto_monitoring_pairs()
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
                                # 重新计算所有自动点位监控的交易对
                                self.recalculate_all_auto_monitoring_pairs()
                                return -trade.stake_amount, f"short_tp3_{sorted_exit_points[2]}"

                            # 处理回撤情况 - 空头
                            elif (
                                exit_stage == 1
                                and current_rate >= cost_price
                                and current_profit >= -0.005
                            ):
                                # 第一阶段回撤到成本价，清仓
                                logger.info(f"{pair} 从第一点位回撤至成本价 {cost_price}，清仓")
                                self.enable_auto_calculation(pair, direction)
                                # 重新计算所有自动点位监控的交易对
                                self.recalculate_all_auto_monitoring_pairs()
                                return -trade.stake_amount, 'short_tp1_pullback_cost'

                            elif exit_stage == 2 and current_rate >= sorted_exit_points[0]:
                                # 第二阶段回撤到第一点位，清仓
                                logger.info(f"{pair} 从第二点位回撤至第一点位 {sorted_exit_points[0]}，清仓")
                                self.enable_auto_calculation(pair, direction)
                                # 重新计算所有自动点位监控的交易对
                                self.recalculate_all_auto_monitoring_pairs()
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

    def _has_monitoring_cfg(self, pair: str, direction: str) -> bool:
        cfgs = self.coin_monitoring.get(pair, []) if hasattr(self, 'coin_monitoring') else []
        for cfg in cfgs:
            if cfg.get('direction', 'long') == direction and (
                cfg.get('exit_points') or cfg.get('entry_points')
            ):
                return True
        return False

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
        - 手动单（manual_open）优先：只按手动点位做最终全退判断，避免被 coin_monitoring 误退
        - 非手动单：按 coin_monitoring 的第三级（或对应最终）退出处理
        - 其它情况回落到原有 _custom_exit_long/_short
        """
        # 1) open_rate 为空 => 首单还没成交完成
        if trade.open_rate is None:
            return None

        # 2) 保险起见，再加几条稳妥的保护（可选）
        if not trade.is_open:
            return None
        if trade.amount is None or trade.amount <= 0:
            return None
        # 如果还有挂着的订单（首单/补仓/减仓）也不要再下新单（可选）
        if getattr(trade, 'open_order_id', None):
            return None

        direction = 'short' if trade.is_short else 'long'

        # ========= ① 手动单优先：只处理“最终全退”，其余交给 adjust_trade_position =========
        if trade.enter_tag and 'manual' in trade.enter_tag:
            manual_cfg = self.manual_open.get(pair, {})
            exit_points = manual_cfg.get('exit_points', []) or []

            if exit_points:
                # 排序：多头升序，空头降序
                if direction == 'long':
                    sorted_exit_points = sorted(exit_points)
                else:
                    sorted_exit_points = sorted(exit_points, reverse=True)

                exit_stage = trade.get_custom_data('exit_stage', default=0)
                n = len(sorted_exit_points)

                # 1 个点位：到价即全退（兜底，避免边界情况下未在 adjust_trade_position 执行）
                if n == 1:
                    trig = (
                        (current_rate >= sorted_exit_points[0])
                        if direction == 'long'
                        else (current_rate <= sorted_exit_points[0])
                    )
                    if trig:
                        # 手动单全退清理
                        if hasattr(self, '_manual_cleanup_after_full_close'):
                            self._manual_cleanup_after_full_close(
                                pair, direction, f"manual_{direction}_single_tp"
                            )
                        return f"manual_{direction}_single_tp_{sorted_exit_points[0]}"

                    # 未到最终点位 -> 不允许落到 coin_monitoring，避免误退
                    return None

                # 2 个点位：仅当已过 TP1（exit_stage==1）且到达 TP2 时全退（兜底）
                if n == 2 and exit_stage == 1:
                    trig = (
                        (current_rate >= sorted_exit_points[1])
                        if direction == 'long'
                        else (current_rate <= sorted_exit_points[1])
                    )
                    if trig:
                        if hasattr(self, '_manual_cleanup_after_full_close'):
                            self._manual_cleanup_after_full_close(
                                pair, direction, f"manual_{direction}_tp2_of_2"
                            )
                        return f"manual_{direction}_tp2_of_2_{sorted_exit_points[1]}"

                    # 未到“最终全退”条件 -> 直接返回，避免 coin_monitoring 误退
                    return None

                # ≥3 个点位：仅当已过 TP2（exit_stage==2）且到达 TP3 时全退（与你原设计一致）
                if n >= 3 and exit_stage == 2:
                    trig = (
                        (current_rate >= sorted_exit_points[2])
                        if direction == 'long'
                        else (current_rate <= sorted_exit_points[2])
                    )
                    if trig:
                        # 手动单全退清理
                        if hasattr(self, '_manual_cleanup_after_full_close'):
                            self._manual_cleanup_after_full_close(
                                pair, direction, f"manual_{direction}_tp3"
                            )
                        # 内部状态更新仅用于记录；真正的清理在 _manual_cleanup_after_full_close 完成
                        trade.set_custom_data('exit_stage', 3)
                        return f"manual_{direction}_tp3_{sorted_exit_points[2]}"

                # 手动单存在，但未满足“最终全退” → 明确不让 coin_monitoring 介入
                return None

            # 没有手动 exit_points，则继续走后续逻辑（可能是老单/被外部清空）
            # 不 return

        # ===== ② coin_monitoring：只处理“最终全退”；否则不下放 =====
        if self._has_monitoring_cfg(pair, direction):
            # 找到该方向的退出点位
            for cfg in self.coin_monitoring.get(pair, []):
                if cfg.get('direction', 'long') != direction:
                    continue
                xs = cfg.get('exit_points', []) or []
                if not xs:
                    continue

                sorted_xs = sorted(xs, reverse=direction == 'short')
                stage = trade.get_custom_data('exit_stage', default=0)
                n = len(sorted_xs)

                # 与原设计一致：≥3点位，已过TP2(stage==2)且到TP3 -> 全退
                if n >= 3 and stage == 2:
                    trig = (
                        (current_rate >= sorted_xs[2])
                        if direction == 'long'
                        else (current_rate <= sorted_xs[2])
                    )
                    if trig:
                        trade.set_custom_data('exit_stage', 3)
                        self.enable_auto_calculation(pair, direction)
                        self.recalculate_all_auto_monitoring_pairs()
                        return f"{direction}_tp3_{sorted_xs[2]}"

                # 无论是否触发最终全退，coin_monitoring 接管了该 pair 的退出。
                # 为避免误触默认 _custom_exit_*，这里必须阻断下放：
                return None

            # 有 coin_monitoring 但没拿到点位（异常情况） -> 允许下放到默认

        # ========= ③ 其余情况：回落到你原有的 _custom_exit_* =========
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
