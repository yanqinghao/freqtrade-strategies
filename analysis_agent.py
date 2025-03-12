import pandas as pd
import os
import requests
import ccxt
from datetime import datetime
import talib
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class CryptoTechnicalAnalyst:
    """
    使用CCXT获取数据的增强版加密货币技术分析系统，提供多时间周期分析和LLM辅助决策
    """

    def __init__(self, api_key, api_secret):
        """
        初始化分析器

        参数:
        api_key: Binance API Key
        api_secret: Binance API Secret
        openai_api_key: OpenAI API Key
        """
        # 初始化CCXT交易所
        self.exchange = ccxt.binance(
            {
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,  # 避免触发频率限制
                'options': {
                    'defaultType': 'future',  # 使用期货API
                },
            }
        )
        llm_api_key = os.environ['LLM_API_KEY']
        llm_base_url = os.environ['LLM_BASE_URL']
        llm_model_name = os.environ['LLM_MODEL_NAME']
        self.llm = ChatOpenAI(
            temperature=0.2, model_name=llm_model_name, base_url=llm_base_url, api_key=llm_api_key
        )

        # 定义要分析的时间周期
        self.timeframes = {'1d': '1d', '4h': '4h', '1h': '1h', '15m': '15m'}

        # 初始化结果存储
        self.analysis_results = {}

        self.raw_data = {}

        # 币种上线时间缓存
        self.listing_times = {}

    def get_recent_data(self):
        """
        Extract recent data from different timeframes:
        - 1d: 7 days
        - 4h: 3 days
        - 1h: 1 day
        - 15m: 1 hour
        """
        result = {}

        # For 1d timeframe, get last 7 days
        if '1d' in self.raw_data and not self.raw_data['1d'].empty:
            result['1d'] = self.raw_data['1d'].tail(7).copy()

        # For 4h timeframe, get last 3 days (18 bars)
        if '4h' in self.raw_data and not self.raw_data['4h'].empty:
            result['4h'] = self.raw_data['4h'].tail(12).copy()  # 6 bars per day * 3 days

        # For 1h timeframe, get last 1 day (24 bars)
        if '1h' in self.raw_data and not self.raw_data['1h'].empty:
            result['1h'] = self.raw_data['1h'].tail(12).copy()

        # For 15m timeframe, get last 1 hour (4 bars)
        if '15m' in self.raw_data and not self.raw_data['15m'].empty:
            result['15m'] = self.raw_data['15m'].tail(12).copy()

        return result

    def format_data_for_llm_markdown(self):
        """
        Format the recent data from all timeframes into markdown tables
        for evaluation by a large language model.
        """
        recent_data = self.get_recent_data()

        if not recent_data:
            return 'No K-line data available.'

        result = []

        # Process each timeframe
        for timeframe, df in recent_data.items():
            if df.empty:
                continue

            # Add timeframe header
            result.append(f"\n## {timeframe} Timeframe Data")

            # Create markdown table header
            table_header = '| Time | Open | High | Low | Close | Volume |'
            table_separator = '| --- | --- | --- | --- | --- | --- |'
            table_rows = []

            # Format the dataframe as markdown table rows
            for idx, row in df.iterrows():
                timestamp = idx
                if hasattr(timestamp, 'strftime'):
                    timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S')

                table_row = f"| {timestamp} | {row['open']:.2f} | {row['high']:.2f} | {row['low']:.2f} | {row['close']:.2f} | {row['volume']:.2f} |"
                table_rows.append(table_row)

            # Combine table parts
            table = [table_header, table_separator] + table_rows
            result.append('\n'.join(table))

        # Combine all data into one string
        return '\n\n'.join(result)

    def fetch_ohlcv(self, symbol, timeframe, limit=100):
        """
        使用CCXT从Binance获取OHLCV数据

        参数:
        symbol: 交易对 (例如 'BTC/USDT')
        timeframe: 时间周期 (例如 '1d', '4h', '1h', '15m')
        limit: 获取的K线数量

        返回:
        DataFrame: OHLCV数据
        """
        # 标准化符号格式(添加斜杠)
        if '/' not in symbol:
            # 假设是USDT对
            if symbol.endswith('USDT'):
                formatted_symbol = f"{symbol[:-4]}/USDT"
            else:
                formatted_symbol = f"{symbol}/USDT"
        else:
            formatted_symbol = symbol

        # 获取OHLCV数据
        ohlcv = self.exchange.fetch_ohlcv(symbol=formatted_symbol, timeframe=timeframe, limit=limit)

        # 将数据转换为DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

        # 转换时间戳
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 设置时间戳为索引
        df.set_index('timestamp', inplace=True)

        return df

    def get_listing_time(self, symbol):
        """
        获取币种的上线时间

        参数:
        symbol: 交易对 (例如 'BTC/USDT')

        返回:
        datetime: 币种上线时间或None
        """
        # 如果已经缓存，直接返回
        if symbol in self.listing_times:
            return self.listing_times[symbol]

        try:
            # 标准化符号格式
            if '/' in symbol:
                base_symbol = symbol.split('/')[0]
                symbol_without_slash = f"{base_symbol}USDT"
            elif symbol.endswith('USDT'):
                base_symbol = symbol[:-4]
                symbol_without_slash = symbol
            else:
                base_symbol = symbol
                symbol_without_slash = f"{symbol}USDT"

            # 从Binance期货API获取币种信息
            response = requests.get('https://fapi.binance.com/fapi/v1/exchangeInfo')
            data = response.json()

            # 查找匹配的交易对
            listing_time = None
            for symbol_info in data['symbols']:
                if symbol_info['symbol'] == symbol_without_slash:
                    # 将毫秒时间戳转换为datetime
                    if 'onboardDate' in symbol_info:
                        listing_time = datetime.fromtimestamp(symbol_info['onboardDate'] / 1000)
                    elif 'listingDate' in symbol_info:
                        listing_time = datetime.fromtimestamp(symbol_info['listingDate'] / 1000)
                    break

            # 缓存结果
            self.listing_times[symbol] = listing_time
            return listing_time

        except Exception as e:
            print(f"获取{symbol}上线时间出错: {e}")
            return None

    def calculate_technical_indicators(self, df):
        """
        计算各种技术指标

        参数:
        df: OHLCV数据的DataFrame

        返回:
        dict: 计算出的各种技术指标
        """
        # 基本指标计算
        indicators = {}

        # 移动平均线
        indicators['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        indicators['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        indicators['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        indicators['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        indicators['ema_26'] = talib.EMA(df['close'], timeperiod=26)

        # MACD
        macd, macd_signal, macd_hist = talib.MACD(
            df['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )
        indicators['macd'] = macd
        indicators['macd_signal'] = macd_signal
        indicators['macd_hist'] = macd_hist

        # RSI
        indicators['rsi_14'] = talib.RSI(df['close'], timeperiod=14)

        # Stochastic
        indicators['slowk'], indicators['slowd'] = talib.STOCH(
            df['high'],
            df['low'],
            df['close'],
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0,
        )

        # Bollinger Bands
        indicators['upperband'], indicators['middleband'], indicators['lowerband'] = talib.BBANDS(
            df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )

        # ADX
        indicators['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # 趋势变化点指标 (CCI)
        indicators['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

        # OBV (On-Balance Volume)
        indicators['obv'] = talib.OBV(df['close'], df['volume'])

        # Ichimoku Cloud
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        indicators['tenkan_sen'] = (high_9 + low_9) / 2

        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        indicators['kijun_sen'] = (high_26 + low_26) / 2

        # 将所有指标转换为DataFrame
        indicators_df = pd.DataFrame(indicators, index=df.index)

        return indicators_df

    def analyze_signals(self, df, indicators_df):
        """
        根据技术指标分析交易信号

        参数:
        df: 原始OHLCV数据
        indicators_df: 计算出的技术指标

        返回:
        dict: 各种交易信号和它们的强度
        """
        signals = {}

        # 最新的价格和指标值
        current_close = df['close'].iloc[-1]

        # --- 趋势跟踪信号 ---
        # 移动平均线趋势
        sma_20 = indicators_df['sma_20'].iloc[-1]
        sma_50 = indicators_df['sma_50'].iloc[-1]
        sma_200 = indicators_df['sma_200'].iloc[-1]

        # MA交叉信号
        ma_cross = 0
        if sma_20 > sma_50 and indicators_df['sma_20'].iloc[-2] <= indicators_df['sma_50'].iloc[-2]:
            ma_cross = 1  # 黄金交叉 (看涨)
        elif (
            sma_20 < sma_50 and indicators_df['sma_20'].iloc[-2] >= indicators_df['sma_50'].iloc[-2]
        ):
            ma_cross = -1  # 死亡交叉 (看跌)

        # 价格相对于MA的位置
        price_vs_ma = 0
        if current_close > sma_200:
            price_vs_ma += 1
        if current_close > sma_50:
            price_vs_ma += 1
        if current_close > sma_20:
            price_vs_ma += 1
        if current_close < sma_200:
            price_vs_ma -= 1
        if current_close < sma_50:
            price_vs_ma -= 1
        if current_close < sma_20:
            price_vs_ma -= 1

        # ADX (趋势强度)
        adx = indicators_df['adx'].iloc[-1]
        trend_strength = min(adx / 50.0, 1.0)  # 归一化为0-1

        # 总体趋势信号
        if price_vs_ma > 0:
            trend_signal = 'bullish'
        elif price_vs_ma < 0:
            trend_signal = 'bearish'
        else:
            trend_signal = 'neutral'

        trend_confidence = abs(price_vs_ma) / 3 * trend_strength

        signals['trend'] = {
            'signal': trend_signal,
            'confidence': round(trend_confidence * 100),
            'metrics': {
                'price_vs_ma': price_vs_ma,
                'ma_cross': ma_cross,
                'adx': float(adx),
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'sma_200': float(sma_200),
            },
        }

        # --- 震荡指标信号 ---
        # RSI
        rsi = indicators_df['rsi_14'].iloc[-1]

        # Stochastic
        slowk = indicators_df['slowk'].iloc[-1]
        slowd = indicators_df['slowd'].iloc[-1]

        # CCI
        cci = indicators_df['cci'].iloc[-1]

        # RSI信号
        if rsi > 70:
            rsi_signal = 'bearish'
            rsi_strength = min((rsi - 70) / 30, 1.0)
        elif rsi < 30:
            rsi_signal = 'bullish'
            rsi_strength = min((30 - rsi) / 30, 1.0)
        else:
            if rsi > 50:
                rsi_signal = 'slightly_bullish'
                rsi_strength = (rsi - 50) / 20
            elif rsi < 50:
                rsi_signal = 'slightly_bearish'
                rsi_strength = (50 - rsi) / 20
            else:
                rsi_signal = 'neutral'
                rsi_strength = 0

        # Stochastic信号
        if slowk > 80 and slowd > 80:
            stoch_signal = 'bearish'
            stoch_strength = min((slowk - 80) / 20, 1.0)
        elif slowk < 20 and slowd < 20:
            stoch_signal = 'bullish'
            stoch_strength = min((20 - slowk) / 20, 1.0)
        else:
            if slowk > slowd and slowk < 80 and slowd < 80:
                stoch_signal = 'slightly_bullish'
                stoch_strength = 0.5
            elif slowk < slowd and slowk > 20 and slowd > 20:
                stoch_signal = 'slightly_bearish'
                stoch_strength = 0.5
            else:
                stoch_signal = 'neutral'
                stoch_strength = 0

        # 震荡器的综合信号
        oscillator_signals = {
            'bullish': 0,
            'slightly_bullish': 0,
            'neutral': 0,
            'slightly_bearish': 0,
            'bearish': 0,
        }

        oscillator_signals[rsi_signal] += rsi_strength
        oscillator_signals[stoch_signal] += stoch_strength

        # 找出最强的信号
        max_signal = max(oscillator_signals, key=oscillator_signals.get)
        osc_confidence = oscillator_signals[max_signal]

        # 简化为基本的三种信号
        if max_signal in ['bullish', 'slightly_bullish']:
            osc_signal = 'bullish'
        elif max_signal in ['bearish', 'slightly_bearish']:
            osc_signal = 'bearish'
        else:
            osc_signal = 'neutral'

        signals['oscillators'] = {
            'signal': osc_signal,
            'confidence': round(osc_confidence * 100),
            'metrics': {
                'rsi': float(rsi),
                'stoch_k': float(slowk),
                'stoch_d': float(slowd),
                'cci': float(cci),
            },
        }

        # --- MACD信号 ---
        macd = indicators_df['macd'].iloc[-1]
        macd_signal = indicators_df['macd_signal'].iloc[-1]
        macd_hist = indicators_df['macd_hist'].iloc[-1]

        # MACD交叉信号
        if (
            macd > macd_signal
            and indicators_df['macd'].iloc[-2] <= indicators_df['macd_signal'].iloc[-2]
        ):
            macd_cross = 'bullish'
            macd_strength = 0.8
        elif (
            macd < macd_signal
            and indicators_df['macd'].iloc[-2] >= indicators_df['macd_signal'].iloc[-2]
        ):
            macd_cross = 'bearish'
            macd_strength = 0.8
        else:
            if macd > macd_signal:
                macd_cross = 'bullish'
                macd_strength = 0.5
            elif macd < macd_signal:
                macd_cross = 'bearish'
                macd_strength = 0.5
            else:
                macd_cross = 'neutral'
                macd_strength = 0

        # MACD柱状图趋势
        macd_hist_trend = 'neutral'
        if len(indicators_df) >= 3:
            hist_values = indicators_df['macd_hist'].iloc[-3:].values
            if all(hist_values[i] > hist_values[i - 1] for i in range(1, len(hist_values))):
                macd_hist_trend = 'bullish'
            elif all(hist_values[i] < hist_values[i - 1] for i in range(1, len(hist_values))):
                macd_hist_trend = 'bearish'

        # 综合MACD信号
        if macd_cross == 'bullish' and macd_hist_trend == 'bullish':
            momentum_signal = 'bullish'
            momentum_confidence = macd_strength
        elif macd_cross == 'bearish' and macd_hist_trend == 'bearish':
            momentum_signal = 'bearish'
            momentum_confidence = macd_strength
        elif macd_cross == 'bullish':
            momentum_signal = 'bullish'
            momentum_confidence = macd_strength * 0.7
        elif macd_cross == 'bearish':
            momentum_signal = 'bearish'
            momentum_confidence = macd_strength * 0.7
        else:
            momentum_signal = 'neutral'
            momentum_confidence = 0.3

        signals['momentum'] = {
            'signal': momentum_signal,
            'confidence': round(momentum_confidence * 100),
            'metrics': {
                'macd': float(macd),
                'macd_signal': float(macd_signal),
                'macd_hist': float(macd_hist),
                'macd_cross': macd_cross,
                'hist_trend': macd_hist_trend,
            },
        }

        # --- 布林带信号 ---
        upper = indicators_df['upperband'].iloc[-1]
        middle = indicators_df['middleband'].iloc[-1]
        lower = indicators_df['lowerband'].iloc[-1]

        # 计算价格相对于布林带的位置 (0 = 下轨, 0.5 = 中轨, 1 = 上轨)
        bb_position = (current_close - lower) / (upper - lower) if upper != lower else 0.5

        # 布林带宽度 (波动率指标)
        bb_width = (upper - lower) / middle

        # 波动率信号
        vol_signal = 'neutral'
        vol_confidence = 0.5

        if bb_position <= 0.1:
            vol_signal = 'bullish'  # 价格接近下轨，超卖
            vol_confidence = 0.7
        elif bb_position >= 0.9:
            vol_signal = 'bearish'  # 价格接近上轨，超买
            vol_confidence = 0.7
        elif bb_width < 0.1:  # 布林带收窄，可能即将爆发
            # 根据趋势确定方向
            if trend_signal == 'bullish':
                vol_signal = 'bullish'
                vol_confidence = 0.6
            elif trend_signal == 'bearish':
                vol_signal = 'bearish'
                vol_confidence = 0.6

        signals['volatility'] = {
            'signal': vol_signal,
            'confidence': round(vol_confidence * 100),
            'metrics': {
                'bb_position': float(bb_position),
                'bb_width': float(bb_width),
                'upperband': float(upper),
                'middleband': float(middle),
                'lowerband': float(lower),
            },
        }

        # --- 成交量信号 ---
        recent_volume = df['volume'].iloc[-5:]
        avg_volume = df['volume'].iloc[-20:-5].mean()
        volume_ratio = recent_volume.mean() / avg_volume if avg_volume > 0 else 1.0

        # OBV趋势
        obv = indicators_df['obv'].iloc[-5:].values
        obv_trend = 0
        if all(obv[i] >= obv[i - 1] for i in range(1, len(obv))):
            obv_trend = 1  # 上升
        elif all(obv[i] <= obv[i - 1] for i in range(1, len(obv))):
            obv_trend = -1  # 下降

        # 成交量确认趋势
        if volume_ratio > 1.5 and obv_trend == 1 and trend_signal == 'bullish':
            volume_signal = 'bullish'
            volume_confidence = min(volume_ratio / 3, 1.0)
        elif volume_ratio > 1.5 and obv_trend == -1 and trend_signal == 'bearish':
            volume_signal = 'bearish'
            volume_confidence = min(volume_ratio / 3, 1.0)
        else:
            volume_signal = 'neutral'
            volume_confidence = 0.4

        signals['volume'] = {
            'signal': volume_signal,
            'confidence': round(volume_confidence * 100),
            'metrics': {'volume_ratio': float(volume_ratio), 'obv_trend': obv_trend},
        }

        # 组合所有信号
        strategy_weights = {
            'trend': 0.30,
            'oscillators': 0.20,
            'momentum': 0.20,
            'volatility': 0.15,
            'volume': 0.15,
        }

        combined_signal = self._combine_signals(signals, strategy_weights)
        signals['combined'] = combined_signal

        return signals

    def _combine_signals(self, signals, weights):
        """
        使用加权方法组合多个交易信号

        参数:
        signals: 各种策略的信号字典
        weights: 各种策略的权重字典

        返回:
        dict: 组合后的信号和置信度
        """
        # 信号值映射
        signal_values = {'bullish': 1, 'neutral': 0, 'bearish': -1}

        weighted_sum = 0
        total_weight = 0

        for strategy, signal_info in signals.items():
            if strategy in weights:
                weight = weights[strategy]
                signal = signal_info['signal']
                confidence = signal_info['confidence'] / 100  # 归一化到0-1

                numeric_signal = signal_values.get(signal, 0)
                weighted_sum += numeric_signal * weight * confidence
                total_weight += weight

        # 计算最终得分
        final_score = weighted_sum / total_weight if total_weight > 0 else 0

        # 转换回信号
        if final_score >= 0.3:
            signal = 'bullish'
        elif final_score <= -0.3:
            signal = 'bearish'
        else:
            signal = 'neutral'

        return {'signal': signal, 'confidence': round(abs(final_score) * 100)}

    def analyze_crypto(self, symbol):
        """
        分析加密货币在多个时间周期上的技术指标

        参数:
        symbol: 交易对 (例如 'BTC/USDT')

        返回:
        dict: 包含多个时间周期分析结果的字典
        """
        all_timeframe_results = {}

        # 获取币种上线时间
        listing_time = self.get_listing_time(symbol)
        listing_time_str = listing_time.strftime('%Y-%m-%d') if listing_time else '未知'

        # 分析每个时间周期
        for timeframe_key, timeframe_value in self.timeframes.items():
            try:
                # 获取K线数据
                df = self.fetch_ohlcv(symbol, timeframe_value)

                self.raw_data[timeframe_key] = df

                # 计算指标
                indicators_df = self.calculate_technical_indicators(df)

                # 分析信号
                signals = self.analyze_signals(df, indicators_df)

                # 保存该时间周期的结果
                all_timeframe_results[timeframe_key] = {
                    'signals': signals,
                    'last_price': float(df['close'].iloc[-1]),
                    'timestamp': df.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                    'listing_time': listing_time_str,
                }

            except Exception as e:
                print(f"分析{timeframe_key}时间周期时出错: {e}")
                all_timeframe_results[timeframe_key] = {'error': str(e)}

        # 保存结果
        self.analysis_results[symbol] = all_timeframe_results

        return all_timeframe_results

    def format_technical_analysis(self, symbol, timeframe_results):
        """
        将技术分析结果格式化为适合LLM理解的文本

        参数:
        symbol: 交易对
        timeframe_results: 各时间周期的分析结果

        返回:
        str: 格式化的分析文本
        """
        # 从任意时间周期获取上线时间(应该所有时间周期都相同)
        listing_time = None
        for _, results in timeframe_results.items():
            if 'error' not in results and 'listing_time' in results:
                listing_time = results['listing_time']
                break

        formatted_text = f"加密货币: {symbol}\n"
        formatted_text += f"上线时间: {listing_time if listing_time else '未知'}\n\n"

        for timeframe, results in timeframe_results.items():
            if 'error' in results:
                formatted_text += f"{timeframe}时间周期分析出错: {results['error']}\n\n"
                continue

            formatted_text += f"====== {timeframe}时间周期 ======\n"
            formatted_text += f"最新价格: {results['last_price']}\n"
            formatted_text += f"时间戳: {results['timestamp']}\n\n"

            signals = results['signals']

            # 组合信号
            combined = signals['combined']
            formatted_text += f"综合信号: {combined['signal']} (置信度: {combined['confidence']}%)\n\n"

            # 各策略信号
            formatted_text += '各策略信号:\n'
            for strategy in ['trend', 'oscillators', 'momentum', 'volatility', 'volume']:
                signal_info = signals[strategy]
                formatted_text += (
                    f"- {strategy}: {signal_info['signal']} (置信度: {signal_info['confidence']}%)\n"
                )

                # 添加关键指标
                formatted_text += '  关键指标:\n'
                for metric, value in signal_info['metrics'].items():
                    formatted_text += f"    {metric}: {value}\n"

            formatted_text += '\n'

        return formatted_text

    def get_llm_analysis(self, symbol):
        """
        使用LLM分析多个时间周期的技术指标并提供入场建议

        参数:
        symbol: 交易对

        返回:
        str: LLM的分析和建议
        """
        # 确保已经有分析结果
        if symbol not in self.analysis_results:
            self.analyze_crypto(symbol)

        timeframe_results = self.analysis_results[symbol]

        # 格式化分析结果
        formatted_analysis = self.format_technical_analysis(symbol, timeframe_results)
        formatted_klines = self.format_data_for_llm_markdown()

        # 创建LLM提示
        prompt_template = """
        作为加密货币交易专家，基于以下{symbol}的技术分析结果提供详细的多时间周期分析和交易入场建议:

        {klines}

        {analysis}

        请提供以下内容:
        1. 币种背景: 考虑币种的上线时间，简要分析其发展阶段和市场成熟度。
        2. 多时间周期分析: 分析各个时间周期(1d, 4h, 1h, 15m)的技术指标，说明信号的一致性或冲突。
        3. 交易机会评估: 是否有任何明显的做多或做空机会？在哪个时间周期上信号最强？
        4. 入场建议: 最佳入场时机和价格水平，最好分时间周期提供明确的建议。
        5. 风险管理: 建议的止损位和目标盈利水平。
        6. 信号冲突解释: 如果不同时间周期之间存在冲突信号，解释可能的原因。

        最后，请总结当前{symbol}的整体交易观点，并用加粗的文本明确指出最终的交易建议。
        """

        prompt = PromptTemplate(
            input_variables=['symbol', 'analysis', 'klines'], template=prompt_template
        )
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        # 获取LLM的分析
        response = llm_chain.run(
            symbol=symbol, analysis=formatted_analysis, klines=formatted_klines
        )

        return response

    def generate_multi_timeframe_table(self, symbol):
        """
        生成多时间周期技术指标的ASCII表格

        参数:
        symbol: 交易对

        返回:
        str: 格式化的ASCII表格
        """
        if symbol not in self.analysis_results:
            self.analyze_crypto(symbol)

        timeframe_results = self.analysis_results[symbol]

        # 从任意时间周期获取上线时间(应该所有时间周期都相同)
        listing_time = None
        for _, results in timeframe_results.items():
            if 'error' not in results and 'listing_time' in results:
                listing_time = results['listing_time']
                break

        # 表格标题和币种信息
        coin_info = f"币种: {symbol} | 上线时间: {listing_time if listing_time else '未知'}\n\n"
        header = '| 时间周期 | 价格 | 信号 | 置信度 | 趋势 | 震荡器 | 动量 | 波动性 | 成交量 |'
        separator = '|----------|----------|----------|----------|----------|----------|----------|----------|----------|'

        rows = []
        for timeframe, results in sorted(timeframe_results.items()):
            if 'error' in results:
                continue

            # 获取信号和价格数据
            price = results['last_price']
            signals = results['signals']

            combined = signals['combined']
            trend = signals['trend']['signal']
            oscillators = signals['oscillators']['signal']
            momentum = signals['momentum']['signal']
            volatility = signals['volatility']['signal']
            volume = signals['volume']['signal']

            # 信号映射为符号
            signal_map = {'bullish': '🔼', 'bearish': '🔽', 'neutral': '◀▶'}

            # 创建表格行
            row = f"| {timeframe} | {price:.2f} | {signal_map.get(combined['signal'], '?')} | {combined['confidence']}% | "
            row += f"{signal_map.get(trend, '?')} | {signal_map.get(oscillators, '?')} | "
            row += f"{signal_map.get(momentum, '?')} | {signal_map.get(volatility, '?')} | {signal_map.get(volume, '?')} |"

            rows.append(row)

        # 组合表格
        table = coin_info + f"{header}\n{separator}\n" + '\n'.join(rows)

        return table


# 使用示例
if __name__ == '__main__':
    # 配置API密钥
    binance_api_key = 'your_binance_api_key'
    binance_api_secret = 'your_binance_api_secret'
    openai_api_key = 'your_openai_api_key'

    # 初始化分析器
    analyst = CryptoTechnicalAnalyst(
        api_key=binance_api_key,
        api_secret=binance_api_secret,
    )

    # 分析比特币
    symbol = 'BTCUSDT'
    analyst.analyze_crypto(symbol)

    # 获取技术指标表格
    table = analyst.generate_multi_timeframe_table(symbol)
    print(table)

    # 获取LLM分析
    analysis = analyst.get_llm_analysis(symbol)
    print('\nLLM分析结果:')
    print(analysis)
