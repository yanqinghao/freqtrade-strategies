import requests
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import time
from pathlib import Path
import glob
import shutil


def update_config(pairs: List[str], config_path: str):
    """更新配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 备份当前配置
    backup_path = f"{config_path}.{int(time.time())}.bak"
    shutil.copy(config_path, backup_path)

    # 更新交易对列表
    config['exchange']['pair_whitelist'] = [p[:-4] + '/USDT:USDT' for p in pairs]

    # 保存新配置
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


def get_local_data(symbol: str, lookback_days: int = 30) -> Optional[pd.DataFrame]:
    """从本地feather文件获取历史数据"""
    # 构建查找模式 - 精确匹配4h数据文件
    base_path = Path('user_data/data/binance')
    pattern = f"{base_path}/**/{symbol.replace('USDT', '_')}*-4h-futures.feather"

    try:
        # 查找所有匹配的feather文件
        feather_files = glob.glob(pattern, recursive=True)

        if not feather_files:
            # 如果没有找到本地数据，直接返回None，表示该币种是新币种，不进行处理
            return None

        # 读取最新的feather文件
        df = pd.read_feather(feather_files[0])

        # 确保数据包含必要的列
        if 'date' not in df.columns or 'close' not in df.columns:
            return None

        # 将4h数据转换为日线数据
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df_daily = df['close'].resample('D').last().dropna()

        # 获取最近N天的数据
        end_date = df_daily.index.max()
        start_date = end_date - timedelta(days=lookback_days)
        df_daily = df_daily[start_date:end_date]

        if len(df_daily) < lookback_days / 2:  # 如果数据量不足期望天数的一半，返回None
            return None

        return df_daily

    except Exception as e:
        print(f"读取本地数据出错: {str(e)}")
        return None


def get_listing_date(symbol: str) -> Optional[datetime]:
    """获取币种上线时间，优先检查本地数据"""
    # 尝试从本地数据获取最早的日期
    base_path = Path('user_data/data/binance')
    pattern = f"{base_path}/**/*{symbol.replace('USDT', '')}*-4h-futures.feather"

    try:
        feather_files = glob.glob(pattern, recursive=True)
        if feather_files:
            earliest_date = None
            for file in feather_files:
                df = pd.read_feather(file)
                if 'date' in df.columns:
                    file_earliest = pd.to_datetime(df['date']).min()
                    if earliest_date is None or file_earliest < earliest_date:
                        earliest_date = file_earliest

            if earliest_date is not None:
                return earliest_date
    except Exception as e:
        print(f"读取本地数据获取上线时间出错: {str(e)}")

    # 如果本地数据不可用，则返回 None，不再请求 API
    print(f"{symbol} 是新币种，直接过滤掉。")
    return None


def get_historical_volatility(symbol: str, days: int = 30) -> float:
    """获取历史波动率，优先使用本地数据"""
    # 首先尝试获取本地数据
    df_local = get_local_data(symbol, days)

    if df_local is not None:
        # 使用本地数据计算波动率
        returns = np.diff(np.log(df_local.values))
        volatility = np.std(returns) * np.sqrt(365) * 100
        return volatility

    # 如果没有本地数据，跳过该币种
    print(f"{symbol} 是新币种，直接过滤掉。")
    return float('inf')  # 直接返回一个无限大的波动率值来过滤掉该币种


def get_futures_exchange_info() -> Dict:
    """获取合约市场所有交易对信息"""
    url = 'https://fapi.binance.com/fapi/v1/exchangeInfo'
    response = requests.get(url)
    return response.json()


def get_24h_ticker() -> List[Dict]:
    """获取24小时价格统计"""
    url = 'https://fapi.binance.com/fapi/v1/ticker/24hr'
    response = requests.get(url)
    return response.json()


def get_funding_rate_stats(symbol: str, days: int = 30) -> Dict:
    """获取资金费率统计数据"""
    end_time = int(time.time() * 1000)
    start_time = end_time - (days * 24 * 60 * 60 * 1000)

    url = 'https://fapi.binance.com/fapi/v1/fundingRate'
    params = {'symbol': symbol, 'startTime': start_time, 'endTime': end_time, 'limit': 1000}

    try:
        response = requests.get(url, params=params)
        rates = response.json()
        if not rates:
            return {'mean': float('inf'), 'std': float('inf'), 'max_abs': float('inf')}

        rates_values = [float(r['fundingRate']) * 100 for r in rates]  # 转换为百分比
        return {
            'mean': np.mean(rates_values),
            'std': np.std(rates_values),
            'max_abs': max(abs(min(rates_values)), abs(max(rates_values))),
        }
    except:
        return {'mean': float('inf'), 'std': float('inf'), 'max_abs': float('inf')}


def get_open_interest_stats(symbol: str) -> Dict:
    """获取持仓量统计"""
    url = 'https://fapi.binance.com/fapi/v1/openInterest'
    params = {'symbol': symbol}

    try:
        response = requests.get(url, params=params)
        data = response.json()
        oi = float(data['openInterest'])

        # 获取当前价格
        ticker_url = f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={symbol}"
        price_response = requests.get(ticker_url)
        price = float(price_response.json()['price'])

        # 计算美元价值
        oi_value = oi * price
        return {'open_interest_usd': oi_value}
    except:
        return {'open_interest_usd': 0}


def filter_and_sort_pairs(
    min_volume: float = 100000000,  # 最小交易量
    max_volatility: float = 80,  # 最大年化波动率
    min_days: int = 365,  # 最小上线天数
    stable_period: int = 30,  # 稳定性检查期限(天)
) -> pd.DataFrame:
    """筛选并排序交易对"""
    print('获取交易所基本信息...')
    exchange_info = get_futures_exchange_info()

    print('获取24小时行情...')
    tickers = get_24h_ticker()

    # 创建DataFrame
    df = pd.DataFrame(tickers)

    # 筛选USDT交易对
    df = df[df['symbol'].str.endswith('USDT')]

    # 转换数据类型
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
    df['quoteVolume'] = pd.to_numeric(df['quoteVolume'], errors='coerce')
    df['priceChangePercent'] = pd.to_numeric(df['priceChangePercent'], errors='coerce')

    # 获取交易对状态
    symbols_info = {s['symbol']: s for s in exchange_info['symbols']}
    df['status'] = df['symbol'].map(lambda x: symbols_info.get(x, {}).get('status', 'UNKNOWN'))

    print('获取上线时间和波动率信息...')
    listing_dates = {}
    volatilities = {}

    for symbol in df['symbol']:
        print(f"处理 {symbol}...")

        # 获取本地数据
        df_local = get_local_data(symbol, stable_period)

        # 如果没有本地数据，跳过该币种
        if df_local is None:
            print(f"{symbol} 没有本地数据，跳过此币种。")
            continue

        # 如果有本地数据，计算波动率和上线时间
        listing_dates[symbol] = get_listing_date(symbol)
        volatilities[symbol] = get_historical_volatility(symbol, stable_period)
        time.sleep(0.1)  # 避免请求频率过高

    df['listing_date'] = df['symbol'].map(listing_dates)
    df = df.dropna(subset=['listing_date'])
    df['listing_date'] = df['listing_date'].dt.tz_localize(None)
    df['days_listed'] = (datetime.now().replace(tzinfo=None) - df['listing_date']).dt.days
    df['volatility'] = df['symbol'].map(volatilities)
    # 筛选条件
    df = df[
        (df['quoteVolume'] > min_volume)
        & (df['status'] == 'TRADING')  # 交易量筛选
        & (df['days_listed'] >= min_days)  # 状态筛选
        & (df['volatility'] <= max_volatility)  # 上线时间筛选  # 波动率筛选
    ]

    # 排序规则：先按照上线时间排序，再按交易量排序
    df = df.sort_values(['days_listed', 'quoteVolume'], ascending=[False, False])

    result_df = df[
        ['symbol', 'quoteVolume', 'priceChangePercent', 'volatility', 'days_listed', 'status']
    ].copy()

    # 转换交易量为百万USDT
    result_df['quoteVolume'] = result_df['quoteVolume'] / 1_000_000

    # 重命名列
    result_df.columns = ['交易对', '24h交易量(百万USDT)', '24h涨跌幅(%)', '年化波动率(%)', '上线天数', '状态']

    return result_df


def advanced_filter_pairs(
    min_volume: float = 100000000,  # 最小交易量
    max_volatility: float = 80,  # 最大年化波动率
    min_days: int = 365,  # 最小上线天数
    min_oi_usd: float = 10000000,  # 最小持仓量(USD)
    max_funding_rate: float = 0.1,  # 最大平均资金费率(绝对值)
    max_funding_std: float = 0.05,  # 最大资金费率标准差
    stability_weight: float = 0.3,  # 稳定性权重
    volume_weight: float = 0.3,  # 交易量权重
    age_weight: float = 0.2,  # 上线时间权重
    oi_weight: float = 0.2,  # 持仓量权重
) -> pd.DataFrame:
    """使用多维度指标筛选老牌币"""

    print('获取基础市场数据...')
    df = filter_and_sort_pairs(
        min_volume=min_volume, max_volatility=max_volatility, min_days=min_days
    )

    print('获取额外指标...')
    funding_stats = {}
    oi_stats = {}

    for symbol in df['交易对']:
        print(f"处理 {symbol}...")
        funding_stats[symbol] = get_funding_rate_stats(symbol)
        oi_stats[symbol] = get_open_interest_stats(symbol)
        time.sleep(0.1)  # 避免请求频率过高

    # 添加新指标
    df['持仓量(USD)'] = df['交易对'].map(lambda x: oi_stats[x]['open_interest_usd'])
    df['平均资金费率'] = df['交易对'].map(lambda x: funding_stats[x]['mean'])
    df['资金费率波动'] = df['交易对'].map(lambda x: funding_stats[x]['std'])
    df['最大资金费率'] = df['交易对'].map(lambda x: funding_stats[x]['max_abs'])

    # 筛选条件
    df = df[
        (df['持仓量(USD)'] >= min_oi_usd)
        & (abs(df['平均资金费率']) <= max_funding_rate)
        & (df['资金费率波动'] <= max_funding_std)
    ]

    # 计算综合得分
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        return (series - min_val) / (max_val - min_val) if max_val > min_val else series

    # 稳定性得分 (越低越好)
    stability_score = normalize(1 / df['年化波动率(%)']) * 0.5 + normalize(1 / df['资金费率波动']) * 0.5

    # 交易活跃度得分
    volume_score = normalize(df['24h交易量(百万USDT)'])

    # 币龄得分
    age_score = normalize(df['上线天数'])

    # 持仓量得分
    oi_score = normalize(df['持仓量(USD)'])

    # 综合得分
    df['综合得分'] = (
        stability_score * stability_weight
        + volume_score * volume_weight
        + age_score * age_weight
        + oi_score * oi_weight
    )

    # 最终排序
    df = df.sort_values('综合得分', ascending=False)

    # 格式化数值
    df['持仓量(百万USD)'] = df['持仓量(USD)'] / 1_000_000
    df = df.drop('持仓量(USD)', axis=1)

    # 重新排列列顺序
    columns = [
        '交易对',
        '综合得分',
        '24h交易量(百万USDT)',
        '持仓量(百万USD)',
        '年化波动率(%)',
        '平均资金费率',
        '资金费率波动',
        '上线天数',
        '24h涨跌幅(%)',
        '状态',
    ]

    return df[columns]


def main():
    # 配置参数
    CONFIG = {
        'min_volume': 5000000,  # 5千万USDT日交易量
        'max_volatility': 100,  # 80%年化波动率
        'min_days': 180,  # 至少上线一年
        'min_oi_usd': 10000000,  # 1千万USD持仓量
        'max_funding_rate': 0.1,  # 0.1%平均资金费率
        'max_funding_std': 0.05,  # 0.05%资金费率标准差
        'stability_weight': 0.3,  # 稳定性权重
        'volume_weight': 0.3,  # 交易量权重
        'age_weight': 0.2,  # 上线时间权重
        'oi_weight': 0.2,  # 持仓量权重
    }

    print('开始分析交易对...')
    df = advanced_filter_pairs(**CONFIG)

    print('\n=== 交易对分析结果 ===')
    print(f"\n筛选条件:")
    print(f"- 最小24h交易量: {CONFIG['min_volume']/1_000_000:,.1f} 百万USDT")
    print(f"- 最大年化波动率: {CONFIG['max_volatility']}%")
    print(f"- 最小上线天数: {CONFIG['min_days']}天")
    print(f"- 最小持仓量: {CONFIG['min_oi_usd']/1_000_000:,.1f} 百万USD")
    print(f"- 最大平均资金费率: {CONFIG['max_funding_rate']}%")
    print(f"- 最大资金费率波动: {CONFIG['max_funding_std']}%")

    print(f"\n符合条件的交易对数量: {len(df)}")
    print('\n=== 前15个交易对 ===')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
    print(df.head(15).to_string(index=False))

    # 确保获取指定数量的交易对
    selected_pairs = df.head(30)['交易对'].tolist()
    output_path = update_config(selected_pairs, 'user_data/config.json')

    print(f"\n配置文件已更新: {output_path}")
    print(f"已添加 {len(selected_pairs)} 个交易对")

    # 保存分析结果
    csv_path = 'user_data/pairs_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n详细分析已保存到: {csv_path}")


if __name__ == '__main__':
    main()
