import requests
import json
import pandas as pd
from typing import List, Dict
import shutil
import time
from datetime import datetime, timedelta

with open('./tools/black_list.json', 'r') as f:
    BLACKLIST_PAIRS = json.load(f)


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


def format_pair(symbol: str) -> str:
    """格式化交易对名称"""
    if symbol.endswith('USDT'):
        base = symbol[:-4]
        return f"{base}/USDT:USDT"
    return symbol


def filter_and_sort_pairs(
    min_volume: float = 100000000, min_listing_days: int = 180
) -> pd.DataFrame:
    """筛选并排序交易对，过滤掉上线时间少于指定天数的币种"""
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

    # 获取交易对状态和上线时间
    current_time = datetime.now()
    symbols_info = {}
    newly_listed = []

    for s in exchange_info['symbols']:
        symbols_info[s['symbol']] = s

        # 尝试获取上线时间
        try:
            # 如果API提供了上线时间字段（示例用onboardDate）
            if 'onboardDate' in s:
                listing_time = datetime.fromtimestamp(s['onboardDate'] / 1000)
            elif 'listingDate' in s:
                listing_time = datetime.fromtimestamp(s['listingDate'] / 1000)
            else:
                # 如果没有直接的上线时间字段，可能需要使用其他方法获取
                continue

            days_since_listing = (current_time - listing_time).days

            if days_since_listing < min_listing_days:
                newly_listed.append(s['symbol'])

        except (KeyError, TypeError):
            # 如果无法获取上线时间，跳过
            continue

    # 将上线时间少于指定天数的币种加入黑名单
    print(f"发现 {len(newly_listed)} 个上线不足 {min_listing_days} 天的币种")
    temp_blacklist = BLACKLIST_PAIRS.copy()
    for symbol in newly_listed:
        formatted = format_pair(symbol)
        if formatted not in temp_blacklist:
            temp_blacklist.append(formatted)

    df['status'] = df['symbol'].map(lambda x: symbols_info.get(x, {}).get('status', 'UNKNOWN'))

    # 筛选条件
    df = df[(df['quoteVolume'] > min_volume) & (df['status'] == 'TRADING')]  # 交易量筛选  # 状态筛选

    # 添加格式化的交易对名称
    df['formatted_symbol'] = df['symbol'].apply(format_pair)

    # 排除黑名单中的交易对
    df = df[~df['formatted_symbol'].isin(temp_blacklist)]

    # 排序并选择列
    df = df.sort_values('quoteVolume', ascending=False)

    result_df = df[
        ['symbol', 'formatted_symbol', 'quoteVolume', 'priceChangePercent', 'status']
    ].copy()

    # 转换交易量为百万USDT
    result_df['quoteVolume'] = result_df['quoteVolume'] / 1_000_000

    # 重命名列
    result_df.columns = ['原始交易对', '交易对', '24h交易量(百万USDT)', '24h涨跌幅(%)', '状态']

    return result_df


def update_config(pairs: List[str], config_path: str):
    """更新配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 备份当前配置
    backup_path = f"{config_path}.{int(time.time())}.bak"
    shutil.copy(config_path, backup_path)

    # 更新交易对列表
    config['exchange']['pair_whitelist'] = pairs

    # 保存新配置
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


def main():
    # 配置
    CONFIG_PATH = 'user_data/config.json'
    MIN_VOLUME = 100000000  # 1亿USDT
    TARGET_COUNT = 50  # 目标交易对数量
    MIN_LISTING_DAYS = 180  # 币种上线至少180天（约半年）

    # 获取并筛选交易对数据
    print('分析交易对...')
    df = filter_and_sort_pairs(min_volume=MIN_VOLUME, min_listing_days=MIN_LISTING_DAYS)

    # 如果交易对数量不足，减少交易量要求
    while len(df) < TARGET_COUNT and MIN_VOLUME > 1000000:  # 最低限制为100万USDT
        MIN_VOLUME = MIN_VOLUME * 0.8  # 每次减少20%
        print(f"\n交易对数量不足，降低交易量要求至: {MIN_VOLUME/1_000_000:.1f}M USDT")
        df = filter_and_sort_pairs(min_volume=MIN_VOLUME, min_listing_days=MIN_LISTING_DAYS)

    # 显示结果
    print('\n=== 交易对分析结果 ===')
    print(f"\n筛选条件:")
    print(f"- 最小24h交易量: {MIN_VOLUME/1_000_000:,.1f} 百万USDT")
    print(f"- 目标数量: {TARGET_COUNT}")
    print(f"- 已排除的交易对: {len(BLACKLIST_PAIRS)}个")
    print(f"- 最短上线时间: {MIN_LISTING_DAYS}天")

    print(f"\n符合条件的交易对数量: {len(df)}")
    print('\n=== 前10个交易对 ===')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
    print(df.head(10).to_string(index=False))

    # 确保获取指定数量的交易对
    selected_pairs = df.head(min(TARGET_COUNT, len(df)))['交易对'].tolist()
    output_path = update_config(selected_pairs, CONFIG_PATH)

    print(f"\n配置文件已更新: {output_path}")
    print(f"已添加 {len(selected_pairs)} 个交易对")

    # 保存分析结果
    csv_path = 'user_data/pairs_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n详细分析已保存到: {csv_path}")


if __name__ == '__main__':
    main()
