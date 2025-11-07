import requests
import json
import pandas as pd
from typing import List, Dict, Tuple, Set
import shutil
import time
from datetime import datetime
import os

# ===== Paths & Constants =====
VOL_BLACKLIST_PATH = './tools/volatility_blacklist.json'  # 30天临时黑名单（自动创建/维护）
VOL_BLACKLIST_DAYS = 30

# Load permanent blacklist
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


def get_top_coins_by_market_cap(limit: int = 100) -> Dict:
    """从CoinGecko获取市值排名前N的加密货币"""
    coins_per_page = 250  # CoinGecko API限制每页最多250个
    pages_needed = (limit + coins_per_page - 1) // coins_per_page

    all_coins = []
    for page in range(1, pages_needed + 1):
        url = f'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page={coins_per_page}&page={page}'
        print(f'获取CoinGecko市值数据，页面 {page}/{pages_needed}...')
        try:
            response = requests.get(url)
            response.raise_for_status()
            coins = response.json()
            all_coins.extend(coins)
        except Exception as e:
            print(f"获取CoinGecko数据时出错: {e}")
            break
    return all_coins[:limit]


def format_pair(symbol: str) -> str:
    """格式化交易对名称"""
    if symbol.endswith('USDT'):
        base = symbol[:-4]
        return f"{base}/USDT:USDT"
    return symbol


def create_market_cap_dict(coins: List[Dict]) -> Dict[str, float]:
    """创建从币种符号到市值的映射字典"""
    market_cap_dict = {}
    for coin in coins:
        symbol = coin['symbol'].upper()
        market_cap = coin.get('market_cap', 0)
        market_cap_dict[symbol] = market_cap
    return market_cap_dict


# ===== 30天临时黑名单（基于日内大波动）维护 =====
def _today_str() -> str:
    return datetime.now().date().isoformat()


def load_vol_blacklist(path: str = VOL_BLACKLIST_PATH) -> Dict[str, str]:
    """加载30天临时黑名单：{ formatted_symbol: last_flagged_date_str }"""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # 兜底：确保类型正确
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
            return {}
    except Exception:
        return {}


def save_vol_blacklist(black: Dict[str, str], path: str = VOL_BLACKLIST_PATH):
    """保存30天临时黑名单"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(black, f, ensure_ascii=False, indent=2)


def prune_expired_vol_blacklist(
    black: Dict[str, str], days: int = VOL_BLACKLIST_DAYS
) -> Dict[str, str]:
    """清理超过 days 天未再次触发的条目"""
    keep = {}
    today = datetime.now().date()
    for sym, date_str in black.items():
        try:
            last = datetime.fromisoformat(date_str).date()
            if (today - last).days <= days:
                keep[sym] = date_str
        except Exception:
            # 解析失败则丢弃
            pass
    return keep


def merge_new_flagged(black: Dict[str, str], flagged: Set[str]):
    """将今日新触发的币对写回/刷新为今天"""
    today = _today_str()
    for sym in flagged:
        black[sym] = today


def active_vol_blacklist_set(black: Dict[str, str], days: int = VOL_BLACKLIST_DAYS) -> Set[str]:
    """返回当前仍在封禁期内的 formatted_symbol 集合"""
    s = set()
    today = datetime.now().date()
    for sym, date_str in black.items():
        try:
            last = datetime.fromisoformat(date_str).date()
            if (today - last).days <= days:
                s.add(sym)
        except Exception:
            continue
    return s


def filter_and_sort_pairs(
    top_coins,
    min_volume: float = 100000000,
    min_listing_days: int = 180,
    max_24h_change: float = 12.0,
    active_vol_blacklist: Set[str] = None,  # 当前仍在封禁期内的 formatted_symbol
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    筛选并排序交易对，基于交易量、上线时间、市值以及单日涨跌幅。
    返回：(结果DataFrame, 今日因涨跌幅超阈值而被过滤的 formatted_symbol 集合)
    """
    print('获取交易所基本信息...')
    exchange_info = get_futures_exchange_info()

    print('获取24小时行情...')
    tickers = get_24h_ticker()

    print('获取市值排名数据...')
    market_cap_dict = create_market_cap_dict(top_coins)
    top_market_cap_symbols = [coin['symbol'].upper() for coin in top_coins]

    df = pd.DataFrame(tickers)
    df = df[df['symbol'].str.endswith('USDT')]

    df['volume'] = pd.to_numeric(df.get('volume', 0), errors='coerce')
    df['quoteVolume'] = pd.to_numeric(df.get('quoteVolume', 0), errors='coerce')
    df['priceChangePercent'] = pd.to_numeric(
        df.get('priceChangePercent', 0), errors='coerce'
    ).fillna(0.0)

    # 上线时间筛
    current_time = datetime.now()
    symbols_info = {}
    newly_listed = []
    for s in exchange_info['symbols']:
        symbols_info[s['symbol']] = s
        try:
            if 'onboardDate' in s:
                listing_time = datetime.fromtimestamp(s['onboardDate'] / 1000)
            elif 'listingDate' in s:
                listing_time = datetime.fromtimestamp(s['listingDate'] / 1000)
            else:
                continue
            days_since_listing = (current_time - listing_time).days
            if days_since_listing < min_listing_days:
                newly_listed.append(s['symbol'])
        except (KeyError, TypeError):
            continue

    print(f"发现 {len(newly_listed)} 个上线不足 {min_listing_days} 天的币种")
    temp_blacklist = BLACKLIST_PAIRS.copy()
    for symbol in newly_listed:
        formatted = format_pair(symbol)
        if formatted not in temp_blacklist:
            temp_blacklist.append(formatted)

    df['status'] = df['symbol'].map(lambda x: symbols_info.get(x, {}).get('status', 'UNKNOWN'))
    df['base_symbol'] = df['symbol'].apply(lambda x: x[:-4] if x.endswith('USDT') else x)
    df['market_cap'] = df['base_symbol'].map(lambda x: market_cap_dict.get(x, 0))
    df['formatted_symbol'] = df['symbol'].apply(format_pair)

    # 先做基础过滤：交易量、状态
    df_base = df[(df['quoteVolume'] > min_volume) & (df['status'] == 'TRADING')]

    # —— 记录“涨跌幅超阈值”的集合（今天需要写回临时黑名单）
    violated = df_base[df_base['priceChangePercent'].abs() > float(max_24h_change)]
    flagged_today: Set[str] = set(violated['formatted_symbol'].tolist())

    # 再应用“涨跌幅”过滤
    df = df_base[df_base['priceChangePercent'].abs() <= float(max_24h_change)]

    # 市值过滤
    df = df[df['base_symbol'].isin(top_market_cap_symbols) | (df['market_cap'] > 0)]

    # 合并永久黑名单 + 当前仍生效的“波动30天黑名单”
    if active_vol_blacklist:
        merged_black = set(temp_blacklist) | set(active_vol_blacklist)
    else:
        merged_black = set(temp_blacklist)

    df = df[~df['formatted_symbol'].isin(merged_black)]

    # 排序
    df = df.sort_values(['market_cap', 'quoteVolume'], ascending=[False, False])

    result_df = df[
        ['symbol', 'formatted_symbol', 'quoteVolume', 'market_cap', 'priceChangePercent', 'status']
    ].copy()
    result_df['quoteVolume'] = result_df['quoteVolume'] / 1_000_000
    result_df['market_cap'] = result_df['market_cap'] / 1_000_000
    result_df.columns = ['原始交易对', '交易对', '24h交易量(百万USDT)', '市值(百万USDT)', '24h涨跌幅(%)', '状态']

    return result_df, flagged_today


def update_config(pairs: List[str], config_path: str):
    """更新配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    backup_path = f"{config_path}.{int(time.time())}.bak"
    shutil.copy(config_path, backup_path)

    config['exchange']['pair_whitelist'] = pairs

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config_path


def main():
    # 配置
    CONFIG_PATH = 'user_data/config.json'
    MIN_VOLUME = 100000000  # 1亿USDT
    TARGET_COUNT = 80
    MIN_LISTING_DAYS = 180
    MARKET_CAP_LIMIT = 200
    MAX_24H_CHANGE = 12.0  # 单日最大涨跌幅（绝对值）阈值，建议10~15%，默认12%

    # === 加载与预处理 30 天临时黑名单 ===
    vol_black = load_vol_blacklist()
    vol_black = prune_expired_vol_blacklist(vol_black, VOL_BLACKLIST_DAYS)
    active_vol_black = active_vol_blacklist_set(vol_black, VOL_BLACKLIST_DAYS)

    # 获取并筛选交易对数据
    print('分析交易对...')
    top_coins = get_top_coins_by_market_cap(limit=MARKET_CAP_LIMIT)

    df, flagged_today = filter_and_sort_pairs(
        top_coins=top_coins,
        min_volume=MIN_VOLUME,
        min_listing_days=MIN_LISTING_DAYS,
        max_24h_change=MAX_24H_CHANGE,
        active_vol_blacklist=active_vol_black,
    )

    # 如果交易对数量不足，降低交易量阈值重试（保持同一套黑名单/阈值策略）
    while len(df) < TARGET_COUNT and MIN_VOLUME > 100000000 * 0.3:
        MIN_VOLUME = int(MIN_VOLUME * 0.8)
        print(f"\n交易对数量不足，降低交易量要求至: {MIN_VOLUME / 1_000_000:.1f}M USDT")
        df, flagged_retry = filter_and_sort_pairs(
            top_coins=top_coins,
            min_volume=MIN_VOLUME,
            min_listing_days=MIN_LISTING_DAYS,
            max_24h_change=MAX_24H_CHANGE,
            active_vol_blacklist=active_vol_black,
        )
        # 兼容：在更低门槛的重试中也可能出现新被“涨跌幅”过滤的币
        flagged_today |= flagged_retry

    # === 写回/刷新 当天触发的“涨跌幅超阈值”临时黑名单 ===
    if flagged_today:
        merge_new_flagged(vol_black, flagged_today)
        save_vol_blacklist(vol_black)
        print(
            f"\n已更新临时黑名单（{VOL_BLACKLIST_DAYS}天）：{len(flagged_today)} 个币对写入/刷新 -> {VOL_BLACKLIST_PATH}"
        )
    else:
        print('\n今日无新触发的临时黑名单币对。')

    # 显示结果
    print('\n=== 交易对分析结果 ===')
    print('\n筛选条件:')
    print(f"- 最小24h交易量: {MIN_VOLUME / 1_000_000:,.1f} 百万USDT")
    print(f"- 市值排名限制: 前 {MARKET_CAP_LIMIT}")
    print(f"- 目标数量: {TARGET_COUNT}")
    print(f"- 永久黑名单: {len(BLACKLIST_PAIRS)} 个")
    print(f"- 最短上线时间: {MIN_LISTING_DAYS} 天")
    print(f"- 单日涨跌幅阈值: ±{MAX_24H_CHANGE} %")
    print(f"- 临时黑名单在封禁期内: {len(active_vol_black)} 个（{VOL_BLACKLIST_DAYS}天内）")

    print(f"\n符合条件的交易对数量: {len(df)}")
    print('\n=== 前10个交易对 ===')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', lambda x: '{:,.2f}'.format(x))
    print(df.head(10).to_string(index=False))

    # 更新配置
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
