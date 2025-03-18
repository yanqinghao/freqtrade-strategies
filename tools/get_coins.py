import requests

# 从 CoinGecko 获取前十币种
gecko_url = 'https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=30&page=1'
top_10 = [coin['symbol'].upper() for coin in requests.get(gecko_url).json()]

# 从 Binance 获取价格
binance_url = 'https://api.binance.com/api/v3/ticker/24hr'
binance_data = requests.get(binance_url).json()
pairs = []
for pair in binance_data:
    if pair['symbol'] in [f"{coin}USDT" for coin in top_10]:
        print(f"{pair['symbol']}: 价格={pair['lastPrice']}")
        pairs.append(pair['symbol'].replace('USDT', '/USDT:USDT'))
print(pairs)
