# all strategies are tested against this config. Tests only done on binance futures

```

{
  "max_open_trades": -1,
  "stake_currency": "USDT",
  "stake_amount": 100,
  "tradable_balance_ratio": 0.99,
  "fiat_display_currency": "USD",
  "dry_run": true,
  "cancel_open_orders_on_exit": false,
  "trading_mode": "futures",
  "margin_mode": "isolated",
  "unfilledtimeout": {
    "entry": 10,
    "exit": 10,
    "exit_timeout_count": 0,
    "unit": "minutes"
  },
  "entry_pricing": {
    "price_side": "same",
    "use_order_book": true,
    "order_book_top": 1,
    "price_last_balance": 0.0,
    "check_depth_of_market": {
      "enabled": false,
      "bids_to_ask_delta": 1
    }
  },
  "exit_pricing": {
    "price_side": "same",
    "use_order_book": true,
    "order_book_top": 1
  },
  "exchange": {
    "name": "binance",
    "key": "",
    "secret": "",
    "ccxt_config": {},
    "ccxt_async_config": {},
    "pair_whitelist": [
      "AUDIO/USDT",
      "AAVE/USDT",
      "ALICE/USDT",
      "ARPA/USDT",
      "AVAX/USDT",
      "ATOM/USDT",
      "ANKR/USDT",
      "AXS/USDT",
      "ADA/USDT",
      "ALGO/USDT",
      "BTS/USDT",
      "BAND/USDT",
      "BEL/USDT",
      "BNB/USDT",
      "BTC/USDT",
      "BLZ/USDT",
      "BAT/USDT",
      "CHR/USDT",
      "C98/USDT",
      "COTI/USDT",
      "CHZ/USDT",
      "COMP/USDT",
      "CRV/USDT",
      "CELO/USDT",
      "DUSK/USDT",
      "DOGE/USDT",
      "DENT/USDT",
      "DASH/USDT",
      "DOT/USDT",
      "DYDX/USDT",
      "ENJ/USDT",
      "EOS/USDT",
      "ETH/USDT",
      "ETC/USDT",
      "ENS/USDT",
      "EGLD/USDT",
      "FIL/USDT",
      "FTM/USDT",
      "FLM/USDT",
      "GRT/USDT",
      "GALA/USDT",
      "HBAR/USDT",
      "HOT/USDT",
      "IOTX/USDT",
      "ICX/USDT",
      "ICP/USDT",
      "IOTA/USDT",
      "IOST/USDT",
      "KLAY/USDT",
      "KAVA/USDT",
      "KNC/USDT",
      "KSM/USDT",
      "LUNA/USDT",
      "LRC/USDT",
      "LINA/USDT",
      "LTC/USDT",
      "LINK/USDT",
      "MATIC/USDT",
      "NEAR/USDT",
      "MANA/USDT",
      "MTL/USDT",
      "NEO/USDT",
      "ONT/USDT",
      "OMG/USDT",
      "OCEAN/USDT",
      "OGN/USDT",
      "ONE/USDT",
      "PEOPLE/USDT",
      "RLC/USDT",
      "RUNE/USDT",
      "RVN/USDT",
      "RSR/USDT",
      "REEF/USDT",
      "ROSE/USDT",
      "SNX/USDT",
      "SAND/USDT",
      "SOL/USDT",
      "SUSHI/USDT",
      "SRM/USDT",
      "SKL/USDT",
      "SXP/USDT",
      "STORJ/USDT",
      "TRX/USDT",
      "TOMO/USDT",
      "TRB/USDT",
      "TLM/USDT",
      "THETA/USDT",
      "UNI/USDT",
      "UNFI/USDT",
      "VET/USDT",
      "YFI/USDT",
      "ZIL/USDT",
      "ZEN/USDT",
      "ZRX/USDT",
      "ZEC/USDT",
      "WAVES/USDT",
      "XRP/USDT",
      "XLM/USDT",
      "XTZ/USDT",
      "XMR/USDT",
      "XEM/USDT",
      "QTUM/USDT",
      "1INCH/USDT"
    ],
    "pair_blacklist": ["BNB/.*"]
  },
  "pairlists": [{ "method": "StaticPairList" }],
  "edge": {
    "enabled": false,
    "process_throttle_secs": 3600,
    "calculate_since_number_of_days": 7,
    "allowed_risk": 0.01,
    "stoploss_range_min": -0.01,
    "stoploss_range_max": -0.1,
    "stoploss_range_step": -0.01,
    "minimum_winrate": 0.6,
    "minimum_expectancy": 0.2,
    "min_trade_number": 10,
    "max_trade_duration_minute": 1440,
    "remove_pumps": false
  },
  "telegram": {
    "enabled": true,
    "token": "",
    "chat_id": ""
  },
  "api_server": {
    "enabled": true,
    "listen_ip_address": "0.0.0.0",
    "listen_port": 8080,
    "verbosity": "error",
    "enable_openapi": false,
    "jwt_secret_key": "556ebba5770ae3a07e80eda0f0f2b55df102896f8a5b86459c3433c1314345c4",
    "CORS_origins": [],
    "username": "",
    "password": ""
  },
  "bot_name": "",
  "initial_state": "running",
  "force_entry_enable": false,
  "internals": {
    "process_throttle_secs": 5
  }
}

```

1. download data

```shell
freqtrade download-data --config user_data/config.json --timerange 20230101-20250110 --timeframe 15m 4h 1h
```

2. backtesting 

```shell
# 基础回测命令
freqtrade backtesting --config config.json --strategy OptimizedStrategy

# 指定时间范围的回测
freqtrade backtesting --config config.json --strategy OptimizedStrategy --timerange 20230101-20240101

# 指定交易对的回测
freqtrade backtesting --config config.json --strategy OptimizedStrategy --pairs BTC/USDT ETH/USDT

# 更详细的回测报告
freqtrade backtesting --config config.json --strategy OptimizedStrategy --timerange 20230101-20240101 --enable-full-position-metrics --enable-protections

# 修改路径
freqtrade backtesting --config user_data/config.json --strategy-path user_data/strategies/futures --strategy FOttStrategy --timerange 20241001-20250108

freqtrade backtesting --config user_data/config.json --strategy-path user_data/strategies/futures --strategy LeverageFOttStrategy --timerange 20241201-20250108 --breakdown day --export signals

freqtrade backtesting --config user_data/config.json --strategy-path user_data/strategies --strategy LeverageSupertrend --timerange 20241210-20250111 --breakdown day --export signals
```

3. 超参数优化

```shell
freqtrade hyperopt --config config.json --strategy OptimizedStrategy --epochs 100 --spaces buy sell --timerange 20230101-20240101
```

4. 结果打印

```shell
freqtrade backtesting-show --filename user_data/backtest_results/backtest-result.json
```

5. 结果分析

```shell
freqtrade backtesting-analysis -c user_data/config.json --filename user_data/backtest_results/backtest-result-2025-01-10_13-28-09.json --analysis-to-csv --analysis-groups 0 1 2 3 4 5
```
