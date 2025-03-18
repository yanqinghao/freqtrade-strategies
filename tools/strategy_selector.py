import json


stable_pairs = [
    'BTC/USDT:USDT',
    'ETH/USDT:USDT',
    'BNB/USDT:USDT',
    'LTC/USDT:USDT',
    'ADA/USDT:USDT',
    'XRP/USDT:USDT',
    'XLM/USDT:USDT',
    'TRX/USDT:USDT',
    'USDC/USDT:USDT',
    'LINK/USDT:USDT',
    'USDS/USDT:USDT',
    'DOGE/USDT:USDT',
    'HBAR/USDT:USDT',
    'BCH/USDT:USDT',
    'SOL/USDT:USDT',
    'DOT/USDT:USDT',
    'AVAX/USDT:USDT',
    'OM/USDT:USDT',
    'SHIB/USDT:USDT',
    'WBTC/USDT:USDT',
    'SUI/USDT:USDT',
    'TON/USDT:USDT',
]


def parse_data(data_str, strategy_type):
    """Parse the CSV-like data string and return a dictionary of pairs to profit values"""
    result = {}

    # Clean the data string and handle potential CSV format issues
    lines = [line.strip() for line in data_str.strip().split('\n') if line.strip()]

    # Skip header line(s)
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('pair,profit_abs_sum') or line.startswith(',pair,profit_abs_sum'):
            start_idx = i + 1
            break

    for line in lines[start_idx:]:
        # Handle both formats: with and without index column
        parts = line.split(',')

        # Skip lines that don't have enough fields
        if len(parts) < 4:
            continue

        # Check if the first column is an index (purely numeric)
        if parts[0].strip().isdigit():
            # Format with index: idx,pair,profit_abs_sum,...
            pair = parts[1].strip()
            profit_abs_sum = float(parts[2].strip())
        else:
            # Format without index: pair,profit_abs_sum,...
            pair = parts[0].strip()
            profit_abs_sum = float(parts[1].strip())

        result[pair] = profit_abs_sum

    return result


def select_strategies(long_data, short_data, threshold=10):
    """
    Compare long and short strategies and select the best one for each pair.
    If long profit > short profit - threshold, select long; otherwise, select short.
    """
    # Combine all unique pairs
    all_pairs = set(long_data.keys()) | set(short_data.keys())

    strategies = {}
    for pair in all_pairs:
        long_profit = long_data.get(pair, 0)
        short_profit = short_data.get(pair, 0)

        # Apply the rule: if long profit > short profit - threshold, select long; otherwise, select short
        if long_profit > short_profit - threshold:
            strategies[pair] = 'long'
        else:
            strategies[pair] = 'short'

    return strategies


def main(threshold=-50):
    # For command-line usage, you can read from files
    with open('long.csv', 'r') as f:
        # For demonstration, paste your data directly here
        long_data_str = f.read()

    with open('short.csv', 'r') as f:
        short_data_str = f.read()

    with open('deploy/strategy_state.json', 'r') as f:
        strategy_state = json.load(f)

    coin_monitoring = strategy_state['coin_monitoring']

    # Parse the data
    long_data = parse_data(long_data_str, 'long')
    short_data = parse_data(short_data_str, 'short')

    # Select the strategies
    strategies = select_strategies(long_data, short_data, threshold=threshold)

    pairs = list(strategies.keys())

    # Build the output JSON
    output = {'pair_strategy_mode': strategies}

    output_with_monitoring = {'pair_strategy_mode': strategies, 'coin_monitoring': coin_monitoring}

    # Print to console
    json_str = json.dumps(output, indent=2)
    print(json_str)

    json_str_with_monitoring = json.dumps(output_with_monitoring, indent=2)
    print(json_str_with_monitoring)

    # Write to file
    with open('user_data/strategy_state.json', 'w') as f:
        f.write(json_str)

    with open('deploy/strategy_state.json', 'w') as f:
        f.write(json_str_with_monitoring)

    with open('user_data/config.json', 'r') as f:
        config = json.load(f)

    config['exchange']['pair_whitelist'] = pairs

    for pair in coin_monitoring:
        if pair not in config['exchange']['pair_whitelist'] and pair in stable_pairs:
            config['exchange']['pair_whitelist'].append(pair)

    with open('user_data/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("\nJSON file 'pair_strategy_mode.json' has been created successfully.")

    # Print statistics
    long_count = list(strategies.values()).count('long')
    short_count = list(strategies.values()).count('short')
    total = len(strategies)
    print('\nStatistics:')
    print(f"Total pairs: {total}")
    print(f"Long strategies: {long_count} ({long_count / total * 100:.1f}%)")
    print(f"Short strategies: {short_count} ({short_count / total * 100:.1f}%)")


if __name__ == '__main__':
    main()
