import json


def main(strategy_mode='short'):
    with open('user_data/config.json') as f:
        config = json.load(f)

    pairs = config['exchange']['pair_whitelist']

    strategy_state = {}
    strategy_state['pair_strategy_mode'] = {}
    for pair in pairs:
        strategy_state['pair_strategy_mode'][pair] = strategy_mode

    with open('user_data/strategy_state.json', 'w') as f:
        json.dump(strategy_state, f, indent=2)

        print('更新策略状态完成')


if __name__ == '__main__':
    main('short')
