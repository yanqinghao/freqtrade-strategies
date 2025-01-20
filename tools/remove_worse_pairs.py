import pandas as pd
import json
import shutil
import time
from typing import List


def update_config(pairs: List[str], config_path: str):
    """更新配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    # 备份当前配置
    backup_path = f'{config_path}.{int(time.time())}.bak'
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
    CSV_PATH = 'user_data/backtest_results/group_3.csv'
    REMOVE_THRESHOLD = -10  # 仅删除表现特别差的交易对
    MAX_PAIRS = 30  # 最多选择70个交易对

    # 读取CSV文件
    print(f'读取交易对数据: {CSV_PATH}')
    df = pd.read_csv(CSV_PATH)
    df = (
        df.groupby('pair')
        .agg({'profit_abs_sum': 'sum', 'num_buys': 'sum', 'mean_profit_pct': 'sum'})
        .reset_index()
    )

    # 记录原始交易对数量
    total_pairs = len(df)

    # 找出表现特别差的交易对
    bad_pairs = df[df['profit_abs_sum'] < REMOVE_THRESHOLD]

    # 保留收益大于阈值的交易对
    good_pairs = df[df['profit_abs_sum'] >= REMOVE_THRESHOLD]

    # 按照收益排序并只取前30个
    good_pairs = good_pairs.sort_values('profit_abs_sum', ascending=False).head(MAX_PAIRS)

    # 显示统计信息
    print('\n=== 交易对分析结果 ===')
    print(f'总交易对数量: {total_pairs}')
    print(f"符合收益阈值的交易对数量: {len(df[df['profit_abs_sum'] >= REMOVE_THRESHOLD])}")
    print(f'最终选择的交易对数量: {len(good_pairs)}')
    print(f'移除的交易对数量: {len(bad_pairs)}')

    if len(bad_pairs) > 0:
        print('\n已移除的交易对（表现特别差）:')
        print(
            bad_pairs[['pair', 'num_buys', 'profit_abs_sum', 'mean_profit_pct']].to_string(
                index=False
            )
        )

    print('\n选择的前30个交易对:')
    print(
        good_pairs[['pair', 'num_buys', 'profit_abs_sum', 'mean_profit_pct']].to_string(index=False)
    )

    # 获取要保留的交易对列表
    selected_pairs = good_pairs['pair'].tolist()

    # 更新配置文件
    output_path = update_config(selected_pairs, CONFIG_PATH)

    print(f'\n配置文件已更新: {output_path}')
    print(f'已保留 {len(selected_pairs)} 个交易对')


if __name__ == '__main__':
    main()
