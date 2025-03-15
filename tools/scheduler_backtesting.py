import subprocess
import datetime
import json
import os
import requests
from dotenv import load_dotenv
from update_pairs import main as update_pairs_main
from strategy_selector import main as strategy_selector_main
from update_strategy_state import main as update_strategy_state_main
from remove_worse_pairs import main as remove_worse_pairs_main
from remove_worse_pairs import data as good_pairs_data
from daily_agent import main as daily_agent_main

# 加载.env文件
load_dotenv()


class FreqtradeScheduler:
    def __init__(self, config_path, strategy_path, docker_compose_path, backtesting_strategy):
        self.config_path = config_path
        self.strategy_path = strategy_path
        self.docker_compose_path = docker_compose_path
        self.tg_bot_token = os.getenv('TG_BOT_TOKEN')
        self.tg_chat_id = os.getenv('TG_CHAT_ID')
        self.backtesting_strategy = backtesting_strategy
        self.bot_config_path = f"{docker_compose_path}/deploy/config.json"

    def run_command(self, command):
        """运行shell命令并返回输出"""
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"命令执行失败: {e}")
            return None

    def update_pairs(self):
        """更新交易对"""
        print('正在更新交易对...')
        update_pairs_main()

    def update_strategy_mode(self, strategy_mode):
        """更新交易对"""
        print(f'正在更新交易对策略为{strategy_mode}...')
        update_strategy_state_main(strategy_mode)

    def download_data(self):
        """下载最新数据"""
        today = datetime.datetime.now().strftime('%Y%m%d')
        start_date = os.getenv('START_DATE', '20240101')
        command = (
            f".venv/bin/python -m freqtrade download-data --config {self.config_path} "
            f"--timerange {start_date}-{today} --timeframe 5m 15m 4h 1h 1d"
        )
        print('正在下载数据...')
        self.run_command(command)

    def run_backtesting(self, timerange=None):
        """运行回测并处理无法回测的交易对"""
        if not timerange:
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=30)
            timerange = f"{start_date.strftime('%Y%m%d')}-{end_date.strftime('%Y%m%d')}"

        command = (
            f".venv/bin/python -m freqtrade backtesting --config {self.config_path} --cache none "
            f"--strategy-path {self.strategy_path} "
            f"--strategy {self.backtesting_strategy} "
            f"--timerange {timerange} --breakdown day --export signals"
        )

        print('正在运行回测...')
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        # 检查输出中是否有无法回测的交易对
        output = result.stdout + result.stderr
        problematic_pairs = []
        for line in output.split('\n'):
            if 'got no leverage tiers available' in line:
                # 提取交易对名称
                pairs = [
                    i.strip()
                    for i in line.split('Pairs ')[1].split(' got no')[0].strip().split(',')
                ]
                problematic_pairs.extend(pairs)

        if problematic_pairs:
            print(f"发现无法回测的交易对: {problematic_pairs}")
            # 从配置文件中移除这些交易对
            self.remove_pairs_from_config(problematic_pairs)
            # 更新黑名单
            self.update_blacklist(problematic_pairs)
            # 重新运行回测
            print('重新运行回测...')
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            return result.stdout

        return result.stdout if result.returncode == 0 else None

    def remove_pairs_from_config(self, pairs_to_remove):
        """从配置文件中移除指定的交易对"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # 移除交易对
            current_pairs = config['exchange']['pair_whitelist']
            new_pairs = [pair for pair in current_pairs if pair not in pairs_to_remove]
            config['exchange']['pair_whitelist'] = new_pairs

            # 保存更新后的配置
            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=4)

            print(f"已从配置文件中移除交易对: {pairs_to_remove}")
        except Exception as e:
            print(f"更新配置文件失败: {e}")

    def update_blacklist(self, new_pairs):
        """更新黑名单文件"""
        try:
            blacklist_file = 'tools/black_list.json'
            with open(blacklist_file, 'r') as f:
                blacklist = json.load(f)

            blacklist = blacklist + new_pairs

            # 保存更新后的文件
            with open(blacklist_file, 'w') as f:
                json.dump(blacklist, f, indent=4)

            print(f"已更新黑名单: {new_pairs}")
        except Exception as e:
            print(f"更新黑名单失败: {e}")

    def analyze_backtesting(self):
        """分析回测结果"""
        command = (
            f".venv/bin/python -m freqtrade backtesting-analysis -c {self.config_path} "
            '--analysis-to-csv --analysis-groups 0 1 2 3 4 5'
        )
        print('正在分析回测结果...')
        self.run_command(command)

    def remove_worse_pairs(self):
        """移除表现差的交易对"""
        print('正在移除表现差的交易对...')
        remove_worse_pairs_main()

    def select_strategies(self, threshold=-50):
        """更新交易对策略"""
        print('更新交易对策略...')
        strategy_selector_main(threshold)

    def get_good_pairs(self, strategy_mode):
        """获取表现好的交易对"""
        print('正在获取表现好的交易对...')
        good_pairs_data(strategy_mode)

    def send_telegram_message(self, message):
        """发送Telegram消息"""
        if not self.tg_bot_token or not self.tg_chat_id:
            print('未设置Telegram配置')
            return

        url = f"https://api.telegram.org/bot{self.tg_bot_token}/sendMessage"
        data = {'chat_id': self.tg_chat_id, 'text': message, 'parse_mode': 'HTML'}
        try:
            response = requests.post(url, json=data)
            response.raise_for_status()
        except Exception as e:
            print(f"发送Telegram消息失败: {e}")

    def update_config_pairs(self):
        """更新配置文件中的交易对"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)

            # 从pair_list.json读取最新的交易对
            with open(self.bot_config_path, 'r') as f:
                bot_config = json.load(f)

            # 只更新exchange部分的pair_list
            if 'exchange' in bot_config:
                bot_config['exchange']['pair_whitelist'] = config['exchange']['pair_whitelist']

            # 保存更新后的配置
            with open(self.bot_config_path, 'w') as f:
                json.dump(bot_config, f, indent=4)

            print('配置文件更新成功')
        except Exception as e:
            print(f"更新配置文件失败: {e}")

    def restart_bot(self):
        """重启机器人"""
        print('正在重启机器人...')
        self.run_command(f"cd {self.docker_compose_path} && docker compose down")
        self.run_command(f"cd {self.docker_compose_path} && docker compose up -d")

    def extract_key_metrics(self, backtesting_output):
        """从回测输出中提取关键统计信息并格式化"""
        try:
            lines = backtesting_output.split('\n')

            # 主要统计信息
            found_summary = False
            key_metrics = {
                'Total/Daily Avg Trades': None,
                'Starting balance': None,
                'Final balance': None,
                'Absolute profit': None,
                'Total profit %': None,
                'Avg. daily profit %': None,
                'Best Pair': None,
                'Worst Pair': None,
                'Best trade': None,
                'Worst trade': None,
                # 'Long / Short': None,
                # 'Total profit Long %': None,
                # 'Total profit Short %': None,
                'Market change': None,
                'Max % of account underwater': None,
                'Absolute Drawdown': None,
            }

            # 回撤信息
            drawdown_info = {
                'Drawdown Start': None,
                'Drawdown End': None,
                'Drawdown high': None,
                'Drawdown low': None,
            }

            # 解析主要统计信息
            for line in lines:
                if 'SUMMARY METRICS' in line:
                    found_summary = True
                    continue

                if found_summary:
                    # 记录回撤信息
                    if any(key in line for key in drawdown_info.keys()) and '│' in line:
                        key = next(k for k in drawdown_info.keys() if k in line)
                        drawdown_info[key] = line.split('│')[2].strip()

                    # 记录其他指标
                    for key in key_metrics.keys():
                        if key in line and '│' in line and not key_metrics[key]:
                            key_metrics[key] = line.split('│')[2].strip()

            return key_metrics, drawdown_info
        except Exception:
            return None, None

    def extract_summary_stats(self, backtesting_output):
        """从回测输出中提取关键统计信息并格式化"""
        try:
            lines = backtesting_output.split('\n')
            message = '======= 📊 回测汇总统计 =======\n\n'

            # 主要统计信息
            found_summary = False
            key_metrics = {
                'Total/Daily Avg Trades': None,
                'Starting balance': None,
                'Final balance': None,
                'Absolute profit': None,
                'Total profit %': None,
                'Avg. daily profit %': None,
                'Best Pair': None,
                'Worst Pair': None,
                'Best trade': None,
                'Worst trade': None,
                # 'Long / Short': None,
                # 'Total profit Long %': None,
                # 'Total profit Short %': None,
                'Market change': None,
                'Max % of account underwater': None,
                'Absolute Drawdown': None,
            }

            # 回撤信息
            drawdown_info = {
                'Drawdown Start': None,
                'Drawdown End': None,
                'Drawdown high': None,
                'Drawdown low': None,
            }

            # 解析主要统计信息
            for line in lines:
                if 'SUMMARY METRICS' in line:
                    found_summary = True
                    continue

                if found_summary:
                    # 记录回撤信息
                    if any(key in line for key in drawdown_info.keys()) and '│' in line:
                        key = next(k for k in drawdown_info.keys() if k in line)
                        drawdown_info[key] = line.split('│')[2].strip()

                    # 记录其他指标
                    for key in key_metrics.keys():
                        if key in line and '│' in line and not key_metrics[key]:
                            key_metrics[key] = line.split('│')[2].strip()

            # 格式化主要统计信息
            message += '💰 收益统计:\n'
            message += f"• 总收益: {key_metrics['Absolute profit']}\n"
            message += f"• 总收益率: {key_metrics['Total profit %']}\n"
            message += f"• 总交易次数: {key_metrics['Total/Daily Avg Trades']}\n"
            message += f"• 日均收益率: {key_metrics['Avg. daily profit %']}\n"
            message += (
                f"• 账户变化: {key_metrics['Starting balance']} → {key_metrics['Final balance']}\n\n"
            )

            message += '📉 回撤统计:\n'
            message += f"• 账户回撤率: {key_metrics['Max % of account underwater']}\n"
            message += f"• 最大回撤额: {key_metrics['Absolute Drawdown']}\n"
            message += (
                f"• 回撤区间: {drawdown_info['Drawdown high']} → {drawdown_info['Drawdown low']}\n"
            )
            message += (
                f"• 回撤时间: {drawdown_info['Drawdown Start']} → {drawdown_info['Drawdown End']}\n"
            )
            message += f"• 市场变化: {key_metrics['Market change']}\n\n"

            # # 多空统计
            # message += "🔄 多空分布:\n"
            # message += f"• 多/空: {key_metrics['Long / Short']}\n"
            # message += f"• 多头收益率: {key_metrics['Total profit Long %']}\n"
            # message += f"• 空头收益率: {key_metrics['Total profit Short %']}\n\n"

            # 提取退出原因统计
            message += '🚪 退出原因统计:\n'
            found_exit = False
            for line in lines:
                if 'EXIT REASON STATS' in line:
                    found_exit = True
                    continue
                if (
                    found_exit
                    and '│' in line
                    and 'Exit Reason' not in line
                    and 'TOTAL' not in line
                    and '──────' not in line
                ):
                    parts = [p.strip() for p in line.split('│') if p.strip()]
                    if len(parts) >= 4:  # 确保有足够的部分
                        reason = parts[0]
                        count = parts[1]
                        profit = parts[3]
                        # 翻译常见退出原因
                        reason_map = {
                            'roi': 'ROI止盈',
                            'exit_signal': '信号退出',
                            'force_exit': '强制退出',
                            'trailing_stop_loss': '追踪止损',
                            'force_exit_time_profit_range': '超时/利润区间退出',
                        }
                        reason = reason_map.get(reason.lower(), reason)
                        message += f"• {reason}: {count}次 ({profit})\n"
                if found_exit and '──────' in line and found_exit:
                    break

            # 提取每日统计
            message += '\n📅 每日收益:\n'
            found_daily = False
            for line in lines:
                if 'DAY BREAKDOWN' in line:
                    found_daily = True
                    continue
                if (
                    found_daily
                    and '│' in line
                    and '/' in line
                    and 'Tot Profit' not in line
                    and '──────' not in line
                ):
                    parts = [p.strip() for p in line.split('│') if p.strip()]
                    if len(parts) >= 2:
                        date = parts[0]
                        profit = parts[1]
                        message += f"• {date}: {profit} USDT\n"
                if found_daily and '──────' in line and found_daily:
                    break

            message += '\n======= 🔚 报告结束 ======='
            return message

        except Exception as e:
            return f"处理统计信息时出错: {str(e)}"

    def daily_task(self):
        """执行每日任务"""
        try:
            print('开始执行每日任务...')

            # 1. 更新交易对
            self.update_pairs()

            # 2. 下载最新数据
            self.download_data()

            # 3. 运行long回测
            self.update_strategy_mode('long')
            self.run_backtesting()

            # 4. 分析回测结果
            self.analyze_backtesting()
            self.get_good_pairs('long')

            # 5. 运行short回测
            self.update_strategy_mode('short')
            self.run_backtesting()
            self.analyze_backtesting()
            self.get_good_pairs('short')

            summaries = []
            results = []
            reports = []
            threshholds = []
            for threshhold in range(-50, 50, 10):
                # 5. 移除表现差的交易对
                self.select_strategies(threshhold)

                # 6. 运行最终回测并发送结果
                final_results = self.run_backtesting()
                if final_results:
                    key_metrics, _ = self.extract_key_metrics(final_results)
                    report = self.extract_summary_stats(final_results)
                    summaries.append(key_metrics)
                    results.append(final_results)
                    reports.append(report)
                    threshholds.append(threshhold)
            max_profit_pos = max(enumerate(summaries), key=lambda x: x[1]['Absolute profit'])[0]
            self.select_strategies(threshholds[max_profit_pos])

            self.send_telegram_message(reports[max_profit_pos])

            import json

            with open('user_data/strategy_state.json', 'r') as f:
                strategy_state = json.load(f)

            self.send_telegram_message(f"交易对多空设置为：{strategy_state['pair_strategy_mode']}")
            self.send_telegram_message(
                f"交易对多空比为：{sum([v == 'long' for v in strategy_state['pair_strategy_mode'].values()])} : {sum([v == 'short' for v in strategy_state['pair_strategy_mode'].values()])}"
            )
            # 7. 更新配置文件
            self.update_config_pairs()

            # 8. 重启机器人
            self.restart_bot()

            pairs = daily_agent_main()

            self.send_telegram_message(f"完成每日智能分析建议对：{','.join(pairs)}")

            print('每日任务执行完成')
        except Exception as e:
            error_message = f"每日任务执行失败: {str(e)}"
            print(error_message)
            self.send_telegram_message(error_message)


def main():
    # 从环境变量获取配置
    config_path = os.getenv('CONFIG_PATH', 'user_data/config.json')
    strategy_path = os.getenv('STRATEGY_PATH', 'user_data/strategies')
    backtesting_strategy = 'KamaFama_Dynamic'
    docker_compose_path = os.getenv('DOCKER_COMPOSE_PATH', './')

    # 检查必要的环境变量是否存在
    required_env_vars = ['TG_BOT_TOKEN', 'TG_CHAT_ID', 'DOCKER_COMPOSE_PATH']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"错误: 缺少必要的环境变量: {', '.join(missing_vars)}")
        print('请确保.env文件中包含所有必要的环境变量')
        return

    scheduler = FreqtradeScheduler(
        config_path=config_path,
        strategy_path=strategy_path,
        docker_compose_path=docker_compose_path,
        backtesting_strategy=backtesting_strategy,
    )

    scheduler.daily_task()


if __name__ == '__main__':
    main()
