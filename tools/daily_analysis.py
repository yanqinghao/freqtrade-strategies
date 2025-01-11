import pandas as pd
import numpy as np
import re

def parse_trading_data(data_str):
    # 使用正则表达式提取表格数据
    pattern = r'│\s*([\d/]+)\s*│\s*([-\d.]+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│\s*(\d+)\s*│'
    matches = re.findall(pattern, data_str)
    
    # 创建数据字典
    data = {
        'Date': [],
        'Daily_Profit': [],
        'Wins': [],
        'Draws': [],
        'Losses': []
    }
    
    # 解析匹配的数据
    for match in matches:
        data['Date'].append(match[0])
        data['Daily_Profit'].append(float(match[1]))
        data['Wins'].append(int(match[2]))
        data['Draws'].append(int(match[3]))
        data['Losses'].append(int(match[4]))
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 转换日期格式
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    return df

def analyze_trading_data(df, initial_balance=1000):
    """分析交易数据并返回结果"""
    
    # 计算各种指标
    df['Balance'] = initial_balance + df['Daily_Profit'].cumsum()
    df['Win_Rate'] = df['Wins'] / (df['Wins'] + df['Losses']) * 100
    df['Total_Trades'] = df['Wins'] + df['Losses']
    df['Cumulative_Profit'] = df['Daily_Profit'].cumsum()
    
    # 计算回撤
    df['Peak_Balance'] = df['Balance'].cummax()
    df['Drawdown'] = (df['Peak_Balance'] - df['Balance']) / df['Peak_Balance'] * 100
    
    # 打印分析结果
    print("=== 交易分析报告 ===")
    print(f"\n初始余额: {initial_balance:,.2f} USDT")
    print(f"最终余额: {df['Balance'].iloc[-1]:,.2f} USDT")
    print(f"总盈亏: {df['Cumulative_Profit'].iloc[-1]:,.2f} USDT")
    print(f"总盈亏率: {(df['Cumulative_Profit'].iloc[-1] / initial_balance * 100):,.2f}%")
    
    print(f"\n最大余额: {df['Balance'].max():,.2f} USDT")
    print(f"最低余额: {df['Balance'].min():,.2f} USDT")
    print(f"最大回撤: {df['Drawdown'].max():,.2f}%")
    
    print(f"\n总交易天数: {len(df)} 天")
    print(f"总交易次数: {df['Total_Trades'].sum():,.0f}")
    print(f"总胜利次数: {df['Wins'].sum():,.0f}")
    print(f"总亏损次数: {df['Losses'].sum():,.0f}")
    print(f"平均胜率: {df['Win_Rate'].mean():,.2f}%")
    
    print("\n=== 每日余额变化 ===")
    print("日期         余额      日盈亏    胜/负   胜率")
    print("-" * 50)
    for _, row in df.iterrows():
        print(f"{row['Date'].strftime('%Y-%m-%d')}  "
              f"{row['Balance']:8,.2f}  "
              f"{row['Daily_Profit']:8,.2f}  "
              f"{row['Wins']:3.0f}/{row['Losses']:<3.0f}  "
              f"{row['Win_Rate']:5.1f}%")
    
    return df

# 使用示例
if __name__ == "__main__":
    # 假设数据字符串存储在 trading_data_str 变量中
    trading_data_str = """┃        Day ┃ Tot Profit USDT ┃ Wins ┃ Draws ┃ Losses ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ 01/12/2024 │           -4.75 │   26 │     0 │      5 │
│ 02/12/2024 │         208.095 │   87 │     0 │     16 │
│ 03/12/2024 │        -295.854 │   64 │     0 │     28 │
│ 04/12/2024 │          65.608 │   45 │     0 │      6 │
│ 05/12/2024 │        -302.901 │   32 │     0 │     22 │
│ 06/12/2024 │          10.061 │   23 │     0 │      3 │
│ 07/12/2024 │          20.194 │   18 │     0 │      2 │
│ 08/12/2024 │         -31.095 │   15 │     0 │      6 │
│ 09/12/2024 │        1298.242 │  172 │     0 │     13 │
│ 10/12/2024 │          439.86 │  181 │     0 │     34 │
│ 11/12/2024 │        -661.444 │   30 │     0 │     33 │
│ 12/12/2024 │         174.211 │   92 │     0 │     11 │
│ 13/12/2024 │         148.305 │   67 │     0 │      6 │
│ 14/12/2024 │         151.764 │  106 │     0 │     12 │
│ 15/12/2024 │         -38.154 │   43 │     0 │      9 │
│ 16/12/2024 │        -118.085 │   52 │     0 │     16 │
│ 17/12/2024 │          36.722 │   70 │     0 │     13 │
│ 18/12/2024 │        1016.194 │  229 │     0 │     10 │
│ 19/12/2024 │        3276.351 │  477 │     0 │      7 │
│ 20/12/2024 │        -171.956 │  303 │     0 │     88 │
│ 21/12/2024 │          842.98 │  239 │     0 │     16 │
│ 22/12/2024 │          61.758 │  158 │     0 │     26 │
│ 23/12/2024 │         -717.92 │   71 │     0 │     45 │
│ 24/12/2024 │         291.049 │  119 │     0 │     10 │
│ 25/12/2024 │         229.098 │  127 │     0 │     24 │
│ 26/12/2024 │          173.42 │  209 │     0 │     26 │
│ 27/12/2024 │        -138.033 │  105 │     0 │     22 │
│ 28/12/2024 │        -109.204 │   81 │     0 │     19 │
│ 29/12/2024 │        -264.304 │   99 │     0 │     33 │
│ 30/12/2024 │        -139.848 │  118 │     0 │     25 │
│ 31/12/2024 │         -20.117 │   81 │     0 │     15 │
│ 01/01/2025 │        -232.603 │   79 │     0 │     25 │
│ 02/01/2025 │          84.757 │  206 │     0 │     31 │
│ 03/01/2025 │          750.62 │  234 │     0 │     12 │
│ 04/01/2025 │         162.626 │  130 │     0 │     15 │
│ 05/01/2025 │         -119.03 │   96 │     0 │     36 │
│ 06/01/2025 │         129.188 │  114 │     0 │     13 │
│ 07/01/2025 │        -246.667 │  182 │     0 │     60 │
│ 08/01/2025 │          66.626 │   36 │     0 │     19 │"""
    
    # 解析数据
    df = parse_trading_data(trading_data_str)
    
    # 分析数据
    analyzed_df = analyze_trading_data(df, initial_balance=1000)