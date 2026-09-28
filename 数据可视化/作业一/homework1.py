# -*- coding: utf-8 -*-
"""
数据可视化 - 作业一完整解答脚本
运行方式：在 PyCharm 中右键直接运行，或者在终端执行: python homework1.py
"""

import sys
import pandas as pd

# 设置标准输出编码为 utf-8，防止在某些终端打印中文时乱码
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def main():
    print("=" * 60)
    print("【第1题】读取数据并设置显示选项")
    print("=" * 60)
    # 读入数据文件
    df = pd.read_csv("智联.csv")

    # 修改显示设置
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)

    # 打印前5行
    print(df.head())

    print("\n" + "=" * 60)
    print("【第2 & 3题】去重处理")
    print("=" * 60)
    # 记录去重前的行数
    len_before = len(df)

    # 构建重复行掩码：keep=False 表示所有重复行都标记出来（包括第一次出现）
    dup_mask = df.duplicated(keep=False)

    # 输出重复行的数据与行号
    dup_indices = df[dup_mask].index.tolist()
    print(f"重复行的行号（索引）: {dup_indices}")
    print(f"去除重复行前的行数: {len_before}")

    # 去除重复行（默认 keep='first'，保留首次出现的记录）
    df_dedup = df.drop_duplicates().copy()
    len_after = len(df_dedup)
    print(f"去除重复行后的行数: {len_after}")
    print(f"共去除了 {len_before - len_after} 条重复记录。")

    print("\n" + "=" * 60)
    print("【第4 & 5题】缺失值检查与统计")
    print("=" * 60)
    # 检查存在缺失值的列
    missing_counts = df_dedup.isnull().sum()
    missing_ratios = (missing_counts / len(df_dedup)) * 100

    missing_df = pd.DataFrame({
        "缺失数量": missing_counts,
        "缺失占比(%)": missing_ratios
    })
    # 过滤出存在缺失的列并降序排列
    missing_df = missing_df[missing_df["缺失数量"] > 0].sort_values(by="缺失占比(%)", ascending=False)
    # 格式化百分比显示
    missing_df["缺失占比(%)"] = missing_df["缺失占比(%)"].map(lambda x: f"{x:.2f}%")
    print(missing_df)

    print("\n" + "=" * 60)
    print("【第6 & 7题】列的处理与 jobTypeLevelName 分布")
    print("=" * 60)
    # 计算每列数据中不同值的个数 (nunique)
    print("--- 各列唯一值(不同值)个数统计 ---")
    unique_counts = df_dedup.nunique()
    print(unique_counts)

    print("\n--- jobTypeLevelName 列的 value_counts() 统计结果 ---")
    job_type_counts = df_dedup["jobTypeLevelName"].value_counts(dropna=False)
    print(job_type_counts)

    print("\n" + "=" * 60)
    print("【第8 & 9题】技能列分割与词频统计")
    print("=" * 60)
    skill_df = df_dedup.copy()

    # 按空白字符拆分（空格、制表符等），也支持逗号、斜杠等
    skill_df['skill'] = skill_df['skillLabel'].astype(str).str.split(r'[,\s/、;；]+')
    # 展开成一行一个技能
    skill_df = skill_df.explode('skill')
    # 清理：去空格、转小写、去空字符串
    skill_df['skill'] = skill_df['skill'].str.strip().str.lower()
    skill_df = skill_df[skill_df['skill'] != '']
    # 过滤掉缺失值转换产生的 'nan' 字符串
    skill_df = skill_df[skill_df['skill'] != 'nan']

    # 统计每种技能出现的频次
    skill_counts = skill_df['skill'].value_counts()

    # 计算频率（出现次数 / 总岗位行数）
    skill_freq = pd.DataFrame({
        "出现次数": skill_counts,
        "出现频率(占总职位数)": (skill_counts / len(df_dedup)).map(lambda x: f"{x:.4f} ({x*100:.2f}%)")
    })

    print("--- 出现频次 Top 20 技能及频率 ---")
    print(skill_freq.head(20))


if __name__ == "__main__":
    main()
