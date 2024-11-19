import pandas as pd
from datetime import datetime
import re

# 读取Excel和CSV文件
file1_path = '1.xlsx'
file2_path = '2.csv'
file3_path = '3.csv'

# 读取数据
data1 = pd.read_excel(file1_path)
data2 = pd.read_csv(file2_path)
data3 = pd.read_csv(file3_path)

# 处理2.csv（2024数据）
current_year = 2024
data2['Age'] = current_year - pd.to_datetime(data2['birth_date']).dt.year
data2 = data2.explode('disciplines')
data2['Year'] = 2024
data2['Season'] = 'Summer'
data2 = data2.rename(columns={'nationality_code': 'NOC', 'disciplines': 'Sport'})
data2['Sport'] = data2['Sport'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).strip())

# 处理3.csv（新的历史数据）
data3['Age'] = current_year - pd.to_datetime(data3['birth_date']).dt.year
data3['Year'] = 2024  # 可以根据需求调整年份，假定也为2024
data3['Season'] = 'Summer'
data3 = data3.rename(columns={'country_code': 'NOC', 'discipline': 'Sport'})

# 合并所有数据
combined_data = pd.concat([data1, data2, data3], ignore_index=True)

# 任务1：每年每项运动20岁以下运动员比例
def calculate_under_20_ratio(df):
    grouped = df.groupby(['Year', 'Sport'])
    results = []
    for (year, sport), group in grouped:
        total_athletes = group.shape[0]
        under_20_athletes = group[group['Age'] < 20].shape[0]
        under_20_ratio = under_20_athletes / total_athletes if total_athletes > 0 else 0
        results.append({'Year': year, 'Sport': sport, 'Under 20 Ratio': under_20_ratio})
    return pd.DataFrame(results)

# 任务2：每年每项运动参与的国家数
def calculate_country_participation(df):
    grouped = df.groupby(['Year', 'Sport'])['NOC'].nunique().reset_index()
    grouped.columns = ['Year', 'Sport', 'Unique Countries']
    return grouped

# 计算并保存结果
under_20_ratio_df = calculate_under_20_ratio(combined_data)
country_participation_df = calculate_country_participation(combined_data)

with pd.ExcelWriter('final_output_all_data_results.xlsx') as writer:
    under_20_ratio_df.to_excel(writer, sheet_name='Under 20 Ratio', index=False)
    country_participation_df.to_excel(writer, sheet_name='Country Participation', index=False)

print("数据处理完成，结果已保存至 final_output_all_data_results.xlsx")
