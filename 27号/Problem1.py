import pandas as pd
from scipy.stats import pearsonr
from collections import Counter

# 读取Excel文件中所需的列
file_path = '1(1).xlsx'
data = pd.read_excel(file_path, usecols=["InvoiceNo","StockCode", "Quantity"])

# 过滤掉Quantity为负数或零的数据
data = data[data['Quantity'] > 0]

# 统计每个商品的总购买数量
item_counts = data.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)

item_counts
# 获取购买数量最多的前五个商品的商品代码
top_5_items = item_counts.head(5).index.tolist()

# 打印前五个商品及其购买数量
print("客户购买最多的前五个商品及其购买数量：")
print(item_counts.head(5))
from mlxtend.frequent_patterns import apriori, association_rules
from  mlxtend.frequent_patterns import fpgrowth

# 假设 `data` 是一个包含 'InvoiceNo'（订单编号）和 'StockCode'（商品代码）的 DataFrame
# 将数据转换为适合 Apriori 算法的数据结构
basket = data.groupby(['InvoiceNo', 'StockCode'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo')
basket = basket.applymap(lambda x: 1 if x > 0 else 0)
# 使用FP-Growth算法寻找频繁项集
frequent_itemsets = fpgrowth(basket, min_support=0.01, use_colnames=True)

# 根据频繁项集生成关联规则
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)

print(rules)
# # 获取支持度最高的前100条关联规则
# top_100_rules = rules.sort_values(by='support', ascending=False).head(100)

# # 显示结果
# print(top_100_rules)
# 提取规则中的前件和后件及其置信度
rules_matrix = rules.pivot(index='antecedents', columns='consequents', values='confidence')

# 填充缺失值为0
rules_matrix = rules_matrix.fillna(0)

# 显示矩阵
print(rules_matrix)

# 将矩阵写入Excel文件
output_file = '关联规则矩阵.xlsx'
rules_matrix.to_excel(output_file)