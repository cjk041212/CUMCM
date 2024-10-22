import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import matplotlib.pyplot as plt

# 读取数据
file_path = '1.xlsx'  # 请确保文件路径正确
data = pd.read_excel(file_path)

# 构建客户-商品矩阵（以销售额表示）
data['Sales'] = data['Quantity'] * data['UnitPrice']
customer_item_matrix = data.pivot_table(index='CustomerID', columns='StockCode', values='Sales', aggfunc='sum',
                                        fill_value=0)

# 去掉全0的行和列
customer_item_matrix = customer_item_matrix.loc[
    (customer_item_matrix.sum(axis=1) > 0), (customer_item_matrix.sum(axis=0) > 0)]

# 计算客户之间的相似度（余弦相似度）
similarity_matrix = cosine_similarity(customer_item_matrix)

# 将相似度矩阵转为DataFrame形式
similarity_df = pd.DataFrame(similarity_matrix, index=customer_item_matrix.index, columns=customer_item_matrix.index)

# 选择相似度大于等于0.05的作为合并条件，AgglomerativeClustering默认使用欧氏距离
clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                     distance_threshold=0.95)
clusters = clustering.fit_predict(1 - similarity_matrix)

# 将聚类结果添加到客户数据中
customer_clusters = pd.DataFrame({'CustomerID': customer_item_matrix.index, 'Cluster': clusters})

# 过滤掉孤立点（即仅有一个客户的类）
cluster_counts = customer_clusters['Cluster'].value_counts()
non_isolated_clusters = cluster_counts[cluster_counts > 1].index
customer_clusters = customer_clusters[customer_clusters['Cluster'].isin(non_isolated_clusters)]


# 构建同类客户的网络模型并计算度数
def get_all_top_degree_nodes():
    all_degree_nodes = []

    # 遍历每个聚类
    for cluster_id in customer_clusters['Cluster'].unique():
        # 筛选同一类别的客户
        cluster_customers = customer_clusters[customer_clusters['Cluster'] == cluster_id]['CustomerID']

        # 构建子图
        G = nx.Graph()

        # 添加节点
        G.add_nodes_from(cluster_customers)

        # 添加边，基于相似度矩阵构建网络
        for i in cluster_customers:
            for j in cluster_customers:
                if i != j and similarity_df.loc[i, j] >= 0.05:  # 仅添加相似度高于0.05的边
                    G.add_edge(i, j, weight=similarity_df.loc[i, j])

        # 计算每个节点的度数
        degree_dict = dict(G.degree())

        # 将该聚类的度数加入全局列表
        all_degree_nodes.extend(degree_dict.items())

    # 返回度数最高的五个节点
    top_5_nodes = sorted(all_degree_nodes, key=lambda x: x[1], reverse=True)[:5]

    return top_5_nodes


# 获取所有聚类中度数最高的五个节点
top_5_nodes = get_all_top_degree_nodes()

# 输出度数最高的五个节点
print(f"Top 5 nodes by degree across all clusters: {top_5_nodes}")

# 获取这五个节点（客户）的ID
top_5_customer_ids = [node[0] for node in top_5_nodes]

# 筛选出这些客户的购买记录
top_customers_data = data[data['CustomerID'].isin(top_5_customer_ids)]
top_5_item_ids = []
# 对每个客户分别计算购买最多的商品
for customer_id in top_5_customer_ids:
    customer_data = top_customers_data[top_customers_data['CustomerID'] == customer_id]
    item_counts = customer_data.groupby('StockCode')['Quantity'].sum()

    top_item = item_counts.idxmax()
    top_item_count = item_counts.max()
    top_5_item_ids.append(top_item)
    print(f"Customer {customer_id} purchased the most of StockCode {top_item} with {top_item_count} purchases")

top_items_data = data[data['StockCode'].isin(top_5_item_ids)]
print(top_items_data)