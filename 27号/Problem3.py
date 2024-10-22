import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

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

print(similarity_df)

# 选择相似度大于等于0.05的作为合并条件，AgglomerativeClustering默认使用欧氏距离
# 1 - similarity_matrix 将相似度矩阵转换为距离矩阵 (因为0.05的相似度相当于95%的差异)
clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                     distance_threshold=0.95)
clusters = clustering.fit_predict(1 - similarity_matrix)

# 将聚类结果添加到客户数据中
customer_clusters = pd.DataFrame({'CustomerID': customer_item_matrix.index, 'Cluster': clusters})

# 过滤掉孤立点（即仅有一个客户的类）
cluster_counts = customer_clusters['Cluster'].value_counts()
non_isolated_clusters = cluster_counts[cluster_counts > 1].index
customer_clusters = customer_clusters[customer_clusters['Cluster'].isin(non_isolated_clusters)]

# 打印聚类结果
print(customer_clusters)

# 计算每个聚类的数量
cluster_counts = customer_clusters['Cluster'].value_counts()

# 获取聚类的总数（即不同的类别数量）
number_of_clusters = cluster_counts.count()

# 输出结果
print(f"聚类总数: {number_of_clusters}")

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


# 构建同类客户的网络模型
def plot_intra_cluster_network(cluster_id):
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

    # 绘制图形
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=300, font_size=8)
    plt.show()

    # 可视化同类客户的网络模型


for cluster_id in customer_clusters['Cluster'].unique():
    plot_intra_cluster_network(cluster_id)


# 聚合同类客户
def calculate_inter_cluster_similarity(customer_clusters, similarity_df):
    # 初始化类别相似度字典
    cluster_similarity = {}

    # 获取所有类别
    unique_clusters = customer_clusters['Cluster'].unique()

    # 遍历每个类别对
    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            cluster_i = unique_clusters[i]
            cluster_j = unique_clusters[j]

            # 筛选出属于这两个类别的客户
            customers_i = customer_clusters[customer_clusters['Cluster'] == cluster_i]['CustomerID']
            customers_j = customer_clusters[customer_clusters['Cluster'] == cluster_j]['CustomerID']

            # 计算类别之间的平均相似度
            total_similarity = 0
            count = 0
            for customer_i in customers_i:
                for customer_j in customers_j:
                    total_similarity += similarity_df.loc[customer_i, customer_j]
                    count += 1

            if count > 0:
                avg_similarity = total_similarity / count
                cluster_similarity[(cluster_i, cluster_j)] = avg_similarity

    return cluster_similarity


# 计算类别之间的相似度
cluster_similarity = calculate_inter_cluster_similarity(customer_clusters, similarity_df)


# 构建跨类别的网络模型
def plot_inter_cluster_network(cluster_similarity):
    G = nx.Graph()

    # 添加类别节点
    clusters = list(set([c for pair in cluster_similarity.keys() for c in pair]))
    G.add_nodes_from(clusters)

    # 添加跨类别的边
    for (cluster_i, cluster_j), similarity in cluster_similarity.items():
        if similarity >= 0.01:  # 仅添加相似度高于0.05的边
            G.add_edge(cluster_i, cluster_j, weight=similarity)

    # 绘制图形
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.6)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=500, font_size=8)

    plt.show()


# 绘制跨类别网络图
plot_inter_cluster_network(cluster_similarity)