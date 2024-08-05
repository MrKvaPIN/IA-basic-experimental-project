import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
                                                                                                                                                                                                      # `读取Excel文件`
df = pd.read_excel('CW_Data.xlsx')
# 初始化变量定义
Total = df['Total']
MCQ = df['MCQ']
programme = df['Programme']
allcolumns = ['Index','Gender','Programme','Grade','Total', 'MCQ','Q1','Q2','Q3','Q4','Q5']
columns = ['Gender','Programme','Grade','Total', 'MCQ','Q1','Q2','Q3','Q4','Q5']

'''
数据预处理
'''
# 数据标准化
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)
# 将标准化后的数据保存为CSV文件
df_scaled.to_csv('scaled_data.csv', index=False)
# 读取csv文件
df_scaled = pd.read_csv('scaled_data.csv')
# 控制分类颜色
colors = ['orange', 'teal', 'turquoise', 'coral']
cmap = ListedColormap(colors)
markers = ['o', '^', 's', '<', '>']
# 图例绘制
legend_elements = [Line2D([0], [0], marker='o', color='w', label='Prog 1',
                          markerfacecolor='turquoise', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Prog 2',
                          markerfacecolor='teal', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Prog 3',
                          markerfacecolor='orange', markersize=10),
                   Line2D([0], [0], marker='o', color='w', label='Prog 4',
                          markerfacecolor='coral', markersize=10)
                   ]

'''
数据降维器
'''
# 使用PCA进行降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_scaled)
df_scaled['programme'] = programme

# 初始化 t-SNE 模型，进行数据降维，用于为给gmm
tsne = TSNE(n_components=2, init='pca', perplexity=70, n_iter=10000, learning_rate=5, early_exaggeration=95,
            verbose=1, random_state=500)
# 对数据进行降维
X_tsne = tsne.fit_transform(df_scaled)


'''
执行函数
'''

def gmm_test():
    lowest_aic = float('inf')
    best_n_components = 0
    aic_values = []
    bic_values = []

    for n_components in range(1, 11):
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(df_scaled)

        # 计算AIC和BIC值
        aic = gmm.aic(df_scaled)
        bic = gmm.bic(df_scaled)
        aic_values.append(aic)
        bic_values.append(bic)

    # 绘制曲线图
    plt.figure()
    plt.plot(range(1, 11), aic_values, label='AIC')
    plt.plot(range(1, 11), bic_values, label='BIC')
    plt.xlabel('Number of Components')
    plt.ylabel('AIC/BIC Value')
    plt.title('AIC/BIC vs Number of Components')
    plt.legend()
    plt.show()
def gmm():
    # 使用GMM进行聚类
    gmm = GaussianMixture(n_components=4)
    gmm.fit(df_scaled)
    labels_gmm = gmm.predict(df_scaled)


    # 可视化聚类结果
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels_gmm)
    for i, color in enumerate(colors):
        for j, marker in enumerate(markers):
            plt.scatter(X_pca[(labels_gmm == i) & (df_scaled['programme'] == j), 0],
                        X_pca[(labels_gmm == i) & (df_scaled['programme'] == j), 1],
                        marker=marker, color=color)
    plt.title('GMM Clustering by pca')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(handles=legend_elements, loc='best')

    plt.figure(figsize=(8, 6))
    for i, color in enumerate(colors):
        for j, marker in enumerate(markers):
            plt.scatter(X_tsne[(labels_gmm == i) & (df_scaled['programme'] == j), 0],
                        X_tsne[(labels_gmm == i) & (df_scaled['programme'] == j), 1],
                        marker=marker, color=color)
    plt.title('GMM Clustering by t-sne')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(handles=legend_elements, loc='best')

    plt.show()

    # 创建交叉表
    cross_table = pd.crosstab(df_scaled['programme'], labels_gmm, normalize='index')
    # 绘制堆积柱状图
    cross_table.plot(kind='bar', stacked=True, figsize=(10, 8))
    plt.title('Distribution of GMM Clusters by Programme')
    plt.xlabel('Programme')
    plt.ylabel('GMM predict')
    plt.legend(title='GMM Cluster')
    plt.show()
def k_means():
    # 使用肘部法则和轮廓系数法则确定最佳聚类数量
    sse = []
    silhouette_scores = []
    k_values = range(2, 10)


    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

    # 绘制SSE和轮廓系数图
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(k_values, sse, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('SSE')
    plt.title('Elbow Method for Optimal k')

    plt.subplot(1, 2, 2)
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Optimal k')

    plt.show()

    # 从上图中选择最佳的k值

    # 使用最佳k值进行K-means聚类
    kmeans = KMeans(n_clusters=8, random_state=42)
    labels_kmeans = kmeans.fit_predict(df_scaled)

    # 可视化聚类结果
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels_kmeans)
    for i in unique_labels:
        plt.scatter(X_tsne[labels_kmeans == i, 0], X_tsne[labels_kmeans == i, 1], label=i)
    plt.legend()
    plt.title('k-means Clustering by t_sne')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # plt.legend(handles=legend_elements, loc='best')
    plt.show()
    # 创建交叉表
    cross_table = pd.crosstab(df_scaled['programme'], labels_kmeans, normalize='index')
    # 绘制堆积柱状图
    cross_table.plot(kind='bar', stacked=True, figsize=(10, 8))
    plt.title('Distribution of K-means Clusters by Programme')
    plt.xlabel('Programme')
    plt.ylabel('K-means predict')
    plt.legend(title='K-means Cluster')
    plt.show()
def Hiera():
    # 使用分层聚类
    agg_cluster = AgglomerativeClustering(n_clusters=4, linkage='ward')
    labels_agg = agg_cluster.fit_predict(df_scaled)
    # 可视化聚类结果
    plt.figure(figsize=(8, 6))
    unique_labels = np.unique(labels_agg)
    for i in unique_labels:
        plt.scatter(X_tsne[labels_agg == i, 0], X_tsne[labels_agg == i, 1], label=i)
    plt.title('Hierarchical Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # 使用层次聚类方法构建聚类
    Z = linkage(df_scaled, method='ward')  # data是您的输入数据

    # 绘制树状图
    plt.figure(figsize=(12, 6))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.show()
    # 创建交叉表
    cross_table = pd.crosstab(df_scaled['programme'], labels_agg, normalize='index')
    # 绘制堆积柱状图
    cross_table.plot(kind='bar', stacked=True, figsize=(10, 8))
    plt.title('Distribution of K-means Clusters by Programme')
    plt.xlabel('Programme')
    plt.ylabel('K-means predict')
    plt.legend(title='K-means Cluster')
    plt.show()


# gmm_test()
# gmm()
# k_means()
Hiera()





