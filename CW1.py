from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


                                                                                                                                                                                                                # `读取Excel文件`
df = pd.read_excel('CW_Data.xlsx')
# 初始化变量定义
Total = df['Total']
MCQ = df['MCQ']
programme = df['Programme']
allcolumns = ['Index','Gender','Programme','Grade','Total', 'MCQ','Q1','Q2','Q3','Q4','Q5']
columns = ['MCQ','Total','Q1','Q2','Q3','Q4','Q5']

# 数据标准化
scaler = StandardScaler()
df_scaled_box = pd.DataFrame(scaler.fit_transform(df[allcolumns]), columns=allcolumns)
df_scaled = pd.DataFrame(scaler.fit_transform(df[columns]), columns=columns)

# 将标准化后的数据保存为CSV文件
df_scaled.to_csv('scaled_data.csv', index=False)
# 读取csv文件
df_scaled = pd.read_csv('scaled_data.csv')


# # 绘制箱型图
# fig, axs = plt.subplots(1, 2, figsize=(10, 6))
# # 绘制原始数据的 Box Plot
# axs[0].boxplot([df[col] for col in allcolumns], vert=True, labels=allcolumns)
# axs[0].set_title('Box Plot of All Columns')
# axs[0].set_ylabel('Marks')
# axs[0].set_xticklabels(allcolumns, rotation=45)
# # 绘制标准化后的数据的 Box Plot
# axs[1].boxplot([df_scaled_box[col] for col in allcolumns], vert=True, labels=allcolumns)
# axs[1].set_title('Box Plot of Scaled Columns')
# axs[1].set_ylabel('Marks')
# axs[1].set_xticklabels(allcolumns, rotation=45)
# # 调整子图之间的间距
# plt.tight_layout()
# # 显示图形
# plt.show()


# 创建PCA对象，n_components指定要保留的主成分数量
pca = PCA(n_components=2)
# 对数据进行PCA
X_pca = pca.fit_transform(df_scaled)
weights = pca.components_
for i, weight in enumerate(weights):
    print(f"特征对主成分的影响：{np.abs(weight)}")
#  将标签programme添加到数据中
df_scaled['programme'] = programme


# 初始化 t-SNE 模型
# tsne = TSNE(n_components=2, random_state=0)
tsne = TSNE(n_components=2, init='pca', perplexity=70, n_iter = 10000,learning_rate=5, early_exaggeration=95,verbose = 1, random_state=500)
# 对数据进行降维
X_tsne = tsne.fit_transform(df_scaled)


# 控制分类颜色
colors = ['red','blue','green','yellow']
cmap = ListedColormap(colors)


# 成分图的可视化
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.scatter(df['Total'], df['MCQ'],c=df_scaled['programme'],cmap=cmap , alpha=1)
plt.xlabel('Total')
plt.ylabel('MCQ')
plt.title('Original Features')

# 缩放特征可视化
plt.subplot(1, 4, 2)
plt.scatter(df_scaled['Total'], df_scaled['MCQ'], c=df_scaled['programme'],cmap=cmap , alpha=1)
plt.xlabel('Scaled Total')
plt.ylabel('Scaled MCQ')
plt.title('Scaled Features')

# 绘制PCA的分布
plt.subplot(1, 4, 3)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df_scaled['programme'],cmap=cmap , alpha=1)
plt.xlabel('principal components 1')
plt.ylabel('principal components 2')
plt.title('PCA on Data')

# 绘制t-sne分布
plt.subplot(1, 4, 4)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1],c=df_scaled['programme'],cmap=cmap , alpha=1)
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE Visualization')

# 自适应布局与显示
plt.tight_layout()
plt.show()

