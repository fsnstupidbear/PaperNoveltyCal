import h5py
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # 设置Matplotlib使用Agg后端
from sklearn.decomposition import IncrementalPCA

# 定义函数以增量方式加载和应用PCA
def incremental_pca(file_path, n_components=2, batch_size=500):
    ipca = IncrementalPCA(n_components=n_components)

    with h5py.File(file_path, 'r') as h5f:
        total_size = h5f['Vector'].shape[0]
        total_batches = total_size // batch_size + (1 if total_size % batch_size else 0)

        for i in range(total_batches):
            start = i * batch_size
            end = min(start + batch_size, total_size)
            embeddings_batch = np.vstack([np.array(emb) for emb in h5f['Vector'][start:end]])
            ipca.partial_fit(embeddings_batch)

    return ipca

# 定义函数以可视化PCA结果
def plot_pca(file_path, ipca):
    with h5py.File(file_path, 'r') as h5f:
        embeddings = np.vstack([np.array(emb) for emb in h5f['Vector'][:]])
        transformed_embeddings = ipca.transform(embeddings)

        plt.scatter(transformed_embeddings[:, 0], transformed_embeddings[:, 1], alpha=0.5, s=10)
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.title('PCA Visualization')
        plt.show()

# 指定文件路径
method_file_path = '../data/method_data_50.h5'
task_file_path = '../data/task_data_50.h5'
method_task_comb_file_path = '../data/method_task_comb_embeddings_50.h5'

# 对方法嵌入应用增量PCA
method_ipca = incremental_pca(method_file_path)
plot_pca(method_file_path, method_ipca)

# 对任务嵌入应用增量PCA
task_ipca = incremental_pca(task_file_path)
plot_pca(task_file_path, task_ipca)

# 对方法任务组合嵌入应用增量PCA
method_task_comb_ipca = incremental_pca(method_task_comb_file_path)
plot_pca(method_task_comb_file_path, method_task_comb_ipca)
