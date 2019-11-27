import numpy as np
import matplotlib.pyplot as plt

def create_sample(k):
    np.random.seed(10)  # 随机数种子，保证随机数生成的顺序一样
    num = 100
    n_dim = 2
    data_mat = np.zeros((k, num, n_dim))
    label = np.zeros((k, num))
    for i in range(k): 
        data_mat[i, :, :] = 3 * (i + 1) * k + 4 * np.random.randn(num, n_dim)
        label[i, :] = i + np.zeros(num)
    data_mat = np.reshape(data_mat, (num * k, n_dim))
    return {'data_mat': data_mat, 'label': label}

# arr[i, :] #取第i行数据
# arr[i:j, :] #取第i行到第j行的数据
# in:arr[:,0] # 取第0列的数据，以行的形式返回的
# in:arr[:,:1] # 取第0列的数据，以列的形式返回的

def k_mean(data_set, k):
    m, n = np.shape(data_set)  # 获取行数，列数
    cluster_assignment = np.mat(np.zeros((m, n)))  # 转换为矩阵， zeros生成指定维数的全0数组，用来记录迭代结果
    centroids = generate_centroids(k, n)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_distance = np.inf  # 无限大的正数
            min_index = -1
            vec_b = np.array(data_set)[i, :]  # i行数据，数据集内的点的位置
            for j in range(k):
                vec_a = np.array(centroids)[j, :]  # j行数据， 中心点位置
                distance = calculate_distance(vec_a, vec_b)
                if distance < min_distance:
                    min_distance = distance
                    min_index = j
            if cluster_assignment[i, 0] != min_index:
                cluster_changed = True
            cluster_assignment[i, :] = min_index, min_distance ** 2
        update_centroids(data_set, cluster_assignment, centroids, k)
    get_result(data_set, cluster_assignment, k)

def generate_centroids(centroid_num, column_num):
    centroids = np.mat(np.zeros((centroid_num, column_num)))  # 生成中心点矩阵
    for index in range(centroid_num):
        # 随机生成中心点， np.random.rand（Random values in a given shape）
        centroids[index, :] = np.mat(np.random.rand(1, column_num))
    return centroids

def calculate_distance(vec_a, vec_b):
    distance = np.sqrt(sum(np.power(vec_a - vec_b, 2)))  # power(x1, x2) 对x1中的每个元素求x2次方。不会改变x1上午shape。
    return distance

def update_centroids(data_set, cluster_assignment, centroids, k):
    for cent in range(k):
        # 取出对应簇
        cluster = np.array(cluster_assignment)[:, 0] == cent
        # np.nonzero取出矩阵中非0的元素坐标
        pts_in_cluster = data_set[np.nonzero(cluster)]
        # mean() 函数功能：求取均值
        # 经常操作的参数为axis，以m * n矩阵举例：
        # axis : 不设置值,  m * n 个数求均值，返回一个实数
        # axis = 0：压缩行，对各列求均值，返回1 * n矩阵
        # axis = 1 ：压缩列，对各行求均值，返回m * 1矩阵
        if len(pts_in_cluster) > 0:
            centroids[cent, :] = np.mean(pts_in_cluster, axis=0)

def get_result(data_set, cluster_assignment, k):
    cs = ['r', 'g', 'b', 'k', 'y']
    for cent in range(k):
        res_id = np.nonzero(np.array(cluster_assignment)[:, 0] == cent)
        plot_data(data_set[res_id], cs[cent])
    plt.show()

def plot_data(samples, color, plot_type='o'):
    plt.plot(samples[:, 0], samples[:, 1], plot_type, markerfacecolor=color, markersize=14)

data = create_sample(4)
k_mean(data['data_mat'], 4)
