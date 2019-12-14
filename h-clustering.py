import numpy as np
import matplotlib.pyplot as plt
import time

start =time.clock()

def create_sample(k):
    # np.random.seed(10)  # 随机数种子，保证随机数生成的顺序一样
    n = 100
    n_dim = 2
    data_mat = np.zeros((k, n, n_dim))
    t = np.random.random_sample(size=n) * 2 * np.pi - np.pi
    x1, x2, x3, x4 = np.cos(t), np.cos(t), np.cos(t), np.cos(t)
    y1, y2, y3, y4 = np.sin(t), np.sin(t), np.sin(t), np.sin(t)
    for i in range(n):
        len = 1 + np.sqrt(np.random.rand())
        data_mat[0, i, 0] = x1[i] * len - 5
        data_mat[0, i, 1] = y1[i] * len
    for i in range(n):
        len = 3 + np.sqrt(np.random.rand())
        data_mat[1, i, 0] = x2[i] * len - 5
        data_mat[1, i, 1] = y2[i] * len
    for i in range(n):
        len = 1 + np.sqrt(np.random.rand())
        data_mat[2, i, 0] = x3[i] * len + 5
        data_mat[2, i, 1] = y3[i] * len
    for i in range(n):
        len = 3 + np.sqrt(np.random.rand())
        data_mat[3, i, 0] = x4[i] * len + 5
        data_mat[3, i, 1] = y4[i] * len
    # print(data_mat)
    data_mat = np.reshape(data_mat, (n * k, n_dim))
    return {'data_mat': data_mat}

# arr[i, :] #取第i行数据
# arr[i:j, :] #取第i行到第j行的数据
# in:arr[:,0] # 取第0列的数据，以行的形式返回的
# in:arr[:,:1] # 取第0列的数据，以列的形式返回的


def create_sample_h(k=4):
    n = 100
    n_dim = 2
    data_mat = np.zeros((k, n, n_dim))
    t = np.random.random_sample(size=n) * 2 * np.pi - np.pi
    x1, x2, x3, x4 = np.cos(t), np.cos(t), np.cos(t), np.cos(t)
    y1, y2, y3, y4 = np.sin(t), np.sin(t), np.sin(t), np.sin(t)
    for i in range(n):
        len = 1 + np.sqrt(np.random.rand())
        data_mat[0, i, 0] = x1[i] * len - 5
        data_mat[0, i, 1] = y1[i] * len
    for i in range(n):
        len = 3 + np.sqrt(np.random.rand())
        data_mat[1, i, 0] = x2[i] * len - 5
        data_mat[1, i, 1] = y2[i] * len
    for i in range(n):
        len = 1 + np.sqrt(np.random.rand())
        data_mat[2, i, 0] = x3[i] * len + 5
        data_mat[2, i, 1] = y3[i] * len
    for i in range(n):
        len = 3 + np.sqrt(np.random.rand())
        data_mat[3, i, 0] = x4[i] * len + 5
        data_mat[3, i, 1] = y4[i] * len
    data_mat = np.reshape(data_mat, (n * k, n_dim))
    groups = {idx:[[x,y]] for idx,(x,y) in enumerate(data_mat.tolist())}
    return groups


def cal_distance(cluster1,cluster2):
    # 采用最小距离作为聚类标准
    min_distance=10000
    for x1,y1 in cluster1:
        for x2,y2 in cluster2:
            distance=(x1-x2)**2+(y1-y2)**2
            if distance<min_distance:
                min_distance=distance
    return min_distance


def h_clustering(groups=create_sample_h()):
    while len(groups) != 1:
        # 判断是不是所有的数据是不是归为了同一类
        min_distance=10000
        for i in list(groups.keys()):
            for j in list(groups.keys()):
                if i>=j:
                    continue
                distance=cal_distance(groups[i], groups[j])
                if distance < min_distance:
                    min_distance = distance
                    min_i = i
                    min_j = j   # 这里的j>i
        groups[min_i].extend(groups.pop(min_j))
        if len(list(groups.keys()))==4:
            break

    cs = ['r', 'g', 'b', 'y']
    cat = 0

    for i in list(groups.keys()):
        plot_data(np.array(groups[i]), cs[cat])
        cat += 1
    plt.show()


def k_mean(data_set, k):
    m, n = np.shape(data_set)  # 获取行数，列数
    cluster_assignment = np.mat(np.zeros((m, n)))
    # 转换为矩阵， zeros生成指定维数的全0数组，用来记录迭代结果
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
    distance = np.sqrt(sum(np.power(vec_a - vec_b, 2)))
    # power(x1, x2) 对x1中的每个元素求x2次方。不会改变x1上午shape。
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
    plt.plot(samples[:, 0], samples[:, 1], plot_type, markerfacecolor=color, markersize=4)


data = create_sample(4)
k_mean(data['data_mat'], 2)
h_clustering()

end = time.clock()
print('Running time: %s Seconds'%(end-start))