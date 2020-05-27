import numpy as np


def init_centroids(k, n_features):
    tmp = np.random.random(k*n_features).reshape((k, n_features))
    return tmp

def distance(point, centroid):
    tmp = (point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2
    return tmp

def upindex(data, centroids):
    n_samples = len(data)
    indices   = np.zeros((n_samples))
    for i in range(len(data)):
        point = data[i]
        index = -1
        diff  = np.float('inf')
        for j in range(len(centroids)):
            dis = distance(point, centroids[j])
            if dis < diff:
                diff = dis
                index = j
        indices[i] = index
    return indices

def uppoints(data, indices):
    k = int(max(indices) + 1)
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        indexs = indices == i
        new_centroids[i] = data[indexs].mean(axis=0)
    return new_centroids

def kmeans(data, k, iters):
    n_features = data.shape[1]
    centroids = init_centroids(k, n_features)
    old_indices = upindex(data, centroids)
    for iter in range(iters):
        new_centroids = uppoints(data, old_indices)
        indices   = upindex(data, new_centroids)
        if np.array_equal(indices, old_indices):
            break
        else:
            centroids = new_centroids
            old_indices = indices

    return centroids, old_indices

if __name__ == '__main__':
    k = 3
    iters = 20

    data = '1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,6,0.403,0.237,7,0.481,0.149,8,0.437,' \
           '0.211,9,0.666,0.091,10,0.243,0.267,11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,' \
           '0.37,16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,21,0.748,0.232,22,0.714,' \
           '0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,' \
           '0.445,30,0.446,0.459'

    a = data.split(',')
    dataset = [(float(a[i]), float(a[i + 1])) for i in range(1, len(a), 3)]
    dataset = np.array(dataset)
    # print(dataset.shape)
    centroids, old_indices = kmeans(dataset, 3, 20)

    import matplotlib.pyplot as plt
    colValue = ['r', 'y', 'g', 'b', 'c', 'k', 'm']
    for i in range(len(centroids)):
        indexs = old_indices == i
        points = dataset[indexs]
        plt.scatter(points[:, 0], points[:, 1], marker='x', color=colValue[i%len(colValue)], label=i)

    plt.show()