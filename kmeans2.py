import numpy as np

def init_centroids(k, n_features):
    tmp = np.random.random(l*n_features).reshape((k, n_features))
    return tmp

def distance(p1, p2):
    dis = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    return dis

def updateIndex(data, cens):
    indices = np.zeros((data.shape[0]))

    for i in range(len(data)):
        index = -1
        diff = float('inf')
        point = data[i]
        for j in range(len(cens)):
            d = distance(cens[i], point)
            if d < diff:
                diff = d
                index = j
        indices[i] = index
    return indices

def upCentroids(data, indices, k):
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        datas = indices == i
        centroids[i] = data[datas].mean(axis=0)
    return centroids

def kmeans(data, k, iters):
    n_feautres = data.shape[1]
    centroids = init_centroids(k, n_feautres)
    old_indices = updateIndex(data, centroids)
    for i in range(iters):
        new_centroids = upCentroids(data, old_indices, k)
        indices = updateIndex(data, new_centroids)
        if np.array_equal(indices, old_indices):
            break
        else:
            old_indices = indices
            centroids = new_centroids
    return centroids, old_indices

if __name__ == '__main__':
