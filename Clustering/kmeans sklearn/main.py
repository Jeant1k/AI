import sklearn as sk
import sklearn.model_selection
from sklearn.datasets import fetch_openml
import sklearn.cluster
import numpy as np
import matplotlib.pyplot as plt


X = np.load('X.npy')
Y = np.load('Y.npy')

X = X.reshape((2062, 64 * 64))

num_clasters = 100

model = sk.cluster.KMeans(n_clusters=num_clasters).fit(X)


# fig, ax = plt.subplots(1, 50)
# for i, im in enumerate(model.cluster_centers_):
#     ax[i].imshow(im.reshape(64, 64))
#     ax[i].axis('off')
# plt.show()


for c in range(num_clasters):
    res = [x for x, l in zip(X, model.labels_) if l == c]

    fig, ax = plt.subplots(5, 4, figsize=(20, 25))
    for i, im in enumerate(res[:20]):
        row = i // 4
        col = i % 4
        ax[row, col].imshow(im.reshape(64, 64))
        ax[row, col].axis('off')
    plt.show()
