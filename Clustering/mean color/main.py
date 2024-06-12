from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
from mpl_toolkits.mplot3d import Axes3D


img = Image.open("metro.jpg")
img = img.resize((150, 200))
img = np.array(img) / 255.0


def show_color(c, ax):
    t = np.zeros(shape=(10, 10, 3))
    t[:] = c
    ax.imshow(t)
    ax.axis('off')


num_clasters = 10
fig, axs = plt.subplots(num_clasters, num_clasters, figsize=(num_clasters * 2 + 1, num_clasters * 2 + 1))

for ax in axs.flatten():
    ax.axis('off')

for n_clusters in range(1, num_clasters + 1):
    X = img.reshape(-1, 3)
    km = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(X)
    for i in range(n_clusters):
        axs[n_clusters - 1, i].axis('on')
        show_color(km.cluster_centers_[i], axs[n_clusters - 1, i])

plt.show()


img_array = np.array(img)

# Создаем 3D график
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Извлекаем цвета из изображения
colors = img_array.reshape(-1, 3)

# Извлекаем координаты RGB
r, g, b = colors[:, 0], colors[:, 1], colors[:, 2]

# Рисуем точки в 3D пространстве
ax.scatter(r, g, b, c=colors, s=5, alpha=1.0, edgecolors='none')

plt.show()
