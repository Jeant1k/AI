import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.datasets
import matplotlib.animation as animation

# Создание данных
X, Y = sk.datasets.make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    n_classes=2
)

fig = plt.figure()  # Создание нового окна для графика

ax1 = fig.add_subplot(1, 2, 1)  # Добавление первого подграфика
ax1.scatter(X[:, 0], X[:, 1], c=['purple' if x else 'orange' for x in Y])

# Определение сигмоидной функции
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Инициализация весов и смещения
W = np.random.normal(size=(2,))
b = np.random.normal(size=(1,))

# Определение функции потерь
def loss(W, b):
    h = sigmoid(np.matmul(X, W) + b)
    return (-Y * np.log(h) - (1 - Y) * np.log(1 - h)).mean()

# Определение скорости обучения и количества образцов
eta = 0.1
n = len(X)

ax2 = fig.add_subplot(1, 2, 2)  # Добавление второго подграфика

# Определение функции анимации
def animate(i):
    global W, b
    print(f"{i + 1}\tloss = {loss(W,b)}")
    h = sigmoid(np.matmul(X, W) + b)
    dldw = np.matmul(X.T, h - Y.T) / n
    dldb = (h - Y.T).mean()
    W -= eta * dldw
    b -= eta * dldb

    ax2.clear()
    ax2.scatter(X[:, 0], X[:, 1], c=['purple' if x else 'orange' for x in Y])
    xs = X[:, 0].min(), X[:, 0].max()
    ax2.plot(xs, [-b / W[1] - x * W[0] / W[1] for x in xs])
    plt.axis('tight')

# Создание анимации
ani = animation.FuncAnimation(fig, animate, frames=500, interval=50, repeat=False)

plt.show()
