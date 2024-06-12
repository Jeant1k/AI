import sklearn as sk
import sklearn.model_selection
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import sklearn.neighbors


X = np.load('X.npy')
Y = np.load('Y.npy')

X = X.reshape((2062, 64 * 64))


X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, train_size=1650, test_size=412, shuffle=True)

res = []
for i in range(1, 10):
    model = sk.neighbors.KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train, Y_train)
    acc = sk.metrics.accuracy_score(Y_test, model.predict(X_test))
    print(f"Accuracy for k={i} is {acc}")
    res.append(acc)

plt.plot(res)
plt.show()