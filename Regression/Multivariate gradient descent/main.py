import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection


CarPricesDataNumeric = pd.read_pickle('CarPricesData.pkl')

data_train, data_test = sk.model_selection.train_test_split(CarPricesDataNumeric, train_size=0.8)

# Предсказание цены относительно всех параметров

X, Y = data_train[["Age", "KM", "Weight", "HP", "MetColor", "CC", "Doors"]], data_train["Price"]
X_t, Y_t = data_test[["Age", "KM", "Weight", "HP", "MetColor", "CC", "Doors"]], data_test["Price"]


def MSE(y, pr_y):
    return np.sqrt(((y - pr_y) * (y - pr_y)).sum() / len(y)) / Y_t.mean()

def MAE(y, pr_y):
    return (abs(y - pr_y)).sum() / len(y) / Y_t.mean()


Y_err = []

*a, b = np.random.normal(size=8)
a = np.array(a)
eta = 0.000000000075
for i in range(50000):
    print(f"Эпоха {i + 1}: MAE = {MAE(X @ a + b, Y)}")
    # print(f"a = {a}, b = {b}")
    Y_err.append(MAE(X @ a + b, Y))
    a, b = a - eta * ((X @ a + b - Y) * X.T).mean(axis=1), b - eta * (X @ a + b - Y).mean()

print(f'a = {a}, b = {b}')
print(f'MSE = {MSE(X_t @ a + b, Y_t)}, MAE = {MAE(X_t @ a + b, Y_t)}')

plt.plot([i for i in range(1, 50001)], Y_err, c='blue')

plt.show()