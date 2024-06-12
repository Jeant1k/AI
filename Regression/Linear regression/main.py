import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection


CarPricesDataNumeric = pd.read_pickle('CarPricesData.pkl')

data_train, data_test = sk.model_selection.train_test_split(CarPricesDataNumeric, train_size=0.8)

# Предсказание цены относительно возраста машины

X, Y = data_train["Age"], data_train["Price"]
X_t, Y_t = data_test["Age"], data_test["Price"]


n = len(X)
a = (X.sum() * Y.sum() - n * (X * Y).sum()) / (X.sum() * X.sum() - n * (X * X).sum())
b = (Y.sum() - a * X.sum()) / n
print(f"a = {a}, b = {b}")


plt.scatter(X, Y)
X_dots = [X.min(), X.max()]
Y_dots = [a * x + b for x in X_dots]
plt.plot(X_dots, Y_dots, c='red')


def MSE(y, pr_y):
    return np.sqrt(((y - pr_y) * (y - pr_y)).sum() / len(y)) / pr_y.mean()

def MAE(y, pr_y):
    return (abs(y - pr_y)).sum() / len(y) / pr_y.mean()

print(f'MSE = {MSE(a * X_t + b, Y_t)}, MAE = {MAE(a * X_t + b, Y_t)}')

plt.show()
