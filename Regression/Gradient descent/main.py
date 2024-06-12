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


def MSE(y, pr_y):
    return np.sqrt(((y - pr_y) * (y - pr_y)).sum() / len(y)) / Y_t.mean()

def MAE(y, pr_y):
    return (abs(y - pr_y)).sum() / len(y) / Y_t.mean()


Y_err = []

a, b = np.random.normal(size=2)
eta = 0.0005
for i in range(50000):
    print(f"Эпоха {i + 1}: MAE = {MAE(a * X + b, Y)}")
    # print(f"a = {a}, b = {b}")
    Y_err.append(MAE(a * X + b, Y))
    a_new = a - eta * ((a * X + b - Y) * X).mean()
    b_new = b - eta * (a * X + b - Y).mean()
    a, b = a_new, b_new

print(f'MSE = {MSE(a * X_t + b, Y_t)}, MAE = {MAE(a * X_t + b, Y_t)}')

plt.scatter(X, Y)
X_dots = [X.min(), X.max()]
Y_dots = [a * x + b for x in X_dots]
plt.plot(X_dots, Y_dots, c='blue')

plt.show()

plt.plot([i for i in range(1, 50001)], Y_err, c='blue')

plt.show()