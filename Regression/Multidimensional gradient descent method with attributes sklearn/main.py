import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
import sklearn.linear_model
import sklearn.compose


CarPricesDataNumeric = pd.read_pickle('CarPricesData.pkl')

data_train, data_test = sk.model_selection.train_test_split(CarPricesDataNumeric, train_size=0.8)

# Предсказание цены относительно всех параметров и производных параметров

numeric = ["Age", "KM", "Weight", "HP", "CC"]
categorical = ["MetColor", "Doors"]

transformer = sk.compose.make_column_transformer(
    (sk.preprocessing.StandardScaler(), ['Age']),
    (sk.preprocessing.StandardScaler(), ['KM']),
    (sk.preprocessing.StandardScaler(), ['Weight']),
    (sk.preprocessing.StandardScaler(), ['HP']),
    (sk.preprocessing.StandardScaler(), ['CC']),
    (sk.preprocessing.OneHotEncoder(), ['MetColor']),
    (sk.preprocessing.OneHotEncoder(), ['Doors'])
)

X = transformer.fit_transform(data_train[numeric + categorical])
Y = data_train["Price"]

poly = sk.preprocessing.PolynomialFeatures()
Xp = poly.fit_transform(X)
print(f'Было признаков = {X.shape[1]}, стало = {Xp.shape[1]}')

model = sk.linear_model.LinearRegression()
model = model.fit(Xp, Y)


X_t = transformer.transform(data_test[numeric + categorical])
X_t_p = poly.fit_transform(X_t)
Y_t = data_test["Price"]

Y_t_pr = model.predict(X_t_p)


def MSE(y, pr_y):
    return np.sqrt(((y - pr_y) * (y - pr_y)).sum() / len(y)) / Y_t.mean()


def MAE(y, pr_y):
    return (abs(y - pr_y)).sum() / len(y) / Y_t.mean()


print(f"Coefficients = {model.coef_}, bias = {model.intercept_}")
print(f'MSE = {MSE(Y_t_pr, Y_t)}, MAE = {MAE(Y_t_pr, Y_t)}')
