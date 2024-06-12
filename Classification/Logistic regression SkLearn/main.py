import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.datasets
import sklearn.linear_model
import sklearn.preprocessing
import sklearn.metrics
import sklearn.multiclass


data = pd.read_csv("winequality-red.csv")

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']

for i in features + ['quality']:
    count = data[i].isna().sum()
    if count != 0:
        print(f'В столбце {i} пропущено {count} значений')

X, Y = data[features], data['quality']
X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, test_size=0.2, shuffle=True)


# scaler = sk.preprocessing.StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

model = sk.linear_model.LogisticRegression(solver='newton-cg', tol=0.1)

model.fit(X_train, Y_train)

print(sk.metrics.accuracy_score(Y_test, model.predict(X_test)))

sk.metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test)
plt.show()

