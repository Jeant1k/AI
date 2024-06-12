import sklearn as sk
import sklearn.model_selection
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt


X = np.load('X.npy')
Y = np.load('Y.npy')

print(X.shape)
print(Y.shape)


def display_two_images(img1, img2):
    img1 = img1.reshape((64, 64))
    img2 = img2.reshape((64, 64))

    fig, axes = plt.subplots(1, 2)

    axes[0].imshow(img1, cmap='gray')
    axes[0].axis('off')

    axes[1].imshow(img2, cmap='gray')
    axes[1].axis('off')

    plt.show()


X = X.reshape((2062, 64 * 64))



def dist(X, Y):
    return np.sum((X - Y) * (X - Y))


def array_to_number(X):
    return np.where(X == 1.)[0][0]


# for _ in range(15):
#     i, j = np.random.randint(0, len(X), size=2)
#     print(f"Distance between {array_to_number(Y[i])} and {array_to_number(Y[j])} is {dist(X[i], X[j])}")
#     display_two_images(X[i], X[j])




X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(X, Y, train_size=1650, test_size=412, shuffle=True)


def classify(input_img):
    i = np.argmin([dist(x, input_img) for x in X_train])
    return Y_train[i]


n = 20
correct = 0
for x, y in zip(X_test[:n], Y_test[:n]):
    r = classify(x)
    print(f"{array_to_number(y)} classified as {array_to_number(r)}")
    correct += (y == r)
print(f"Accuracy: {np.mean(correct / n)}")

