import numpy as np
import matplotlib.pyplot as plot
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def loss_func(h, y):
    return -y * np.log(h) - (1-y)*np.log(1-h)

def gradient(X,h,y):
    return np.dot(X.T, (h-y))/y.shape[0]

def logistic_regr(X, y, steps=100, lr=0.1):

    weights = np.zeros(X.shape[1])

    for i in range(steps):
        z = np.dot(X, weights)
        h = sigmoid(z)
        gradient_val = gradient(X, h, y)
        weights -= lr * gradient_val
        loss = np.mean(loss_func(h, y))
        print(f"Itr:{i}, weights:{weights}, loss:{loss}")
    
    return weights

iris = datasets.load_iris()

X = iris.data[:, :2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

sc = StandardScaler()

X_train_std = sc.fit_transform(X_train)
X_test_std = sc.fit_transform(X_test)

weigths = logistic_regr(X_train_std, y_train, steps=200, lr=0.01)

y_pred = sigmoid(np.dot(X_test_std, weigths)) > 0.5

print(f'Accuracy: %.4f' % np.mean(y_pred == y_test))

