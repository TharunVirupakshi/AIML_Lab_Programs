from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np


class KNN():
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def __predict(self, x):
        distances = [np.linalg.norm(x-x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_lables = [self.y_train[i] for i in k_indices]
        most_common = Counter(k_nearest_lables).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        y_pred = [self.__predict(x) for x in X]
        return np.array(y_pred)

iris = load_iris()
X, y = iris.data, iris.target
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

knn = KNN(k=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Classification Report\n")
print(classification_report(y_test, y_pred))
print("Confusion Matrix\n")
print(confusion_matrix(y_test, y_pred))
    
