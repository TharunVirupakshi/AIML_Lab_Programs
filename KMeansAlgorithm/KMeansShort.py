import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
class_names = iris.target_names


def KMeans(X, k):
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    for _ in range(100):
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

    return centroids, labels

k = 3
centroids, labels = KMeans(X, k)

fig, axs = plt.subplots(1,2, figsize=(12,5))

axs[0].scatter(X[:, 0], X[:, 1])
axs[0].set_title('Original Data')
axs[0].set_xlabel('Sepal Length')
axs[0].set_ylabel('Sepal Width')

colors = ['r', 'g', 'b']

for i in range(k):
    axs[1].scatter(X[labels == i, 0], X[labels == i, 1], c=colors[i], label=f"Cluster {i}")

axs[1].scatter(centroids[:,0],centroids[:,1], c="black", marker="x", label="Centroids")
axs[1].set_title('Clustered Data')
axs[1].set_xlabel('Sepal Length')
axs[1].set_ylabel('Sepal Width')
axs[1].legend()
plt.tight_layout()
plt.show()
