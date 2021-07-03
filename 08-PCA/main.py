import numpy as np
from sklearn import datasets
from pca import PCA
from matplotlib import pyplot as plt
from termcolor import colored

data = datasets.load_iris()
X = data.data
y = data.target
print(colored(f"X data size : {colored(X.shape, 'magenta', attrs = ['bold'])}", 'cyan'))

pca = PCA(2)
pca.fit(X)
X_tranform  = pca.transform(X)
print(colored(f"X data size after pca transform : {colored(X_tranform.shape, 'magenta', attrs = ['bold'])}", 'cyan'))

x1 = X_tranform[:, 0]
x2 = X_tranform[:, 1]

plt.scatter(x = x1, y = x2, c = y, edgecolors='none', alpha=0.8, cmap=plt.cm.get_cmap('viridis', 3))
plt.colorbar()
plt.show()