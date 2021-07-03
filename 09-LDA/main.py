import numpy as np
from sklearn import  datasets
from lda import LDA
from matplotlib import pyplot as plt
from termcolor import colored

data = datasets.load_iris()
X = data.data
y = data.target
print(colored(f"X data size : {colored(X.shape, 'magenta', attrs = ['bold'])}", 'cyan'))

lda = LDA(2)
lda.fit(X, y)
X_projected = lda.transform(X)
print(colored(f"X data size after LDA transform : {colored(X_projected.shape, 'magenta', attrs = ['bold'])}", 'cyan'))


x1 = X_projected[:, 0]
x2 = X_projected[:, 1]

plt.scatter(x = x1, y = x2, c = y, edgecolors='none', alpha=0.8, cmap = plt.cm.get_cmap('viridis', 3))
plt.colorbar()
plt.show()

