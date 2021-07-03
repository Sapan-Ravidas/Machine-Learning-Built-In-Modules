import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from termcolor import colored
from knn import KNN

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()

X_data, target_data = iris['data'], iris['target']
print(colored(f"Actual Data Size (excluding taret column): (Row X Columns) : { colored(X_data.shape, 'green', attrs = ['bold'])}", 'red'))

# when random_state set to an integer, train_test_split will return same results for each execution.
# when random_state set to an None, train_test_split will return different results for each execution
# So, I don't want my model to generate same set of data every time, so will not use random state
X_train, X_test, y_train, y_test = train_test_split(
    X_data,
    target_data,
    test_size = 0.25, 
    # random_state = 1234
    )

print(colored(f"Training X data size (row x columns): {colored(X_train.shape, 'green', attrs = ['bold'])}", 'red'))

model = KNN(K = 3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)


# TODO: 
# Two scatter graph will be ploted
# 1. PCA-plot   2. LDA-plot  
fig, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter(
    x = X_data[:, 0], 
    y = X_data[:, 1],
    c = target_data,
    cmap = cmap, 
    s = 20
    )

plt.show()


