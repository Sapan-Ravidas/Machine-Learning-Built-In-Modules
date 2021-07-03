import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt
from termcolor import colored
from linearReg import LinearRegression
from utils import mse

# noise -> float, default=0.0
# The standard deviation of the gaussian noise applied to the output.
X_data, target_data = datasets.make_regression(n_samples=100, n_features=1, noise=20)
X_train, X_test, y_train, y_test = train_test_split(X_data, target_data, test_size=0.25)

print(colored(f"Actual Data Size (excluding taret column): (Row X Columns) : { colored(X_data.shape, 'green', attrs = ['bold'])}", 'red'))
print(colored(f"Training X data size (row x columns): {colored(X_train.shape, 'green', attrs = ['bold'])}", 'red'))

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

mse_value = mse(y_test, predictions)
print(mse_value)

cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(12, 7))
plt.scatter(
    x = X_train,
    y = y_train,
    color = cmap(0.9),
    marker='o',
    s = 30
)

plt.scatter(
    x = X_test,
    y = y_test,
    color=cmap(0.5),
    marker='o',
    s = 30
    )

plt.plot(X_data, model.predict(X_data), linewidth=2, label="prediction")

plt.show()