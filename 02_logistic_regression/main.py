import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from matplotlib import pyplot as plt
from termcolor import colored
from logistic import LogisticRegression

cancer = datasets.load_breast_cancer()
X_data, target_data = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X_data, target_data, test_size=0.25)


print(colored(f"Actual Data Size (excluding taret column): (Row X Columns) : { colored(X_data.shape, 'green', attrs = ['bold'])}", 'red'))
print(colored(f"Training X data size (row x columns): {colored(X_train.shape, 'green', attrs = ['bold'])}", 'red'))

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy

model = LogisticRegression(lr = 0.0001, n_iters=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accr = accuracy(y_test, predictions)

print(colored(f"Out of {colored(X_data.shape[0], 'green', attrs = ['bold'])}, {colored(X_train.shape[0], 'green', attrs = ['bold'])} are correctly classified", 'red'))
print(colored(f"Score: {colored(accr, 'green', attrs = ['bold'])}", 'blue'))