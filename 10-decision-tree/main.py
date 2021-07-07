import numpy as np
from decision_tree import DecisionTree
from sklearn.model_selection import train_test_split
from sklearn import datasets

def accuracy(y_true, y_predicted):
    accuracy = np,sum(y_true == y_predicted) / len(y_true)
    return accuracy

data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = DecisionTree(max_depth = 10)
model.fit(X_train, y_train)

y_predicted= model.predict(X_test)
accu = accuracy(y_test, y_predicted)

print("Accuracy: ", accu)

