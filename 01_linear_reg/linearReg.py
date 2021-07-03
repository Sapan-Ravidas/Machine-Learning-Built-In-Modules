import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.01, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        
        # here weights refers to slope and bias refers to intercept on the line
        # Y = weights * x + bias
        self.weights = np.zeros(n_features) # can also use random values
        self.bias = 0
        
        for _ in range(self.n_iters):
            # y = mx + c
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # change in Error function wrt weights
            dw = (1/ n_samples) * np.dot(X.T, y_predicted - y)
            
            # change in Error function wrt bias
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted