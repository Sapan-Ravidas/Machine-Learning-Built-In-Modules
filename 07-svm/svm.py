import numpy as np

class SVM:
    def __init__(self, lr = 0.001, lambda_param = 0.01, n_iters = 1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weight = None
        self.bias = None
    
    def fit(self, X, y):
        y_ = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        
        self.weight = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            for index, xi in enumerate(X):
                condition = y_[index] + (np.dot(xi, self.weight) - self.bias) >= 1
                if condition:
                    self.weight -= self.lr * (2 * self.lambda_param * self.weight)
                else:
                    self.weight -= self.lr * (2 * self.lambda_param * self.weight - np.dot(xi, y_[index]))
                    self.bias -= self.lr * y_[index]
            
    
    def predict(self, X):
        linear_output = np.dot(X, self.weight) - self.bias
        return np.sign(linear_output)
        