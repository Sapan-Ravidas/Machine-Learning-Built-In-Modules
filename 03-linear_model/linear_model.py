import numpy as np
import decimal

class LinearModel:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights, self.bias = np.zeros(n_features), 0
        
        for _ in range(self.n_iters):
            y_predicted = self._approximations(X, self.weights, self.bias)
            
            # rate of change of Error function wrt weights
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            
            # rate of change in Error function wrt bias
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    
    def predict(self, X):
        return self._predict(X, self.weights, self.bias)
            
    def _approximations(self, X, weight, bias):
        raise NotImplementedError()
    
    def _predict(self, X, weight, bias):
        raise NotImplementedError()
    
    
# Linear Regression 
class LinearRegression(LinearModel):
    def _approximations(self, X, weights, bias):
        return np.dot(X, weights) + bias
    
    def _predict(self, X, weights, bias):
        return np.dot(X, weights) + bias
    

# Logistic Regression
class LogisticRegression(LinearModel):
    def _approximations(self, X, weights, bias):
        linear_model = np.dot(X, weights) + bias
        return self._sigmoid(linear_model)
    
    def _predict(self, X, weights, bias):
        linear_model = np.dot(X, weights) + bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [ 1 if probability > 0.5 else 0 for probability in y_predicted ]
        return np.array(y_predicted_cls)
    
    def _sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)
    
    

        