from collections import Counter
import numpy as np
from utils import euclidean_distance

class KNN:
    def __init__(self, K):
        self.K = K
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
        
    def _predict(self, x):
        # compute distances 
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[: self.K]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # get majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
        