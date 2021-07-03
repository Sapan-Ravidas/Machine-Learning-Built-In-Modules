# In PCA we try to find new component axes onto which we project our data such that maximize the variance on the new axis
# PCA in unsupervised techniques

# In LDA we want to find such axis so the class separation is maximized
# in LDA we know the feature labels, so it is a supervised technique
import numpy as np

class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        
        # eigen vectors
        self.linear_discriminants = None
        
    def fit(self, X, y):
        ''' y is passed because it is a supervised technique'''
        n_features = X.shape[1]
        class_label = np.unique(y)
        
        # scatter-matrix for within class S_W, scatter-matrix for between class W_B
        mean_overall = np.mean(X, axis=0)
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        for c in class_label:
            Xc = X[y == c]
            mean_c = np.mean(Xc, axis=0)
            
            # details behind this in notebook 
            # (4, n_c) * (n_c, 4) = (4, 4)
            S_W += (Xc - mean_c).T.dot(Xc - mean_c)
            
            n_c = Xc.shape[0]
            
            # (4, 1) * (4, 1)T = (4, 4)
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * (mean_diff).dot(mean_diff.T)
            
        A = np.linalg.inv(S_W).dot(S_B)
        
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvectors = eigenvectors.T
        indecis = np.argsort(abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[indecis]
        eigenvectors = eigenvectors[indecis]
        
        self.linear_discriminants = eigenvectors[: self.n_components]
        
    
    def transform(self, X):
        return np.dot(X, self.linear_discriminants.T)
    