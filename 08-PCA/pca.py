import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # mean
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        
        # covariance
        # rows = 1, columns = feature
        cov = np.cov(X.T)
        
        # eigen vector, eigen values
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # eigen vecotors are in returned in form of column matrix
        # sort eigen vecotors
        eigenvectors = eigenvectors.T
        indecis = np.argsort(eigenvalues)[::-1]
        sorted_eigen_values = eigenvalues[indecis]
        eigenvectors = eigenvectors[indecis]
        
        # stire first n eigen vectors
        self.components = eigenvectors[0 : self.n_components]
        # print(self.components)
    
    def transform(self, X):
        X = X - self.mean
        
        # Note: we already transpose the eigen vectors from column vectors to row vectors
        # Now for dot product, we want our column vector
        # print("tranform")
        # print(X)
        return np.dot(X, self.components.T)