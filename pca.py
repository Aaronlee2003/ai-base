import numpy as np
#pca s reductionof dimensionality technique
#information about the dataset 
#information amount mean variance ratio
#first we project to certain axis defined by the principal components
# we choose the axis that maximizes the variance of the projected data
# and second axis is chosen orthogonal to the first one with maximum variance
# this process continues for all principal components
#why we use this  reduce the variance while retaining most of the information
#reduce multicollinearity by projecting data onto orthogonal axes
#visualize the variance explained by each principal component
# 100 demension reductiin to  2 or 3 for visualization purposes


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None #main components vectors
        self.mean = None
        
        
    def fit(self, X):
        #mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        #covariance matrix
        n_samples = X.shape[0]
        X_centered = X - self.mean
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        #eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        #sort eigenvectors by eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        self.components = eigenvectors[:, :self.n_components]
        
        
        
        self.components=eigenvectors[:, :self.n_components]
        
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = (
            eigenvalues[: self.n_components] / total_variance
        )

        return self
    def transform(self, X):
        # move to center the data by subtracting the mean
        X = X - self.mean
        # project the centered data onto the principal components
        return np.dot(X, self.components)
    def fit_transform(self, X):
        #
        self.fit(X)
        return self.transform(X)