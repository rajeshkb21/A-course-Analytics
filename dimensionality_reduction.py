import numpy as np

def perform_PCA(X, k):
    cov = np.cov(X)
    eigvecs, eigvals = np.linalg.eigvals(cov)
    sorted_inds = eigvals.argsort()[::-1]
    eigvals = eigvals[sorted_inds]
    eigvecs = eigvals[:,sorted_inds]
    W = eigvecs[:,0:k]
    X_transf = np.matmul(X,W)
    return X_transf, eigvals


def perform_kernel_PCA(X,k):
    return