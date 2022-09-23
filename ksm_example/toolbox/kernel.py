import numpy as np


def kernel_similarity(kernel_base, X, X_test):
    K = rbf_kernel(X, X_test)  # calculate kernel of test data
    k_proj = np.dot(K.T, kernel_base).T  # project on base
    k_norm = np.linalg.norm(K, axis=0)  # calculate norm

    return ((k_proj ** 2) / (k_norm ** 2)).sum(axis=0)


def l2_kernel(X, Y):
    # |x-y|^2 = |x|^2 - 2x@y + |y|^2
    XX = (X ** 2).sum(axis=0)[:, None]
    YY = (Y ** 2).sum(axis=0)[None, :]
    x = XX - 2 * (X.T @ Y) + YY

    return x


def rbf_kernel(X, Y, sigma=None):

    n_dims = X.shape[0]
    if sigma is None:
        sigma = np.sqrt(n_dims / 2)

    x = l2_kernel(X, Y)

    K = np.exp(-0.5 * x / (sigma ** 2))
    # print(K.shape)
    return K


def kernel_subspace_bases(X, n_subdims=None, sigma=None, eps=1e-6):
    K = rbf_kernel(X, X, sigma=sigma)

    K_corr = K.T @ K
    e, A = np.linalg.eigh(K_corr)

    e[(e < eps)] = eps
    A = A / np.sqrt(e)
    V = K @ A

    return np.fliplr(V[:, -n_subdims:])