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

    #寄与率計算
    value, vec = e[::-1], A[:, ::-1]                # 要素を逆順に並び替える
    rank = np.linalg.matrix_rank(K_corr) 
    print("rank", rank)
    value, vec = value[:rank], vec[:, :rank]

    # ret_dim = np.zeros(6)
    # flg = np.zeros(6)

    # c_rate = value/np.sum(value)
    # for j in range(len(c_rate)):      
    #     if np.sum(c_rate[:j]) > 0.60 and flg[0] == 0:
    #         ret_dim[0] =  j
    #         flg[0] = 1
    #     if np.sum(c_rate[:j]) > 0.70 and flg[1] == 0:
    #         ret_dim[1] =  j
    #         flg[1] = 1
    #     if np.sum(c_rate[:j]) > 0.80 and flg[2] == 0:
    #         ret_dim[2] =  j
    #         flg[2] = 1
    #     if np.sum(c_rate[:j]) > 0.85 and flg[3] == 0:
    #         ret_dim[3] =  j
    #         flg[3] = 1
    #     if np.sum(c_rate[:j]) > 0.90 and flg[4] == 0:
    #         ret_dim[4] =  j
    #         flg[4] = 1
    #     if np.sum(c_rate[:j]) > 0.95 and flg[5] == 0:
    #         ret_dim[5] =  j
    #         flg[5] = 1
    #     if flg[5] != 0:
    #         break
    # print("kiyoritsu subdim sigma=", sigma)
    # print(ret_dim)

    return np.fliplr(V[:, -n_subdims:])