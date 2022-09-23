import numpy as np


def subspace_bases(X, n_subdims=1, eps=1e-6):
    # X - matrix, NUM_FEATURES x NUM_SAMPLES
    # print(f"Input shape: {X.shape}")

    # if dimensionality is smaller than number of samples perform regular PCA
    if X.shape[0] <= X.shape[1]:
        print("Performing regular PCA.")
        X_corr = X @ X.T  # correlation matrix, NUM_FEATURES x NUM_FEATURES
        w, V = np.linalg.eigh(X_corr)  # eigenvalue decomposition
    # otherwise perform PCA on dual vectors
    else:
        print("Performing dual-vector PCA.")
        X_corr = X.T @ X  # correlation matrix, NUM_SAMPLES x NUM_SAMPLES
        e, A = np.linalg.eigh(X_corr)  # eigenvalue decomposition

        # replace if there are too small eigenvalues
        e[(e < eps)] = eps

        # TODO: what is this actually?
        A = A / np.sqrt(e)
        V = X @ A

    return np.fliplr(V[:, -n_subdims:])  # select top n_subdim eigenvectors


def gds_bases(all_bases, n_gds_dims, eps=1e-6):
    # X - matrix, NUM_FEATURES x NUM_SAMPLES
    # print(f"Input shape: {all_bases.shape}")

    # if dimensionality is smaller than number of samples perform regular PCA
    if all_bases.shape[0] <= all_bases.shape[1]:
        # print("Performing regular PCA.")
        X_corr = (
            all_bases @ all_bases.T
        )  # correlation matrix, NUM_FEATURES x NUM_FEATURES
        w, V = np.linalg.eigh(X_corr)  # eigenvalue decomposition
    # otherwise perform PCA on dual vectors
    else:
        # print("Performing dual-vector PCA.")
        X_corr = (
            all_bases.T @ all_bases
        )  # correlation matrix, NUM_SAMPLES x NUM_SAMPLES
        e, A = np.linalg.eigh(X_corr)  # eigenvalue decomposition

        # replace if there are too small eigenvalues
        e[(e < eps)] = eps

        # TODO: what is this actually?
        A = A / np.sqrt(e)
        V = all_bases @ A

    # TODO: Eliminate eigenvectors with small eigenvalues to eliminate noise
    return np.fliplr(V[:, 0:n_gds_dims])  # select lowest n_gds_dims eigenvectors


def gds_bases_projection(gds, bases):

    # bases_proj, (n_gds_dims, n_subdims)
    bases_proj = gds.T @ bases
    orth_bases = [np.linalg.qr(base_proj)[0] for base_proj in bases_proj]

    return np.array(orth_bases)

def gds_data_projection(gds, X):
    return gds.T @ X


def similarity(X, bases):
    proj = np.dot(bases.transpose(0, 2, 1), X)
    norm = np.linalg.norm(X, axis=0)
    sim = ((proj ** 2) / (norm ** 2)).sum(axis=1)

    return sim.T


def predict(similarities):
    return np.argmax(similarities, axis=1)
