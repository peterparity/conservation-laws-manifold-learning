import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn import neighbors


def weight_matrix(dist_mat, epsilon=None, n=5, alpha=1):
    if epsilon is None:
        # Mean distance to {n}th nearest neighbor
        nn_mean = np.mean(np.sort(dist_mat)[:, n + 1])
        print(f"nn_mean = {nn_mean}")
        epsilon = 2 * nn_mean**2
        print(f"epsilon = {epsilon}")

    # Gaussian weight kernel
    weight_mat = np.exp(-(dist_mat**2) / epsilon)

    # Normalization
    # alpha = 1 (Laplace-Beltrami), alpha = 0.5 (Fokker--Planck diffusion)
    alpha_norm = weight_mat.sum(axis=-1) ** alpha
    weight_mat /= alpha_norm
    weight_mat /= alpha_norm[:, None]

    return weight_mat


def diffusion_map(dist_mat, n_components=20, epsilon=None, n=5, alpha=1, robust=True):
    sym_laplacian = weight_matrix(dist_mat, epsilon=epsilon, n=n, alpha=alpha)

    if robust:
        # Noise robust diffusion map
        # https://cpsc.yale.edu/sites/default/files/files/tr1497.pdf
        # https://arxiv.org/pdf/1405.6231.pdf
        np.fill_diagonal(sym_laplacian, 0)

    # Normalize L = D^(-1/2) W D^(-1/2)
    sqrt_norm = np.sqrt(sym_laplacian.sum(axis=-1))
    sym_laplacian /= sqrt_norm
    sym_laplacian /= sqrt_norm[:, None]

    # Graph laplacian GL = 1 - L
    sym_laplacian *= -1
    if robust:
        np.fill_diagonal(sym_laplacian, 1)
    else:
        np.fill_diagonal(sym_laplacian, sym_laplacian.diagonal() + 1)

    # Compute largest eigenvalues/eigenvectors of -GL (maximum eigenvalue of 1)
    evals, embedding = eigsh(-sym_laplacian, k=n_components + 1, which="LM", sigma=1.0)

    # Drop largest eigenvalue = 1 and
    # rescale eigenvectors to obtain eigenvectors of D^-1 W
    embedding = embedding[:, n_components - 1 :: -1] / sqrt_norm[:, None]
    evals = evals[n_components - 1 :: -1]

    # Renormalize eigenvectors
    embedding = (
        np.sqrt(embedding.shape[0]) * embedding / np.linalg.norm(embedding, axis=0)
    )

    return evals, embedding


def heuristic_importance_score(evals, embedding, threshold=0.8, n_neighbors=5):
    n_components = evals.shape[0]
    weights = np.sqrt(evals[0] / evals)

    n_trajectories = embedding.shape[0]
    n_off_diag_entries = n_trajectories * (n_trajectories - 1)
    mean_pairwise_dist_0 = np.sum(
        np.abs(embedding[:, None, 0] - embedding[None, :, 0]) / n_off_diag_entries
    )

    nearest_neighbors = neighbors.KNeighborsTransformer(n_neighbors=n_neighbors)

    embed_list = [0]
    scores_pass = [mean_pairwise_dist_0 * weights[0]]
    assert scores_pass[0] >= threshold
    scores_fail = [0]
    for i in range(1, n_components):
        current_embedding = embedding[:, embed_list]
        candidate_vec = embedding[:, i]
        nearest_neighbors.fit(current_embedding)
        nn_ind = nearest_neighbors.kneighbors(return_distance=False)
        score = (
            np.mean(np.abs(candidate_vec[:, None] - candidate_vec[nn_ind])) * weights[i]
        )
        if score > threshold:
            embed_list.append(i)
            scores_pass.append(score)
            scores_fail.append(0)
        else:
            scores_fail.append(score)
            scores_pass.append(0)

    return embed_list, scores_pass, scores_fail
