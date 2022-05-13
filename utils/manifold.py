import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn import neighbors


def gaussian_weight_matrix(dist_mat, epsilon="max", n_neighbors=5, alpha=1):
    if epsilon == "mean":
        # Mean distance to nth nearest neighbor
        nn_mean = np.mean(np.sort(dist_mat)[:, n_neighbors + 1])
        print(f"nn_mean = {nn_mean}")
        epsilon = 2 * nn_mean**2
        print(f"epsilon = {epsilon}")
    elif epsilon == "max":
        # Max distance to nth nearest neighbor
        nn_max = np.max(np.sort(dist_mat)[:, n_neighbors + 1])
        print(f"nn_max = {nn_max}")
        epsilon = 2 * nn_max**2
        print(f"epsilon = {epsilon}")

    # Gaussian weight kernel
    weight_mat = np.exp(-(dist_mat**2) / epsilon)

    # Normalize
    # alpha = 1.0   (Laplace-Beltrami)
    # alpha = 0.5   (Fokker--Planck diffusion)
    # alpha = 0.0   (classical graph Laplacian)
    alpha_norm = weight_mat.sum(axis=-1) ** alpha
    weight_mat /= alpha_norm
    weight_mat /= alpha_norm[:, None]

    return weight_mat, epsilon


def diffusion_map(
    dist_mat,
    n_components=20,
    epsilon="max",
    n_neighbors=5,
    alpha=1,
    robust=True,
    v0=None,
):
    sym_laplacian, epsilon = gaussian_weight_matrix(
        dist_mat, epsilon=epsilon, n_neighbors=n_neighbors, alpha=alpha
    )

    if robust:
        # Noise robust diffusion map
        # https://cpsc.yale.edu/sites/default/files/files/tr1497.pdf
        # https://arxiv.org/pdf/1405.6231.pdf
        diag = sym_laplacian.diagonal().copy()
        np.fill_diagonal(sym_laplacian, 0)

    # Normalize L = D^(-1/2) W D^(-1/2)
    sqrt_norm = np.sqrt(sym_laplacian.sum(axis=-1))  # - sym_laplacian.diagonal())
    sym_laplacian /= sqrt_norm
    sym_laplacian /= sqrt_norm[:, None]

    # Graph laplacian GL = 1 - L
    sym_laplacian *= -1
    if robust:
        mean_shift = np.mean(diag / sqrt_norm**2)
        print("mean_shift =", mean_shift)
        np.fill_diagonal(sym_laplacian, 1 - mean_shift)
    else:
        np.fill_diagonal(sym_laplacian, 1 + sym_laplacian.diagonal())

    # Compute largest eigenvalues/eigenvectors of -GL (maximum eigenvalue of 1)
    evals, embedding = eigsh(
        -sym_laplacian, k=n_components + 1, which="LM", sigma=1.0, v0=None
    )

    # Drop largest eigenvalue = 1 and
    # rescale eigenvectors to obtain eigenvectors of D^-1 W
    embedding = embedding[:, n_components - 1 :: -1] / sqrt_norm[:, None]
    evals = evals[n_components - 1 :: -1]

    # Renormalize eigenvectors
    embedding = (
        np.sqrt(embedding.shape[0]) * embedding / np.linalg.norm(embedding, axis=0)
    )

    return evals, embedding, epsilon


def heuristic_importance_score(
    evals,
    embedding,
    threshold=0.8,
    n_neighbors=5,
    weights="adjusted",
    reweight=None,
    output_raw_scores=False,
):
    n_components = evals.shape[0]
    if weights == "simple":
        weights = np.sqrt(evals[0] / evals)
    elif weights == "adjusted":
        relevant_idx = evals > -1
        weights = np.empty_like(evals)
        weights[~relevant_idx] = 0
        weights[relevant_idx] = np.sqrt(
            np.log(1 + evals[0]) / np.log(1 + evals[relevant_idx])
        )
    elif weights == "constant":
        weights = np.ones_like(evals)

    if reweight is not None:
        weights *= reweight

    n_trajectories = embedding.shape[0]
    n_off_diag_entries = n_trajectories * (n_trajectories - 1)
    mean_pairwise_dist_0 = np.sum(
        np.abs(embedding[:, None, 0] - embedding[None, :, 0]) / n_off_diag_entries
    )

    nearest_neighbors = neighbors.KNeighborsTransformer(n_neighbors=n_neighbors)

    embed_list = [0]
    scores_pass = [mean_pairwise_dist_0 * weights[0]]
    if output_raw_scores:
        raw_scores = [mean_pairwise_dist_0]
    assert scores_pass[0] >= threshold
    scores_fail = [0]

    for i in range(1, n_components):
        current_embedding = embedding[:, embed_list]
        candidate_vec = embedding[:, i]
        nearest_neighbors.fit(current_embedding)
        nn_ind = nearest_neighbors.kneighbors(return_distance=False)

        score = np.mean(np.abs(candidate_vec[:, None] - candidate_vec[nn_ind]))

        if output_raw_scores:
            raw_scores.append(score)

        score *= weights[i]

        if score > threshold:
            embed_list.append(i)
            scores_pass.append(score)
            scores_fail.append(0)
        else:
            scores_fail.append(score)
            scores_pass.append(0)

    if output_raw_scores:
        return embed_list, scores_pass, scores_fail, raw_scores, weights

    return embed_list, scores_pass, scores_fail
