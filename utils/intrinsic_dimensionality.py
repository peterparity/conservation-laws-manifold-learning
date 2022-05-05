import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import random

from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from sklearn.linear_model import LinearRegression

from ott.core.sinkhorn import sinkhorn
from ott.geometry.epsilon_scheduler import Epsilon
from ott.geometry.geometry import Geometry


def local_PCA(dist_mat, n_neighbors=20, n_mds_components=20):
    embedding = MDS(
        n_components=n_mds_components, dissimilarity="precomputed", random_state=0
    ).fit_transform(dist_mat)

    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(embedding)
    _, nn_ind = nn.kneighbors()
    nn_ind = np.concatenate((np.arange(embedding.shape[0])[:, None], nn_ind), axis=1)
    pca = PCA(n_components=n_neighbors)

    explained_variance_ratios = []
    for i in range(nn_ind.shape[0]):
        pca.fit(embedding[nn_ind[i, :]])
        explained_variance_ratios.append(pca.explained_variance_ratio_)
    explained_variance_ratios = np.stack(explained_variance_ratios)

    def c(x):
        return np.cos(x) * (x < np.pi / 2)

    estimated_dim = np.sum(
        1 - c(np.pi * n_mds_components * explained_variance_ratios.mean(0))
    )
    return estimated_dim, explained_variance_ratios


def MLE_estimator(dist_mat, n_neighbors=20):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed")
    nn.fit(dist_mat)
    nn_dist, _ = nn.kneighbors()

    log_nn_dist = np.log(nn_dist)
    m = 1 / (
        1 / (n_neighbors - 1) * np.sum(log_nn_dist[:, [-1]] - log_nn_dist, axis=-1)
    )
    estimated_dim = m.mean()
    return estimated_dim, m


def wasserstein_estimator(
    dist_mat, n_neighbors=5, n_samples=100, ns=np.arange(2, 90), key=random.PRNGKey(0)
):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed")
    nn.fit(dist_mat)
    knn_graph = nn.kneighbors_graph(mode="distance")
    manifold_dist = jnp.asarray(shortest_path(knn_graph, directed=False))

    sinkhorn_W1 = jit(
        lambda p1, p2: sinkhorn(
            Geometry(
                cost_matrix=manifold_dist[jnp.ix_(p1, p2)],
                epsilon=Epsilon(
                    target=1e-3,
                    init=5.0,
                    decay=0.999,
                ),
            ),
            threshold=1e-3,  # 1e-2,
            max_iterations=10000,  # 2000,
        )
    )

    def W1(keys, n):
        p1 = random.choice(keys[0], dist_mat.shape[0], (n,), replace=False)
        p2 = random.choice(keys[1], dist_mat.shape[0], (n,), replace=False)
        sol = sinkhorn_W1(p1, p2)
        return sol.reg_ot_cost, sol.converged

    W1 = vmap(W1, in_axes=(0, None))

    log_ns = np.log(ns)
    keys = random.split(key, len(ns) * n_samples * 2).reshape(len(ns), n_samples, 2, 2)
    log_W1s = jnp.log(
        jnp.stack(list(map(lambda keys, n: jnp.mean(W1(keys, n)[0]), keys, ns)))
    )

    reg = LinearRegression().fit(log_ns[:, None], log_W1s)
    estimated_dim = -1 / reg.coef_[0]
    return estimated_dim, log_ns, log_W1s
