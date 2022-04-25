import jax.numpy as jnp
from jax import vmap

from ott.core.sinkhorn import sinkhorn
from ott.geometry.epsilon_scheduler import Epsilon
from ott.geometry.pointcloud import PointCloud


def wasserstein_metric(
    x,
    y,
    p=2,
    target=1e-1,
    init=10.0,
    decay=0.995,
    threshold=1e-2,
    max_iterations=600,
    **sinkhorn_kwargs
):
    sol = sinkhorn(
        PointCloud(
            x,
            y,
            power=p,
            epsilon=Epsilon(
                target=target**p,
                init=init**p,
                decay=decay**p,
            ),
        ),
        threshold=threshold,
        max_iterations=max_iterations,
        **sinkhorn_kwargs
    )

    return sol.reg_ot_cost ** (1 / p), sol.converged, 10 * jnp.sum(sol.errors > -1)


def distance_matrix(data, metric=wasserstein_metric):
    def pairwise_distances(x, y=None):
        if y is None:
            y = x
        return vmap(
            vmap(lambda x, y: metric(x, y), in_axes=[None, 0]), in_axes=[0, None]
        )(x, y)

    dist_mat, converged, steps = pairwise_distances(jnp.asarray(data))
    return dist_mat, converged, steps
