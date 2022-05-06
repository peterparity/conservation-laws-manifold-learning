import jax
import jax.numpy as jnp
from ott.geometry.costs import CostFn


@jax.tree_util.register_pytree_node_class
class PeriodicEuclidean(CostFn):
    """Squared Periodic Euclidean distance CostFn."""

    def __init__(self, periods):
        super().__init__()
        self._periods = periods

    def pairwise(self, x, y):
        s = jnp.mod(jnp.abs(x - y), self._periods)
        s = jnp.minimum(s, self._periods - s)
        return jnp.sum(s**2)

    def tree_flatten(self):
        return (), (self._periods,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(aux_data[0])


@jax.tree_util.register_pytree_node_class
class MMD(CostFn):
    """MMD distance CostFn."""

    def __init__(self, n_dims, kernel_type="gaussian", scale=1.0, p=1):
        super().__init__()
        self._n_dims = n_dims
        self._kernel_type = kernel_type
        self._scale = scale
        self._p = p

    def norm(self, x):
        x = x.reshape(*x.shape[:-1], -1, self._n_dims)
        xx = x @ jnp.swapaxes(x, -2, -1)
        x_diag = jnp.diagonal(xx, 0, -2, -1)
        dxx = x_diag[..., None] + x_diag[..., None, :] - 2 * xx
        if self._kernel_type == "gaussian":
            XX = jnp.exp(-0.5 * dxx / self._scale**2).mean(axis=(-2, -1))
        elif self._kernel_type == "energy":
            dxx = jnp.clip(dxx, 0.0, None)
            XX = -(dxx ** (0.5 * self._p)).mean(axis=(-2, -1)) / self._scale
        return XX

    def pairwise(self, x, y):
        x = x.reshape(*x.shape[:-1], -1, self._n_dims)
        y = y.reshape(*y.shape[:-1], -1, self._n_dims)
        xy = x @ jnp.swapaxes(y, -2, -1)
        dxy = (
            jnp.sum(x**2, axis=-1)[..., None]
            + jnp.sum(y**2, axis=-1)[..., None, :]
            - 2 * xy
        )
        if self._kernel_type == "gaussian":
            XY = jnp.exp(-0.5 * dxy / self._scale**2).mean(axis=(-2, -1))
        elif self._kernel_type == "energy":
            dxy = jnp.clip(dxy, 0.0, None)
            XY = -(dxy ** (0.5 * self._p)).mean(axis=(-2, -1)) / self._scale
        return -2 * XY

    def tree_flatten(self):
        return (), (self._n_dims, self._kernel_type, self._scale, self._p)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        del children
        return cls(aux_data[0], aux_data[1], aux_data[2], aux_data[3])
