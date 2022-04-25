import numpy as np
from scipy.special import ellipj
import os


def generate_data(
    n_trajs,
    n_samples=100,
    noise=0.0,
    check=False,
    params=(None,),
    rng=np.random.default_rng(0),
):
    time_samples = rng.uniform(0, 10.0 * n_samples, size=(n_trajs, n_samples))
    (E,) = params
    E = (
        rng.uniform(0, 2, size=(n_trajs, 1))
        if E is None
        else np.broadcast_to(E, (n_trajs, 1))
    )

    k2 = E / 2
    sn, cn, dn, _ = ellipj(time_samples, k2)
    q = 2 * np.arcsin(np.sqrt(k2) * sn)
    p = 2 * np.sqrt(k2) * cn * dn / np.sqrt(1 - k2 * sn**2)
    data = np.stack((q, p), axis=-1)

    if check:
        H = 0.5 * p**2 - np.cos(q) + 1
        diffE = H - E
        print("max diffE = ", np.max(np.abs(diffE)))
        assert np.allclose(H, E)

    if noise > 0:
        data += noise * rng.standard_normal(size=data.shape)
    return data, E


if __name__ == "__main__":
    n_trajs = 100
    data, params = generate_data(n_trajs, n_samples=100)

    data_dir = os.path.dirname(os.path.realpath(__file__))
    save_file = os.path.join(data_dir, f"pendulum_{n_trajs}.npz")
    print("Saving to: ", save_file)
    np.savez(save_file, data=data, params=params)
