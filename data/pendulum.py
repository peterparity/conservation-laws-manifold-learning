import numpy as np
from scipy.special import ellipj
import os
import argparse


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
    parser = argparse.ArgumentParser()
    parser.add_argument("n_trajs", type=int, default=200, help="number of trajectories")
    parser.add_argument(
        "n_samples",
        type=int,
        default=200,
        help="number of time samples per trajectory",
    )
    parser.add_argument(
        "--noise", type=np.double, default="0.0", help="stdv of additive noise"
    )
    parser.add_argument("--save", default=None, help="save file")
    args = parser.parse_args()

    data, params = generate_data(args.n_trajs, n_samples=args.n_samples)

    if args.save is None:
        data_dir = os.path.dirname(os.path.realpath(__file__))
        file_name = f"pendulum_{args.n_trajs}_{args.n_samples}"
        if args.noise > 0:
            file_name += f"_noise{args.noise}"
        save_file = os.path.join(data_dir, file_name + ".npz")
    else:
        save_file = args.save

    print("Saving to: ", save_file)
    np.savez(save_file, data=data, params=params)
