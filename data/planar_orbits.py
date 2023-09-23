import numpy as np
import os
import argparse


def eccentric_anomaly_from_mean(e, M, tol=1e-14, MAX_ITERATIONS=1000):
    """Convert mean anomaly to eccentric anomaly.
    Implemented from [A Practical Method for Solving the Kepler Equation][1]
    by Marc A. Murison from the U.S. Naval Observatory
    [1]: http://murison.alpheratz.net/dynamics/twobody/KeplerIterations_summary.pdf
    """
    Mnorm = np.fmod(M, 2 * np.pi)
    E0 = np.fmod(
        M
        + (-1 / 2 * e**3 + e + (e**2 + 3 / 2 * np.cos(M) * e**3) * np.cos(M))
        * np.sin(M),
        2 * np.pi,
    )
    dE = tol + 1
    count = 0
    while np.any(dE > tol):
        t1 = np.cos(E0)
        t2 = -1 + e * t1
        t3 = np.sin(E0)
        t4 = e * t3
        t5 = -E0 + t4 + Mnorm
        t6 = t5 / (1 / 2 * t5 * t4 / t2 + t2)
        E = E0 - t5 / ((1 / 2 * t3 - 1 / 6 * t1 * t6) * e * t6 + t2)
        dE = np.abs(E - E0)
        E0 = np.fmod(E, 2 * np.pi)
        count += 1
        if count == MAX_ITERATIONS:
            print("Current dE: ", dE[dE > tol])
            print("eccentricity: ", np.repeat(e, dE.shape[-1], axis=-1)[dE > tol])
            raise RuntimeError(
                f"Did not converge after {MAX_ITERATIONS} iterations"
                f" with tolerance {tol}."
            )
    return E


def generate_data(
    n_trajs,
    n_samples=100,
    noise=0.0,
    check=False,
    params=(None, None, None),
    rng=np.random.default_rng(0),
):
    # randomly sampled observation times
    time_samples = rng.uniform(0, 10.0 * n_samples, size=(n_trajs, n_samples))

    mu = 1.0  # standard gravitational parameter, i.e. G*M

    # Set/generate parameters
    H, L, phi0 = params

    # H = (
    #     -mu / 2 * (0.5 + 0.5 * rng.uniform(size=(n_trajs, 1)))
    #     if H is None
    #     else H * np.ones((n_trajs, 1))
    # )
    H = (
        -mu / 2 * (0.3 + 0.7 * rng.uniform(size=(n_trajs, 1)))
        if H is None
        else H * np.ones((n_trajs, 1))
    )
    L = rng.uniform(size=(n_trajs, 1)) if L is None else L * np.ones((n_trajs, 1))

    a = -mu / (2 * H)  # semi-major axis
    e = np.sqrt(1 - L**2 / (mu * a))

    phi0 = (
        2 * np.pi * rng.uniform(size=(n_trajs, 1))
        if phi0 is None
        else phi0 * np.ones((n_trajs, 1))
    )

    # https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
    T = 2 * np.pi * np.sqrt(a**3 / mu)  # period
    M = np.fmod(2 * np.pi * time_samples / T, 2 * np.pi)  # mean anomaly
    E = eccentric_anomaly_from_mean(e, M)  # eccentric anomaly
    phi = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2)
    )  # true anomaly/angle
    r = (a * (1 - e**2)) / (1 + e * np.cos(phi))  # radius
    pos = np.stack(
        (r * np.cos(phi + phi0), r * np.sin(phi + phi0)), axis=-1
    )  # position rotated by phi0

    vel = np.expand_dims(np.sqrt(mu * a) / r, axis=-1) * np.stack(
        (-np.sin(E), np.sqrt(1 - e**2) * np.cos(E)), axis=-1
    )  # velocity
    c, s = np.cos(phi0), np.sin(phi0)
    R = np.stack((c, -s, s, c), axis=-1).reshape(n_trajs, 1, 2, 2)
    vel = np.squeeze(R @ np.expand_dims(vel, axis=-1), axis=-1)  # rotated by phi0

    data = np.concatenate((pos, vel), axis=-1)

    if check:
        assert np.allclose(M, E - e * np.sin(E))

        p = np.sqrt(mu * (2 / r - 1 / a))  # speed/specific momentum
        diffp = p - np.linalg.norm(vel, axis=-1)
        # print(e[np.any(np.isnan(diffp), axis=-1)])
        # print(diffp[np.any(np.abs(diffp) > 1e-8, axis=-1)])
        # print(e[np.any(np.abs(diffp) > 1e-8, axis=-1)])
        assert np.allclose(diffp, np.zeros_like(diffp), atol=1e-6)

        L = np.sqrt(mu * a * (1 - e**2))  # specific angular momentum
        diffL = L - np.cross(pos, vel)
        assert np.allclose(diffL, np.zeros_like(diffL))

        H = -mu / (2 * a)  # specific energy
        diffH = H - (
            0.5 * np.linalg.norm(vel, axis=-1) ** 2 - mu / np.linalg.norm(pos, axis=-1)
        )
        assert np.allclose(diffH, np.zeros_like(diffH))

    if noise > 0:
        data += noise * rng.standard_normal(size=data.shape)
    return data, np.concatenate((H, L, phi0, e, a), axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_trajs", type=int, default=400, help="number of trajectories")
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

    data, params = generate_data(args.n_trajs, n_samples=args.n_samples, check=True)

    if args.save is None:
        data_dir = os.path.dirname(os.path.realpath(__file__))
        file_name = f"planar_orbits_{args.n_trajs}_{args.n_samples}"
        if args.noise > 0:
            file_name += f"_noise{args.noise}"
        save_file = os.path.join(data_dir, file_name + ".npz")
    else:
        save_file = args.save

    print("Saving to: ", save_file)
    np.savez(save_file, data=data, params=params)
