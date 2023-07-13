# Discovering Conservation Laws using Optimal Transport and Manifold Learning

This is an implementation of our method for discovering conservation laws directly from trajectory samples. We reformulate this task as a manifold learning problem and propose a non-parametric approach, combining the Wasserstein metric from optimal transport with diffusion maps, to discover conserved quantities that vary across trajectories sampled from a dynamical system. The Wasserstein distances are efficiently computed using the Sinkhorn algorithm implemented by the [OTT-JAX](https://github.com/ott-jax/ott) library, and the diffusion maps are implemented using NumPy/SciPy.

Please cite "**Discovering Conservation Laws using Optimal Transport and Manifold Learning**" (https://arxiv.org/abs/2208.14995) and see the paper for more details. This is the official repository for the paper.
