using SparseArrays, LinearAlgebra, Statistics
using OptimalTransport, Distances
using ManifoldLearning

using CUDA
CUDA.allowscalar(false)

using WGLMakie

include("pendulum_data.jl")
import .Pendulum
data, params = Pendulum.generate_data(100, noise_stdv = 0.05)

# fig = Figure(resolution = (1000, 800))
# Axis(fig[1, 1], autolimitaspect = 1)
# for i in [1, 3]
#     scatter!(data[:, :, i]')
# end
# display(fig);


# cost matrix
data = eachslice(data, dims = 3)
C = [pairwise(SqEuclidean(), x, y; dims = 2) for x in data, y in data]
# C = CuArray.(C)

# uniform histograms
μ = fill(1 / 100, 100)
ν = fill(1 / 100, 100)
# μ = CUDA.fill(1 / 100, 100)
# ν = CUDA.fill(1 / 100, 100)

# regularization parameter
ε = 0.1

# solve entropically regularized optimal transport problem
sinkhorn2.([μ], [ν], C, ε, maxiter = 1_000)
