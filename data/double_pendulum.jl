using Distributed, Random, NPZ

total_procs = Sys.CPU_THREADS
addprocs(total_procs - nprocs(), exeflags = "--project=$(Base.active_project())")

@everywhere begin
    using Statistics
    include("utils.jl")

    function energy(u)
        θ1, ω1, θ2, ω2 = u
        ω1 .^ 2 .+ 0.5 .* ω2 .^ 2 .+ ω1 .* ω2 .* cos.(θ1 .- θ2) .- 2 .* cos.(θ1) .- cos.(θ2)
    end

    function generate_data(tvec, u0 = [π / 2, 0, 0, 0.5])
        ds = Systems.double_pendulum(u0; G = 1.0, L1 = 1.0, L2 = 1.0, M1 = 1.0, M2 = 1.0)

        tr = transpose(
            Matrix(trajectory_tvec(ds, tvec; diffeq = (abstol = 1e-9, reltol = 1e-9))),
        )
        en = energy.(eachcol(tr))
        # println(std(en))
        @assert isapprox(std(en), 0.0; atol = 1e-5)
        # return tr, (mean(en), std(en))
        return tr, [mean(en)]
    end
end

function init(rng, n_trajs, n_samples; total_time = 100 * n_samples)
    tvecs = sort(total_time .* rand(rng, n_samples, n_trajs), dims = 1)

    u0s = transpose(
        hcat(
            2π .* (rand(rng, n_trajs) .- 0.5),
            0.0 .* randn(rng, n_trajs),
            2π .* (rand(rng, n_trajs) .- 0.5),
            0.0 .* randn(rng, n_trajs),
        ),
    )

    # u0s = transpose(
    #     hcat(
    #         rand(rng,(-1, 1), n_trajs) .* acos.(2 .* (rand(rng,n_trajs) .- 0.5)),
    #         0.0 .* randn(rng,n_trajs),
    #         rand(rng,(-1, 1), n_trajs) .* acos.(2 .* (rand(rng,n_trajs) .- 0.5)),
    #         0.0 .* randn(rng,n_trajs),
    #     ),
    # )

    tvecs, u0s
end

function batch_generate(n_trajs; n_samples = 1000, rng = MersenneTwister(0))
    tvecs, u0s = init(rng, n_trajs, n_samples)

    out = pmap(x -> generate_data(x...), zip(eachcol(tvecs), eachcol(u0s)))
    trajs = cat(map(x -> x[1], out)..., dims = 3)
    params = cat(map(x -> x[2], out)..., dims = 2)

    return trajs, params
end

n_trajs = 1000
data, params = batch_generate(n_trajs; n_samples = 1000)

save_file = joinpath(@__DIR__, "double_pendulum_$(n_trajs).npz")
println("Saving to: ", save_file)
npzwrite(
    save_file,
    Dict("data" => permutedims(data, (3, 2, 1)), "params" => permutedims(params, (2, 1))),
)
