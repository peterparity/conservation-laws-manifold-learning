using Distributed, Random, NPZ

total_procs = Sys.CPU_THREADS
addprocs(total_procs - nprocs(), exeflags = "--project=$(Base.active_project())")

@everywhere begin
    using FFTW, LinearAlgebra, FourierTools, Statistics
    include("utils.jl")

    @views function conserved_quantities(traj, l, n)
        @assert size(traj)[1] == n
        iω = 2π * im * rfftfreq(n, n / l)
        fwd = plan_rfft(ones(n))
        inv = plan_irfft(fwd * ones(n), n)

        c1 = l .* mean(traj, dims = 1)
        c2 = l .* mean(traj .^ 2, dims = 1)
        c3 =
            l .* mean(
                traj .^ 3 .-
                0.5 .*
                (mapslices(
                    x -> inv * x,
                    iω .* mapslices(x -> fwd * x, traj, dims = 1),
                    dims = 1,
                )) .^ 2,
                dims = 1,
            )
        # c3_fd =
        #     l .* mean(
        #         traj .^ 3 .- 0.5 .* (n / l .* diff(vcat(traj, traj[[1], :]), dims = 1)) .^ 2,
        #         dims = 1,
        #     )
        return vcat(c1, c2, c3)#, c3_fd)
    end

    @views function generate_data(
        tvec,
        v0;
        n_dims = 200,
        l = 20.0,
        n = 1000,
        total_time = 100,
        dt = 1e-3,
    )
        iω = 2π * im * rfftfreq(n, n / l)
        fwd = plan_rfft(ones(n))
        inv = plan_irfft(fwd * ones(n), n)

        A = DiffEqArrayOperator(Diagonal(-(iω .^ 3)))
        f = (v, p, t) -> -6 .* (fwd * ((inv * v) .* (inv * (iω .* v))))
        ds = SplitODEProblem(A, f, v0, (0.0, total_time))

        sol = solve(ds, ETDRK4(); dt = dt, saveat = tvec)

        u = map(x -> inv * x, sol.u)
        traj = hcat(u...)

        c = conserved_quantities(traj, l, n)
        params = mean(c, dims = 2)
        @assert all(isapprox.(std(c, dims = 2), 0.0; atol = 1e-2)) "violation of conservation: $(params) +/- $(std(c, dims = 2)), init cond: $(v0)"

        if n_dims != n
            u_out = resample.(u, n_dims)
            traj_out = hcat(u_out...)
        else
            traj_out = traj
        end

        return traj_out, params
    end
end

function init(rng, n_trajs, n_samples; l = 20.0, n = 1000, total_time = 100)
    tvecs = sort(total_time .* rand(rng, n_samples, n_trajs), dims = 1)

    iω = 2π * im * rfftfreq(n, n / l)
    s = 5 .+ 10 .* rand(rng, 1, n_trajs)
    # print(s)
    v0s =
        0.5 .* n ./ (sqrt(2π) .* s .* 2π ./ l) .*
        exp.(0.5 .* iω .^ 2 ./ (s .* 2π ./ l) .^ 2) .*
        (randn(rng, size(iω)[1], n_trajs) .+ im * randn(rng, size(iω)[1], n_trajs))

    tvecs, v0s
end

@views function batch_generate(
    n_trajs,
    n_samples;
    n_dims = 200,
    l = 20.0,
    n = 1000,
    total_time = 100,
    dt = 1e-3,
    rng = MersenneTwister(0),
)
    tvecs, v0s =
        init(rng, n_trajs, n_samples; l = l, n = n, total_time = total_time)

    out = pmap(
        x -> generate_data(
            x...;
            n_dims = n_dims,
            l = l,
            n = n,
            total_time = total_time,
            dt = dt,
        ),
        zip(eachcol(tvecs), eachcol(v0s)),
    )
    trajs = cat(map(x -> x[1], out)..., dims = 3)
    params = cat(map(x -> x[2], out)..., dims = 2)

    return trajs, params
end

rng = MersenneTwister(0)
n_trajs = 400
n_samples = 200
n_dims = 200


l = 20.0
n = 1000
total_time = 10
dt = 1e-4

data, params = batch_generate(
    n_trajs,
    n_samples;
    n_dims = n_dims,
    l = l,
    n = n,
    total_time = total_time,
    dt = dt,
)

save_file = joinpath(@__DIR__, "kdv_$(n_trajs)_$(n_samples)_$(n_dims)_ver2.npz")
println("Saving to: ", save_file)
npzwrite(
    save_file,
    Dict("data" => permutedims(data, (3, 2, 1)), "params" => permutedims(params, (2, 1))),
)

# tvecs, v0s = init(rng, n_trajs, n_samples; l = l, n = n, total_time = total_time)
# traj, params = generate_data(
#     # tvecs[:, 1],
#     0:dt:total_time,
#     v0s[:, 1];
#     n_dims = n_dims,
#     l = l,
#     n = n,
#     total_time = total_time,
#     dt = dt,
# )

# samples = vcat(data[newaxis, :, :, :], circshift(data, (-1, 0))[newaxis, :, :, :])
# samples = vcat(
#     samples[[1], :, :, :],
#     n_dims / l .* (samples[[2], :, :, :] .- samples[[1], :, :, :]),
# )
