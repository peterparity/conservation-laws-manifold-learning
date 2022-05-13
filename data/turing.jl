using Distributed, Random, NPZ

total_procs = Sys.CPU_THREADS
addprocs(total_procs - nprocs(), exeflags = "--project=$(Base.active_project())")

@everywhere begin
    using FFTW, LinearAlgebra, FourierTools, Statistics, BlockArrays
    include("utils.jl")

    @views function conserved_quantities(traj, l, n)
        x = rfft(mean(traj, dims = 2)[1:n])[4]
        @assert abs(x) > 20
        return [angle(x)]
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
        vsize = size(iω)[1]
        fwd = plan_rfft(ones(n))
        inv = plan_irfft(fwd * ones(n), n)

        D = 0.08
        C = -1.5
        H = 3.0

        A = DiffEqArrayOperator(
            mortar(
                reshape(
                    [
                        Diagonal(D .* iω .^ 2 .+ 1),
                        Diagonal(H .* ones(vsize)),
                        Diagonal(-ones(vsize)),
                        Diagonal(iω .^ 2 .- 1.5),
                    ],
                    (2, 2),
                ),
            ),
        )
        @views function f(v, p, t)
            u1 = inv * v[1:vsize]
            u2 = inv * v[vsize+1:end]
            out = fwd * (-C .* u1 .* u2 .- u1 .* u2 .^ 2)
            return vcat(out, -out)
        end
        ds = SplitODEProblem(A, f, v0, (0.0, total_time))

        sol = solve(ds, ETDRK4(); dt = dt, saveat = tvec)

        u = map(x -> vcat(inv * x[1:vsize], inv * x[vsize+1:end]), sol.u)
        traj = hcat(u...)

        params = conserved_quantities(traj, l, n)

        if n_dims != n
            u1_out = resample.(eachcol(traj[1:n, :]), n_dims)
            u2_out = resample.(eachcol(traj[(n+1):end, :]), n_dims)
            traj_out = vcat(hcat(u1_out...), hcat(u2_out...))
        else
            traj_out = traj
        end

        return traj_out, params
    end
end

function init(
    rng,
    n_trajs,
    n_samples;
    l = 20.0,
    n = 1000,
    total_time = 100,
    transient_time = 150,
)
    tvecs =
        transient_time .+
        (total_time - transient_time) .* sort(rand(rng, n_samples, n_trajs), dims = 1)

    iω = 2π * im * rfftfreq(n, n / l)
    v0s = randn(rng, size(iω)[1], 2, n_trajs) .+ im * randn(rng, size(iω)[1], 2, n_trajs)

    tvecs, vcat(v0s[:, 1, :], v0s[:, 2, :])
end

@views function batch_generate(
    n_trajs,
    n_samples;
    n_dims = 200,
    l = 20.0,
    n = 1000,
    total_time = 100,
    transient_time = 0,
    dt = 1e-3,
    rng = MersenneTwister(0),
)
    tvecs, v0s = init(
        rng,
        n_trajs,
        n_samples;
        l = l,
        n = n,
        total_time = total_time,
        transient_time = transient_time,
    )

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
n_dims = 50


l = 8.0
n = 50
transient_time = 300
total_time = transient_time + 1000
dt = 1e-2

data, params = batch_generate(
    n_trajs,
    n_samples;
    n_dims = n_dims,
    l = l,
    n = n,
    total_time = total_time,
    transient_time = transient_time,
    dt = dt,
)

save_file = joinpath(@__DIR__, "turing_$(n_trajs)_$(n_samples)_$(n_dims)_l$(l).npz")
println("Saving to: ", save_file)
npzwrite(
    save_file,
    Dict("data" => permutedims(data, (3, 2, 1)), "params" => permutedims(params, (2, 1))),
)

# tvecs, v0s = init(
#     rng,
#     n_trajs,
#     n_samples;
#     l = l,
#     n = n,
#     total_time = total_time,
#     transient_time = transient_time,
# )
# traj, params = generate_data(
#     # tvecs[:, 1],
#     transient_time:dt:total_time,
#     v0s[:, 1];
#     n_dims = n_dims,
#     l = l,
#     n = n,
#     total_time = total_time,
#     dt = dt,
# )
