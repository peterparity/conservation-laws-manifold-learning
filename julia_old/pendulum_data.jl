module Pendulum

using Random, Elliptic

@views function generate_data(
    n_trajectories::Int64;
    n_traj_samples::Int64 = 100,
    noise_stdv::Float64 = 0.0,
    check_energy::Bool = false,
    params::Tuple = (nothing,),
    random_seed::Int64 = 42,
)
    Random.seed!(random_seed)

    energy, = params
    if energy === nothing
        energy = 2 .* rand(Float64, n_trajectories)
    elseif energy isa Number
        energy = fill(energy, n_trajectories)
    end
    params = (energy,)

    time_samples =
        10 .* n_traj_samples .* rand(Float64, (1, n_traj_samples, n_trajectories))
    k2 = 0.5 .* reshape(energy, 1, 1, n_trajectories)
    sn, cn, dn =
        eachslice(
            reshape(
                reduce(hcat, ellipj.(time_samples, k2) .|> collect),
                3,
                1,
                n_traj_samples,
                n_trajectories,
            ),
            dims = 1,
        ) |> collect
    q = @. 2 * asin(sqrt(k2) * sn)
    p = @. 2 * sqrt.(k2) * cn * dn / sqrt(1 - k2 * sn^2)
    data = vcat(q, p)

    if check_energy
        E = @. 0.5 * p^2 - cos(q) + 1
        ΔE = @. E - energy
        println("max ΔE = ", maximum(abs.(ΔE)))
        @assert all(isapprox.(E, energy))
    end

    if noise_stdv > 0
        data .+= noise_stdv .* randn(Float64, size(data))
    end

    return data, params
end

end
