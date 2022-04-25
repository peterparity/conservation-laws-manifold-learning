using DynamicalSystems
using DifferentialEquations

svector_access(::Nothing) = nothing
svector_access(x::AbstractArray) = SVector{length(x),Int}(x...)
svector_access(x::Int) = SVector{1,Int}(x)
obtain_access(u, ::Nothing) = u
obtain_access(u, i::SVector) = u[i]

function trajectory_tvec(
    ds::ContinuousDynamicalSystem,
    tvec,
    u = ds.u0;
    save_idxs = nothing,
    diffeq = NamedTuple(),
    kwargs...,
)

    if !isempty(kwargs)
        # @warn DIFFEQ_DEP_WARN
        diffeq = NamedTuple(kwargs)
    end

    sv_acc = svector_access(save_idxs)
    integ = integrator(ds, u; diffeq)
    dimvector = ones(SVector{dimension(ds),Int})
    trajectory_continuous_tvec(integ, tvec; sv_acc, dimvector)
end

function trajectory_continuous_tvec(
    integ,
    tvec,
    u0 = nothing;
    sv_acc = nothing,
    dimvector = nothing,
    diffeq = nothing,
)
    !isnothing(u0) && reinit!(integ, u0)
    # This hack is to get type-stable `D` from integrator
    # (ODEIntegrator doesn't have `D` as type parameter)
    D = isnothing(dimvector) ? dimension(integ) : length(dimvector)
    X = isnothing(sv_acc) ? D : length(sv_acc)
    ET = eltype(get_state(integ))
    sol = Vector{SVector{X,ET}}(undef, length(tvec))
    for (i, t) in enumerate(tvec)
        while t > current_time(integ)
            step!(integ)
        end
        sol[i] = SVector{X,ET}(obtain_access(integ(t), sv_acc))
    end
    return Dataset(sol)
end
