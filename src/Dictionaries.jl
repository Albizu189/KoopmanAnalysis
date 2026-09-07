module Dictionaries

using LinearAlgebra
using Random
using Clustering: kmeans
using DynamicPolynomials: @polyvar
using MultivariateBases: maxdegree_basis, FullBasis, ProbabilistsHermite, PhysicistsHermite
using MultivariatePolynomials: polynomial
using Base.Threads: @spawn, nthreads

export get_dim_psi, hermite_basis, Psi_Hermite,
       cluster_data, Psi_RBF,
       RFFBasis, build_rff_basis, Psi_RFF,
       construct_projection_operator_hermite,
       lift_state, Psi_slice   # ← ADDED

# ---------------------------------------------------------------------------
# Thread-parallelism helpers (shared pattern across the toolbox)
# ---------------------------------------------------------------------------

# Split 1:n into ~4 chunks per available thread. The oversubscription factor
# gives dynamic load balancing through Julia's work-stealing scheduler.
function _thread_chunks(n::Int)
    T = nthreads()
    T <= 1 && return UnitRange{Int}[1:n]
    nchunks = clamp(4T, 1, n)
    base, rem = divrem(n, nchunks)
    ranges = UnitRange{Int}[]
    start = 1
    for c in 1:nchunks
        len = base + (c <= rem ? 1 : 0)
        len == 0 && break
        push!(ranges, start:(start + len - 1))
        start += len
    end
    return ranges
end

# ---------------------------------------------------------------------------
# Hermite polynomial dictionary
# ---------------------------------------------------------------------------

function get_dim_psi(n::Int, max_deg::Int)
    dim = 1
    for d in 1:max_deg
        dim += factorial(n + d - 1) ÷ (factorial(d) * factorial(n - 1))
    end
    return dim
end

function hermite_basis(n::Int, max_deg::Int; basis_type::Symbol=:probabilist)
    @polyvar pv[1:n]
    BasisType = basis_type == :physicist ? PhysicistsHermite : ProbabilistsHermite
    full_basis = FullBasis{BasisType}(pv)
    basis = maxdegree_basis(full_basis, max_deg)
    dimPsi = get_dim_psi(n, max_deg)
    linear_indices = reverse(collect(2:(n + 1)))
    return basis, dimPsi, linear_indices
end

"""
    Psi_Hermite(X, max_deg; basis_type=:probabilist)

Thread-parallel over columns. Each task owns disjoint column blocks of `Ψ`,
so no synchronisation is needed in the hot loop.

Performance notes
- Single-column calls (`size(X,2)==1`) take the serial fast path — this keeps
  the per-step lifting inside prediction loops free of task overhead.
- The basis is built ONCE per call and shared read-only across tasks
  (DynamicPolynomials polynomials are immutable — safe to share).
- A serial warm-up on column 1 triggers DynamicPolynomials' lazy internal
  tables before any task touches them.

RAM cost: identical to the serial version — the output `Ψ (dimPsi × m)`
dominates. Per-column temporary vectors of the original implementation are
eliminated entirely (results are written straight into `Ψ`).
"""
function Psi_Hermite(X::AbstractMatrix, max_deg::Int; basis_type::Symbol=:probabilist)
    n, m = size(X)
    basis, dimPsi, _ = hermite_basis(n, max_deg; basis_type=basis_type)
    poly_basis = [polynomial(b) for b in basis]
    Ψ = zeros(dimPsi, m)

    if m < 512 || nthreads() == 1
        return _psi_hermite_cols!(Ψ, X, poly_basis, 1:m)
    end

    # Serial warm-up (DynamicPolynomials builds lookup tables lazily on the
    # first evaluation; doing it here avoids first-touch contention).
    _psi_hermite_cols!(Ψ, X, poly_basis, 1:1)

    tasks = map(_thread_chunks(m)) do rng
        @spawn _psi_hermite_cols!($Ψ, $X, $poly_basis, $rng)
    end
    foreach(wait, tasks)
    return Ψ
end

# Column-block worker: evaluates every basis polynomial on columns `cols`.
function _psi_hermite_cols!(Ψ::AbstractMatrix, X::AbstractMatrix,
                            poly_basis::Vector, cols::UnitRange{Int})
    dimPsi = length(poly_basis)
    @inbounds for k in cols
        xk = @view X[:, k]
        for p in 1:dimPsi
            Ψ[p, k] = poly_basis[p](xk)
        end
    end
    return Ψ
end

# ---------------------------------------------------------------------------
# Thin-plate RBF dictionary
# ---------------------------------------------------------------------------

"""
    cluster_data(X, nRBF; max_points=nothing)

Optional column subsampling before k-means. Centroid placement is statistically
robust to strong subsampling, while k-means cost scales linearly with the
number of samples and Clustering.jl is single-threaded. Pass e.g.
`max_points=50_000` on very long series to cut clustering time ≈ m/50_000×
with negligible effect on the centres. Default behaviour is unchanged.
"""
function cluster_data(X::AbstractMatrix, nRBF::Int;
                      max_points::Union{Nothing,Int}=nothing)
    m = size(X, 2)
    if !isnothing(max_points) && max_points < m
        X = X[:, randperm(m)[1:max_points]]
    end
    R = kmeans(X, nRBF; maxiter=2000)
    return R.centers
end

"""
    Psi_RBF(X, centers; include_states=true, state_indices=nothing)

Thread-parallel over columns. Tasks own disjoint COLUMN blocks of `Ψ`
(rows `offset+1 : offset+nRBF` across their block), so every output element
has exactly one writer and no atomics/locks are involved.

RAM cost: each task allocates its own `dists` buffer for its block only, so
the combined scratch space across ALL threads equals the single `m`-vector the
serial version used (8·m bytes). Nothing else is duplicated.
"""
function Psi_RBF(X::AbstractMatrix, centers::AbstractMatrix;
                 include_states::Bool=true, state_indices::Union{Nothing,Vector{Int}}=nothing)
    n, m = size(X)
    nRBF = size(centers, 2)
    local offset::Int
    if include_states
        state_indices = isnothing(state_indices) ? collect(1:n) : state_indices
        nPsi = nRBF + 1 + length(state_indices)
        Ψ = zeros(nPsi, m)
        Ψ[1, :] .= 1.0
        for (j, idx) in enumerate(state_indices)
            Ψ[1 + j, :] .= X[idx, :]
        end
        offset = 1 + length(state_indices)
    else
        nPsi = nRBF
        Ψ = zeros(nPsi, m)
        offset = 0
    end

    if m < 1024 || nthreads() == 1
        return _psi_rbf_cols!(Ψ, X, centers, offset, 1:m)
    end
    tasks = map(_thread_chunks(m)) do rng
        @spawn _psi_rbf_cols!($Ψ, $X, $centers, $offset, $rng)
    end
    foreach(wait, tasks)
    return Ψ
end

# Column-block worker: fills the thin-plate rows for columns `cols`.
# Numerically identical to the original serial triple loop (same accumulation
# order, same +1e-12 regularizer, same r²·log(r) evaluation).
function _psi_rbf_cols!(Ψ::AbstractMatrix, X::AbstractMatrix, centers::AbstractMatrix,
                        offset::Int, cols::UnitRange{Int})
    n = size(X, 1)
    nRBF = size(centers, 2)
    len = length(cols)
    dists = Vector{Float64}(undef, len)
    @inbounds for k in 1:nRBF
        c = @view centers[:, k]
        li = 0
        for i in cols
            li += 1
            s = 0.0
            for j in 1:n
                d = X[j, i] - c[j]
                s += d * d
            end
            dists[li] = sqrt(s) + 1e-12
        end
        li = 0
        for i in cols
            li += 1
            r = dists[li]
            Ψ[offset + k, i] = r * r * log(r)
        end
    end
    return Ψ
end

# ---------------------------------------------------------------------------
# Random Fourier Feature (RFF) dictionary
# ---------------------------------------------------------------------------
# NOT hand-parallelized — see PARALLELIZATION_NOTES.md §"not parallelized":
# the dominant `basis.W * X` is a BLAS gemm (already multithreaded) and the
# elementwise cos pass is memory-bandwidth-bound, leaving no headroom for
# task-level threading.
# ---------------------------------------------------------------------------

struct RFFBasis
    W::Matrix{Float64}
    b::Vector{Float64}
    D::Int
    sigma::Float64
end

function build_rff_basis(n::Int, D::Int, sigma::Real)
    W = randn(D, n) ./ float(sigma)
    b = 2π .* rand(D)
    return RFFBasis(W, b, D, float(sigma))
end

function Psi_RFF(X::AbstractMatrix, basis::RFFBasis; include_states::Bool=false)
    Z = basis.W * X .+ basis.b
    Ψ = sqrt(2.0 / basis.D) .* cos.(Z)
    if include_states
        Ψ = vcat(X, Ψ)
    end
    return Ψ
end

# ---------------------------------------------------------------------------
# Analytic projection for Hermite dictionary
# ---------------------------------------------------------------------------

function construct_projection_operator_hermite(state_dim::Int, max_deg::Int; basis_type::Symbol=:probabilist)
    n = state_dim
    @polyvar pv[1:n]
    BasisType = basis_type == :physicist ? PhysicistsHermite : ProbabilistsHermite
    full_basis = FullBasis{BasisType}(pv)
    basis = maxdegree_basis(full_basis, max_deg)
    nPsi = length(basis)
    B = zeros(n, nPsi)
    lin_indices = reverse(collect(2:(n + 1)))
    for i in 1:n
        B[i, lin_indices[i]] = 1.0
    end
    return B, nPsi
end

# ---------------------------------------------------------------------------
# Generic state lifting (moved from Hankel.jl)
# ---------------------------------------------------------------------------

"""
    lift_state(x::AbstractVector, dict_info::NamedTuple)

Lift a single state vector `x` into the dictionary space defined by
`dict_info`.  Returns a vector of length `nPsi`.

Single-column calls route into the SERIAL fast paths of `Psi_Hermite` /
`Psi_RBF`, so this stays allocation-light inside tight prediction loops and
is safe to call from any thread.
"""
function lift_state(x::AbstractVector, dict_info::NamedTuple)
    X = reshape(x, :, 1)
    if dict_info.type == :hermite
        return Psi_Hermite(X, dict_info.max_deg; basis_type=dict_info.basis_type)[:, 1]
    elseif dict_info.type == :rbf
        return Psi_RBF(X, dict_info.centers;
                       include_states=dict_info.include_states,
                       state_indices=dict_info.state_indices)[:, 1]
    elseif dict_info.type == :rff
        return Psi_RFF(X, dict_info.basis)[:, 1]
    else
        error("Unknown dict_info.type: $(dict_info.type)")
    end
end

# ---------------------------------------------------------------------------
# Partial-state grid lifting (for eigenfunction slice evaluation)
# ---------------------------------------------------------------------------

"""
    Psi_slice(X_slice::AbstractMatrix, dict_info::NamedTuple;
              full_dim::Union{Int,Nothing}=nothing,
              active_dims::AbstractVector{Int}=1:size(X_slice,1),
              fixed_values::Union{AbstractVector,Nothing}=nothing)

Lift a partial-state matrix into the full state space before applying the dictionary.

- `X_slice`: `n_slice × N` matrix of grid points in the active subspace.
- `dict_info`: the dictionary metadata tuple from `hankel_edmd`.
- `full_dim`: total state dimension (auto-inferred for RBF/RFF).
- `active_dims`: which full-space dimensions `X_slice` rows correspond to.
- `fixed_values`: values for the remaining dimensions (default: zeros).

Returns an `nPsi × N` lifted matrix.

Threaded for free: the final dispatch lands on the parallel `Psi_*` kernels
above, so large grid evaluations scale with available cores automatically.
"""
function Psi_slice(X_slice::AbstractMatrix, dict_info::NamedTuple;
                   full_dim::Union{Int,Nothing}=nothing,
                   active_dims::AbstractVector{Int}=1:size(X_slice,1),
                   fixed_values::Union{AbstractVector,Nothing}=nothing)
    n_slice, N = size(X_slice)

    if isnothing(full_dim)
        full_dim = _infer_state_dim(dict_info)
    end

    @assert length(active_dims) == n_slice "active_dims length must match n_slice=$n_slice"
    @assert maximum(active_dims) <= full_dim "active_dims cannot exceed full_dim=$full_dim"

    X_full = zeros(full_dim, N)
    X_full[active_dims, :] .= X_slice

    if !isnothing(fixed_values)
        inactive = setdiff(1:full_dim, active_dims)
        @assert length(fixed_values) == length(inactive) "fixed_values length must equal number of inactive dims ($(length(inactive)))"
        for (i, d) in enumerate(inactive)
            X_full[d, :] .= fixed_values[i]
        end
    end

    return _psi_dispatch(X_full, dict_info)
end

function _infer_state_dim(dict_info::NamedTuple)
    if dict_info.type == :rbf
        return size(dict_info.centers, 1)
    elseif dict_info.type == :rff
        return size(dict_info.basis.W, 2)
    else
        error("Cannot infer state dimension for dict_info.type=$(dict_info.type); pass full_dim explicitly")
    end
end

function _psi_dispatch(X::AbstractMatrix, dict_info::NamedTuple)
    if dict_info.type == :hermite
        return Psi_Hermite(X, dict_info.max_deg; basis_type=dict_info.basis_type)
    elseif dict_info.type == :rbf
        return Psi_RBF(X, dict_info.centers;
                       include_states=dict_info.include_states,
                       state_indices=get(dict_info, :state_indices, nothing))
    elseif dict_info.type == :rff
        return Psi_RFF(X, dict_info.basis;
                       include_states=get(dict_info, :include_states, false))
    else
        error("Unknown dict_info.type: $(dict_info.type)")
    end
end

end # module