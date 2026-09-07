module EDMD

using LinearAlgebra
using Random
using Statistics
using Base.Threads: @spawn, nthreads

export compute_koopman_operator, construct_projection_operator,
       rbf_kernel, median_heuristic_sigma, kernel_feature_vector,
       kernel_edmd_rbf,
       edmd_predict_from_psi

# ---------------------------------------------------------------------------
# Thread-parallelism helpers (same pattern as Dictionaries._thread_chunks)
# ---------------------------------------------------------------------------

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
# EDMD — Extended Dynamic Mode Decomposition
# ---------------------------------------------------------------------------

# NOTE ON PARALLELIZATION — deliberately NOT hand-threaded:
# Both branches are dense linear algebra (`pinv`, gemm `\`) executed by
# BLAS/LAPACK, which are ALREADY multithreaded. Wrapping them in Julia tasks
# would oversubscribe cores and thrash caches. If you run this while other
# task-parallel stages are active, control the split explicitly:
#   BLAS.set_num_threads(...)   # linear-algebra parallelism
#   JULIA_NUM_THREADS=...       # task-level parallelism (Psi/FNN/kernel fill)
function compute_koopman_operator(ΨX::AbstractMatrix, ΨY::AbstractMatrix;
                                  method::Symbol=:ridge, alpha::Real=1e-3)
    if method == :pinv
        K = pinv(ΨX') * ΨY'
    elseif method == :ridge
        nPsi = size(ΨX, 1)
        K = (ΨX * ΨX' + float(alpha) * I(nPsi)) \ (ΨX * ΨY')
    else
        error("Unknown EDMD method: $method. Use :pinv or :ridge.")
    end
    return K
end

function compute_koopman_operator(X::AbstractMatrix, Y::AbstractMatrix, Psi_func::Function;
                                  method::Symbol=:pinv, alpha::Real=1e-3)
    ΨX = Psi_func(X)      # dispatches to the thread-parallel dictionaries
    ΨY = Psi_func(Y)
    K = compute_koopman_operator(ΨX, ΨY; method=method, alpha=alpha)
    return K, ΨX, ΨY
end

# Same rationale as compute_koopman_operator: pure BLAS (`*`, `/`) — already
# threaded by the BLAS backend.
function construct_projection_operator(state_dim::Int, ΨX::AbstractMatrix, X_train::AbstractMatrix;
                                       alpha::Real=1e-6)
    nPsi = size(ΨX, 1)
    G = ΨX * ΨX' + float(alpha) * I(nPsi)
    B = (X_train * ΨX') / G
    return B
end

# ---------------------------------------------------------------------------
# RBF Kernel EDMD
# ---------------------------------------------------------------------------

function rbf_kernel(x, y, sigma)
    return exp(-sum((x .- y).^2) / (2 * sigma^2))
end

"""
    median_heuristic_sigma(X; n_sample=1000)

Thread-parallel pairwise-distance fill. Each task owns a contiguous stripe of
the global `dists` array at precomputed offsets — single writer per slot, so
results are numerically identical to the serial version.

RAM: the `dists` vector (8 · n_pairs bytes ≈ 4 MB for the default 1000-sample
heuristic) is the same allocation as before; threading adds only tiny per-task
buffers.
"""
function median_heuristic_sigma(X::AbstractMatrix; n_sample::Int=1000)
    m = size(X, 2)
    idx = randperm(m)[1:min(n_sample, m)]
    n_s = length(idx)
    n_pairs = n_s * (n_s - 1) ÷ 2
    n_dim = size(X, 1)

    dists = Vector{Float64}(undef, n_pairs)

    # Row i contributes (n_s − i) pairs; prefix offsets let every task write
    # into its own contiguous window of `dists`.
    offs = Vector{Int}(undef, n_s + 1)
    acc = 0
    @inbounds for i in 1:n_s
        offs[i] = acc
        acc += n_s - i
    end
    offs[n_s + 1] = acc
    @assert acc == n_pairs

    if n_pairs < 100_000 || nthreads() == 1
        _pdist_rows!(dists, X, idx, offs, 1:n_s)
    else
        tasks = map(_thread_chunks(n_s)) do rng
            @spawn _pdist_rows!($dists, $X, $idx, $offs, $rng)
        end
        foreach(wait, tasks)
    end

    return median(dists)
end

# Distances for all pairs whose FIRST index lies in `rows`; written into the
# global `dists` at precomputed offsets via a running cursor.
function _pdist_rows!(dists::Vector{Float64}, X::AbstractMatrix,
                      idx::Vector{Int}, offs::Vector{Int}, rows::UnitRange{Int})
    n_s = length(idx)
    n_dim = size(X, 1)
    xi = Vector{Float64}(undef, n_dim)
    l = offs[first(rows)]                      # cursor inside this stripe
    @inbounds for i in rows
        xi .= @view X[:, idx[i]]
        for jj in (i + 1):n_s
            xj = @view X[:, idx[jj]]
            s = 0.0
            for d in 1:n_dim
                dd = xi[d] - xj[d]
                s += dd * dd
            end
            dists[l] = sqrt(s)
            l += 1
        end
    end
    return nothing
end

function kernel_feature_vector(x, X_dict::AbstractMatrix, sigma)
    N = size(X_dict, 2)
    kx = zeros(N)
    @inbounds for j in 1:N
        kx[j] = rbf_kernel(x, X_dict[:, j], sigma)
    end
    return kx
end

"""
    kernel_edmd_rbf(X_dict, Y_dict, sigma; alpha=1e-6)

The O(N²) Gram-matrix construction is the heavy stage and is now
**thread-parallel over columns**: each task fills the columns of ITS stripe
in both `K_ZZ` and `K_ZY`. Every ordered pair is computed exactly once (no
symmetric mirroring), every matrix element has exactly one writer → no races,
no redundant kernel evaluations, and results match the serial computation.

RAM cost (dominant terms, Float64):
    K_ZZ        N²·8 B          (output — same as serial)
    K_ZY        N²·8 B          (output — same as serial)
    G_reg       N²·8 B          (K_ZZ + αI copy, needed to keep K_ZZ intact)
    solve       K (N²·8), LAPACK workspace ~N²·8
Threading adds NOTHING on top: tasks write directly into the two output
matrices. The final solves are BLAS/LAPACK — leave those to the backend.

⇒ Peak ≈ 48·N² bytes during the solve phase (≈1.2 GB at N=5 000, ≈19 GB at
N=20 000). Use `hankel_kernel_edmd(...; N_subsample=...)` to cap N.
"""
function kernel_edmd_rbf(X_dict::AbstractMatrix, Y_dict::AbstractMatrix,
                         sigma::Real; alpha::Real=1e-6)
    N = size(X_dict, 2)
    @assert size(Y_dict, 2) == N "Y_dict must have same number of columns as X_dict"

    K_ZZ = zeros(N, N)
    K_ZY = zeros(N, N)

    # Precompute exactly like rbf_kernel's `2 * sigma^2` denominator so the
    # values agree bit-for-bit with the scalar rbf_kernel.
    two_sigma2 = 2 * float(sigma)^2

    if N < 2048 || nthreads() == 1
        _fill_kernel_stripes!(K_ZZ, K_ZY, X_dict, Y_dict, two_sigma2, 1:N)
    else
        tasks = map(_thread_chunks(N)) do rng
            @spawn _fill_kernel_stripes!($K_ZZ, $K_ZY, $X_dict, $Y_dict, $two_sigma2, $rng)
        end
        foreach(wait, tasks)
    end

    G_reg = K_ZZ + float(alpha) * I(N)
    K = G_reg \ K_ZY        # LAPACK gesv — multithreaded via BLAS
    B = X_dict / G_reg      # LAPACK trsm — multithreaded via BLAS
    return K, B, K_ZZ, K_ZY
end

# Column-stripe worker: computes K_ZZ[:, cols] and K_ZY[:, cols].
# Inner accumulation mirrors rbf_kernel (sequential sum over dimensions),
# division by the precomputed `two_sigma2` keeps values bit-faithful.
function _fill_kernel_stripes!(K_ZZ::AbstractMatrix, K_ZY::AbstractMatrix,
                               X_dict::AbstractMatrix, Y_dict::AbstractMatrix,
                               two_sigma2::Float64, cols::UnitRange{Int})
    N = size(K_ZZ, 1)
    n_dim = size(X_dict, 1)
    xi = Vector{Float64}(undef, n_dim)
    @inbounds for i in 1:N
        xi .= @view X_dict[:, i]
        for j in cols
            s = 0.0
            for d in 1:n_dim
                dd = xi[d] - X_dict[d, j]
                s += dd * dd
            end
            K_ZZ[i, j] = exp(-(s / two_sigma2))

            s = 0.0
            for d in 1:n_dim
                dd = xi[d] - Y_dict[d, j]
                s += dd * dd
            end
            K_ZY[i, j] = exp(-(s / two_sigma2))
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Generic EDMD multi-step prediction (moved from Hankel.jl)
# ---------------------------------------------------------------------------

"""
    edmd_predict_from_psi(ΨX, K_edmd, B_full, S, start_indices, n_pred)

Multi-step prediction by iterating the Koopman operator **directly in the
lifted (dictionary) space**.  No re-lifting, no projection drift.

Thread-parallel over trajectories (they are fully independent); the
per-trajectory recursion itself is sequential by nature. Writes go to
disjoint trajectory slices of the output arrays.

# Arguments
- `ΨX`: nPsi × (n_snap-1) matrix from `hankel_edmd(...; return_ψ=true)`.
- `K_edmd`: nPsi × nPsi Koopman matrix.
- `B_full`: m_embed × nPsi projection back to full delay space.
- `S`: original m_embed × n_snap Hankel matrix (for ground truth only).
- `start_indices`: column indices in `ΨX` to start from.
- `n_pred`: number of steps to predict.

# Returns
`(X_true, X_pred)` with shape `(1, n_pred+1, n_traj)`.
"""
function edmd_predict_from_psi(ΨX::AbstractMatrix, K_edmd::AbstractMatrix,
                                 B_full::AbstractMatrix, S::AbstractMatrix,
                                 start_indices::AbstractVector{Int}, n_pred::Int)
    nPsi, n_lifted = size(ΨX)
    n_traj = length(start_indices)
    m_embed, n_snap = size(S)

    X_true = zeros(1, n_pred + 1, n_traj)
    X_pred = zeros(1, n_pred + 1, n_traj)

    if n_traj < 32 || nthreads() == 1
        _pred_psi_chunk!(X_true, X_pred, ΨX, K_edmd, B_full, S, start_indices,
                         n_pred, 1:n_traj)
    else
        tasks = map(_thread_chunks(n_traj)) do rng
            @spawn _pred_psi_chunk!($X_true, $X_pred, $ΨX, $K_edmd, $B_full, $S,
                                    $start_indices, $n_pred, $rng)
        end
        foreach(wait, tasks)
    end

    return X_true, X_pred
end

# Trajectory-block worker (identical inner math to the original serial loop).
function _pred_psi_chunk!(X_true::Array{Float64,3}, X_pred::Array{Float64,3},
                          ΨX::AbstractMatrix, K_edmd::AbstractMatrix,
                          B_full::AbstractMatrix, S::AbstractMatrix,
                          start_indices::AbstractVector{Int}, n_pred::Int,
                          trajs::UnitRange{Int})
    n_snap = size(S, 2)
    for j in trajs
        idx = start_indices[j]
        @assert idx + n_pred <= n_snap "Index $idx + $n_pred exceeds $n_snap snapshots"

        # Ground truth (scalar observable = first row of S)
        X_true[1, :, j] = S[1, idx:idx+n_pred]

        # Start from the pre-computed lifted state
        ψ = ΨX[:, idx]
        X_pred[1, 1, j] = (B_full * ψ)[1]

        # Iterate purely in ψ-space
        for k in 1:n_pred
            ψ = K_edmd' * ψ
            X_pred[1, k+1, j] = (B_full * ψ)[1]
        end
    end
    return nothing
end

end # module