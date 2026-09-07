module Utils

using LinearAlgebra
using Statistics
using Printf
using Random
using ProgressMeter
using Base.Threads: @spawn, nthreads, Atomic, atomic_add!

export normalize_vector, signed_area, meshgrid_2d, classify_fixed_point_2d,
       finite_difference_jacobian,
       mutual_information, find_first_minimum,
       false_nearest_neighbors,
       # --- zero-level-set solvers (added) ---
       grad_Psi_RBF, grad_phi, find_zls_gradient_descent

"""
    normalize_vector(x; lb=0.35)

Normalize `x` to the interval `[lb/(1+lb), 1]` so the minimum value stays
slightly above zero.  This is useful for colour-mapping eigenfunctions while
preserving contrast near zero.
"""
function normalize_vector(x; lb=0.35)
    dx = maximum(x) - minimum(x)
    iszero(dx) && return fill(lb / (1 + lb), size(x))
    return ((((x .- minimum(x)) ./ dx) .+ lb) ./ (1 + lb))
end

"""
    signed_area(x, y)

Signed area enclosed by the closed polygon `(x, y)`.  Used to orient PCA
projections consistently with the original (v, w) trajectory.
"""
function signed_area(x, y)
    n = length(x)
    length(y) == n || throw(DimensionMismatch("x and y must have the same length"))
    s = 0.0
    for i in 1:(n - 1)
        s += x[i] * y[i+1] - x[i+1] * y[i]
    end
    s += x[n] * y[1] - x[1] * y[n]
    return s / 2
end

"""
    meshgrid_2d(xrange, yrange)

Return `(Xgrid, Ygrid)` matrices such that `Xgrid[i,j] == xrange[j]` and
`Ygrid[i,j] == yrange[i]`.
"""
function meshgrid_2d(xrange, yrange)
    X = repeat(collect(xrange)', length(yrange), 1)
    Y = repeat(collect(yrange), 1, length(xrange))
    return X, Y
end

"""
    classify_fixed_point_2d(eigenvalues)

Return a human-readable classification of a planar fixed point from its
Jacobian eigenvalues.
"""
function classify_fixed_point_2d(eigenvalues)
    λ1, λ2 = eigenvalues
    if all(isreal, eigenvalues)
        r1, r2 = real(λ1), real(λ2)
        if r1 * r2 < 0
            return "saddle"
        elseif r1 < 0 && r2 < 0
            return r1 ≈ r2 ? "stable node (star/degenerate)" : "stable node"
        elseif r1 > 0 && r2 > 0
            return "unstable node"
        else
            return "non-hyperbolic"
        end
    else
        r = real(λ1)
        if r < 0
            return "stable spiral/focus"
        elseif r > 0
            return "unstable spiral/focus"
        else
            return "center"
        end
    end
end

"""
    finite_difference_jacobian(f, x; h=1e-6)

Compute an approximate Jacobian of `f: ℝⁿ → ℝⁿ` at `x` using central
finite differences.
"""
function finite_difference_jacobian(f, x; h=1e-6)
    n = length(x)
    J = zeros(n, n)
    fx = f(x)
    for j in 1:n
        xh = copy(x)
        xh[j] += h
        xh_m = copy(x)
        xh_m[j] -= h
        J[:, j] .= (f(xh) .- f(xh_m)) ./ (2h)
    end
    return J
end

# ---------------------------------------------------------------------------
# Mutual Information (for time-delay embedding)
# ---------------------------------------------------------------------------

"""
    mutual_information(x, τ; n_bins=50)

Histogram-based mutual information (natural log, nats) between a scalar
time series `x` and its τ-delayed copy `x(t−τ)`.

Returns `NaN` if τ == 0, and `0.0` if the signal is constant.
"""
function mutual_information(x::AbstractVector, τ::Int; n_bins::Int=50)
    N = length(x)
    τ == 0 && return NaN
    τ >= N && return 0.0

    x1 = x[1:N-τ]
    x2 = x[1+τ:N]

    xmin, xmax = minimum(x), maximum(x)
    dx = (xmax - xmin) / n_bins
    iszero(dx) && return 0.0

    joint = zeros(n_bins, n_bins)
    for i in eachindex(x1)
        b1 = min(n_bins, floor(Int, (x1[i] - xmin) / dx) + 1)
        b2 = min(n_bins, floor(Int, (x2[i] - xmin) / dx) + 1)
        joint[b1, b2] += 1
    end

    n_total = sum(joint)
    n_total == 0 && return 0.0

    joint_p = joint ./ n_total
    p_x = sum(joint_p, dims=2)[:]
    p_y = sum(joint_p, dims=1)[:]

    mi = 0.0
    for i in 1:n_bins, j in 1:n_bins
        pij = joint_p[i, j]
        if pij > 0.0
            mi += pij * log(pij / (p_x[i] * p_y[j]))
        end
    end
    return mi
end

"""
    find_first_minimum(v)

Return the index of the first local minimum of vector `v`.
If no local minimum exists, return the global minimum index.
"""
function find_first_minimum(v::AbstractVector)
    for i in 2:(length(v)-1)
        if v[i] < v[i-1] && v[i] <= v[i+1]
            return i
        end
    end
    return argmin(v)
end

# ---------------------------------------------------------------------------
# False Nearest Neighbors (for embedding dimension selection)
# ---------------------------------------------------------------------------

"""
    false_nearest_neighbors(x, m, τ; rtol=10.0, atol=2.0)

Fraction of false nearest neighbors for embedding dimension `m` with
delay `τ` (in samples).  Uses the Kennel criterion: a neighbor is false
if the additional coordinate in R^{m+1} separates the points by more
than `rtol` times their distance in R^m.
"""
function false_nearest_neighbors(x::AbstractVector, m::Int, τ::Int;
                                 rtol::Real=10.0, atol::Real=2.0)
    N = length(x)
    max_lag = m * τ
    n_points = N - max_lag
    n_points <= 0 && return 1.0

    # Delay vectors in R^m and R^{m+1}
    Y_m   = zeros(m,   n_points)
    Y_mp1 = zeros(m+1, n_points)
    for i in 1:n_points
        for j in 0:(m-1)
            Y_m[j+1, i] = x[i + j*τ]
        end
        for j in 0:m
            Y_mp1[j+1, i] = x[i + j*τ]
        end
    end

    n_false = 0
    n_valid = 0

    for i in 1:n_points
        # Nearest neighbor in R^m (excluding self)
        d_min = Inf
        j_min = -1
        yi = Y_m[:, i]
        for j in 1:n_points
            i == j && continue
            d = norm(yi - Y_m[:, j])
            if d < d_min
                d_min = d
                j_min = j
            end
        end

        d_min < 1e-12 && continue
        n_valid += 1

        dx_extra = abs(Y_mp1[m+1, i] - Y_mp1[m+1, j_min])
        if dx_extra / d_min > rtol
            n_false += 1
        end
    end

    n_valid == 0 && return 1.0
    return n_false / n_valid
end

# ===========================================================================
#  Zero-level-set solvers for Koopman eigenfunctions  (added)
# ---------------------------------------------------------------------------
#  These three functions form a self-contained ZLS-solving trio:
#
#    grad_Psi_RBF  -- analytic Jacobian of the thin-plate RBF dictionary
#                     (bit-faithful to Dictionaries.Psi_RBF, including the
#                     +1e-12 regularizer on r so gradients match the actual
#                     computed function exactly).
#
#    grad_phi      -- gradient of a single Koopman eigenfunction
#                     φ(s) = ξ' Ψ(s);  just J(s) * ξ.
#
#    find_zls_gradient_descent  -- multi-start gradient descent on
#                     f(s) = Σ_j φ_j(s)²  → points where all selected
#                     eigenfunctions vanish simultaneously.  Parallelized
#                     across Julia threads via Threads.@spawn with dynamic
#                     load balancing.
#
#  IMPORTANT: launch Julia with multiple threads BEFORE running:
#
#      JULIA_NUM_THREADS=auto julia -O3 script.jl
#
#  and call `BLAS.set_num_threads(1)` once from your script so BLAS does
#  not oversubscribe the cores.
# ===========================================================================

# ---------------------------------------------------------------------------
#  Analytic Jacobian of the thin-plate RBF dictionary
# ---------------------------------------------------------------------------
#  Layout (must match Dictionaries.Psi_RBF):
#
#    include_states=true:   Ψ(s) = [1; s[idx_1]; ...; s[idx_K];
#                                    r_1²·log(r_1); ...; r_M²·log(r_M)]
#                          where r_k = ||s - c_k|| + 1e-12
#
#    include_states=false:  Ψ(s) = [r_1²·log(r_1); ...; r_M²·log(r_M)]
#
#  Gradient of the thin-plate basis ψ_k(s) = r_k²·log(r_k):
#
#    dψ_k/ds_j = (2·log(r_eff) + 1) · (r_eff / r_raw) · (s_j - c_k_j)
#
#  where r_raw = ||s - c_k||  and  r_eff = r_raw + 1e-12.  At the center
#  (r_raw → 0) the limit is 0 by smoothness, so we skip that column.
# ---------------------------------------------------------------------------
"""
    grad_Psi_RBF(s, centers; include_states=true, state_indices=nothing)

Analytic Jacobian `J` of the thin-plate RBF dictionary `Ψ(s)` (as defined
in `Dictionaries.Psi_RBF`) evaluated at a single state vector `s`.

`J[i, j] = ∂Ψ_j/∂s_i`, shape `(n_dim, nPsi)` where `n_dim = length(s)`
and `nPsi = 1 + length(state_indices) + nRBF` if `include_states` else `nRBF`.

This is the gradient companion to `Dictionaries.Psi_RBF` — keep the two in
sync if you ever change the basis.
"""
function grad_Psi_RBF(s::AbstractVector, centers::AbstractMatrix;
                      include_states::Bool=true,
                      state_indices::Union{Nothing,Vector{Int}}=nothing)
    n_dim = length(s)
    nRBF  = size(centers, 2)

    if include_states
        state_indices = isnothing(state_indices) ? collect(1:n_dim) :
                                                  collect(Int, state_indices)
        nPsi   = nRBF + 1 + length(state_indices)
        offset = 1 + length(state_indices)
    else
        state_indices = Int[]
        nPsi   = nRBF
        offset = 0
    end

    J = zeros(n_dim, nPsi)

    if include_states
        # Constant term: zero gradient (already zeroed)
        # Linear terms: ∂s[idx_k]/∂s_j = δ_{j, idx_k}
        for (k, idx) in enumerate(state_indices)
            1 <= idx <= n_dim || throw(BoundsError("state_indices entry $idx out of range 1:$n_dim"))
            J[idx, 1 + k] = 1.0
        end
    end

    # Thin-plate RBF terms
    @inbounds for k in 1:nRBF
        # diff = s - c_k
        r_raw_sq = 0.0
        @simd for j in 1:n_dim
            d = s[j] - centers[j, k]
            r_raw_sq += d * d
        end
        r_raw = sqrt(r_raw_sq)
        if r_raw < 1e-10
            # At the center: smooth limit of thin-plate RBF gradient is zero
            continue
        end
        r_eff = r_raw + 1e-12   # match Psi_RBF's regularizer exactly
        coef  = (2 * log(r_eff) + 1) * (r_eff / r_raw)
        @simd for j in 1:n_dim
            J[j, offset + k] = coef * (s[j] - centers[j, k])
        end
    end

    return J
end


# ---------------------------------------------------------------------------
#  Gradient of a single Koopman eigenfunction  φ(s) = ξ' Ψ(s)
# ---------------------------------------------------------------------------
"""
    grad_phi(s, ξ, dict_info)

Gradient `∇φ(s)` of the Koopman eigenfunction  φ(s) = ξ' Ψ(s)
where `Ψ` is the thin-plate RBF dictionary specified by `dict_info`.

`dict_info` must contain at least:
- `centers`        : RBF centers matrix, `n_dim × nRBF`
- `include_states` : Bool, whether the dictionary prepends `[1; s[idxs]]`
- `state_indices`  : (optional) which entries of `s` to include as linear terms

Returns a real-valued vector of length `n_dim`.
"""
function grad_phi(s::AbstractVector, ξ::AbstractVector, dict_info::NamedTuple)
    centers        = dict_info.centers
    include_states = dict_info.include_states
    state_indices  = get(dict_info, :state_indices, nothing)

    J = grad_Psi_RBF(s, centers;
                     include_states=include_states,
                     state_indices=state_indices)
    return real.(J * ξ)
end


# ---------------------------------------------------------------------------
#  Parallel multi-start gradient descent on f(s) = Σ_j φ_j(s)²
# ---------------------------------------------------------------------------
#  Each start is dispatched to a Julia thread via Threads.@spawn; the runtime
#  work-stealing scheduler gives dynamic load balancing across cores, which
#  matters here because some starts converge in 5 iterations and others run
#  the full n_iter budget.
#
#  Per-start state lives in pre-allocated per-index slots so no two threads
#  ever touch the same memory.  Only the progress-bar counter, the converged
#  count, and `f_max` are shared — all guarded by atomics + a small lock.
# ---------------------------------------------------------------------------
"""
    find_zls_gradient_descent(S, Psi_func, Ξ, j_modes, dict_info; kwargs...)

Find points on the zero level set of one or more Koopman eigenfunctions by
minimizing  f(s) = Σ_j φ_j(s)²  via gradient descent with backtracking line
search (Armijo condition).

# Arguments
- `S`          : `n_dim × n_snap` snapshot matrix; start points are sampled
                 from its columns.
- `Psi_func`   : callable `X -> Ψ(X)` returning the lifted dictionary
                 matrix (the same closure you use elsewhere in the pipeline).
- `Ξ`          : matrix of Koopman eigenvectors (columns are eigenvectors).
- `j_modes`    : single Int (minimize φ_j²) or Vector{Int} (minimize Σ φ_j²).
- `dict_info`  : NamedTuple consumed by `grad_phi` (centers, include_states,
                 state_indices).

# Keyword arguments
- `n_starts` : number of starts sampled from `S` (default 500).
- `n_iter`   : max gradient-descent iterations per start (default 200).
- `lr`       : base learning rate (default 0.1).
- `tol`      : gradient-norm convergence threshold (default 1e-8).
- `verbose`  : print iteration log for start #1 only (default true).

# Returns
- `Vector{Vector{Float64}}` of de-duplicated converged points (radius 0.05).

# Threading
Set `JULIA_NUM_THREADS=auto` (or run Julia with `-t auto`) BEFORE launching.
Also call `BLAS.set_num_threads(1)` from your script to avoid BLAS
oversubscription.
"""
function find_zls_gradient_descent(S::AbstractMatrix, Psi_func::Function,
                                    Ξ::AbstractMatrix, j_modes, dict_info::NamedTuple;
                                    n_starts::Int=500, n_iter::Int=200,
                                    lr::Float64=0.1, tol::Float64=1e-8,
                                    verbose::Bool=true)
    # ---- Normalize j_modes to a vector ---------------------------------
    if j_modes isa Integer
        j_modes = [Int(j_modes)]
    else
        j_modes = collect(Int, j_modes)
    end
    n_modes = length(j_modes)

    n_dim, n_snap = size(S)
    ξs = [Ξ[:, j] for j in j_modes]   # read-only column views (one per mode)

    mode_str = n_modes == 1 ? "φ_$(j_modes[1])" :
               n_modes == 2 ? "φ_$(j_modes[1])² + φ_$(j_modes[2])²" :
               "Σ φ_j² (j=$j_modes)"

    n_threads = nthreads()
    println("    [grad-desc] Modes: $j_modes  (objective: f = $mode_str)")
    println("    [grad-desc] Starting from $n_starts points, $n_iter iterations each")
    if n_threads > 1
        println("    [grad-desc] Parallelizing across $n_threads threads (dynamic schedule)")
    else
        println("    [grad-desc] Running serially (JULIA_NUM_THREADS not set)")
    end

    # ---- Compute scale for convergence threshold -----------------------
    sample_idx = unique(rand(1:n_snap, min(200, n_snap)))
    f_scale = 0.0
    for idx in sample_idx
        Ψ_sample = Psi_func(reshape(S[:, idx], :, 1))
        for ξ in ξs
            f_scale += real(dot(ξ, Ψ_sample[:, 1]))^2
        end
    end
    f_scale = f_scale / length(sample_idx) + 1e-12

    if verbose
        println("    [grad-desc] Verbose safety printing for start #1:")
        println("    ┌──────┬──────────────┬──────────────" * repeat("┬──────────────", n_modes) * "┬──────────┬──────────┐")
        println("    │ iter │      f       │    ‖∇f‖      " *
                join([@sprintf("│    φ_%-4d    ", j) for j in j_modes]) *
                "│    α     │   ‖s‖    │")
        println("    ├──────┼──────────────┼──────────────" * repeat("┼──────────────", n_modes) * "┼──────────┼──────────┤")
    end

    start_idx = unique(rand(1:n_snap, min(n_starts, n_snap)))
    n_starts_actual = length(start_idx)

    # ---- Pre-allocate per-start result slots (one writer per slot) -----
    # Each task writes ONLY to its own index; no data race.
    per_start_s    = Vector{Vector{Float64}}(undef, n_starts_actual)
    per_start_f    = Vector{Float64}(undef, n_starts_actual)
    per_start_conv = Vector{Bool}(undef, n_starts_actual)
    fill!(per_start_conv, false)

    # ---- Inner worker: run ONE start to convergence --------------------
    # Pure function — no shared state mutated; returns (s, f_val, converged).
    function _run_one(i::Int, idx::Int)
        s = copy(S[:, idx])

        for iter in 1:n_iter
            Ψ_s = Psi_func(reshape(s, :, 1))
            φs = [real(dot(ξ, Ψ_s[:, 1])) for ξ in ξs]

            # ∇f = Σ_k 2 φ_k ∇φ_k
            ∇f = zeros(n_dim)
            for (k, ξ) in enumerate(ξs)
                ∇φ_k = grad_phi(s, ξ, dict_info)
                ∇f .+= 2 .* φs[k] .* ∇φ_k
            end

            f_curr = sum(φs .^ 2)
            gnorm = norm(∇f)

            if gnorm < tol || f_curr < 1e-12 * f_scale
                if verbose && i == 1
                    @printf("    │ %4d │ %12.6e │ %12.6e │  CONVERGED (f < 1e-12 * scale)\n",
                            iter, f_curr, gnorm)
                end
                break
            end

            # Backtracking line search (Armijo condition)
            α = max(lr, 1.0 / (gnorm + 1e-12))
            α_used = 0.0
            for _ in 1:30
                s_new = s .- α .* ∇f
                Ψ_new = Psi_func(reshape(s_new, :, 1))
                φs_new = [real(dot(ξ, Ψ_new[:, 1])) for ξ in ξs]
                f_new = sum(φs_new .^ 2)
                if f_new < f_curr - 1e-4 * α * gnorm^2
                    s = s_new
                    α_used = α
                    break
                end
                α *= 0.5
            end

            # Only one task owns i==1 — these prints cannot race with themselves
            if verbose && i == 1
                φ_str = join([@sprintf("%12.6e │", φ) for φ in φs])
                @printf("    │ %4d │ %12.6e │ %12.6e │%s %8.4f │ %8.4f │\n",
                        iter, f_curr, gnorm, φ_str, α_used, norm(s))
            end

            if α_used == 0.0
                α_fixed = 0.01 / (gnorm + 1e-12)
                s = s .- α_fixed .* ∇f
                if verbose && i == 1
                    @printf("    │      │  LINE SEARCH FAILED — taking fixed step α=%.4e\n", α_fixed)
                end
            end
        end

        if verbose && i == 1
            println("    └──────┴──────────────┴──────────────" * repeat("┴──────────────", n_modes) * "┴──────────┴──────────┘")
            println("    (verbose printing disabled for starts #2 onward)")
        end

        # Final objective value
        Ψ_final = Psi_func(reshape(s, :, 1))
        φs_final = [real(dot(ξ, Ψ_final[:, 1])) for ξ in ξs]
        f_val = sum(φs_final .^ 2)
        conv = f_val < 1e-6 * f_scale

        return (s, f_val, conv)
    end

    # ---- Progress bar (thread-safe via lock) ---------------------------
    p_c = Progress(n_starts_actual;
                   desc="Alg C ($n_modes mode" * (n_modes > 1 ? "s" : "") * ")",
                   dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=30)

    n_done   = Atomic{Int}(0)
    n_conv   = Atomic{Int}(0)
    f_max_ref = Ref{Float64}(0.0)
    prog_lock = ReentrantLock()

    # ---- Dispatch each start to a thread (dynamic) ----------------------
    # Tasks that finish early automatically pick up the next pending start,
    # because @spawn uses the runtime scheduler work-stealing queue.
    tasks = Vector{Task}(undef, n_starts_actual)
    for (i, idx) in enumerate(start_idx)
        tasks[i] = @spawn begin
            s, f_val, conv = _run_one($i, $idx)

            # Write into OUR OWN slot (no other task touches this index)
            per_start_s[$i]    = s
            per_start_f[$i]    = f_val
            per_start_conv[$i] = conv

            # Aggregate atomics + progress bar (shared state → lock)
            _ = atomic_add!(n_done, 1)
            if conv
                _ = atomic_add!(n_conv, 1)
                lock(prog_lock) do
                    if f_val > f_max_ref[]
                        f_max_ref[] = f_val
                    end
                end
            end
            lock(prog_lock) do
                cv = n_conv[]
                next!(p_c; showvalues=[
                    ("converged", cv),
                    ("f_max",     cv == 0 ? "—" : @sprintf("%.2e", f_max_ref[])),
                ])
            end
        end
    end

    # Wait for all starts to complete (no blocking on individual results)
    foreach(wait, tasks)
    finish!(p_c)

    # ---- Aggregate per-start results into final lists -------------------
    results = Vector{Float64}[]
    f_final = Float64[]
    for i in 1:n_starts_actual
        if per_start_conv[i]
            push!(results, per_start_s[i])
            push!(f_final, per_start_f[i])
        end
    end

    if !isempty(results)
        dedup_radius = 0.05
        unique_pts = Vector{Float64}[]
        for p in results
            is_dup = any(norm(p .- q) < dedup_radius for q in unique_pts)
            if !is_dup
                push!(unique_pts, p)
            end
        end
        println("    [grad-desc] $(length(results)) converged, $(length(unique_pts)) unique after dedup")
        if !isempty(f_final)
            println("    [grad-desc] Final objective: mean=$(round(mean(f_final), sigdigits=4)), max=$(round(maximum(f_final), sigdigits=4))")
        end
        return unique_pts
    end
    println("    [grad-desc] No points converged to f < tol")
    return Vector{Float64}[]
end

end # module