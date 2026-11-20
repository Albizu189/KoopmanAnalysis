module Hankel

using LinearAlgebra
using Random
using Statistics   # ← NEW: needed for median() in select_svd_rank
using ..EDMD: compute_koopman_operator, construct_projection_operator, edmd_predict_from_psi,
                        kernel_edmd_rbf, median_heuristic_sigma
using ..Dictionaries: Psi_Hermite, Psi_RBF, Psi_RFF, cluster_data, build_rff_basis,
                        construct_projection_operator_hermite, lift_state
using ..Utils: normalize_states, apply_norm_stats

export build_hankel, hankel_dmd, hankel_edmd, hankel_kernel_edmd, havok_dmd, havok_predict,
        delay_space_edmd_prediction, select_svd_rank,
        build_hankel_multichannel, delay_embed_training_data,
        delay_embed_pairs, pair_views, dmd_pairs, edmd_pairs, havok_pairs

# =============================================================================
# Build_hankel and hankel_dmd
# =============================================================================

function build_hankel(v_series::AbstractVector, m_embed::Int, tau_delay::Int)
    N = length(v_series)
    max_lag = (m_embed - 1) * tau_delay
    n_snap = N - max_lag
    n_snap <= 0 && error("Time series too short for the requested embedding.")
    S = zeros(m_embed, n_snap)
    @inbounds for t in 1:n_snap
        base = t + max_lag
        for d in 0:(m_embed - 1)
            S[d + 1, t] = v_series[base - d * tau_delay]
        end
    end
    return S
end

# ---------------------------------------------------------------------------
# Multi-channel delay embedding (N-dimensional state)
# ---------------------------------------------------------------------------

"""
    build_hankel_multichannel(X::AbstractMatrix, m_embed::Int, tau_delay::Int)

Build a multi-channel Hankel matrix from an n-dimensional state trajectory.
`X` is `n × T`.  Returns `(n·m_embed) × (T - (m_embed-1)·tau_delay)` matrix.
"""
function build_hankel_multichannel(X::AbstractMatrix, m_embed::Int, tau_delay::Int)
    n, T = size(X)
    max_lag = (m_embed - 1) * tau_delay
    n_snap = T - max_lag
    n_snap <= 0 && error("Trajectory too short: T=$T < max_lag=$max_lag")
    S = zeros(n * m_embed, n_snap)
    for d in 1:n
        H = build_hankel(X[d, :], m_embed, tau_delay)
        S[(d-1)*m_embed+1 : d*m_embed, :] .= H
    end
    return S
end

function hankel_dmd(S::AbstractMatrix; r::Union{Nothing,Int}=nothing, dt::Real=1.0)
    F = svd(S, full=false)
    r_actual = isnothing(r) ? length(F.S) : min(r, length(F.S))
    U_r = F.U[:, 1:r_actual]
    Σ_r = Diagonal(F.S[1:r_actual])
    Vt_r = F.Vt[1:r_actual, :]
    X_r = Σ_r * Vt_r[:, 1:end-1]
    Y_r = Σ_r * Vt_r[:, 2:end]
    K_tilde = Y_r * pinv(X_r)
    K = U_r * K_tilde * U_r'
    B_proj = U_r'
    return K, K_tilde, X_r, B_proj, U_r
end

# =============================================================================
# hankel_edmd
# =============================================================================

"""
    hankel_edmd(S; r, dt, dict_type, dict_params, edmd_method, edmd_alpha,
                proj_alpha, return_ψ, use_svd)

Hankel-EDMD with an explicit dictionary.

# Memory layout (important for long series)
`X_r`/`Y_r` are consecutive column ranges of ONE coordinate matrix `Vc`
(`Vc = Σ_r·Vt_r` with SVD, or `S` itself without), so they are stored as
**views** — no copies.

Because the dictionary acts columnwise, `ψ(Y_r)` is exactly `ψ(Vc)` shifted
by one column. The lifted data is therefore computed **once** into a single
matrix `Ψall = ψ(Vc)`, and `ΨX`/`ΨY` are returned as **column views** of it:
`ΨX ≡ Ψall[:, 1:end-1]`, `ΨY ≡ Ψall[:, 2:end]`.

Compared to materializing `X_r`, `Y_r`, `ΨX`, `ΨY` separately this saves,
e.g. for the full-RBF case, ~15 GiB of copies plus ~11 GiB of duplicate
lifted data — and it also halves the dictionary computation time, since
`Psi_*` is evaluated once instead of twice.
"""
function hankel_edmd(S::AbstractMatrix;
    r::Union{Nothing,Int}=nothing,
    dt::Real=1.0,
    dict_type::Symbol=:hermite,
    dict_params::NamedTuple=NamedTuple(),
    edmd_method::Symbol=:ridge,
    edmd_alpha::Real=1e-3,
    proj_alpha::Real=1e-6,
    return_ψ::Bool=false,
    use_svd::Bool=true
)
    dict_type = Symbol(lowercase(String(dict_type)))

    m_embed, n_snap = size(S)

    # ── Reduced coordinates (views, no copies) ──
    if use_svd
        F = svd(S, full=false)
        r_actual = isnothing(r) ? length(F.S) : min(r, length(F.S))
        U_r = F.U[:, 1:r_actual]
        Vc = Diagonal(F.S[1:r_actual]) * F.Vt[1:r_actual, :]   # r × n_snap
    else
        r_actual = m_embed
        U_r = Matrix{Float64}(I, m_embed, m_embed)
        Vc = S                                                # no copy
    end
    X_r = view(Vc, :, 1:n_snap-1)
    Y_r = view(Vc, :, 2:n_snap)

    # ── Dictionary lift, computed ONCE (see docstring) ──
    Ψall, B_reduced, dict_info = _build_dict_and_project(
        Vc, 1:n_snap-1, dict_type, dict_params, r_actual, proj_alpha
    )
    ΨX = view(Ψall, :, 1:n_snap-1)
    ΨY = view(Ψall, :, 2:n_snap)

    if dict_type == :hermite && r_actual > 100
        nPsi = size(Ψall, 1)
        @warn "Hermite dictionary with r=$r_actual produces nPsi=$nPsi observables. " *
              "Consider reducing `r` or `max_deg`."
    end

    K_edmd = compute_koopman_operator(ΨX, ΨY; method=edmd_method, alpha=edmd_alpha)

    B_full = use_svd ? U_r * B_reduced : B_reduced

    if return_ψ
        return K_edmd, B_reduced, B_full, U_r, X_r, dict_info, ΨX, ΨY
    else
        return K_edmd, B_reduced, B_full, U_r, X_r, dict_info
    end
end

# =============================================================================
# Hankel Kernel-EDMD based in RBF Gaussian Kernel
# =============================================================================

function hankel_kernel_edmd(S::AbstractMatrix;
    r::Union{Nothing,Int}=100,
    dt::Real=1.0,
    sigma::Real=1.0,
    alpha::Real=1e-6,
    use_svd::Bool=true,
    N_subsample::Union{Nothing,Int}=nothing,
    seed::Union{Nothing,Int}=nothing
)
    isnothing(seed) || Random.seed!(seed)
    if use_svd
        F = svd(S, full=false)
        r_actual = isnothing(r) ? length(F.S) : min(r, length(F.S))
        U_r = F.U[:, 1:r_actual]
        Σ_r = Diagonal(F.S[1:r_actual])
        Vt_r = F.Vt[1:r_actual, :]
        X = Σ_r * Vt_r[:, 1:end-1]
        Y = Σ_r * Vt_r[:, 2:end]
    else
        r_actual = size(S, 1)
        U_r = nothing
        X = S[:, 1:end-1]
        Y = S[:, 2:end]
    end

    # --- subsample for exact kernel feasibility ---
    if !isnothing(N_subsample) && N_subsample < size(X, 2)
        idx = randperm(size(X, 2))[1:N_subsample]
        X = X[:, idx]
        Y = Y[:, idx]
    end

    K, B, K_ZZ, K_ZY = kernel_edmd_rbf(X, Y, sigma; alpha=alpha)
    return K, B, X, Y, U_r
end

# =============================================================================
# HAVOK — Hankel Alternative View of Koopman
# (continuous-time formulation, following Brunton et al. 2017, arXiv:1608.05306)
# =============================================================================
#
# Pipeline implemented here, matching the paper:
#   1. SVD of the Hankel matrix  →  POD modes U_r and coordinates V_r = U_r'S.
#      (Computed via eigen(S*S') + a projection instead of svd(S): identical
#       U_r up to irrelevant sign conventions, but avoids materializing the
#       full m_embed × n_snap Vt factor, which is several GiB for long series.)
#   2. Automatic choice of the number r of modes to keep, from the singular
#      value spectrum (select_svd_rank: cumulative energy / Gavish–Donoho
#      hard threshold / largest spectral gap).
#   3. Least-squares regression for the CONTINUOUS-time linear system with
#      forcing,  d/dt v = A v + B v_input,  where v_input = v_r is the last
#      POD coordinate. Time derivatives of the POD coordinates are estimated
#      with 4th-order central differences, as in the paper.
#   4. Prediction by integrating that linear system with RK4 (see
#      havok_predict).

"""
    select_svd_rank(σ; method=:energy, energy=0.99, threshold=1e-10, max_rank=nothing)

Select SVD rank from singular values σ using :energy, :hard_threshold, or :gap method.
"""
function select_svd_rank(σ::AbstractVector; method::Symbol=:energy,
                         energy::Real=0.999, aspect::Real=1.0)
    if method == :energy
        cume = cumsum(σ .^ 2) ./ sum(σ .^ 2)
        r = findfirst(>=(energy), cume)
        return isnothing(r) ? length(σ) : r

    elseif method == :hard_threshold
        β = clamp(aspect, 1e-6, 1.0)
        ω = 0.56*β^3 - 0.95*β^2 + 1.82*β + 1.43
        τ = ω * median(σ)
        return max(count(>(τ), σ), 2)

    elseif method == :gap
        pos = σ .> 0
        logσ = log.(σ[pos])
        length(logσ) < 3 && return max(length(logσ), 2)
        return max(argmax(diff(logσ)), 2)

    else
        error("Unknown rank-selection method: $method. " *
              "Use :energy, :hard_threshold, or :gap.")
    end
end

"""
    havok_dmd(S; r=nothing, dt=1.0, rank_method=:energy, energy=0.999, ridge_alpha=0.0)

Hankel Alternative View of Koopman (HAVOK) analysis, **continuous-time**
formulation following Brunton et al. (2017), arXiv:1608.05306.

Decomposes the delay-embedded dynamics into a linear model in the leading
`r−1` POD coordinates with forcing from the `r`-th coordinate:

    d/dt v^{(1:r-1)} = A v^{(1:r-1)} + B v^{(r)}

# Steps
1. SVD of the Hankel matrix (via `eigen(S*S')` — memory-efficient, see above).
2. If `r === nothing`, the number of modes is chosen automatically from the
   singular value spectrum with `select_svd_rank(; method=rank_method)`.
3. `A` and `B` are obtained by least-squares regression of the time
   derivatives of the POD coordinates (4th-order central differences, step `dt`)
   onto `[v_{1:r-1}; v_r]` (optionally ridge-regularised with `ridge_alpha`).

# Returns `(K, A, B, U_r, V_r, u)`
- `A`: `(r−1)×(r−1)` continuous-time dynamics matrix.
- `B`: `(r−1)×1` continuous-time forcing matrix.
- `U_r`: `m_embed × r` POD modes.
- `V_r`: `r × n_snap` POD coordinates of the training data.
- `u`: the full forcing signal `V_r[r, :]`.
- `K`: `r×r` discrete flow map `exp([A B; 0 0]·dt)` of the augmented
  autonomous system over one snapshot interval. Provided only so that
  spectral diagnostics (`spectrum`, eigenvalue plots) can treat the HAVOK
  model like the other (discrete-time) methods; predictions should use
  `havok_predict` with `A` and `B`, not `K`.
"""
function havok_dmd(S::AbstractMatrix; r::Union{Nothing,Int}=nothing, dt::Real=1.0,
                   rank_method::Symbol=:energy, energy::Real=0.999,
                   ridge_alpha::Real=0.0)
    m_embed, n_snap = size(S)

    # ── 1. SVD of the Hankel matrix ──
    # U and σ from eigen(S*S') (BLAS syrk): only an m_embed×m_embed allocation
    # on top of S, instead of the full n_snap-sized Vt factor of svd(S).
    G = S * S'
    E = eigen(Symmetric(G))
    p = sortperm(E.values; rev=true)
    σ = sqrt.(max.(E.values[p], 0.0))
    U_all = E.vectors[:, p]

    # ── 2. Number of modes to keep ──
    r_actual = if isnothing(r)
        select_svd_rank(σ; method=rank_method, energy=energy,
                        aspect=m_embed / n_snap)
    else
        min(r, length(σ))
    end
    r_actual = min(r_actual, n_snap)
    r_actual < 2 && error("HAVOK requires r ≥ 2 (need at least one linear coordinate + forcing).")
    @info "HAVOK: keeping r = $r_actual POD modes (selection = $rank_method)"

    U_r = U_all[:, 1:r_actual]
    V_r = U_r' * S                       # r × n_snap POD coordinates (= Σ_r Vt_r)

    # ── 3. Regression for dv/dt = A v + B v_input ──
    r_lin = r_actual - 1
    # 4th-order central differences on interior columns, as in the paper:
    #   dv/dt|_t ≈ (−v[t+2] + 8v[t+1] − 8v[t−1] + v[t−2]) / (12 dt)
    n_int = n_snap - 4
    X  = V_r[1:r_lin, 3:n_snap-2]         # linear states at interior times
    u  = vec(V_r[r_actual, 3:n_snap-2])   # forcing at interior times
    Vd = zeros(r_lin, n_int)              # their time derivatives
    @inbounds for t in 3:n_snap-2
        for j in 1:r_lin
            Vd[j, t-2] = (-V_r[j, t+2] + 8V_r[j, t+1] - 8V_r[j, t-1] + V_r[j, t-2]) / (12*dt)
        end
    end

    Z = vcat(X, reshape(u, 1, n_int))     # r × n_int regression input
    AB = if ridge_alpha > 0
        (Vd * Z') / (Z * Z' + float(ridge_alpha) * I(r_actual))
    else
        Vd * pinv(Z)                      # least squares, as in the paper
    end
    A = AB[:, 1:r_lin]
    B = AB[:, r_actual:r_actual]

    # Discrete flow map of the augmented autonomous system over one snapshot
    # interval — for spectral diagnostics only (see docstring).
    𝒜 = zeros(r_actual, r_actual)
    𝒜[1:r_lin, :] .= AB
    K = Matrix(exp(𝒜 * dt))

    return K, A, B, U_r, V_r, vec(V_r[r_actual, :])
end

"""
    havok_predict(A, B, U_r, S, start_indices, n_pred; dt=1.0, use_true_forcing=true)

Predict trajectories by **integrating the HAVOK linear system with RK4**:

    d/dt v = A v + B u(t),     u(t) = v_r(t)  (the forcing POD coordinate)

One RK4 step of size `dt` is taken per snapshot interval, with the forcing
sampled at the step endpoints and linearly interpolated at the midpoint.

If `use_true_forcing=true` (default), the forcing is extracted from the true
delay-embedded data — the *reconstruction* mode of the paper. If `false`,
the forcing is frozen at its initial value (fully autonomous model).

Returns `(X_true, X_pred)` with shape `(1, n_pred+1, n_traj)`, where the
observable is the first row of the reconstructed delay state `U_r * [v; u]`.
"""
function havok_predict(A::AbstractMatrix, B::AbstractMatrix, U_r::AbstractMatrix,
                       S::AbstractMatrix, start_indices::AbstractVector{Int},
                       n_pred::Int; dt::Real=1.0, use_true_forcing::Bool=true)
    m_embed, n_snap = size(S)
    n_traj = length(start_indices)
    r = size(U_r, 2)
    r_lin = r - 1
    Bv = vec(B)

    # True forcing signal over all training snapshots: v_r(t) = U_r[:,r]' * s(t)
    u_true = vec(U_r[:, r]' * S)

    X_true = zeros(1, n_pred + 1, n_traj)
    X_pred = zeros(1, n_pred + 1, n_traj)

    for j in 1:n_traj
        idx = start_indices[j]
        @assert idx + n_pred <= n_snap "Index $idx + $n_pred exceeds $n_snap snapshots"

        X_true[1, :, j] = S[1, idx:idx+n_pred]

        # Initial POD state
        v_full = U_r' * S[:, idx]
        v = collect(v_full[1:r_lin])
        u_curr = v_full[r]

        X_pred[1, 1, j] = S[1, idx]

        for k in 1:n_pred
            # Forcing at the endpoints of this step
            u_next = use_true_forcing ? u_true[idx + k] : u_curr
            u_mid  = 0.5 * (u_curr + u_next)

            # ── RK4 integration of dv/dt = A v + B u over one snapshot ──
            k1 = A * v                      .+ Bv .* u_curr
            k2 = A * (v .+ 0.5 * dt .* k1)  .+ Bv .* u_mid
            k3 = A * (v .+ 0.5 * dt .* k2)  .+ Bv .* u_mid
            k4 = A * (v .+ dt .* k3)        .+ Bv .* u_next
            v .+= (dt / 6) .* (k1 .+ 2 .* (k2 .+ k3) .+ k4)

            u_curr = u_next

            # Reconstruct the delay state and take the observable
            s_next = U_r * vcat(v, u_curr)
            X_pred[1, k+1, j] = s_next[1]
        end
    end

    return X_true, X_pred
end

# =============================================================================
# Delay-space EDMD multi-trajectory data generator
# =============================================================================

"""
    delay_embed_training_data(X_train, Y_train, m_embed, tau_delay; n_trajectories=1)

Build delay-embedded training matrices from multi-trajectory data.
"""
function delay_embed_training_data(X_train::AbstractMatrix, Y_train::AbstractMatrix,
                                    m_embed::Int, tau_delay::Int;
                                    n_trajectories::Int=1)
    n, total_cols = size(X_train)
    m_traj = div(total_cols, n_trajectories)
    @assert m_traj * n_trajectories == total_cols "Total columns not divisible by n_trajectories"

    max_lag = (m_embed - 1) * tau_delay
    T_full = m_traj + 1
    @assert T_full > max_lag "Trajectory length $T_full too short for embedding"

    n_delay = n * m_embed
    n_pairs = T_full - max_lag - 1

    X_delay = zeros(n_delay, n_pairs * n_trajectories)
    Y_delay = zeros(n_delay, n_pairs * n_trajectories)

    for i in 1:n_trajectories
        start_idx = (i - 1) * m_traj + 1
        # Reconstruct full trajectory i from X_train / Y_train
        X_traj = zeros(n, T_full)
        X_traj[:, 1] = X_train[:, start_idx]
        X_traj[:, 2:end] = Y_train[:, start_idx : start_idx + m_traj - 1]

        S = build_hankel_multichannel(X_traj, m_embed, tau_delay)

        out_start = (i - 1) * n_pairs + 1
        out_end   = i * n_pairs
        X_delay[:, out_start:out_end] = S[:, 1:end-1]
        Y_delay[:, out_start:out_end] = S[:, 2:end]
    end

    return X_delay, Y_delay
end

# =============================================================================
# Delay-space EDMD prediction (re-lifting version for new initial conditions)
# =============================================================================

"""
    delay_space_edmd_prediction(S, K_edmd, B_full, U_r, start_indices, n_pred, dict_info)

EDMD prediction for **new initial conditions** not in the training set.
At each step the state is projected, re-lifted, evolved, and projected back.
Uses `Dictionaries.lift_state` for the lifting step.
"""
function delay_space_edmd_prediction(S::AbstractMatrix, K_edmd::AbstractMatrix,
                                      B_full::AbstractMatrix, U_r::AbstractMatrix,
                                      start_indices::AbstractVector{Int}, n_pred::Int,
                                      dict_info::NamedTuple)
    m_embed, n_snap = size(S)
    n_traj = length(start_indices)

    X_true = zeros(1, n_pred + 1, n_traj)
    X_pred = zeros(1, n_pred + 1, n_traj)

    for j in 1:n_traj
        idx = start_indices[j]
        @assert idx + n_pred <= n_snap "Index $idx + $n_pred exceeds $n_snap snapshots"

        X_true[1, :, j] = S[1, idx:idx+n_pred]

        s_curr = S[:, idx]
        X_pred[1, 1, j] = s_curr[1]

        for k in 1:n_pred
            x_r = U_r' * s_curr
            ψ = lift_state(x_r, dict_info)
            ψ_next = K_edmd' * ψ
            s_next = B_full * ψ_next
            X_pred[1, k+1, j] = s_next[1]
            s_curr = s_next
        end
    end

    return X_true, X_pred
end

# =============================================================================
# Internal helpers
# =============================================================================

# Lift the full coordinate matrix ONCE and return it as Ψall; the caller
# derives its ΨX/ΨY as views:  hankel_edmd uses the contiguous shift
# x_cols = 1:n_snap-1,  edmd_pairs passes the multi-trajectory index pd.x.
#
# Vc:      full coordinate matrix (r × n_cols).  The X-side columns are
#          Vc[:, x_cols] — used for centre clustering and the projection
#          operator.  The Y side is never needed here: the dictionary acts
#          columnwise, so ψ of any Y column is already a column of Ψall.
# x_cols:  column indices of Vc forming the X side of the snapshot pairs.
#          Any index set is accepted — a contiguous UnitRange (legacy
#          Hankel path) or an arbitrary Vector{Int} (multi-trajectory path).
#
# Returns (Ψall, B_reduced, dict_info) with the SAME contract as before the
# pair refactor: Ψall = ψ(Vc) over ALL columns (the caller takes its own
# column views of it); B_reduced maps Ψ-space back to the `state_dim`
# Vc-coordinates, fit on the X columns only.
function _build_dict_and_project(Vc, x_cols::AbstractVector{Int},
                                 dict_type, dict_params, state_dim, proj_alpha)
    n_snap = size(Vc, 2)
    (isempty(x_cols) || minimum(x_cols) < 1 || maximum(x_cols) > n_snap) &&
        error("_build_dict_and_project: x_cols must be non-empty column " *
              "indices in 1:$n_snap (got $(length(x_cols)) indices).")

    if dict_type == :hermite
        max_deg = get(dict_params, :max_deg, 2)
        basis_type = get(dict_params, :basis_type, :probabilist)
        Ψall = Psi_Hermite(Vc, max_deg; basis_type=basis_type)
        B_reduced, _ = construct_projection_operator_hermite(state_dim, max_deg;
                                                             basis_type=basis_type)
        dict_info = (type=:hermite, max_deg=max_deg, basis_type=basis_type)

    elseif dict_type == :rbf
        nRBF = get(dict_params, :nRBF, 200)
        include_states = get(dict_params, :include_states, true)
        state_indices = get(dict_params, :state_indices, nothing)
        kernel_type = Symbol(lowercase(String(get(dict_params, :kernel_type, :thinplate))))
        sigma = get(dict_params, :sigma, nothing)
        seed = get(dict_params, :seed, nothing)
        max_points = get(dict_params, :max_points, nothing)
        normalize = get(dict_params, :normalize, false)

        # Optional per-coordinate standardization, fit ONCE on the full
        # coordinate matrix so the ΨX/ΨY views (column subsets of Ψall) agree.
        norm_stats = nothing
        Vw = Vc
        if normalize
            include_states ||
                throw(ArgumentError("dict_params.normalize=true requires include_states=true: " *
                                    "the constant feature row carries the mean offset that maps " *
                                    "predictions back to original coordinates."))
            Vw, norm_stats = normalize_states(Vc)
        end

        if kernel_type == :gaussian && (isnothing(sigma) || sigma == :auto)
            sigma = median_heuristic_sigma(Vw; seed=seed)
            @info "RBF sigma auto-set to $sigma via median heuristic"
        end

        # accept pre-computed centres, else run k-means on the X columns only
        # (x_cols) — same convention as the old contiguous shift 1:n_snap-1.
        centers = get(dict_params, :centers, nothing)
        if isnothing(centers)
            centers = cluster_data(Vw[:, x_cols], nRBF;
                                   max_points=max_points, seed=seed)
        else
            @info "Using user-supplied RBF centres ($(size(centers,2)) points)"
        end
        Ψall = Psi_RBF(Vw, centers;
                       include_states=include_states, state_indices=state_indices,
                       kernel_type=kernel_type, sigma=sigma)

        # X side of the lifted data for the projection operator.
        # Contiguous x_cols (hankel_edmd's 1:n_snap-1) → strided view: zero
        # copy AND BLAS-compatible, exactly the old behaviour.  Arbitrary
        # index vectors (edmd_pairs' pd.x) are NOT strided, so materialise a
        # contiguous copy — otherwise the products inside
        # construct_projection_operator fall back to the slow generic matmul.
        ΨX = x_cols isa AbstractRange ? view(Ψall, :, x_cols) : Ψall[:, x_cols]
        B_reduced = construct_projection_operator(state_dim, ΨX, Vw[:, x_cols];
                                                  alpha=proj_alpha)
        if normalize
            # Push the affine correction into B so predictions come back in
            # ORIGINAL coordinates: x = s .* x̂ + mu, with mu absorbed by the
            # constant feature row (row 1 of Ψ). B_full = U_r * B_reduced in
            # hankel_edmd then inherits the correction automatically.
            B_reduced = diagm(norm_stats.std) * B_reduced
            B_reduced[:, 1] .+= norm_stats.mean
        end
        dict_info = (type=:rbf, centers=centers,
                     include_states=include_states, state_indices=state_indices,
                     kernel_type=kernel_type, sigma=sigma, norm_stats=norm_stats)

    elseif dict_type == :rff
        D = get(dict_params, :D, 500)
        sigma = get(dict_params, :sigma, 1.0)
        include_states = get(dict_params, :include_states, false)
        basis = build_rff_basis(state_dim, D, sigma;
                                seed=get(dict_params, :seed, nothing))
        Ψall = Psi_RFF(Vc, basis; include_states=include_states)

        # Same strided-view / contiguous-copy rule as the :rbf branch.
        # For the contiguous shift, X_side is literally the old X_r view
        # (zero copy); for an arbitrary index vector it is one BLAS-friendly
        # copy of the X-side coordinates.
        ΨX     = x_cols isa AbstractRange ? view(Ψall, :, x_cols) : Ψall[:, x_cols]
        X_side = x_cols isa AbstractRange ? view(Vc,    :, x_cols) : Vc[:,    x_cols]
        B_reduced = construct_projection_operator(state_dim, ΨX, X_side;
                                                  alpha=proj_alpha)
        dict_info = (type=:rff, basis=basis, include_states=include_states)

    else
        error("Unknown dict_type: $dict_type. Use :hermite, :rbf, or :rff.")
    end

    return Ψall, B_reduced, dict_info
end

# --- legacy signature (kept so pre-refactor calls keep working) ---------------
# The old API took the X-side DATA matrix as second argument
# (X_r = view(Vc, :, 1:n_snap-1), as built by hankel_edmd).  For that call
# pattern this forwarder is numerically identical to the old implementation —
# it simply routes to the index-based method with the contiguous shift, so
# hankel_edmd works unchanged even BEFORE you apply the (2d) call-site edit.
function _build_dict_and_project(Vc, X_r::AbstractMatrix, dict_type, dict_params,
                                 state_dim, proj_alpha)
    n_snap = size(Vc, 2)
    size(X_r) == (size(Vc, 1), n_snap - 1) ||
        error("legacy _build_dict_and_project(Vc, X_r, …) requires " *
              "X_r == Vc[:, 1:n_snap-1]; got size(X_r) = $(size(X_r)) for Vc " *
              "with $n_snap columns — pass the X column indices instead.")
    return _build_dict_and_project(Vc, 1:n_snap-1, dict_type, dict_params,
                                   state_dim, proj_alpha)
end

# =============================================================================
# MULTI-TRAJECTORY PAIR FITTING (X/Y paradigm)
# =============================================================================

"""
    delay_embed_pairs(X_train, Y_train, m_embed, tau_delay; n_trajectories=1)
        -> (S = S_all, x = x_idx, y = y_idx, cols = traj_cols,
            n_pairs = ..., m_traj = ...)

Multi-trajectory delay embedding, memory-shaped for the pair fitters
(`dmd_pairs`, `edmd_pairs`, `havok_pairs`, `Config.pair_analysis`).

`X_train`/`Y_train` are the raw state-space pairs produced by
`edmd_training_data` (n × (m_traj·n_trajectories), trajectory-major).  Each
trajectory is reconstructed from its consecutive pairs, delay-embedded with
`build_hankel_multichannel`, and the PER-TRAJECTORY Hankel matrices are
concatenated into ONE snapshot matrix

    S_all = [S_1 S_2 … S_P]        (n·m_embed) × ((n_pairs+1)·P)

so trajectory boundaries never contaminate a delay vector.  The training
PAIRS are INDEX VIEWS of S_all:

    X = view(S_all, :, x_idx),   Y = view(S_all, :, y_idx),
    Y[:, k] = one-snapshot successor of X[:, k] INSIDE the same trajectory.

Compared with `delay_embed_training_data` (which materialises X_delay and
Y_delay as two separate copies) this stores the delay data ONCE, and lets the
downstream fitters lift/project the shared parent a single time — the same
memory contract that `hankel_edmd` guarantees through its Ψall views.

`cols[j]` is the column range of trajectory j in S_all (needed by
`havok_pairs`); with `m_embed = 1, tau_delay = 1` the result is the plain
state-space concatenation, so the same machinery also serves the
no-embedding reference methods (DMD / EDMD on the raw shown state).
"""
function delay_embed_pairs(X_train::AbstractMatrix, Y_train::AbstractMatrix,
                           m_embed::Int, tau_delay::Int;
                           n_trajectories::Int=1)
    n, total_cols = size(X_train)
    size(Y_train) == (n, total_cols) ||
        error("delay_embed_pairs: X_train and Y_train must have the same shape")
    m_traj = div(total_cols, n_trajectories)
    m_traj * n_trajectories == total_cols ||
        error("delay_embed_pairs: total columns ($total_cols) not divisible " *
              "by n_trajectories ($n_trajectories)")

    max_lag = (m_embed - 1) * tau_delay
    T_full  = m_traj + 1                     # reconstructed trajectory length
    T_full > max_lag + 1 ||
        error("delay_embed_pairs: trajectory length $T_full too short for " *
              "m_embed=$m_embed, tau_delay=$tau_delay")

    n_delay = n * m_embed
    n_pairs = T_full - max_lag - 1           # pairs per trajectory

    S_all = zeros(n_delay, (n_pairs + 1) * n_trajectories)
    x_idx = Vector{Int}(undef, n_pairs * n_trajectories)
    y_idx = Vector{Int}(undef, n_pairs * n_trajectories)
    cols  = Vector{UnitRange{Int}}(undef, n_trajectories)

    X_traj = zeros(n, T_full)                # reusable scratch
    @inbounds for i in 1:n_trajectories
        src = (i - 1) * m_traj + 1
        # Reconstruct trajectory i from its consecutive pairs (same
        # convention as delay_embed_training_data): first column from X,
        # the remaining m_traj columns from Y.
        X_traj[:, 1]     .= @view X_train[:, src]
        X_traj[:, 2:end] .= @view Y_train[:, src:(src + m_traj - 1)]
        # Per-trajectory Hankel block — boundaries never enter a delay vector
        H = build_hankel_multichannel(X_traj, m_embed, tau_delay)
        dst = (i - 1) * (n_pairs + 1)
        cols[i] = (dst + 1):(dst + n_pairs + 1)
        S_all[:, cols[i]] .= H
        out = (i - 1) * n_pairs
        x_idx[out+1:out+n_pairs] .= (dst + 1):(dst + n_pairs)       # S_i[:, 1:end-1]
        y_idx[out+1:out+n_pairs] .= (dst + 2):(dst + n_pairs + 1)   # S_i[:, 2:end]
    end

    return (S = S_all, x = x_idx, y = y_idx, cols = cols,
            n_pairs = n_pairs, m_traj = m_traj)
end

"""Convenience views of a `delay_embed_pairs` dataset: `(X, Y) = pair_views(pd)`."""
pair_views(pd::NamedTuple) = view(pd.S, :, pd.x), view(pd.S, :, pd.y)

# ---------------------------------------------------------------------------

"""
    dmd_pairs(pd; r=nothing, use_svd=true)

Linear DMD on the multi-trajectory pair views of `pd` (see
`delay_embed_pairs`).  Mirrors `hankel_dmd` — same return contract
`(K, K_tilde, X_r, B_proj, U_r)` — but the snapshot pairs never cross a
trajectory boundary.

* `use_svd=false`: full operator `K = Y·pinv(X)`, `U_r = I` — the honest
  rank-n operator on the raw pairs (`use_svd` is HONOURED here, unlike
  `hankel_dmd`, which always SVDs).
* `use_svd=true, r=nothing`: POD basis from `G = S_all·S_all'` (BLAS syrk +
  `eigen`, never materialising the n_state × n_cols `Vt` factor of `svd`), all
  `n_state` modes kept, `K = U_r·(Y_r·pinv(X_r))·U_r'` — mathematically equal
  to `Y·pinv(X)` (exact-rotation invariant) but numerically better
  conditioned, and it keeps the POD modes `U_r` for diagnostics.
* `use_svd=true, r=R`: same with the leading `R` POD modes only
  (rank-truncated DMD, e.g. `r = r_havok`).

In the single-trajectory limit this reproduces `hankel_dmd(S; r)` exactly
(sign/rotation freedom of the POD basis cancels in `U_r·K_tilde·U_r'`).
"""
function dmd_pairs(pd::NamedTuple; r::Union{Nothing,Int}=nothing, use_svd::Bool=true)
    n_state = size(pd.S, 1)
    X = view(pd.S, :, pd.x)
    Y = view(pd.S, :, pd.y)

    if !use_svd
        K = Y * pinv(X)
        Id = Matrix{Float64}(I, n_state, n_state)
        return K, K, X, Id, Id
    end

    # POD basis of the data — same memory-safe route as havok_dmd
    G = pd.S * pd.S'
    E = eigen(Symmetric(G))
    p = sortperm(E.values; rev=true)
    σ = sqrt.(max.(E.values[p], 0.0))
    r_actual = isnothing(r) ? length(σ) : min(r, length(σ))

    U_r = E.vectors[:, p][:, 1:r_actual]
    X_r = U_r' * X                          # POD coordinates of the X side
    Y_r = U_r' * Y
    K_tilde = Y_r * pinv(X_r)
    K = U_r * K_tilde * U_r'
    return K, K_tilde, X_r, U_r', U_r
end

# ---------------------------------------------------------------------------

"""
    edmd_pairs(pd; r, dict_type, dict_params, edmd_method, edmd_alpha,
               proj_alpha, return_ψ, use_svd)

Pair version of `hankel_edmd` with the SAME memory contract: the dictionary
is evaluated ONCE on the shared coordinate matrix, and `ΨX`/`ΨY` are
returned as INDEX VIEWS of that single parent (`ΨX = Ψall[:, pd.x]`,
`ΨY = Ψall[:, pd.y]`), so downstream offload logic that deduplicates shared
storage keeps working.

Coordinates: `Vc = S_all` when `use_svd=false` (no copy); otherwise the POD
coordinates `Vc = U_r'·S_all` with `U_r` from `eigen(Symmetric(S_all·S_all'))`
(leading `r` modes; `r === nothing` keeps all).  RBF centres are clustered on
the X columns `Vc[:, pd.x]` only, and live in Vc coordinates — exactly the
convention of `hankel_edmd` (map back with `U_r` when `use_svd=true`).

Returns the `hankel_edmd` contract
`(K_edmd, B_reduced, B_full, U_r, X_r, dict_info, ΨX, ΨY)` with
`return_ψ=true`, otherwise without the last two.
"""
function edmd_pairs(pd::NamedTuple;
    r::Union{Nothing,Int}=nothing,
    dt::Real=1.0,
    dict_type::Symbol=:hermite,
    dict_params::NamedTuple=NamedTuple(),
    edmd_method::Symbol=:ridge,
    edmd_alpha::Real=1e-3,
    proj_alpha::Real=1e-6,
    return_ψ::Bool=false,
    use_svd::Bool=true
)
    dict_type = Symbol(lowercase(String(dict_type)))
    S_all = pd.S
    n_state = size(S_all, 1)

    # ── Reduced coordinates ──
    if use_svd
        G = S_all * S_all'
        E = eigen(Symmetric(G))
        p = sortperm(E.values; rev=true)
        σ = sqrt.(max.(E.values[p], 0.0))
        r_actual = isnothing(r) ? length(σ) : min(r, length(σ))
        U_r = E.vectors[:, p][:, 1:r_actual]
        Vc = U_r' * S_all                    # r × (n_pairs + P)   (copy)
    else
        r_actual = n_state
        U_r = Matrix{Float64}(I, n_state, n_state)
        Vc = S_all                           # no copy
    end
    X_r = view(Vc, :, pd.x)                  # X columns of the shared parent

    # ── Dictionary lift, computed ONCE on the shared parent (see docstring) ──
    Ψall, B_reduced, dict_info = _build_dict_and_project(
        Vc, pd.x, dict_type, dict_params, r_actual, proj_alpha
    )
    ΨX = view(Ψall, :, pd.x)
    ΨY = view(Ψall, :, pd.y)

    if dict_type == :hermite && r_actual > 100
        nPsi = size(Ψall, 1)
        @warn "Hermite dictionary with r=$r_actual produces nPsi=$nPsi observables. " *
              "Consider reducing `r` or `max_deg`."
    end

    K_edmd = compute_koopman_operator(ΨX, ΨY; method=edmd_method, alpha=edmd_alpha)
    B_full = use_svd ? U_r * B_reduced : B_reduced

    if return_ψ
        return K_edmd, B_reduced, B_full, U_r, X_r, dict_info, ΨX, ΨY
    else
        return K_edmd, B_reduced, B_full, U_r, X_r, dict_info
    end
end

# ---------------------------------------------------------------------------

"""
    havok_pairs(pd; r=nothing, dt=1.0, rank_method=:energy, energy=0.999,
                ridge_alpha=0.0)

Multi-trajectory HAVOK.  Identical pipeline to `havok_dmd` — POD of the
delay-embedded data, automatic rank selection, least-squares fit of the
continuous-time forced system `d/dt v = A v + B u` — except that:

* the POD basis and singular values come from the concatenated
  per-trajectory matrix `S_all` (`pd.S`);
* the POD coordinates `V_r = U_r'·S_all` are BLOCK-structured (one block per
  trajectory, `pd.cols`), and the 4th-order central-difference derivatives
  are taken INSIDE each block only — the stencil (t±1, t±2) never crosses a
  trajectory boundary.

Returns the `havok_dmd` contract `(K, A, B, U_r, V_r, u)`; `u` is the
concatenated forcing signal over all blocks.  `K` is the diagnostics-only
discrete flow map `exp([A B; 0 0]·dt)` — predictions should integrate `(A, B)`
with RK4, as in `havok_predict`.
"""
function havok_pairs(pd::NamedTuple; r::Union{Nothing,Int}=nothing, dt::Real=1.0,
                     rank_method::Symbol=:energy, energy::Real=0.999,
                     ridge_alpha::Real=0.0)
    S_all = pd.S
    n_state, n_cols = size(S_all)

    # ── 1. POD of the concatenated data (memory-safe: eigen(S·S'), no Vt) ──
    G = S_all * S_all'
    E = eigen(Symmetric(G))
    p = sortperm(E.values; rev=true)
    σ = sqrt.(max.(E.values[p], 0.0))
    U_all = E.vectors[:, p]

    # ── 2. Number of modes to keep (same machinery as havok_dmd) ──
    r_actual = if isnothing(r)
        select_svd_rank(σ; method=rank_method, energy=energy,
                        aspect=n_state / n_cols)
    else
        min(r, length(σ))
    end
    r_actual = min(r_actual, n_cols)
    r_actual < 2 && error("HAVOK requires r ≥ 2 (need at least one linear " *
                          "coordinate + forcing).")
    @info "HAVOK (pairs): keeping r = $r_actual POD modes (selection = $rank_method)"

    U_r = U_all[:, 1:r_actual]
    V_r = U_r' * S_all                       # r × n_cols, block-structured

    # ── 3. Regression for dv/dt = A v + B v_input, per trajectory block ──
    #   dv/dt|_t ≈ (−v[t+2] + 8v[t+1] − 8v[t−1] + v[t−2]) / (12 dt)
    # evaluated only at interior columns t of each block (t±1, t±2 in-block).
    r_lin  = r_actual - 1
    blocks = pd.cols
    n_int  = sum(max(length(b) - 4, 0) for b in blocks)
    n_int > 0 || error("havok_pairs: trajectories too short for the 5-point " *
                       "stencil (need ≥ 5 columns per block)")

    Z  = Matrix{Float64}(undef, r_actual, n_int)   # [v_{1:r-1}; u] at t
    Vd = Matrix{Float64}(undef, r_lin, n_int)      # dv/dt at t
    out = 0
    @inbounds for b in blocks
        t0, t1 = first(b), last(b)
        for t in (t0 + 2):(t1 - 2)
            out += 1
            for j in 1:r_lin
                Vd[j, out] = (-V_r[j, t+2] + 8V_r[j, t+1] - 8V_r[j, t-1] + V_r[j, t-2]) / (12dt)
                Z[j, out]  = V_r[j, t]
            end
            Z[r_actual, out] = V_r[r_actual, t]
        end
    end

    AB = if ridge_alpha > 0
        (Vd * Z') / (Z * Z' + float(ridge_alpha) * I(r_actual))
    else
        Vd * pinv(Z)                        # least squares, as in the paper
    end
    A = AB[:, 1:r_lin]
    B = AB[:, r_actual:r_actual]

    # Discrete flow map of the augmented autonomous system — diagnostics only
    𝒜 = zeros(r_actual, r_actual)
    𝒜[1:r_lin, :] .= AB
    K = Matrix(exp(𝒜 * dt))

    return K, A, B, U_r, V_r, vec(V_r[r_actual, :])
end

end # module
