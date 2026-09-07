module Hankel

using LinearAlgebra
using Random
using Statistics   # ← NEW: needed for median() in select_svd_rank
using ..EDMD: compute_koopman_operator, construct_projection_operator, edmd_predict_from_psi,
                        kernel_edmd_rbf, median_heuristic_sigma
using ..Dictionaries: Psi_Hermite, Psi_RBF, Psi_RFF, cluster_data, build_rff_basis,
                        construct_projection_operator_hermite, lift_state

export build_hankel, hankel_dmd, hankel_edmd, hankel_kernel_edmd, havok_dmd, havok_predict,
    delay_space_edmd_prediction, select_svd_rank,   # ← NEW: select_svd_rank
    build_hankel_multichannel, delay_embed_training_data

# =============================================================================
# build_hankel and hankel_dmd
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
        Vc, X_r, dict_type, dict_params, r_actual, proj_alpha
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
    N_subsample::Union{Nothing,Int}=nothing
)
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
    select_svd_rank(σ; method=:energy, energy=0.999, aspect=1.0)

Choose how many leading singular values / POD modes to keep, given the
singular values `σ` in descending order.

# Methods
- `:energy` — smallest r whose modes capture the fraction `energy`
  (e.g. 0.999) of the total energy Σσᵢ².
- `:hard_threshold` — Gavish–Donoho optimal singular-value hard threshold
  with unknown noise level: keep σᵢ > ω(β)·median(σ), with β = `aspect`
  (min(m,n)/max(m,n)) and ω(β) ≈ 0.56β³ − 0.95β² + 1.82β + 1.43.
- `:gap` — keep r at the largest relative drop (elbow) of the spectrum,
  i.e. the argmax of diff(log σ).
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
    delay_embed_training_data(X_train, Y_train, m_embed, tau_delay; n_trajectories)

Build delay-embedded EDMD training pairs from the output of `edmd_training_data`.
Assumes trajectories are concatenated contiguously with equal length.
Returns `(X_delay, Y_delay)` with shape `(n·m_embed, N_pairs)`.
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
# derives ΨX = Ψall[:, 1:end-1] and ΨY = Ψall[:, 2:end] as views.
# Vc: full coordinate matrix (r × n_snap); X_r: its first n_snap-1 columns
# (a view), used for centre clustering and the projection operator.
function _build_dict_and_project(Vc, X_r, dict_type, dict_params, state_dim, proj_alpha)
    n_snap = size(Vc, 2)

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
        sigma = get(dict_params, :sigma, nothing)
        if isnothing(sigma) || sigma == :auto
            sigma = median_heuristic_sigma(X_r)
            @info "RBF sigma auto-set to $sigma via median heuristic"
        end
        # accept pre-computed centres, else run k-means
        centers = get(dict_params, :centers, nothing)
        if isnothing(centers)
            centers = cluster_data(X_r, nRBF)
        else
            @info "Using user-supplied RBF centres ($(size(centers,2)) points)"
        end
        Ψall = Psi_RBF(Vc, centers;
                       include_states=include_states, state_indices=state_indices)
        ΨX = view(Ψall, :, 1:n_snap-1)
        B_reduced = construct_projection_operator(state_dim, ΨX, X_r; alpha=proj_alpha)
        dict_info = (type=:rbf, centers=centers,
                     include_states=include_states, state_indices=state_indices)

    elseif dict_type == :rff
        D = get(dict_params, :D, 500)
        sigma = get(dict_params, :sigma, 1.0)
        include_states = get(dict_params, :include_states, false)
        basis = build_rff_basis(state_dim, D, sigma)
        Ψall = Psi_RFF(Vc, basis; include_states=include_states)
        ΨX = view(Ψall, :, 1:n_snap-1)
        B_reduced = construct_projection_operator(state_dim, ΨX, X_r; alpha=proj_alpha)
        dict_info = (type=:rff, basis=basis, include_states=include_states)

    else
        error("Unknown dict_type: $dict_type. Use :hermite, :rbf, or :rff.")
    end

    return Ψall, B_reduced, dict_info
end

end # module