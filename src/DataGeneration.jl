module DataGeneration

using LinearAlgebra
using Random

using ..Regimes: regime_config
using ..Systems: fixed_point, rk4, 
                 fhn_rhs, duffing_rhs, epileptor3d_rhs,
                 lorenz_rhs, vanderpol_rhs, rossler_rhs
using ..Hankel: build_hankel, build_hankel_multichannel

export edmd_training_data, hankel_training_data, state_space_predictions,
        state_space_predict_from_psi, delay_space_predictions,
        sample_seeds_window_integrated

# ---------------------------------------------------------------------------
# EDMD training data
# ---------------------------------------------------------------------------

function edmd_training_data(system::String, regime::String;
                            m_train::Int=1000,
                            n_trajectories::Int=10,
                            window::Real=1.0,
                            dt::Real=0.01,
                            nLag::Int=1,
                            center::Union{Nothing,AbstractVector}=nothing,
                            save_dir::Union{Nothing,String}=nothing,
                            seed::Union{Nothing,Int}=nothing)
    isnothing(seed) || Random.seed!(seed)
    cfg = regime_config(system, regime; save_dir=save_dir)
    params = cfg.params
    rhs = if system == "FHN"
        fhn_rhs(params)
    elseif system == "Duffing"
        duffing_rhs(params)
    elseif system == "Epileptor3D"
        epileptor3d_rhs(params)
    elseif system == "Lorenz"
        lorenz_rhs(params)
    elseif system == "VanderPol"
        vanderpol_rhs(params)
    elseif system == "Rossler"
        rossler_rhs(params)
    else
        error("Unknown system: $system")
    end

    X_fixed = fixed_point(system, params)
    c = isnothing(center) ? X_fixed : vec(center)
    n = length(c)

    X_train = zeros(n, m_train * n_trajectories)
    Y_train = zeros(n, m_train * n_trajectories)

    for i in 1:n_trajectories
        x0 = c .+ window .* (2 .* rand(n) .- 1)
        X = rk4(rhs, x0, dt, m_train; nLag=nLag)
        idx = ((i - 1) * m_train + 1):(i * m_train)
        X_train[:, idx] .= X[:, 1:end-1]
        Y_train[:, idx] .= X[:, 2:end]
    end

    return X_train, Y_train, cfg, rhs, X_fixed
end

# ---------------------------------------------------------------------------
# Hankel-EDMD training data
# ---------------------------------------------------------------------------

"""
    hankel_training_data(system, regime; ...)

Build training data for Hankel-EDMD.  Returns `(v_train, S, cfg, rhs)` where
`S` is the full Hankel matrix ready for `hankel_dmd` or `hankel_edmd`.

`seed` optionally fixes the random generator before the initial condition is
drawn, making the training series reproducible across runs.
"""
function hankel_training_data(system::String, regime::String;
                              m_embed::Int=10,
                              tau_delay::Int=2,
                              m_train::Int=2000,
                              window::Real=1.0,
                              dt::Real=0.01,
                              nLag::Int=1,
                              observed_state::Int=1,
                              center::Union{Nothing,AbstractVector}=nothing,
                              save_dir::Union{Nothing,String}=nothing,
                              seed::Union{Nothing,Int}=nothing)
    isnothing(seed) || Random.seed!(seed)
    cfg = regime_config(system, regime; save_dir=save_dir)
    params = cfg.params
    rhs = if system == "FHN"
        fhn_rhs(params)
    elseif system == "Duffing"
        duffing_rhs(params)
    elseif system == "Epileptor3D"
        epileptor3d_rhs(params)
    elseif system == "Lorenz"
        lorenz_rhs(params)
    elseif system == "VanderPol"
        vanderpol_rhs(params)
    elseif system == "Rossler"
        rossler_rhs(params)
    else
        error("Unknown system: $system")
    end

    X_fixed = fixed_point(system, params)
    c = isnothing(center) ? X_fixed : vec(center)
    n = length(c)

    x0 = c .+ window .* (2 .* rand(n) .- 1)
    X_long = rk4(rhs, x0, dt, m_train; nLag=nLag)
    v_train = X_long[observed_state, :]

    S = build_hankel(v_train, m_embed, tau_delay)

    # Return S directly so the caller can pass it to hankel_edmd / hankel_dmd
    return v_train, S, cfg, rhs
end

# ---------------------------------------------------------------------------
# Koopman prediction generators
# ---------------------------------------------------------------------------

function state_space_predictions(rhs, K::AbstractMatrix, x0s::AbstractMatrix,
                                  dt::Real, n_pred::Int;
                                  Psi_func::Function, B_proj::AbstractMatrix,
                                  nLag::Int=1)
    n_state, n_traj = size(x0s)
    n_obs = size(B_proj, 1)

    X_true = zeros(n_obs, n_pred + 1, n_traj)
    X_pred = zeros(n_obs, n_pred + 1, n_traj)

    for j in 1:n_traj
        x0 = x0s[:, j]

        # True trajectory via RK4
        Xt = rk4(rhs, x0, dt, n_pred; nLag=nLag)
        X_true[:, :, j] = Xt[1:n_obs, :]

        # Predicted trajectory via Koopman
        x_curr = x0[1:n_obs]
        X_pred[:, 1, j] = x_curr
        for k in 1:n_pred
            # Lift and evolve
            Ψ_curr = Psi_func(reshape(x_curr, n_obs, 1))[:, 1]
            Ψ_next = K' * Ψ_curr
            x_next = B_proj * Ψ_next
            X_pred[:, k+1, j] = x_next
            x_curr = x_next
        end
    end

    return X_true, X_pred
end

"""
    state_space_predict_from_psi(X_test, K, Psi_func, B_proj, start_indices, n_pred)

Predict state-space trajectories using pre-computed observable matrices.
"""
function state_space_predict_from_psi(X_test::AbstractMatrix, K::AbstractMatrix,
                                      Psi_func::Function, B_proj::AbstractMatrix,
                                      start_indices::AbstractVector{Int}, n_pred::Int)
    n_state, n_snap = size(X_test)
    n_traj = length(start_indices)

    X_true = zeros(n_state, n_pred + 1, n_traj)
    X_pred = zeros(n_state, n_pred + 1, n_traj)

    for j in 1:n_traj
        idx = start_indices[j]
        @assert idx + n_pred <= n_snap "Index $idx + $n_pred exceeds $n_snap snapshots"

        # Ground truth from pre-computed test data
        X_true[:, :, j] = X_test[:, idx:idx+n_pred]

        # Lift initial condition ONCE
        ψ = Psi_func(reshape(X_test[:, idx], n_state, 1))[:, 1]
        X_pred[:, 1, j] = B_proj * ψ

        # Iterate purely in ψ-space (no re-lifting)
        for k in 1:n_pred
            ψ = K' * ψ
            X_pred[:, k+1, j] = B_proj * ψ
        end
    end

    return X_true, X_pred
end

function delay_space_predictions(S::AbstractMatrix, K::AbstractMatrix,
                                  start_indices::AbstractVector{Int}, n_pred::Int)
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
            s_next = K * s_curr
            X_pred[1, k+1, j] = s_next[1]
            s_curr = s_next
        end
    end

    return X_true, X_pred
end

"""
    sample_seeds_window_integrated(f_dyn, x0_center, window_ic, dim, showdim,
                                   dt_internal, nLag, m_embed, tau_delay;
                                   n_starts, direction, rng,
                                   max_consecutive_failures)

Sample random initial conditions in the physical `dim`-D window

    x0[v] ~ Uniform(x0_center[v] - window_ic, x0_center[v] + window_ic),
    v in showdim   (non-shown components are pinned at x0_center),

then fill the `(m_embed - 1) * tau_delay` older delay coordinates by
integrating the true dynamics.  Two directions:

  `direction = :forward`  — integrate `f_dyn` forward from `x0`; the drawn
                            uniform point becomes the OLDEST delay coordinate
                            and the anchor ("current") is the evolved point.
  `direction = :backward` — integrate the time-reversed drift `-f_dyn`; the
                            drawn uniform point IS the anchor and the delays
                            are its true past.

Unlike iid-delay sampling, every returned point lies on the system's
trajectory manifold.  NaN / Inf trajectories are rejected and resampled
(with a `max_consecutive_failures` guard to avoid infinite loops when the
window is too large).

Returns an `n_state × n_starts` matrix where `n_state = length(showdim) * m_embed`.
"""
function sample_seeds_window_integrated(f_dyn, x0_center::AbstractVector,
                                        window_ic::Real, dim::Int,
                                        showdim::AbstractVector{<:Integer},
                                        dt_internal::Real, nLag::Int,
                                        m_embed::Int, tau_delay::Int;
                                        n_starts::Int,
                                        direction::Symbol=:backward,
                                        rng::AbstractRNG=Random.default_rng(),
                                        max_consecutive_failures::Int=10)

    @assert direction in (:forward, :backward) "direction must be :forward or :backward"
    @assert max_consecutive_failures >= 1

    n_show  = length(showdim)
    n_state = n_show * m_embed
    n_fill  = (m_embed - 1) * tau_delay

    # Backward = forward integration of time-reversed drift
    f_int = direction == :forward ? f_dyn : (x -> -f_dyn(x))

    S0 = Matrix{Float64}(undef, n_state, n_starts)
    consecutive_failures = 0
    total_failures = 0

    for i in 1:n_starts
        while true
            # Draw full physical IC; non-shown coords pinned at x0_center
            x0 = Vector{Float64}(x0_center[1:dim])
            for v in 1:dim
                x0[v] = x0_center[v] + window_ic * (2 * rand(rng) - 1)
            end

            candidate = try
                X_seg = rk4(f_int, x0, dt_internal, n_fill; nLag=nLag)

                if direction == :backward
                    X_seg = reverse(X_seg; dims=2)
                    X_seg[:, end] = x0   # preserve drawn point as anchor
                end

                if !all(isfinite, X_seg)
                    nothing
                else
                    S_seg = build_hankel_multichannel(
                        X_seg[showdim, :], m_embed, tau_delay)
                    s = S_seg[:, end]
                    all(isfinite, s) ? s : nothing
                end
            catch err
                err isa InterruptException && rethrow()
                nothing
            end

            if !isnothing(candidate)
                S0[:, i] = candidate
                consecutive_failures = 0
                break
            end

            total_failures += 1
            consecutive_failures += 1

            if consecutive_failures >= max_consecutive_failures
                error("Delay filling produced $consecutive_failures consecutive " *
                      "non-finite trajectories with window_ic=$window_ic and " *
                      "direction=$direction. Use a tighter physical window.")
            end
        end
    end

    if total_failures > 0
        println("  Integrated-window sampling rejected $total_failures non-finite trajectories.")
    end

    return S0
end

end # module
