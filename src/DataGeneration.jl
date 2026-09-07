module DataGeneration

using LinearAlgebra

using ..Regimes: regime_config
using ..Systems: fixed_point, rk4, 
                 fhn_rhs, duffing_rhs, epileptor3d_rhs,
                 lorenz_rhs, vanderpol_rhs, rossler_rhs
using ..Hankel: build_hankel

export edmd_training_data, hankel_training_data, state_space_predictions,
        state_space_predict_from_psi, delay_space_predictions

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
                            save_dir::Union{Nothing,String}=nothing)
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
                              save_dir::Union{Nothing,String}=nothing)
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

State-space analogue of `edmd_predict_from_psi`.

# Arguments
- `X_test`: n_state × n_snap_test matrix of pre-computed test trajectories
- `K`: nPsi × nPsi Koopman matrix
- `Psi_func`: lifting function `Psi_func(x::Matrix) -> Ψ` (must accept n_state × 1)
- `B_proj`: n_state × nPsi projection matrix
- `start_indices`: vector of column indices in `X_test` to start predictions from
- `n_pred`: number of steps to predict

# Returns
`(X_true, X_pred)` with shape `(n_state, n_pred+1, n_traj)`
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

end # module