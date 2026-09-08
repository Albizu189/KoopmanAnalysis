module Config

using LinearAlgebra
using Statistics
using KrylovKit: eigsolve

using ..Hankel: build_hankel, hankel_dmd, hankel_edmd, hankel_kernel_edmd, delay_space_edmd_prediction,
            havok_dmd, havok_predict, select_svd_rank
using ..EDMD: compute_koopman_operator, construct_projection_operator,
            kernel_edmd_rbf, rbf_kernel, edmd_predict_from_psi
using ..Dictionaries: Psi_Hermite, Psi_RBF, Psi_RFF, cluster_data, build_rff_basis, construct_projection_operator_hermite
using ..Spectral: koopman_eigendecomposition, find_all_harmonic_branches
using ..DataGeneration: delay_space_predictions

export KoopmanConfig, AnalysisResult, validate!,
       hankel_analysis, state_analysis, predict, spectrum, embed, all_harmonic_branches,
       select_svd_rank

# ---------------------------------------------------------------------------
# 1. Configuration struct
# ---------------------------------------------------------------------------

Base.@kwdef struct KoopmanConfig
    m_embed::Int = 5000
    tau_delay::Int = 10
    r::Union{Nothing,Int} = 100
    use_svd::Bool = true
    dict_type::Symbol = :RBF
    dict_params::NamedTuple = (nRBF=500, include_states=true)
    edmd_method::Symbol = :ridge
    edmd_alpha::Float64 = 1e-2
    proj_alpha::Float64 = 1e-6
    dt::Float64 = 0.1
    return_ψ::Bool = true
    verbose::Bool = true
    parallel::Bool = true
    seed::Union{Nothing,Int} = nothing
end

function Base.show(io::IO, cfg::KoopmanConfig)
    print(io, "KoopmanConfig(")
    print(io, "m=$(cfg.m_embed), τ=$(cfg.tau_delay), ")
    print(io, "r=$(cfg.r), dict=$(cfg.dict_type), ")
    print(io, "method=$(cfg.edmd_method), α=$(cfg.edmd_alpha))")
end

function validate!(cfg::KoopmanConfig)
    cfg.m_embed > 0 || throw(ArgumentError("m_embed must be > 0, got $(cfg.m_embed)"))
    cfg.tau_delay > 0 || throw(ArgumentError("tau_delay must be > 0, got $(cfg.tau_delay)"))
    cfg.dt > 0 || throw(ArgumentError("dt must be > 0, got $(cfg.dt)"))
    cfg.edmd_alpha >= 0 || throw(ArgumentError("edmd_alpha must be >= 0, got $(cfg.edmd_alpha)"))
    cfg.proj_alpha >= 0 || throw(ArgumentError("proj_alpha must be >= 0, got $(cfg.proj_alpha)"))
    (cfg.r === nothing || cfg.r > 0) || throw(ArgumentError("r must be > 0 or nothing, got $(cfg.r)"))
    Symbol(lowercase(String(cfg.dict_type))) in (:hermite, :rbf, :rff, :none, :kernel, :havok) ||
        throw(ArgumentError("dict_type must be one of :hermite, :rbf, :rff, :none, :kernel, :havok, got $(cfg.dict_type)"))
    cfg.edmd_method in (:pinv, :ridge) ||
        throw(ArgumentError("edmd_method must be :pinv or :ridge, got $(cfg.edmd_method)"))
    return cfg
end

# ---------------------------------------------------------------------------
# 2. Result container
# ---------------------------------------------------------------------------

# NOTE: X_r, ΨX and ΨY are typed as AbstractMatrix because hankel_edmd now
# returns VIEWS for them (X_r/Y_r are consecutive column ranges of one
# coordinate matrix; ΨY is ΨX shifted by one column — see Hankel.jl).
# Everything downstream (matmul, indexing, KrylovKit, JLD2 via the parent
# array) works identically on views.
mutable struct AnalysisResult
    K::Matrix{<:Number}
    B_full::Union{Matrix{Float64},Nothing}
    B_reduced::Union{Matrix{Float64},Nothing}
    U_r::Union{Matrix{Float64},Nothing}
    X_r::Union{AbstractMatrix{Float64},Nothing}
    dict_info::Union{NamedTuple,Nothing}
    ΨX::Union{AbstractMatrix{Float64},Nothing}
    ΨY::Union{AbstractMatrix{Float64},Nothing}
    S::Union{Matrix{Float64},Nothing}
    λ::Union{Vector{ComplexF64},Nothing}
    Ξ::Union{Matrix{ComplexF64},Nothing}
    cfg::KoopmanConfig
end

function AnalysisResult(K, B_full, B_reduced, U_r, X_r, dict_info, ΨX, ΨY, S, cfg)
    AnalysisResult(K, B_full, B_reduced, U_r, X_r, dict_info, ΨX, ΨY, S, nothing, nothing, cfg)
end

function Base.iterate(res::AnalysisResult, state=1)
    fields = (res.K, res.B_full, res.B_reduced, res.U_r, res.X_r,
              res.dict_info, res.ΨX, res.ΨY, res.S)
    state > length(fields) && return nothing
    return (fields[state], state + 1)
end
Base.length(::AnalysisResult) = 9
Base.eltype(::Type{AnalysisResult}) = Any

# ---------------------------------------------------------------------------
# 3. High-level analysis functions
# ---------------------------------------------------------------------------

function embed(v::AbstractVector, cfg::KoopmanConfig)
    return build_hankel(v, cfg.m_embed, cfg.tau_delay)
end

function hankel_analysis(v::AbstractVector, cfg::KoopmanConfig)
    cfg.verbose && @info "Building Hankel matrix (m=$(cfg.m_embed), τ=$(cfg.tau_delay))..."
    S = build_hankel(v, cfg.m_embed, cfg.tau_delay)
    return hankel_analysis(S, cfg)
end

function hankel_analysis(S::AbstractMatrix, cfg::KoopmanConfig)
    if cfg.dict_type == :none
        # ── M1: Linear Hankel-DMD ──
        cfg.verbose && @info "Running linear Hankel-DMD (r=$(cfg.r))..."
        K, K_tilde, X_r, B_proj, U_r = hankel_dmd(S; r=cfg.r, dt=cfg.dt)
        return AnalysisResult(
            K, U_r, Matrix(B_proj),
            Matrix{Float64}(I(size(K,1))), X_r, (type=:linear,),
            nothing, nothing, S, cfg
        )

    elseif cfg.dict_type == :kernel
        # ── M6: Hankel-Kernel-EDMD ──
        # Delegates SVD + kernel regression to Hankel.hankel_kernel_edmd
        cfg.verbose && @info "Running Hankel-Kernel-EDMD (sigma=$(cfg.dict_params.sigma), r=$(cfg.r))..."

        sigma_k      = float(get(cfg.dict_params, :sigma, 1.0))
        alpha_kernel = float(get(cfg.dict_params, :alpha, 1e-6))
        N_subsample  = get(cfg.dict_params, :N_subsample, nothing)

        K, B, X, Y, U_r = hankel_kernel_edmd(S;
            r=cfg.r, dt=cfg.dt,
            sigma=sigma_k, alpha=alpha_kernel,
            use_svd=cfg.use_svd,
            N_subsample=N_subsample
        )

        # Recompute Gram matrices for storage in ΨX/ΨY slots (optional, for diagnostics)
        K_ZZ, K_ZY = kernel_edmd_rbf(X, Y, sigma_k; alpha=alpha_kernel)[3:4]
        dict_info = (type=:kernel, X_dict=X, sigma=sigma_k, alpha=alpha_kernel)

        return AnalysisResult(K, B, B, U_r, X, dict_info, K_ZZ, K_ZY, S, cfg)

    elseif cfg.dict_type == :havok
        # ── M8: HAVOK — continuous-time linear model with intermittent ──
        # forcing (Brunton et al. 2017, arXiv:1608.05306).
        # Pipeline: (1) SVD of the Hankel matrix → (2) automatic choice of
        # the number r of POD modes from the singular value spectrum →
        # (3) least-squares regression for dv/dt = A v + B v_input →
        # (4) prediction by RK4 integration of that linear system
        # (see _havok_predict below).
        cfg.verbose && @info "Running HAVOK analysis (r=$(cfg.r))..."

        rank_method = get(cfg.dict_params, :rank_method, :energy)
        energy      = get(cfg.dict_params, :energy, 0.999)
        ridge_alpha = get(cfg.dict_params, :ridge_alpha, 0.0)

        K, A, B, U_r, V_r, u = havok_dmd(S;
            r=cfg.r, dt=cfg.dt,
            rank_method=rank_method, energy=energy,
            ridge_alpha=ridge_alpha
        )
        dict_info = (type=:havok, A=A, B=B, forcing=u, r_lin=size(A, 1))
        return AnalysisResult(
            K, U_r, nothing,
            U_r, V_r, dict_info,
            nothing, nothing, S, cfg
        )

    else
        # ── M2–M5, M7: Explicit dictionary EDMD ──
        cfg.verbose && @info "Running Hankel-EDMD (dict=$(cfg.dict_type), r=$(cfg.r))..."
        if cfg.return_ψ
            K_edmd, B_reduced, B_full, U_r, X_r, dict_info, ΨX, ΨY = hankel_edmd(S;
                r=cfg.r, dt=cfg.dt,
                dict_type=cfg.dict_type, dict_params=cfg.dict_params,
                edmd_method=cfg.edmd_method, edmd_alpha=cfg.edmd_alpha,
                proj_alpha=cfg.proj_alpha, return_ψ=true, use_svd=cfg.use_svd
            )
            return AnalysisResult(K_edmd, B_full, B_reduced, U_r, X_r,
                                  dict_info, ΨX, ΨY, S, cfg)
        else
            K_edmd, B_reduced, B_full, U_r, X_r, dict_info = hankel_edmd(S;
                r=cfg.r, dt=cfg.dt,
                dict_type=cfg.dict_type, dict_params=cfg.dict_params,
                edmd_method=cfg.edmd_method, edmd_alpha=cfg.edmd_alpha,
                proj_alpha=cfg.proj_alpha, return_ψ=false, use_svd=cfg.use_svd
            )
            return AnalysisResult(K_edmd, B_full, B_reduced, U_r, X_r,
                                  dict_info, nothing, nothing, S, cfg)
        end
    end
end

function state_analysis(X::AbstractMatrix, Y::AbstractMatrix, cfg::KoopmanConfig)
    cfg.verbose && @info "Running state-space EDMD (dict=$(cfg.dict_type))..."
    n, m = size(X)
    dict_type = Symbol(lowercase(String(cfg.dict_type)))

    if dict_type == :hermite
        max_deg = get(cfg.dict_params, :max_deg, 2)
        basis_type = get(cfg.dict_params, :basis_type, :probabilist)
        ΨX = Psi_Hermite(X, max_deg; basis_type=basis_type)
        ΨY = Psi_Hermite(Y, max_deg; basis_type=basis_type)
        B_reduced, _ = construct_projection_operator_hermite(n, max_deg; basis_type=basis_type)
        dict_info = (type=:hermite, max_deg=max_deg, basis_type=basis_type)

    elseif dict_type == :rbf
        nRBF = get(cfg.dict_params, :nRBF, 200)
        include_states = get(cfg.dict_params, :include_states, true)
        state_indices = get(cfg.dict_params, :state_indices, nothing)
        centers = cluster_data(X, nRBF)
        ΨX = Psi_RBF(X, centers; include_states=include_states, state_indices=state_indices)
        ΨY = Psi_RBF(Y, centers; include_states=include_states, state_indices=state_indices)
        B_reduced = construct_projection_operator(n, ΨX, X; alpha=cfg.proj_alpha)
        dict_info = (type=:rbf, centers=centers, include_states=include_states, state_indices=state_indices)

    elseif dict_type == :rff
        D = get(cfg.dict_params, :D, 500)
        sigma = get(cfg.dict_params, :sigma, 1.0)
        include_states = get(cfg.dict_params, :include_states, false)
        basis = build_rff_basis(n, D, sigma)
        ΨX = Psi_RFF(X, basis; include_states=include_states)
        ΨY = Psi_RFF(Y, basis; include_states=include_states)
        B_reduced = construct_projection_operator(n, ΨX, X; alpha=cfg.proj_alpha)
        dict_info = (type=:rff, basis=basis, include_states=include_states)

    else
        error("Unknown dict_type: $dict_type. Use :hermite, :rbf, :rff, or :none.")
    end

    K_edmd = compute_koopman_operator(ΨX, ΨY; method=cfg.edmd_method, alpha=cfg.edmd_alpha)

    if cfg.return_ψ
        return AnalysisResult(K_edmd, B_reduced, B_reduced, nothing, nothing,
                              dict_info, ΨX, ΨY, nothing, cfg)
    else
        return AnalysisResult(K_edmd, B_reduced, B_reduced, nothing, nothing,
                              dict_info, nothing, nothing, nothing, cfg)
    end
end

# ---------------------------------------------------------------------------
# 4. Post-processing
# ---------------------------------------------------------------------------

function predict(res::AnalysisResult, start_indices::AbstractVector{Int}, n_pred::Int)
    if !isnothing(res.dict_info) && get(res.dict_info, :type, :none) == :kernel
        return _kernel_delay_predict(res, start_indices, n_pred)
    elseif !isnothing(res.dict_info) && get(res.dict_info, :type, :none) == :havok
        return _havok_predict(res, start_indices, n_pred)
    elseif isnothing(res.ΨX)
        return delay_space_predictions(res.S, res.K, start_indices, n_pred)
    else
        return edmd_predict_from_psi(res.ΨX, res.K, res.B_full, res.S, start_indices, n_pred)
    end
end

function _kernel_delay_predict(res::AnalysisResult, start_indices::AbstractVector{Int}, n_pred::Int)
    m_embed, n_snap = size(res.S)
    n_traj = length(start_indices)
    X_dict = res.dict_info.X_dict
    sigma  = res.dict_info.sigma
    U_r    = res.U_r

    X_true = zeros(1, n_pred + 1, n_traj)
    X_pred = zeros(1, n_pred + 1, n_traj)

    for j in 1:n_traj
        idx = start_indices[j]
        @assert idx + n_pred <= n_snap "Index $idx + $n_pred exceeds $n_snap snapshots"

        X_true[1, :, j] = res.S[1, idx:idx+n_pred]

        s_curr = res.S[:, idx]
        X_pred[1, 1, j] = s_curr[1]

        for k in 1:n_pred
            x_r = isnothing(U_r) ? s_curr : U_r' * s_curr
            N = size(X_dict, 2)
            k_vec = Float64[rbf_kernel(x_r, X_dict[:, i], sigma) for i in 1:N]
            ψ_next = res.K' * k_vec
            x_next = res.B_full * ψ_next
            s_next = isnothing(U_r) ? x_next : U_r * x_next

            X_pred[1, k+1, j] = s_next[1]
            s_curr = s_next
        end
    end
    return X_true, X_pred
end

# HAVOK prediction: RK4 integration of dv/dt = A v + B v_input
# (step 4 of the HAVOK pipeline; see Hankel.havok_predict).
function _havok_predict(res::AnalysisResult, start_indices::AbstractVector{Int}, n_pred::Int;
                        use_true_forcing::Bool=true)
    return havok_predict(res.dict_info.A, res.dict_info.B, res.U_r, res.S,
                         start_indices, n_pred;
                         dt=res.cfg.dt, use_true_forcing=use_true_forcing)
end

function spectrum(res::AnalysisResult; howmany::Int=200, krylovdim::Int=300, tol::Real=1e-8)
    if !isnothing(res.λ) && !isnothing(res.Ξ)
        return res.λ, res.Ξ
    end

    n = size(res.K, 1)

    # ── Small matrix: dense is faster and more reliable ──
    if n < 10000
        λ, Ξ = koopman_eigendecomposition(res.K)
        res.λ = λ
        res.Ξ = Ξ
        return λ, Ξ
    end

    # ── Large matrix: iterative Arnoldi via KrylovKit ──
    @info "Matrix size $n×$n > 2000; using iterative eigsolve (KrylovKit)..."

    # howmany cannot exceed matrix dimension
    howmany = min(howmany, n)
    krylovdim = min(krylovdim, n)

    # Target largest magnitude (Koopman eigenvalues near unit circle)
    vals, vecs, info = eigsolve(res.K, howmany, :LM;
                                krylovdim=krylovdim,
                                tol=tol,
                                maxiter=1000,
                                verbosity=0)

    if info.converged < howmany
        @warn "Only $(info.converged)/$howmany eigenvalues converged. " *
              "Increase krylovdim (currently $krylovdim) or tol."
    end

    # KrylovKit returns vecs as Vector{Vector}; pack into a matrix
    n_conv = length(vals)
    Ξ = similar(res.K, eltype(vals[1]), n, n_conv)
    for j in 1:n_conv
        Ξ[:, j] .= vecs[j]
    end
    λ = convert(Vector{ComplexF64}, vals)

    # Sort by |λ| descending (same convention as koopman_eigendecomposition)
    idx = reverse(sortperm(λ; by=abs))
    λ = λ[idx]
    Ξ = Ξ[:, idx]

    res.λ = λ
    res.Ξ = Ξ
    return λ, Ξ
end

# ---------------------------------------------------------------------------
# Convenience: spectral analysis directly from AnalysisResult
# ---------------------------------------------------------------------------

function all_harmonic_branches(res::AnalysisResult; kwargs...)
    λ, _ = spectrum(res; howmany=min(10000, size(res.K, 1)))
    return find_all_harmonic_branches(λ, res.cfg.dt; kwargs...)
end

end # module
