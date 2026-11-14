# ============================================================================
# Prediction accuracy comparison across Koopman fitting methods
# — multi-trajectory training + common random test starts —
# ============================================================================
# Training paradigm: multi-trajectory with perturbed initial conditions
#   (avoids the flat-line artefact from starting exactly at a silent fixed
#   point).  All methods that support multi-trajectory data use the same
#   pooled training set.  HAVOK keeps a single continuous orbit because the
#   continuous-time forcing model requires trajectory continuity.
#
# Test paradigm: one fresh continuous trajectory from a perturbed IC.
#   Test start indices are drawn ONCE and reused by every method so the
#   comparison is fair.
# ============================================================================

using KoopmanAnalysis
using KoopmanAnalysis.Systems
using KoopmanAnalysis.Config
using KoopmanAnalysis.Plotting
using KoopmanAnalysis.Figures
using KoopmanAnalysis.Regimes
using KoopmanAnalysis.Utils
using KoopmanAnalysis.Spectral
using KoopmanAnalysis.Dictionaries
using KoopmanAnalysis.EDMD
using KoopmanAnalysis.DataGeneration
using KoopmanAnalysis.Hankel

using LinearAlgebra: BLAS
BLAS.set_num_threads(1)   # critical when using Julia threads

using CairoMakie
using Statistics
using Random
using LinearAlgebra
using Printf
using JLD2
using LaTeXStrings
using Dates

CairoMakie.activate!(type="png")
set_koopman_theme!()

const SEED = 42
Random.seed!(SEED)

# ============================================================================
# DISK-BACKING UTILITIES
# ============================================================================

fmt_bytes(b::Integer) = b >= 1<<30 ? @sprintf("%.2f GiB", b/2^30) :
                        b >= 1<<20 ? @sprintf("%.2f MiB", b/2^20) :
                        @sprintf("%.2f KiB", b/2^10)

print_mem(tag::AbstractString) =
    @printf("%-55s free RAM: %.2f GiB\n", tag, Sys.free_memory()/2^30)

const HEAVY_RESULT_FIELDS = (:S, :ΨX, :ΨY, :X_r)

function offload_result!(f, res, mid; heavy_fields=HEAVY_RESULT_FIELDS)
    pairs  = Pair{Symbol,Any}[]
    stored = Dict{UInt,String}()
    for fname in fieldnames(typeof(res))
        val = getfield(res, fname)
        if fname in heavy_fields && val isa AbstractArray
            target = val isa SubArray ? parent(val) : val
            key = objectid(target)
            if haskey(stored, key)
                @printf("    %-8s shares storage with %s (view — not duplicated)\n",
                        String(fname), stored[key])
            else
                if target !== val
                    path = "fit/$mid/$(String(fname))_full"
                    f[path] = target
                    @printf("    offloaded %-8s -> %s (%s, shared parent of views)\n",
                            String(fname), path, fmt_bytes(sizeof(target)))
                else
                    path = "fit/$mid/$(String(fname))"
                    f[path] = target
                    @printf("    offloaded %-8s -> %s (%s)\n",
                            String(fname), path, fmt_bytes(sizeof(target)))
                end
                stored[key] = path
            end
            try
                setfield!(res, fname, nothing)
            catch
            end
            push!(pairs, fname => nothing)
        else
            push!(pairs, fname => val)
        end
    end
    return (; pairs...)
end

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

experiment_type = "StateSpace-LargeExp-multitraj-delays"
regime_system   = "FHN"
regime_name     = length(ARGS) > 0 ? ARGS[1] : "stable-limit-cycle"

showdim = [1, 2]
n_show  = length(showdim)

# --- Local path configuration (see LocalPaths.jl.example) ---
const _LOCAL_PATHS = joinpath(@__DIR__, "LocalPaths.jl")
include(isfile(_LOCAL_PATHS) ? _LOCAL_PATHS : joinpath(@__DIR__, "LocalPaths.jl.example"))
import .LocalPaths: DATA_ROOT, RESULTS_ROOT

save_dir = default_save_dir(regime_system, regime_name;
                            base_dir=RESULTS_ROOT, subdir=experiment_type)
mkpath(save_dir)
println("Output directory: ", save_dir)

data_save_dir = default_save_dir(regime_system, regime_name;
                                 base_dir=DATA_ROOT, subdir=experiment_type)
mkpath(data_save_dir)

save_path = joinpath(data_save_dir,
                     "prediction_accuracy_results-$regime_system-$regime_name.jld2")
isfile(save_path) && rm(save_path)
println("Datasets directory: ", save_path)

print_mem("start")

# ============================================================================
# SYSTEM PARAMETERS
# ============================================================================

p_det = if regime_system == "VanderPol"
    vanderpol_parameters(regime_name; noise_type=:none, sigma=0.0, noise_mask=[1.0,0.0,0.0])
elseif regime_system == "Epileptor3D"
    epileptor3d_parameters(regime_name; noise_type=:none, sigma=0.0, noise_mask=[1.0,0.0,0.0])
elseif regime_system == "Lorenz"
    lorenz_parameters(regime_name; noise_type=:none, sigma=0.0, noise_mask=[1.0,1.0,1.0])
elseif regime_system == "FHN"
    fhn_parameters(regime_name; noise_type=:none, sigma=0.0, noise_mask=[1.0,0.0])
elseif regime_system == "Duffing"
    duffing_parameters(regime_name; noise_type=:none, sigma=0.0, noise_mask=[1.0,0.0])
elseif regime_system == "Rossler"
    rossler_parameters("stable-limit-cycle"; noise_type=:none, sigma=0.0, noise_mask=[1.0,0.0,0.0])
else
    error("Unknown regime_system: $regime_system")
end

f_det = if regime_system == "VanderPol"
    vanderpol_drift(p_det)
elseif regime_system == "Epileptor3D"
    epileptor3d_drift(p_det)
elseif regime_system == "Lorenz"
    lorenz_drift(p_det)
elseif regime_system == "FHN"
    fhn_drift(p_det)
elseif regime_system == "Duffing"
    duffing_drift(p_det)
elseif regime_system == "Rossler"
    rossler_drift(p_det)
end

a_param = if regime_system == "FHN"
    p_det.a
elseif regime_system == "VanderPol"
    p_det.mu
elseif regime_system == "Lorenz"
    p_det.σ
elseif regime_system == "Epileptor3D"
    p_det.c
elseif regime_system == "Duffing"
    p_det.delta
else
    0.0
end

b_param = if regime_system == "FHN"
    p_det.b
elseif regime_system == "Lorenz"
    p_det.ρ
elseif regime_system == "Epileptor3D"
    p_det.d_star
elseif regime_system == "Duffing"
    p_det.beta
else
    0.0
end

c_param = if regime_system == "FHN"
    p_det.epsilon
elseif regime_system == "Lorenz"
    p_det.β
elseif regime_system == "Epileptor3D"
    p_det.tau0
elseif regime_system == "Duffing"
    p_det.alpha
else
    0.0
end

d_param = if regime_system == "FHN"
    p_det.I_ext
else
    0.0
end

sys_param_str = if regime_system == "FHN"
    "FHN: a=$a_param, b=$b_param, ε=$c_param, I=$d_param"
elseif regime_system == "VanderPol"
    "VdP: μ=$a_param"
elseif regime_system == "Lorenz"
    "Lorenz: σ=$a_param, ρ=$b_param, β=$(round(c_param,digits=2))"
elseif regime_system == "Epileptor3D"
    "Epileptor3D: c=$a_param, d★=$b_param, τ₀=$c_param"
elseif regime_system == "Duffing"
    "Duffing: δ=$a_param, β=$b_param, α=$c_param"
else
    ""
end

println("Parameters: ", sys_param_str)

# ============================================================================
# TIME / DATA SIZE
# ============================================================================

dt_internal = 0.01
nLag        = 10
dt_eff      = dt_internal * nLag

n_train  = 1500000
n_test   = 10000
n_pred   = 5000

m_embed   = 20
tau_delay = 1
max_lag   = (m_embed - 1) * tau_delay

burn_in  = 0   # discarded snapshots from the start of each trajectory

n_traj       = 2000
window_ic    = 2.0

# Length of each training trajectory (including lag buffer for delay embedding)
m_traj = div(n_train, n_traj) + max_lag + 1
n_pairs_per_traj = m_traj - max_lag

n_traj_test = 10
window_ic_test = 1.0
x0_center = [0.0,0.5]

println("dt = ", dt_internal, " s  |  nLag = ", nLag, "  |  dt_eff = ", dt_eff, " s")
println("Embedding: m=$m_embed, tau=$tau_delay, max_lag=$max_lag")
println("Train: $n_train snapshots  |  Test: $n_test snapshots  |  Predict: $n_pred steps")
println("Trajectories: $n_traj × $m_traj steps  ($n_pairs_per_traj pairs each)")

# ============================================================================
# FIXED POINTS
# ============================================================================

x0_silent = fixed_point(regime_system, p_det)
println("Silent-state fixed point: x0 = ", x0_silent)

# ============================================================================
# TRAINING DATA — multi-trajectory with perturbed ICs
# ============================================================================
println("\n" * "="^60)
println("Generating multi-trajectory training data ...")
println("="^60)

X_train_raw, Y_train_raw, _, _, _ = edmd_training_data(
    regime_system, regime_name;
    m_train        = m_traj,
    n_trajectories = n_traj,
    window         = window_ic,
    dt             = dt_internal,
    nLag           = nLag,
    center         = x0_silent
)

n_xy_total = size(X_train_raw, 2)
println("Raw training data: ", size(X_train_raw), "  (", n_xy_total, " snapshot pairs)")
print_mem("after multi-trajectory generation")

# ---------------------------------------------------------------------------
# Figure 01 — Original training data (multi-trajectory pool)
# ---------------------------------------------------------------------------
fig = fig_training_data(X_train_raw;
    dim=2, projection_dims=(1, 2),
    title="Multi-trajectory training data pool",
    subtitle="$regime_system $regime_name | $n_traj trajectories × $m_traj steps",
    param_str="dt=$dt_internal, nLag=$nLag, window_ic=$window_ic")
save(joinpath(save_dir, "01_original_training_data.png"), fig)
println("Saved: 01_original_training_data.png")
fig = nothing; GC.gc()

# ============================================================================
# HAVOK training — single continuous trajectory (needs continuity for forcing)
# ============================================================================
println("\nGenerating single continuous trajectory for HAVOK ...")

x0_havok    = x0_silent .+ ones(length(x0_silent))
n_havok     = n_train + max_lag + n_pred + burn_in
X_havok_raw = rk4(f_det, x0_havok, dt_internal, n_havok; nLag=nLag)

# ============================================================================
# TRAINING DELAY MATRIX — built per-trajectory to avoid cross-boundary leakage
# ============================================================================
println("\nBuilding per-trajectory delay matrix from training pool ...")

n_pairs_per_traj = m_traj - max_lag
S_train_list = Matrix{Float64}[]
for t in 1:n_traj
    start_col = (t - 1) * m_traj + 1
    end_col   = t * m_traj
    X_traj    = X_train_raw[showdim, start_col:end_col]
    S_traj    = build_hankel_multichannel(X_traj, m_embed, tau_delay)
    push!(S_train_list, S_traj)
end
S_train = hcat(S_train_list...)
println("S_train: ", size(S_train), "  (", n_traj, " trajectories × ", n_pairs_per_traj, " delay states)")

# ---------------------------------------------------------------------------
# Figure 02 — Training data delay-conformed (current-time values only)
# ---------------------------------------------------------------------------
phys_rows = [(j - 1) * m_embed + 1 for j in 1:n_show]
X_train_delay = S_train[phys_rows, :]
fig = fig_training_data(X_train_delay;
    dim=2, projection_dims=(1, 2),
    title="Training data — delay-conformed, current-time values",
    subtitle="$regime_system $regime_name",
    param_str="m_embed=$m_embed, tau_delay=$tau_delay, n_pairs=$(size(S_train,2))")
save(joinpath(save_dir, "02_training_data_delay_conformed.png"), fig)
println("Saved: 02_training_data_delay_conformed.png")
fig = nothing; GC.gc()

# ============================================================================
# HAVOK RANK from energy threshold (0.99) on the HAVOK delay matrix
# ============================================================================
println("\nComputing HAVOK energy-based rank ...")

H_havok = build_hankel_multichannel(X_havok_raw[showdim, :], m_embed, tau_delay)
F_havok = svd(H_havok)
Σ       = F_havok.S
total_energy = sum(Σ.^2)
cum_energy   = cumsum(Σ.^2)
r_havok      = findfirst(cum_energy .>= 0.99 * total_energy)
if isnothing(r_havok)
    r_havok = length(Σ)
end
println("  HAVOK singular values (first 10): ", round.(Σ[1:min(10,length(Σ))], digits=4))
println("  Energy threshold: 0.99  →  rank r = $r_havok  (captures " *
        "$(round(100*cum_energy[r_havok]/total_energy, digits=2))% of variance)")

# ============================================================================
# TEST STARTING POINTS — window-integrated backward sampling
# ============================================================================
println("\nGenerating test starting points via window-integrated sampling ...")

S0_test = sample_seeds_window_integrated(
    f_det, x0_center, window_ic_test, length(x0_center), showdim,
    dt_internal, nLag, m_embed, tau_delay;
    n_starts=n_traj_test, direction=:backward,
    rng=MersenneTwister(2024), max_consecutive_failures=20)

println("  S0_test: $(size(S0_test))  ($n_traj_test seeds)")

# Physical anchor indices within each delay vector
phys_idx = [(j - 1) * m_embed + 1 for j in 1:n_show]

# Pre-generate TRUE trajectories by forward integration from anchors
X_true_all = zeros(n_show, n_pred + 1, n_traj_test)
S_test_for_havok = Matrix{Float64}[]   # per-trajectory delay matrices for HAVOK

for j in 1:n_traj_test
    x0_anchor = S0_test[phys_idx, j]
    # True trajectory: integrate forward n_pred effective steps
    X_seg = rk4(f_det, x0_anchor, dt_internal, n_pred; nLag=nLag)
    X_true_all[:, :, j] = X_seg[showdim, :]

    # HAVOK forcing needs a delay matrix of the true trajectory
    # Build with enough lag buffer for embedding
    X_seg_havok = rk4(f_det, x0_anchor, dt_internal,
                      n_pred + max_lag; nLag=nLag)
    S_havok_j = build_hankel_multichannel(X_seg_havok[showdim, :], m_embed, tau_delay)
    push!(S_test_for_havok, S_havok_j)
end

println("  True trajectories pre-generated for all $n_traj_test seeds")

# ============================================================================
# STATE-SPACE DICTIONARY (RBF) — subsampled before clustering to avoid OOM
# ============================================================================
const RBF_DICT_PARAMS = (nRBF=1000, include_states=true,
                         kernel_type=:thinplate, seed=42,
                         max_points=50000)

# ============================================================================
# METHOD CONFIGURATION
# ============================================================================

METHOD_CONFIGS = Dict(
    "DMD"           => (dict=:none, m_embed=1,       tau_delay=1,          rank=nothing, extra=(; use_svd=false, return_ψ=false)),
    "Hankel_DMD"    => (dict=:none, m_embed=m_embed, tau_delay=tau_delay,  rank=nothing, extra=(; use_svd=true,  return_ψ=false)),
    "HAVOK"         => (dict=:havok, m_embed=m_embed, tau_delay=tau_delay, rank=:havok,  extra=(; use_svd=true,  return_ψ=false)),
    "Hankel_DMD_r"  => (dict=:none, m_embed=m_embed, tau_delay=tau_delay,  rank=:havok,  extra=(; use_svd=true,  return_ψ=false)),
    "EDMD"          => (dict=:RBF,  m_embed=1,       tau_delay=1,          rank=nothing, extra=(; use_svd=false, return_ψ=true)),
    "Hankel_EDMD"   => (dict=:RBF,  m_embed=m_embed, tau_delay=tau_delay,  rank=nothing, extra=(; use_svd=true,  return_ψ=true)),
    "Hankel_EDMD_r" => (dict=:RBF,  m_embed=m_embed, tau_delay=tau_delay,  rank=:havok,  extra=(; use_svd=true,  return_ψ=true)),
)

function build_method_config(mid; dt, r_havok)
    haskey(METHOD_CONFIGS, mid) || error("Unknown method $mid")
    c = METHOD_CONFIGS[mid]
    r_use = (c.rank == :havok) ? r_havok : c.rank
    KoopmanConfig(;
        m_embed     = c.m_embed,
        tau_delay   = c.tau_delay,
        r           = r_use,
        use_svd     = c.extra.use_svd,
        dict_type   = c.dict,
        dict_params = (c.dict == :RBF) ? RBF_DICT_PARAMS : NamedTuple(),
        edmd_method = :ridge,
        edmd_alpha  = 1e-1,
        proj_alpha  = 1e-6,
        dt          = dt,
        return_ψ    = c.extra.return_ψ,
        verbose     = true,
    )
end

# ============================================================================
# PREDICTION HELPERS — explicit initial conditions from integrated sampling
# ============================================================================

function predict_dmd(K::AbstractMatrix, X_ics::AbstractMatrix, n_pred::Int)
    n_traj = size(X_ics, 2)
    n_state = size(X_ics, 1)
    X_pred = zeros(1, n_pred + 1, n_traj)
    x = Vector{Float64}(undef, n_state)
    for j in 1:n_traj
        @inbounds for i in 1:n_state
            x[i] = X_ics[i, j]
        end
        X_pred[1, 1, j] = x[1]
        @inbounds for k in 1:n_pred
            x .= K * x
            X_pred[1, k+1, j] = x[1]
        end
    end
    return X_pred
end

function predict_hankel_dmd(K::AbstractMatrix, S_ics::AbstractMatrix, n_pred::Int)
    n_traj = size(S_ics, 2)
    m = size(S_ics, 1)
    X_pred = zeros(1, n_pred + 1, n_traj)
    s = Vector{Float64}(undef, m)
    s_next = Vector{Float64}(undef, m)
    for j in 1:n_traj
        @inbounds for i in 1:m
            s[i] = S_ics[i, j]
        end
        X_pred[1, 1, j] = s[1]
        @inbounds for k in 1:n_pred
            s_next .= K * s
            s, s_next = s_next, s
            X_pred[1, k+1, j] = s[1]
        end
    end
    return X_pred
end

function predict_edmd_rbf(res, S_ics::AbstractMatrix, n_pred::Int)
    n_traj = size(S_ics, 2)
    n_state = size(S_ics, 1)
    nPsi = size(res.K, 1)
    X_pred = zeros(1, n_pred + 1, n_traj)

    s = Vector{Float64}(undef, n_state)
    s_next = Vector{Float64}(undef, n_state)
    n_reduced = isnothing(res.U_r) ? n_state : size(res.U_r, 2)
    x_r = Vector{Float64}(undef, n_reduced)
    ψ = Vector{Float64}(undef, nPsi)
    ψ_next = Vector{Float64}(undef, nPsi)

    for j in 1:n_traj
        @inbounds for i in 1:n_state
            s[i] = S_ics[i, j]
        end
        X_pred[1, 1, j] = s[1]

        if isnothing(res.U_r)
            x_r .= s
        else
            mul!(x_r, res.U_r', s)
        end
        ψ_tmp = lift_state(x_r, res.dict_info)
        ψ .= ψ_tmp

        @inbounds for k in 1:n_pred
            mul!(ψ_next, res.K', ψ)
            x_next = res.B_reduced * ψ_next
            if isnothing(res.U_r)
                s_next .= x_next
            else
                mul!(s_next, res.U_r, x_next)
            end
            X_pred[1, k+1, j] = s_next[1]
            s .= s_next
            ψ .= ψ_next
        end
    end
    return X_pred
end

function compute_rmse(X_true::AbstractArray, X_pred::AbstractArray)
    diff = X_true .- X_pred
    return sqrt(mean(diff .^ 2))
end

# ============================================================================
# FIT & PREDICT — method loop
# ============================================================================

method_list = [
    ("DMD",           "DMD (no embedding)"),
    ("Hankel_DMD",    "Hankel-DMD (full rank)"),
    ("HAVOK",         "HAVOK (continuous, RK4)"),
    ("Hankel_DMD_r",  "Hankel-DMD (r = HAVOK rank)"),
    ("EDMD",          "RBF-EDMD (no embedding)"),
    ("Hankel_EDMD",   "Hankel RBF-EDMD (full)"),
    ("Hankel_EDMD_r", "Hankel RBF-EDMD (r = HAVOK rank)"),
]

all_results = Dict{String,Any}()
K_mats      = Dict{String,AbstractMatrix}()
spectral_table = Dict{String, NamedTuple}()

for (mid, tag) in method_list
    println("\n", "="^70)
    println("Method: $mid  ($tag)")
    println("="^70)

    cfg = build_method_config(mid; dt=dt_eff, r_havok=r_havok)
    println(cfg)

    # -----------------------------------------------------------------------
    # Training data selection and fit
    # -----------------------------------------------------------------------
    if mid in ("DMD", "EDMD")
        # Plain methods: raw state-space data
        X_fit = X_train_raw[showdim, :]
        Y_fit = Y_train_raw[showdim, :]
        println("  Using multi-trajectory pool: ", size(X_fit))
        res = state_analysis(X_fit, Y_fit, cfg)

    elseif mid == "HAVOK"
        # HAVOK: single continuous trajectory (needs continuity for forcing)
        S_havok = build_hankel_multichannel(X_havok_raw[showdim, :], m_embed, tau_delay)
        println("  Using single continuous trajectory for HAVOK: ", size(S_havok))
        res = hankel_analysis(S_havok, cfg)

    else
        # All other Hankel methods: delay-embedded multi-trajectory pairs
        pd = delay_embed_pairs(X_train_raw, Y_train_raw, cfg.m_embed, cfg.tau_delay;
                               n_trajectories=n_traj)
        println("  Using delay-embedded pairs: ", size(pd.S))
        res = pair_analysis(pd, cfg)
    end
    println("  Koopman matrix size: $(size(res.K))")

    # ---- Spectrum (before offload) ----
    println("  Computing spectrum...")
    nK = size(res.K, 1)
    λ, Ξ = spectrum(res; howmany=min(1000, nK), krylovdim=min(nK, 1250))
    spectral_table[mid] = (λ=λ, Ξ=Ξ)

    # -----------------------------------------------------------------------
    # Predictions (explicit ICs from window-integrated sampling)
    # -----------------------------------------------------------------------
    println("  Predicting from window-integrated test seeds ...")

    if mid == "DMD"
        # DMD: plain, no embedding — use physical anchors only
        X_ics = S0_test[phys_idx, :]   # n_show × n_traj_test
        X_pred = predict_dmd(res.K, X_ics, n_pred)
        X_true = reshape(X_true_all[1, :, :], 1, n_pred + 1, n_traj_test)

    elseif mid in ("Hankel_DMD", "Hankel_DMD_r")
        X_pred = predict_hankel_dmd(res.K, S0_test, n_pred)
        X_true = reshape(X_true_all[1, :, :], 1, n_pred + 1, n_traj_test)

    elseif mid == "HAVOK"
        A = res.dict_info.A
        B = res.dict_info.B
        U_r = res.U_r
        X_pred = zeros(1, n_pred + 1, n_traj_test)
        X_true = zeros(1, n_pred + 1, n_traj_test)
        for j in 1:n_traj_test
            X_true_j, X_pred_j = havok_predict(A, B, U_r, S_test_for_havok[j],
                                                [1], n_pred;
                                                dt=dt_eff, use_true_forcing=true)
            X_true[:, :, j] = X_true_j
            X_pred[:, :, j] = X_pred_j
        end

    elseif mid == "EDMD"
        # EDMD: plain RBF, no embedding
        X_ics = S0_test[phys_idx, :]
        X_pred = predict_edmd_rbf(res, X_ics, n_pred)
        X_true = reshape(X_true_all[1, :, :], 1, n_pred + 1, n_traj_test)

    elseif mid in ("Hankel_EDMD", "Hankel_EDMD_r")
        X_pred = predict_edmd_rbf(res, S0_test, n_pred)
        X_true = reshape(X_true_all[1, :, :], 1, n_pred + 1, n_traj_test)

    else
        error("Unknown method: $mid")
    end

    # -----------------------------------------------------------------------
    # Error metrics
    # -----------------------------------------------------------------------
    errs = abs.(X_true .- X_pred)
    max_err = maximum(errs)
    mean_err = mean(errs)
    rmse = compute_rmse(X_true, X_pred)
    println("  max error:  $(round(max_err, sigdigits=4))")
    println("  mean error: $(round(mean_err, sigdigits=4))")
    println("  RMSE:       $(round(rmse, sigdigits=4))")

    # -----------------------------------------------------------------------
    # Store
    # -----------------------------------------------------------------------
    all_results[mid] = Dict(
        "tag"      => tag,
        "X_true"   => X_true,
        "X_pred"   => X_pred,
        "rmse"     => rmse,
        "max_err"  => max_err,
        "mean_err" => mean_err,
        "res"      => res,
    )
    K_mats[mid] = res.K

    # -----------------------------------------------------------------------
    # Figure — individual method
    # -----------------------------------------------------------------------
    fig_title = "$tag | $sys_param_str"
    fig_sub   = "FHN $regime_name — $n_traj_test trajectories, $n_pred-step horizon"
    fig_par   = "dt=$(dt_eff)s, n_train=$n_train, m=$m_embed, τ=$tau_delay"

    fig = fig_prediction(X_true, X_pred;
        dt = dt_eff,
        dim_labels = ["v(t)"],
        title = fig_title,
        subtitle = fig_sub,
        param_str = fig_par,
        max_trajectories = n_traj_test
    )
    save(joinpath(save_dir, "pred_$(mid).png"), fig)
    println("  Saved: pred_$(mid).png")
    fig = nothing; GC.gc()

    # -----------------------------------------------------------------------
    # Save to JLD2
    # -----------------------------------------------------------------------
    jldopen(save_path, "a+") do f
        f["results/$mid/X_true"] = X_true
        f["results/$mid/X_pred"] = X_pred
        f["results/$mid/rmse"]   = rmse
        f["results/$mid/tag"]    = tag
    end

    # -----------------------------------------------------------------------
    # Offload heavy fields before next method
    # -----------------------------------------------------------------------
    jldopen(save_path, "a+") do f
        offload_result!(f, res, mid)
    end
    GC.gc()
    print_mem("after $mid")
end

# ============================================================================
# FIGURE 10: TRAJECTORY COMPARISON + RMSE
# ============================================================================
println("\nBuilding Figure 10: Prediction trajectory comparison...")

X_true_ref = all_results[method_list[1][1]]["X_true"]

fig1 = Figure(size=(1800, 200 + 500 * (ceil(Int, (length(method_list) + 1) / 2) + 1)))

row_start = 0
Label(fig1[row_start, :], "Prediction accuracy comparison — v(t) only",
      fontsize=24, font=:bold, tellwidth=false, halign=:center)
row_start += 1
Label(fig1[row_start, :],
      "$regime_system $regime_name — $n_traj_test trajectories, $n_pred-step horizon",
      fontsize=18, tellwidth=false, halign=:center)
row_start += 1

ax_real = Axis(fig1[row_start, 1],
    title="Real trajectories", xlabel="Step", ylabel="v(t)",
    titlesize=18, titlefont=:bold)

cols_real = blue_tones(n_traj_test)
cols_pred = orange_tones(n_traj_test)
for j in 1:n_traj_test
    lines!(ax_real, 0:n_pred, X_true_ref[1, :, j], color=cols_real[j], linewidth=1.5)
end

n_method_rows = ceil(Int, (length(method_list) + 1) / 2)

METHOD_PALETTE = Makie.wong_colors()
method_colors = Dict(mid => METHOD_PALETTE[mod1(i, length(METHOD_PALETTE))]
                     for (i, (mid, _)) in enumerate(method_list))

for (i, (mid, tag)) in enumerate(method_list)
    row = row_start + i ÷ 2
    col = i % 2 + 1

    cfg_mid = build_method_config(mid; dt=dt_eff, r_havok=r_havok)
    sub = "m=$(cfg_mid.m_embed), τ=$(cfg_mid.tau_delay)"
    if cfg_mid.dict_type == :none
        sub *= ", linear"
    elseif cfg_mid.dict_type == :havok
        sub *= ", r=$r_havok"
    elseif cfg_mid.dict_type == :RBF
        sub *= ", $(cfg_mid.dict_params.nRBF) RBF"
        if cfg_mid.use_svd && !isnothing(cfg_mid.r)
            sub *= ", r=$(cfg_mid.r)"
        end
    end

    ax = Axis(fig1[row, col], title="$mid: $tag | $sub",
        xlabel="Step", ylabel="v(t)", titlesize=12, titlefont=:bold)

    X_pred = all_results[mid]["X_pred"]
    for j in 1:n_traj_test
        lines!(ax, 0:n_pred, X_pred[1, :, j], color=cols_pred[j], linewidth=1.2)
        lines!(ax, 0:n_pred, X_true_ref[1, :, j],
               color=(cols_real[j], 0.5), linewidth=1.0, linestyle=:dash)
    end
end

err_row = row_start + n_method_rows
ax_err = Axis(fig1[err_row, :], title="Average RMSE — v(t)",
    xlabel="Step", ylabel="RMSE", titlesize=18, titlefont=:bold)

for (mid, _) in method_list
    X_pred = all_results[mid]["X_pred"]
    rmse_t = sqrt.(mean((X_true_ref .- X_pred).^2, dims=3))[1, :, 1]
    lines!(ax_err, 0:n_pred, rmse_t, label=mid,
           color=method_colors[mid], linewidth=2)
end
axislegend(ax_err, position=:rt, framevisible=true)

save(joinpath(save_dir, "10_prediction_trajectory_comparison.png"), fig1)
println("Saved: 10_prediction_trajectory_comparison.png")
fig1 = nothing; GC.gc()

# ============================================================================
# FIGURE 11: SPECTRUM COMPARISON
# ============================================================================
println("\nBuilding Figure 11: Spectrum comparison...")

fig_spec = Figure(size=(1800, 800))
Label(fig_spec[0, :], "Koopman spectrum comparison — all methods",
      fontsize=22, font=:bold, tellwidth=false, halign=:center)

ax1 = Axis(fig_spec[1, 1], title="Discrete eigenvalues λ",
    xlabel="Re(λ)", ylabel="Im(λ)", aspect=DataAspect(),
    titlesize=16, titlefont=:bold)
θ = range(0.0, 2π, length=200)
lines!(ax1, cos.(θ), sin.(θ), color=(:black, 0.4), linewidth=1, linestyle=:dash)
hlines!(ax1, 0.0, color=(:black, 0.2), linewidth=0.5)
vlines!(ax1, 0.0, color=(:black, 0.2), linewidth=0.5)

ax2 = Axis(fig_spec[1, 2],
    title="Continuous-time analogue  μ = log(λ)/dt_eff",
    xlabel="Re(μ) (growth rate, 1/s)", ylabel="Im(μ) (angular frequency, rad/s)",
    titlesize=16, titlefont=:bold)
hlines!(ax2, 0.0, color=(:black, 0.2), linewidth=0.5)
vlines!(ax2, 0.0, color=(:black, 0.4), linewidth=1, linestyle=:dash)

for (mid, _) in method_list
    if haskey(spectral_table, mid)
        λ = spectral_table[mid].λ
        μ = log.(λ) ./ dt_eff
        scatter!(ax1, real.(λ), imag.(λ),
                 color=(method_colors[mid], 0.6), markersize=6, label=mid)
        scatter!(ax2, real.(μ), imag.(μ),
                 color=(method_colors[mid], 0.6), markersize=6, label=mid)
    end
end

Legend(fig_spec[1, 3],
       [MarkerElement(color=method_colors[mid], marker=:circle)
        for (mid, _) in method_list],
       [mid for (mid, _) in method_list], "Method")

save(joinpath(save_dir, "11_spectrum_comparison.png"), fig_spec)
println("Saved: 11_spectrum_comparison.png")
fig_spec = nothing; GC.gc()

# ============================================================================
# FIGURE 12: RBF CENTRES per method
# ============================================================================
println("\nBuilding Figure 12: RBF centres...")

rbf_methods = [mid for (mid, _) in method_list
               if build_method_config(mid; dt=dt_eff, r_havok=r_havok).dict_type == :RBF]

for mid in rbf_methods
    res_l = all_results[mid]["res"]
    (!haskey(res_l, :dict_info) || !haskey(res_l.dict_info, :centers)) && continue

    cd = res_l.dict_info.centers
    if !isnothing(res_l.U_r)
        cd = res_l.U_r * cd
    end
    ktype = get(res_l.dict_info, :kernel_type, :thinplate)

    fig = Figure(size=(800, 800))
    ax = Axis(fig[1, 1], title="$mid — $(ktype) RBF centres ($(size(cd, 2)))",
        xlabel="v", ylabel="w", titlesize=16, titlefont=:bold)

    n_bg = min(10000, size(X_train_raw, 2))
    idx_bg = rand(1:size(X_train_raw, 2), n_bg)
    scatter!(ax, X_train_raw[1, idx_bg], X_train_raw[2, idx_bg],
             color=(:gray, 0.1), markersize=2)

    if size(cd, 1) == 2
        scatter!(ax, cd[1, :], cd[2, :],
                 color=:red, markersize=8, marker=:x, label="centres")
    elseif size(cd, 1) == n_show * m_embed
        v_c = cd[1, :]
        w_c = cd[(2 - 1) * m_embed + 1, :]
        scatter!(ax, v_c, w_c,
                 color=:red, markersize=8, marker=:x, label="centres")
    end
    axislegend(ax)

    save(joinpath(save_dir, "12_rbf_centers_$mid.png"), fig)
    println("Saved: 12_rbf_centers_$mid.png")
    fig = nothing; GC.gc()
end

# ============================================================================
# AVERAGE RMSE SUMMARY
# ============================================================================
println("\n", "="^60)
println("Average RMSE summary")
println("="^60)

avg_rmse = Dict{String,Float64}()
for (mid, _) in method_list
    res_d = all_results[mid]
    X_true = res_d["X_true"]
    X_pred = res_d["X_pred"]
    n_t = size(X_true, 3)
    rmse_per_traj = [compute_rmse(X_true[:, :, j], X_pred[:, :, j]) for j in 1:n_t]
    avg_rmse[mid] = mean(rmse_per_traj)
    println("  $mid: ", round(avg_rmse[mid], sigdigits=4))
end

# ============================================================================
# SAVE METADATA
# ============================================================================
jldopen(save_path, "a+") do f
    f["metadata/sys_param_str"] = sys_param_str
    f["metadata/dt_internal"]   = dt_internal
    f["metadata/dt_eff"]        = dt_eff
    f["metadata/nLag"]          = nLag
    f["metadata/n_train"]       = n_train
    f["metadata/n_test"]        = n_test
    f["metadata/n_pred"]        = n_pred
    f["metadata/m_embed"]       = m_embed
    f["metadata/tau_delay"]           = tau_delay
    f["metadata/n_traj"]        = n_traj
    f["metadata/n_traj_test"]   = n_traj_test
    f["metadata/test_starts_embed"] = test_starts_embed
    f["metadata/test_starts_raw"] = test_starts_raw
    f["metadata/avg_rmse"]      = avg_rmse
    f["metadata/method_list"]   = [m for (m, _) in method_list]
end

println("\nAll results saved to: $save_path")
println("Figures saved to:     $save_dir")
print_mem("done")