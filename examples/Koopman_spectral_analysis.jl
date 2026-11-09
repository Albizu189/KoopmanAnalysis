# ============================================================================
# Koopman_spectral_analysis.jl  —  Script 2 of the KoopmanAnalysis usage-case
# pipeline (rewrites Koopman_Classic_analysis_zls_full.jl on top of the public
# package API, with the POD-of-ZLS and phase-colormap additions).
#
# INPUT : the JLD2 dataset written by Script 1 (Koopman_fitting_methods.jl),
#           fitting_methods-$regime_system-$regime_name-show$showdim_tag.jld2
#         located under
#           default_save_dir(regime_system, regime_name;
#                            base_dir=DATA_ROOT, subdir=experiment_name)
#         (see SHARED_BRIEF.md data contract for the exact key layout).
# OUTPUT: spectral_analysis-$regime_system-$regime_name-show$showdim_tag-$spectral_method.jld2
#         in the same data directory, plus PNG figures under RESULTS_ROOT.
#
# Pipeline:
#   1.  Load Script-1 dataset; select the fitted Koopman operator via the
#       top-level `spectral_method` variable; rebuild the plotting-space
#       machinery (P_plot, phys_idx, plot_labels) from the saved metadata.
#   2.  Identify ALL real eigenvalues (|Im λ| < tol) inside the unit circle
#       of the selected operator and extract the zero level set (ZLS) of each
#       corresponding eigenfunction with find_zls_gradient_descent
#       (multi-start gradient descent, seeded).  Each ZLS is written to the
#       output JLD2 immediately (`zls/j_<j>`) so interrupted runs resume.
#   3.  Compute a clean reference limit cycle (burn-in + rk4, exactly two
#       periods, delay-embedded with build_hankel_multichannel).
#   4.  Huge grid figure of all ZLS with limit-cycle overlay.
#   5.  Transversality angle matrix between the ZLS of every eigenvalue pair
#       (grad_phi on limit-cycle samples) + heatmap figure.
#   6.  Joint ZLS of the most transverse pair + intersection figure.
#   7.  Cross-method spectrum comparison (discrete + continuous analogue)
#       from Script 1's spectral_table, highlighting the selected method.
#   8.  Full POD projection of the stacked ZLS point cloud (eigen of the
#       covariance — never svd's Vt), POD spectrum figure.
#   9.  Phase color map: φ_phase (from select_phase_amplitude_modes) on every
#       ZLS point; θ = angle(φ) shown in plot space AND in POD space.
#   10. Console summary (phase eigenvalue, LC period/frequency, ZLS counts,
#       best pair, POD spectrum).
#
# Reproducibility (SHARED_BRIEF rule ii): one top-level `const SEED = 42`;
# every stochastic call (ZLS start sampling inside find_zls_gradient_descent,
# all figure/POD subsampling) is explicitly seeded — see the comments at each
# call site.
# ============================================================================

using KoopmanAnalysis
using KoopmanAnalysis.Utils: find_zls_gradient_descent, grad_phi
using KoopmanAnalysis.Dictionaries: Psi_slice
using KoopmanAnalysis.Systems
using KoopmanAnalysis.Hankel
using KoopmanAnalysis.Plotting
using KoopmanAnalysis.Spectral: select_phase_amplitude_modes
using CairoMakie
using LinearAlgebra
using Statistics
using Random
using JLD2
using Printf
using ProgressMeter
using Base.Threads: nthreads

CairoMakie.activate!(type="png")
set_koopman_theme!()

# --- Local path configuration (see LocalPaths.jl.example) ---
const _LOCAL_PATHS = joinpath(@__DIR__, "LocalPaths.jl")
include(isfile(_LOCAL_PATHS) ? _LOCAL_PATHS : joinpath(@__DIR__, "LocalPaths.jl.example"))
import .LocalPaths: DATA_ROOT, RESULTS_ROOT

# One global seed for the whole script (SHARED_BRIEF rule ii).  All
# stochastic calls below either take `seed=SEED` or use a dedicated
# MersenneTwister derived from SEED (the exact stream is stated at each site).
const SEED = 42
Random.seed!(SEED)

# Recommended by KoopmanAnalysis.Utils to prevent BLAS oversubscription when
# find_zls_gradient_descent parallelises starts over Julia threads.
BLAS.set_num_threads(1)
println("Running with $(nthreads()) Julia threads.")

# ============================================================================
# UTILITIES (memory instrumentation — SHARED_BRIEF rule i)
# ============================================================================

fmt_bytes(b::Integer) = b >= 1<<30 ? @sprintf("%.2f GiB", b/2^30) :
                        b >= 1<<20 ? @sprintf("%.2f MiB", b/2^20) :
                                     @sprintf("%.2f KiB", b/2^10)

print_mem(tag::AbstractString) =
    @printf("%-55s free RAM: %.2f GiB\n", tag, Sys.free_memory()/2^30)

"""
    seeded_subsample_idx(n, n_max, stream_tag) -> Vector{Int}

Seeded, sorted column subset of `1:n` capped at `n_max` entries.  Uses a
DEDICATED `MersenneTwister(SEED ⊻ stream_tag)` so the draw is reproducible
independently of the global RNG stream (which find_zls_gradient_descent
re-seeds internally).  Identity when `n <= n_max`.
"""
function seeded_subsample_idx(n::Int, n_max::Int, stream_tag::Integer)
    n <= n_max && return collect(1:n)
    rng = MersenneTwister(SEED ⊻ stream_tag)   # dedicated stream (rule ii)
    return sort!(rand(rng, 1:n, n_max))
end

"""
    get_entry(tbl, mid)

Fetch `tbl[mid]` tolerating String/Symbol key types in Script 1's Dicts.
"""
function get_entry(tbl, mid)
    haskey(tbl, mid) && return tbl[mid]
    haskey(tbl, string(mid)) && return tbl[string(mid)]
    haskey(tbl, Symbol(mid)) && return tbl[Symbol(mid)]
    error("Key '$mid' not found (available: $(collect(keys(tbl))))")
end

"""
    jld_write!(f, key, val; overwrite=false)

Idempotent JLD2 write: skips existing keys unless `overwrite=true` (then
deletes first).  Keeps the incremental-save sections resumable.
"""
function jld_write!(f, key, val; overwrite::Bool=false)
    if haskey(f, key)
        overwrite || return false
        delete!(f, key)
    end
    f[key] = val
    return true
end

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

experiment_name = "FittingMethods"         # must match Script 1's subdir
regime_system   = "Epileptor3D"
regime_name     = length(ARGS) > 0 ? ARGS[1] : "c4b-SN-FLC"

# Which fitted operator of Script 1 to analyse.  Any id in the dataset's
# `method_ids` is valid; the analysis below requires a THIN-PLATE RBF
# dictionary (grad_phi / find_zls_gradient_descent are thinplate-only —
# API_REFERENCE GOTCHAS #3), which is Script 1's default for the EDMD family.
const spectral_method = length(ARGS) > 1 ? ARGS[2] : "Hankel_EDMD"

# Dimension-control parameters (defaults per the SHARED_BRIEF data contract)
dim     = 3
showdim = [1, 2, 3]                  # which physical coords Script 1 trained with
n_show  = length(showdim)
n_hidden = dim - n_show
n_plot  = dim                        # total plotting directions

@assert issorted(showdim) "showdim must be sorted in increasing order"
@assert all(1 .<= showdim .<= dim) "showdim entries must lie in 1:dim ($dim)"
@assert n_show <= dim "n_show cannot exceed dim"

showdim_tag = join(string.(showdim))

# Memory guards (rule i): caps for heavy point clouds in POD / figures.
const MAX_ZLS_POD  = 30_000    # ZLS columns entering the POD covariance
const MAX_FIG_PTS  = 20_000    # scatter points per figure panel

# Paths — results (figures) vs heavy data (JLD2) roots from LocalPaths.
save_dir = default_save_dir(regime_system, regime_name;
                            base_dir=RESULTS_ROOT, subdir=experiment_name)
mkpath(save_dir)
data_save_dir = default_save_dir(regime_system, regime_name;
                                 base_dir=DATA_ROOT, subdir=experiment_name)
mkpath(data_save_dir)

println("Output directory (figures): ", save_dir)
println("Data directory (JLD2):      ", data_save_dir)
println("spectral_method = $spectral_method")

# --- Locate Script 1's dataset ------------------------------------------------
fit_filename = "fitting_methods-$regime_system-$regime_name-show$showdim_tag.jld2"
save_path_fit = joinpath(data_save_dir, fit_filename)
if !isfile(save_path_fit)
    # Fallback: search DATA_ROOT for the file (tolerates a different
    # experiment_name choice in Script 1).  Deterministic: first sorted hit.
    hits = String[]
    isdir(DATA_ROOT) && for (root, _, files) in walkdir(DATA_ROOT)
        fit_filename in files && push!(hits, joinpath(root, fit_filename))
    end
    isempty(hits) &&
        error("Script 1 output not found: $save_path_fit " *
              "(and no '$fit_filename' anywhere under DATA_ROOT=$DATA_ROOT). " *
              "Run Koopman_fitting_methods.jl first or check experiment_name.")
    save_path_fit = sort!(hits)[1]
    data_save_dir = dirname(save_path_fit)
    @warn "experiment_name mismatch — using Script 1 file found at $save_path_fit"
end
println("Script 1 dataset: $save_path_fit")

save_path_out = joinpath(data_save_dir,
    "spectral_analysis-$regime_system-$regime_name-show$showdim_tag-$spectral_method.jld2")
println("Output dataset:   $save_path_out")
print_mem("start")

# ============================================================================
# 1. LOAD SCRIPT 1 DATA (metadata + selected method + spectral table)
# ============================================================================

println("\nLoading Script 1 metadata and the '$spectral_method' operator...")

# Top-level metadata (per SHARED_BRIEF data contract) + the lightweight
# per-method analysis results and the cross-method spectral table.
s1 = jldopen(save_path_fit, "r") do f
    Dict(
        # metadata
        :experiment_name => get(f, "experiment_name", experiment_name),
        :dim             => f["dim"],
        :showdim         => f["showdim"],
        :n_show          => f["n_show"],
        :n_hidden        => f["n_hidden"],
        :n_state         => f["n_state"],
        :n_plot          => f["n_plot"],
        :phys_idx        => f["phys_idx"],
        :P_plot          => f["P_plot"],
        :plot_labels     => f["plot_labels"],
        :dt_internal     => f["dt_internal"],
        :nLag            => f["nLag"],
        :dt_eff          => f["dt_eff"],
        :m_embed         => f["m_embed"],
        :tau_delay       => f["tau_delay"],
        :method_ids      => f["method_ids"],
        :SEED_fit        => get(f, "SEED", SEED),
        # per-method lightweight results + spectral summary
        :analysis_results => f["analysis_results"],
        :spectral_table   => f["spectral_table"],
    )
end

# --- Consistency assertions (feature 1) ---------------------------------------
@assert s1[:dim] == dim "dim mismatch: Script 1 used dim=$(s1[:dim]), configured dim=$dim"
@assert s1[:showdim] == showdim "showdim mismatch: Script 1 used $(s1[:showdim]), configured $showdim"
@assert s1[:n_show] == n_show "n_show mismatch"
@assert s1[:n_plot] == n_plot "n_plot mismatch"

dim        = s1[:dim]
showdim    = s1[:showdim]
n_state    = s1[:n_state]
n_plot     = s1[:n_plot]
phys_idx   = s1[:phys_idx]
P_plot     = s1[:P_plot]
plot_labels = s1[:plot_labels]
dt_eff     = s1[:dt_eff]
dt_internal = s1[:dt_internal]
nLag       = s1[:nLag]
m_embed    = s1[:m_embed]
tau_delay  = s1[:tau_delay]
method_ids = s1[:method_ids]
spectral_table = s1[:spectral_table]
analysis_results = s1[:analysis_results]

@assert spectral_method in method_ids ||
        string(spectral_method) in string.(method_ids) ||
        haskey(analysis_results, spectral_method) ||
        haskey(analysis_results, string(spectral_method)) ||
        haskey(analysis_results, Symbol(spectral_method)) """
    spectral_method="$spectral_method" is not among Script 1's methods:
    $(method_ids).  Set the top-level `spectral_method` variable accordingly."""

# Selected method's lightweight results (K, B_full, B_reduced, U_r, dict_info,
# cfg, λ, Ξ — heavy fields were offloaded to fit/<mid>/ by Script 1).
res = get_entry(analysis_results, spectral_method)

K         = res.K
B_reduced = res.B_reduced
U_r       = res.U_r
dict_info = res.dict_info
cfg_fit   = res.cfg
println("  Fit config: dict_type=$(cfg_fit.dict_type), r=$(cfg_fit.r), " *
        "use_svd=$(cfg_fit.use_svd), edmd_alpha=$(cfg_fit.edmd_alpha), " *
        "dict_params=$(cfg_fit.dict_params)")
λ         = res.λ
Ξ         = res.Ξ

(isnothing(λ) || isnothing(Ξ)) &&
    error("Selected method '$spectral_method' has no cached spectrum (λ/Ξ) in " *
          "analysis_results — Script 1 must run spectrum() for every method.")

# --- Plotting-space assertions -------------------------------------------------
@assert size(P_plot, 1) == n_plot "P_plot has $(size(P_plot,1)) rows, expected n_plot=$n_plot"
@assert size(P_plot, 2) == n_state "P_plot has $(size(P_plot,2)) columns, expected n_state=$n_state"
@assert length(plot_labels) >= n_plot "plot_labels too short"
@assert length(phys_idx) == n_show "phys_idx must have n_show=$n_show entries"
@assert n_state == n_show * m_embed "n_state=$n_state != n_show*m_embed=$(n_show*m_embed)"
# phys_idx points at the current-time shown coords of the delay state
# (build_hankel_multichannel places the LATEST sample first in each block).
@assert phys_idx == [(j - 1) * m_embed + 1 for j in 1:n_show] "phys_idx inconsistent with m_embed"

println("  n_state = $n_state  (n_show=$n_show × m_embed=$m_embed),  n_plot = $n_plot")
println("  phys_idx = $phys_idx")
println("  plot_labels = $plot_labels")
println("  dt_eff = $dt_eff s  (dt_internal=$dt_internal, nLag=$nLag),  tau_delay=$tau_delay")
println("  Operator: K $(size(K)),  $(length(λ)) eigenvalues,  " *
        "dict type = $(dict_info.type)")

# Helper for projecting delay states to plotting coords (used by all figures)
plot_coords(S::AbstractMatrix) = P_plot * S
plot_coords(s::AbstractVector) = P_plot * s

print_mem("after loading Script 1 data")

# ============================================================================
# DICTIONARY CHECKS + LIFT/GRADIENT SPACE MACHINERY
# ============================================================================
# grad_phi and find_zls_gradient_descent are THINPLATE-RBF ONLY
# (API_REFERENCE GOTCHAS #3: grad_Psi_RBF has no kernel_type/sigma kwargs and
# would silently compute wrong gradients for gaussian dictionaries).  Fail
# loudly here instead of producing garbage ZLS.

dict_info.type == :rbf || error(
    "spectral_method = \"$spectral_method\" uses dictionary type " *
    ":$(dict_info.type).  The ZLS machinery (grad_phi, find_zls_gradient_descent) " *
    "requires a thin-plate RBF dictionary (:rbf).  Choose an EDMD-family method " *
    "(e.g. \"Hankel_EDMD\") — Script 1's default dictionary is thin-plate RBF.")

const _kernel_type = get(dict_info, :kernel_type, :thinplate)
_kernel_type == :thinplate || error(
    "spectral_method = \"$spectral_method\" uses RBF kernel_type = $_kernel_type. " *
    "grad_phi / find_zls_gradient_descent support ONLY :thinplate " *
    "(grad_Psi_RBF is thinplate-only, API_REFERENCE GOTCHAS #3).  Re-fit the " *
    "method with dict_params.kernel_type = :thinplate or pick another method.")

get(dict_info, :norm_stats, nothing) === nothing || error(
    "spectral_method = \"$spectral_method\" uses a normalized RBF dictionary " *
    "(dict_info.norm_stats present).  grad_phi does NOT apply the " *
    "de-normalization, so gradients would be inconsistent with Psi_slice.  " *
    "Re-fit with normalize=false (Script 1's default).")

# Native dimension of the dictionary (== reduced dim r for SVD-truncated
# Hankel-EDMD, == n_state for full-state / state-space EDMD).
n_dict = size(dict_info.centers, 1)
_Ur_str = U_r === nothing ? "nothing" :
          (U_r isa UniformScaling ? "I (use_svd=false)" : string(size(U_r)))
println("  Dictionary space dimension: n_dict = $n_dict " *
        "(n_state = $n_state, U_r = $(_Ur_str))")

# Dictionary lift acting on a BATCH of dictionary-space states.
# Psi_slice with default active_dims = 1:n_dict reduces to the plain
# dictionary evaluation (full_dim is auto-inferred from the RBF centers) and
# — unlike lift_state — handles dict_params consistently (GOTCHAS #12).
Psi_method(X::AbstractMatrix) = Psi_slice(X, dict_info)

# --- Maps between delay space (n_state) and dictionary space (n_dict) --------
# The fit convention (hankel_edmd, use_svd=true): dictionary input is
# x_r = U_r' * s  and the delay state reconstructs as s = U_r * x_r.
# U_r has orthonormal columns, so the round trip x_r -> s -> x_r is exact.
# With use_svd=false (U_r = I) or state-space EDMD (U_r = nothing) the two
# spaces coincide and both maps are the identity.
to_dict_space(S::AbstractMatrix) =
    size(S, 1) == n_dict ? Matrix(S) :
    (isnothing(U_r) ? error("Cannot map $(size(S,1))-D states to the " *
                            "$n_dict-D dictionary space: U_r is nothing.") :
                     Matrix(U_r)' * S)
to_delay_space(X::AbstractMatrix) =
    size(X, 1) == n_state ? Matrix(X) :
    (isnothing(U_r) ? error("Cannot map $(size(X,1))-D dictionary states to " *
                            "the $n_state-D delay space: U_r is nothing.") :
                     Matrix(U_r) * X)

# ============================================================================
# RECONSTRUCT SYSTEM PARAMETERS (for limit cycle integration)
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

x0_silent = fixed_point(regime_system, p_det)
println("Silent-state fixed point: x0 = ", round.(x0_silent, digits=4))

# ============================================================================
# PHASE/AMPLITUDE MODE SELECTION (needed early: j_phase sets the LC period)
# ============================================================================
# j_phase comes from the package's select_phase_amplitude_modes (5-tuple
# return, see API_REFERENCE); fall back to Script 1's spectral_table entry if
# the selection fails, and to 0 (= autocorrelation fallback below) otherwise.

j_phase = 0
mu_phase = NaN + 0.0im
j_amp = 0
mu_amp = NaN + 0.0im
branch_idx = Int[]

try
    global j_phase, mu_phase, j_amp, mu_amp, branch_idx =
        select_phase_amplitude_modes(λ, dt_eff;
            TOL_UNITY=1e-4, TOL_IMAG=1e-4, TOL_HARMONIC=0.05, tol_growth=1.0)
    println("Phase mode:      j=$j_phase,  λ=$(round(λ[j_phase], digits=6))")
    println("Amplitude mode:  j=$j_amp,  λ=$(round(λ[j_amp], digits=6))")
catch e
    @warn "select_phase_amplitude_modes failed: $e"
    entry = get_entry(spectral_table, spectral_method)
    if get(entry, :j_phase, 0) > 0
        global j_phase = entry.j_phase
        global mu_phase = log(λ[j_phase]) / dt_eff
        println("  Falling back to Script 1's j_phase=$j_phase " *
                "(λ=$(round(λ[j_phase], digits=6)))")
    else
        @warn "No phase eigenvalue available — limit-cycle period will be " *
              "estimated from the test-trajectory autocorrelation."
    end
end

# ============================================================================
# 2. PREPARE OUTPUT JLD2 + START SNAPSHOTS FOR GRADIENT DESCENT
# ============================================================================
# The output file is created up-front with all lightweight metadata; heavy
# results (per-eigenvalue ZLS, LC, POD, phase) are appended incrementally so
# an interrupted run can resume (SHARED_BRIEF rules i + contract).

if !isfile(save_path_out)
    jldopen(save_path_out, "w") do f
        f["regime_system"]   = regime_system
        f["regime_name"]     = regime_name
        f["showdim_tag"]     = showdim_tag
        f["experiment_name"] = experiment_name
        f["spectral_method"] = spectral_method
        f["SEED"]            = SEED
        # Selected operator + plotting machinery (data contract)
        f["lambda"]      = λ
        f["Xi"]          = Ξ
        f["dict_info"]   = dict_info
        f["K"]           = K
        f["B_reduced"]   = B_reduced
        f["U_r"]         = U_r
        f["P_plot"]      = P_plot
        f["plot_labels"] = plot_labels
        f["dim"]         = dim
        f["n_state"]     = n_state
        f["showdim"]     = showdim
        f["phys_idx"]    = phys_idx
        f["dt_eff"]      = dt_eff
        f["dt_internal"] = dt_internal
        f["nLag"]        = nLag
        f["m_embed"]     = m_embed
        f["tau_delay"]   = tau_delay
        f["script_complete"] = false
    end
    println("Output JLD2 initialized: $save_path_out")
else
    println("Output JLD2 exists — resuming: $save_path_out")
    # Sanity: refuse to mix results from different configurations.
    jldopen(save_path_out, "r") do f
        if haskey(f, "spectral_method")
            @assert read(f, "spectral_method") == spectral_method """
                $save_path_out belongs to spectral_method="$(read(f, "spectral_method"))".
                Delete it or fix the `spectral_method` variable."""
        end
    end
end

# --- Start snapshots (dictionary space) for find_zls_gradient_descent -------
# First positional argument of find_zls_gradient_descent is the FIT snapshot
# matrix whose columns seed the multi-start descent.  We reload the heavy
# fields Script 1 offloaded under fit/<spectral_method>/ (X_r_full = parent of
# the reduced-state view, S = training Hankel matrix) and fall back to a fresh
# delay embedding of the shown training block X_train.
println("\nLoading gradient-descent start snapshots (dictionary space)...")

S_starts = jldopen(save_path_fit, "r") do f
    if haskey(f, "fit/$spectral_method/X_r_full")
        Xr_full = read(f, "fit/$spectral_method/X_r_full")
        println("  fit/$spectral_method/X_r_full: $(size(Xr_full)) " *
                "($(fmt_bytes(sizeof(Xr_full))))")
        Xr_full[:, 1:end-1]   # drop last column → X_r (offload contract)
    elseif haskey(f, "fit/$spectral_method/X_r")
        read(f, "fit/$spectral_method/X_r")
    elseif haskey(f, "fit/$spectral_method/S")
        read(f, "fit/$spectral_method/S")
    else
        nothing
    end
end

if isnothing(S_starts)
    println("  No heavy fit group for '$spectral_method' — delay-embedding " *
            "the shown training block X_train instead.")
    X_train_shown = jldopen(f -> read(f, "X_train"), save_path_fit, "r")
    S_starts = build_hankel_multichannel(X_train_shown, m_embed, tau_delay)
    X_train_shown = nothing
    GC.gc()
end

S_starts = to_dict_space(S_starts)
@assert size(S_starts, 1) == n_dict """
    Start snapshots have $(size(S_starts,1)) rows but the dictionary lives in
    $n_dict dimensions — inconsistent fit/$spectral_method heavy fields."""
println("  Start snapshots (dict space): $(size(S_starts)) " *
        "($(fmt_bytes(sizeof(S_starts))))")
print_mem("after loading start snapshots")

# ============================================================================
# 3. ZLS FOR ALL REAL EIGENVALUES INSIDE THE UNIT CIRCLE
# ============================================================================
# Selection identical to zls_full: purely real (|Im λ| < 1e-5), non-trivial
# (not the constant mode λ≈1), strictly inside the unit circle (|Re λ| < 1).

const TOL_IMAG_REAL = 1e-5
real_eig_indices = findall(x -> abs(imag(x)) < TOL_IMAG_REAL &&
                                abs(real(x) - 1.0) > 1e-4 &&
                                abs(real(x)) < 1.0, λ)
println("\n" * "="^60)
println("Found $(length(real_eig_indices)) real eigenvalues inside the unit circle:")
println("="^60)
for j in real_eig_indices
    println("  j=$j,  λ=$(round(real(λ[j]), digits=6))")
end

jldopen(save_path_out, "a+") do f
    jld_write!(f, "real_eig_indices", real_eig_indices; overwrite=true)
end

# All ZLS kept in DELAY space (n_state × n_j per eigenvalue) — the canonical
# storage of the data contract; dictionary-space points are recovered exactly
# via to_dict_space when gradients/eigenfunctions are evaluated.
all_zls_results = Dict{Int, Matrix{Float64}}()

println("\nComputing ZLS for all real eigenvalues " *
        "(find_zls_gradient_descent, n_starts=100000, n_iter=500, lr=0.1, tol=1e-18)...")
println("  Each call is seeded with seed=SEED=$SEED: the package re-seeds the")
println("  global RNG before start sampling, so every eigenvalue's ZLS is")
println("  reproducible independently of execution order / resumed runs.")

p_zls = Progress(length(real_eig_indices); desc="Computing ZLS", dt=1.0,
                 barglyphs=BarGlyphs("[=> ]"), barlen=40)

for j in real_eig_indices
    key = "zls/j_$j"
    resumed = jldopen(save_path_out, "r") do f
        haskey(f, key) ? read(f, key) : nothing
    end

    if !isnothing(resumed)
        # Resumable: this eigenvalue's ZLS is already on disk.
        all_zls_results[j] = resumed
        next!(p_zls; showvalues=[("Current j", j), ("λ", round(λ[j], digits=4)),
                                 ("ZLS pts", size(resumed, 2)), ("status", "loaded from disk")])
        continue
    end

    zls_pts_dict = find_zls_gradient_descent(
        S_starts,            # dictionary-space fit snapshots (start points)
        Psi_method,          # Psi_slice-based lift for this method's dict_info
        Ξ,
        j,
        dict_info;
        n_starts = 100000,
        n_iter   = 500,
        lr       = 0.1,
        tol      = 1e-18,
        verbose  = false,
        seed     = SEED,     # seeds the global RNG for start sampling (rule ii)
    )

    # Map the converged dictionary-space points back to the delay state.
    zls_matrix = isempty(zls_pts_dict) ?
                 Matrix{Float64}(undef, n_state, 0) :
                 to_delay_space(reduce(hcat, zls_pts_dict))
    all_zls_results[j] = zls_matrix

    # Incremental save (resumability — SHARED_BRIEF rule i).
    jldopen(save_path_out, "a+") do f
        f[key] = zls_matrix
    end

    next!(p_zls; showvalues=[("Current j", j), ("λ", round(λ[j], digits=4)),
                             ("ZLS pts", size(zls_matrix, 2))])
end
finish!(p_zls)
GC.gc()
print_mem("after ZLS extraction")

# ============================================================================
# CLEAN LIMIT CYCLE (reference trajectory for all figures)
# ============================================================================
# Machinery from zls_full: period from the phase eigenvalue (fallback:
# autocorrelation of the test trajectory), 10-period burn-in, then exactly
# max_lag delay-fill + 2 periods of rk4 integration, and delay embedding of
# the shown coordinates with build_hankel_multichannel.

println("\n--- Computing clean limit cycle ---")

if j_phase > 0
    ω_lc = abs(imag(log(λ[j_phase])) / dt_eff)
    T_lc = 2π / ω_lc
    println("  Phase eigenvalue: λ = $(round(λ[j_phase], digits=6))")
    println("  Period T = $(round(T_lc, digits=4)) s  (f = $(round(1/T_lc, digits=4)) Hz)")
else
    @warn "No phase eigenvalue — estimating period from X_test_raw autocorrelation"
    X_test_raw = jldopen(f -> read(f, "X_test_raw"), save_path_fit, "r")
    v_test = X_test_raw[1, :]
    X_test_raw = nothing
    n_ac = min(2000, length(v_test) - 1)
    ac = [sum((v_test[1:end-k] .- mean(v_test)) .* (v_test[1+k:end] .- mean(v_test)))
          for k in 0:n_ac]
    k_zero = findfirst(ac .< 0)
    T_lc = isnothing(k_zero) ? 20.0 : 2 * k_zero * dt_eff
    println("  Estimated period T ≈ $(round(T_lc, digits=4)) s")
end

n_burn_steps  = ceil(Int, 10 * T_lc / dt_eff)
n_lc_steps    = ceil(Int, 2 * T_lc / dt_eff)     # exactly two periods
max_lag_local = (m_embed - 1) * tau_delay
@assert n_lc_steps > 0 "Limit-cycle window is empty (T_lc=$T_lc, dt_eff=$dt_eff)"

# Integrate max_lag_local EXTRA samples before the limit-cycle window so the
# delay embedding can be built without shortening X_lc.  The extra samples
# are absorbed into the delay buffer and do NOT grow the number of plotted
# columns (which stays at n_lc_steps).
x0_lc = x0_silent .+ ones(dim)
println("  Integrating burn-in ($n_burn_steps steps ≈ $(round(n_burn_steps*dt_eff, digits=1)) s) " *
        "+ $max_lag_local delay-fill + $n_lc_steps limit-cycle steps...")

X_lc_full = rk4(f_det, x0_lc, dt_internal,
                n_burn_steps + max_lag_local + n_lc_steps; nLag=nLag)
# Drop burn-in; keep max_lag_local + n_lc_steps columns
X_lc = X_lc_full[:, n_burn_steps+1:end]
X_lc_full = nothing

println("  Clean limit cycle: $(size(X_lc, 2)) snapshots  " *
        "($(round(size(X_lc,2)*dt_eff, digits=2)) s ≈ $(round(size(X_lc,2)*dt_eff/T_lc, digits=2)) periods)")

# Build delay state from shown coords.  The first max_lag_local columns are
# consumed by the delay buffer; the remaining n_lc_steps columns are the
# delay-embedded limit cycle used for plotting.
S_lc_shown = build_hankel_multichannel(X_lc[showdim, :], m_embed, tau_delay)
@assert size(S_lc_shown, 1) == n_state "S_lc_shown has $(size(S_lc_shown,1)) rows, expected n_state=$n_state"
@assert size(S_lc_shown, 2) >= n_lc_steps "max_lag check failed: delay embedding " *
    "returned $(size(S_lc_shown,2)) columns, expected >= n_lc_steps=$n_lc_steps"

lc_plot = plot_coords(S_lc_shown)            # n_plot × n_lc_steps
lc_len  = size(lc_plot, 2)
lc_x = lc_plot[1, :]
lc_y = n_plot >= 2 ? lc_plot[2, :] : zeros(lc_len)
lc_z = n_plot >= 3 ? lc_plot[3, :] : zeros(lc_len)
println("  Delay-embedded LC: $lc_len points (n_lc_steps preserved, delay buffer absorbed)")

jldopen(save_path_out, "a+") do f
    jld_write!(f, "S_lc_shown", S_lc_shown; overwrite=true)
    jld_write!(f, "X_lc", X_lc; overwrite=true)
end

X_lc = nothing
GC.gc()
print_mem("after limit cycle computation")

# ============================================================================
# 4. HUGE FIGURE — ZLS OF ALL REAL EIGENVALUES
# ============================================================================
println("\nGenerating huge figure (ZLS of all real eigenvalues)...")

# Filter out eigenfunctions that yielded no ZLS points
valid_js = sort!(filter(j -> !isempty(all_zls_results[j]) &&
                             size(all_zls_results[j], 2) > 0,
                        collect(keys(all_zls_results))))
n_plots  = length(valid_js)

# Plot-space training coordinates for grid axes (tiny) — only needed when the
# eigenfunction background grid is drawn.
X_train_plot = nothing

if n_plots == 0
    println("No zero-level sets found!")
else
    # Eigenfunction background grid + contour only when the EDMD state space
    # coincides with the plotting space (no delay embedding / degenerate case
    # n_show·m_embed == dim).  Grid axes are the first two PLOT-space
    # coordinates, i.e. rows of P_plot * s.
    do_grid_eval = (n_state == n_plot) && (n_dict == n_plot)
    do_grid = do_grid_eval && (dim == 2)
    ngrid = 101  # grid resolution (used only when do_grid)
    if do_grid
        X_train_shown = jldopen(f -> read(f, "X_train"), save_path_fit, "r")
        X_train_plot = plot_coords(build_hankel_multichannel(X_train_shown, m_embed, tau_delay)[:, 1:50:end])
        X_train_shown = nothing
        GC.gc()

        ngrid = 101
        lo1, hi1 = extrema(X_train_plot[1, :])
        lo2, hi2 = extrema(X_train_plot[2, :])
        g1 = range(lo1, hi1; length=ngrid)
        g2 = range(lo2, hi2; length=ngrid)
        Xg1 = (ones(ngrid)' .* g1)'   # grid of plotting coord 1
        Xg2 = (g2' .* ones(ngrid))'   # grid of plotting coord 2
        φ_grid = zeros(ngrid, ngrid, n_plots)
    end

    # Grid size (as square as possible)
    n_cols = ceil(Int, sqrt(n_plots))
    n_rows = ceil(Int, n_plots / n_cols)

    fig = Figure(size = (n_cols * 450, n_rows * 450))
    Label(fig[0, 1:n_cols], "Zero Level Sets for All Real Eigenfunctions — $spectral_method",
          fontsize=24, font=:bold)

    # Sort by eigenvalue magnitude for a cleaner layout
    sorted_valid_js = sort(valid_js, by=j -> abs(λ[j]))

    for (idx, j) in enumerate(sorted_valid_js)
        row = ceil(Int, idx / n_cols)
        col = ((idx - 1) % n_cols) + 1

        n_pts_j = size(all_zls_results[j], 2)
        title_j = "j=$j, λ=$(round(real(λ[j]), digits=4))\n$n_pts_j pts"

        # Plot-space projection of this eigenfunction's ZLS; subsample for the
        # scatter with a DEDICATED seeded stream (rule ii) — independent of the
        # global RNG state left behind by find_zls_gradient_descent.
        pts_plot = plot_coords(all_zls_results[j])                 # n_plot × n_pts_j
        idx_fig = seeded_subsample_idx(n_pts_j, MAX_FIG_PTS, 0x10 ⊻ j)
        pts_plot = pts_plot[:, idx_fig]

        if dim == 3
            ax = Axis3(fig[row, col];
                title = title_j, titlegap = 5,
                xlabel = plot_labels[1], ylabel = plot_labels[2], zlabel = plot_labels[3],
                xlabelsize=10, ylabelsize=10, zlabelsize=10, titlesize=12)

            scatter!(ax, pts_plot[1, :], pts_plot[2, :],
                     n_plot >= 3 ? pts_plot[3, :] : zeros(size(pts_plot, 2));
                     color=(:orange, 0.6), markersize=4)
            lines!(ax, lc_x, lc_y, lc_z, color=(COLOR_PRIMARY, 1.0), linewidth=1.5)

        else # dim == 2
            ax = Axis(fig[row, col];
                title = title_j, titlegap = 5,
                xlabel = plot_labels[1], ylabel = plot_labels[2],
                xlabelsize=10, ylabelsize=10, titlesize=12)

            if do_grid
                for i in 1:ngrid
                    φ_grid[i, :, idx] .= vec(real.(Ξ[:, j]') *
                        Psi_method([Xg1[i, :]'; Xg2[i, :]']))
                end
                v = @view φ_grid[:, :, idx]
                scatter!(ax, vec(Xg1), vec(Xg2),
                         color = vec(normalize_vector(v)), colorrange = (0, 1),
                         markersize = 5, colormap = :thermal)
                contour!(ax, Xg1, Xg2, φ_grid[:, :, idx], levels=[0.0],
                         color=:white, linewidth=1.5)
            end

            scatter!(ax, pts_plot[1, :], pts_plot[2, :],
                     color=COLOR_ACCENT, strokecolor=:black, strokewidth=0.5,
                     markersize=6)
            lines!(ax, lc_x, lc_y, color=(:black, 1.0), linestyle=:dash, linewidth=1.5)
        end
    end

    save_path_fig = joinpath(save_dir, "ZLS_All_Real_Eigenvalues.png")
    save(save_path_fig, fig)
    println("Saved huge figure to: $save_path_fig")
    fig = nothing
    X_train_plot = nothing
    GC.gc()
end

jldopen(save_path_out, "a+") do f
    jld_write!(f, "valid_js", valid_js; overwrite=true)
end
print_mem("after huge figure")

# ============================================================================
# 5. TRANSVERSALITY ANALYSIS (find the best pair to intersect)
# ============================================================================
# For every pair of eigenfunctions with non-empty ZLS, the mean angle between
# ∇φ_j1 and ∇φ_j2 along the (subsampled) limit cycle — computed with the
# package's grad_phi (thinplate RBF, checked above).  Gradients live in the
# dictionary space, so the delay-embedded LC samples are mapped through
# to_dict_space first (exact round trip for SVD-reduced fits).

println("\n" * "="^60)
println("Computing Transversality Matrix for all real eigenfunction pairs...")
println("="^60)

# We use the delay-embedded limit cycle S_lc_shown as our reference trajectory
# Sample every 10th point to keep computation fast
S_lc_sample = to_dict_space(S_lc_shown[:, 1:10:end])
n_lc_pts = size(S_lc_sample, 2)

n_valid = length(valid_js)
angle_matrix = Matrix{Float64}(undef, n_valid, n_valid)
fill!(angle_matrix, NaN)

if n_valid >= 2
    # Pre-compute ALL gradients along the limit cycle once (same values as
    # zls_full's per-row caching, without the O(n_valid²) recomputation).
    println("  Pre-computing gradients for $n_valid eigenfunctions × $n_lc_pts LC points...")
    grads_all = Vector{Matrix{Float64}}(undef, n_valid)
    for (idx, j) in enumerate(valid_js)
        ξj = Ξ[:, j]
        grads_all[idx] = reduce(hcat, grad_phi(S_lc_sample[:, i], ξj, dict_info)
                                for i in 1:n_lc_pts)
    end

    p_angle = Progress(n_valid; desc="Computing angles", dt=1.0,
                       barglyphs=BarGlyphs("[=> ]"), barlen=40)
    for idx1 in 1:n_valid
        G1 = grads_all[idx1]
        for idx2 in 1:n_valid
            idx1 == idx2 && continue
            G2 = grads_all[idx2]
            angles = Float64[]
            sizehint!(angles, n_lc_pts)
            for i in 1:n_lc_pts
                g1 = @view G1[:, i]
                g2 = @view G2[:, i]
                norm1 = norm(g1)
                norm2 = norm(g2)
                # Avoid division by zero if gradient vanishes
                if norm1 > 1e-8 && norm2 > 1e-8
                    cos_theta = abs(dot(g1, g2)) / (norm1 * norm2)
                    # Clamp to [0, 1] to avoid numerical errors with acos
                    cos_theta = clamp(cos_theta, 0.0, 1.0)
                    push!(angles, acos(cos_theta) * 180 / π)  # degrees
                end
            end
            if !isempty(angles)
                angle_matrix[idx1, idx2] = mean(angles)
            end
        end
        next!(p_angle)
    end
    finish!(p_angle)
    grads_all = nothing
    GC.gc()
else
    @warn "Fewer than 2 eigenfunctions with ZLS points — transversality matrix is empty."
end

# Find the pair with the maximum average angle (closest to 90 degrees)
println("\nTop most transverse pairs (closest to 90°):")

# Extract upper triangle pairs and their angles
pair_angles = NamedTuple{(:j1, :j2, :angle), Tuple{Int, Int, Float64}}[]
for i in 1:n_valid, j in i+1:n_valid
    if !isnan(angle_matrix[i, j])
        push!(pair_angles, (j1=valid_js[i], j2=valid_js[j], angle=angle_matrix[i, j]))
    end
end

# Sort by angle descending
sort!(pair_angles, by=x -> x.angle, rev=true)

for (rank, pair) in enumerate(pair_angles[1:min(50, length(pair_angles))])
    println("  Rank $rank: j=$(pair.j1) (λ=$(round(real(λ[pair.j1]), digits=4))) & " *
            "j=$(pair.j2) (λ=$(round(real(λ[pair.j2]), digits=4))) -> Avg Angle = $(round(pair.angle, digits=2))°")
end

# Save the matrix for reference
jldopen(save_path_out, "a+") do f
    jld_write!(f, "angle_matrix", angle_matrix; overwrite=true)
    jld_write!(f, "valid_js", valid_js; overwrite=true)
end

# --- Heatmap figure of the transversality angle matrix -----------------------
if n_valid >= 2 && !isempty(pair_angles)
    fig = Figure(size=(900, 750))
    Label(fig[0, :], "Transversality angle between ZLS gradients — $spectral_method",
          fontsize=16, font=:bold)
    ax = Axis(fig[1, 1];
        title = "Mean angle between ∇φ_j1 and ∇φ_j2 along the limit cycle (degrees)",
        xlabel = "eigenvalue index j", ylabel = "eigenvalue index j",
        xticks = (1:n_valid, string.(valid_js)),
        yticks = (1:n_valid, string.(valid_js)),
        xticklabelsize = 8, yticklabelsize = 8, titlesize = 11,
        aspect = DataAspect())
    hm = heatmap!(ax, 1:n_valid, 1:n_valid, angle_matrix;
                  colormap = :viridis, colorrange = (0, 90), nan_color = :lightgray)
    Colorbar(fig[1, 2], hm; label = "mean angle (°)", width = 15)
    save_path_fig = joinpath(save_dir, "ZLS_Transversality_Angle_Matrix.png")
    save(save_path_fig, fig)
    println("Saved transversality heatmap to: $save_path_fig")
    fig = nothing
    GC.gc()
end
print_mem("after transversality analysis")

# ============================================================================
# 6. BEST-PAIR JOINT INTERSECTION
# ============================================================================

if isempty(pair_angles)
    @warn "No valid transverse pair — skipping joint ZLS intersection."
    jldopen(save_path_out, "a+") do f
        jld_write!(f, "best_pair", nothing; overwrite=true)
        jld_write!(f, "joint_zls", Matrix{Float64}(undef, n_state, 0); overwrite=true)
    end
else
    # Take the best pair from the output above
    best_pair = pair_angles[1]
    j1_best = best_pair.j1
    j2_best = best_pair.j2

    println("\nRunning joint gradient descent for best pair: j=$j1_best and j=$j2_best...")

    joint_zls_dict = find_zls_gradient_descent(
        S_starts,
        Psi_method,
        Ξ,
        [j1_best, j2_best],   # minimize φ_1² + φ_2²
        dict_info;
        n_starts = 100000,
        n_iter   = 500,
        lr       = 0.1,
        tol      = 1e-12,
        verbose  = false,
        seed     = SEED,      # seeds the global RNG for start sampling (rule ii)
    )

    # Data contract: joint_zls stored as an n_state × n matrix (delay space).
    joint_zls = isempty(joint_zls_dict) ?
                Matrix{Float64}(undef, n_state, 0) :
                to_delay_space(reduce(hcat, joint_zls_dict))
    println("  Joint ZLS: $(size(joint_zls, 2)) points")

    jldopen(save_path_out, "a+") do f
        jld_write!(f, "joint_zls", joint_zls; overwrite=true)
        jld_write!(f, "best_pair", best_pair; overwrite=true)
    end

    # Plot the intersection (dim-aware: 3-D view only when dim == 3)
    idx_fig = seeded_subsample_idx(size(joint_zls, 2), MAX_FIG_PTS, 0x20)  # seeded (rule ii)
    pts_pd = plot_coords(joint_zls[:, idx_fig])

    fig = Figure(size = (800, 600))

    if dim == 3
        ax = Axis3(fig[1, 1];
            title = "1D Limit Cycle Intersection (j=$j1_best ∩ j=$j2_best)\nAvg Angle: $(round(best_pair.angle, digits=1))°",
            xlabel = plot_labels[1], ylabel = plot_labels[2], zlabel = plot_labels[3])
        scatter!(ax, pts_pd[1, :], pts_pd[2, :],
                 n_plot >= 3 ? pts_pd[3, :] : zeros(size(pts_pd, 2));
                 color = :orange, markersize = 6, label = "Intersection ZLS")
        lines!(ax, lc_x, lc_y, lc_z, color = COLOR_PRIMARY, linewidth = 2,
               label = "True Limit Cycle")
        axislegend(ax)
    else
        ax = Axis(fig[1, 1];
            title = "Limit Cycle Intersection (j=$j1_best ∩ j=$j2_best)\nAvg Angle: $(round(best_pair.angle, digits=1))°",
            xlabel = plot_labels[1], ylabel = plot_labels[2])
        scatter!(ax, pts_pd[1, :], pts_pd[2, :],
                 color = :orange, markersize = 6, label = "Intersection ZLS")
        lines!(ax, lc_x, lc_y, color = COLOR_PRIMARY, linewidth = 2,
               label = "True Limit Cycle")
        axislegend(ax)
    end

    save(joinpath(save_dir, "Best_Pair_Intersection.png"), fig)
    println("Saved intersection figure.")
    fig = nothing
    pts_pd = nothing
    joint_zls_dict = nothing
    GC.gc()
end
print_mem("after best-pair intersection")

# ============================================================================
# 7. CROSS-METHOD SPECTRUM COMPARISON (all methods, selected highlighted)
# ============================================================================
# Script 1 also produces this figure; it is preserved here to document the
# spectral context of the SELECTED operator.  Two panels: the discrete
# spectrum (unit circle) and the continuous-time analogue μ = log(λ)/dt_eff
# (the package's hankel_dmd ignores dt, so the conversion is done manually —
# API_REFERENCE "WHAT THE PACKAGE LACKS" #4).  Point alpha encodes the
# per-method Koopman mode amplitude v_modes from the saved spectral_table.

println("\nBuilding cross-method spectrum comparison figure...")

method_palette = Makie.wong_colors()

fig = Figure(size = (1800, 800))
Label(fig[0, :], "Koopman spectra — all fitting methods " *
                 "(selected: $spectral_method)  |  $regime_system $regime_name",
      fontsize = 16, font = :bold)

θ_circle = range(0, 2π; length = 200)

ax1 = Axis(fig[1, 1];
    title = "Discrete spectrum (unit circle)",
    xlabel = "Re(λ)", ylabel = "Im(λ)", aspect = DataAspect())
lines!(ax1, cos.(θ_circle), sin.(θ_circle),
       color = (:black, 0.3), linewidth = 1, linestyle = :dash)
hlines!(ax1, 0.0, color = (:black, 0.2), linewidth = 0.5)
vlines!(ax1, 0.0, color = (:black, 0.2), linewidth = 0.5)

ax2 = Axis(fig[1, 2];
    title = "Continuous-time analogue  μ = log(λ)/dt_eff",
    xlabel = "Re(μ)", ylabel = "Im(μ)")
vlines!(ax2, 0.0, color = (:black, 0.3), linewidth = 1, linestyle = :dash)
hlines!(ax2, 0.0, color = (:black, 0.2), linewidth = 0.5)

# Non-selected methods first, selected method last (on top).
mids_sorted = vcat([m for m in method_ids if string(m) != string(spectral_method)],
                   [m for m in method_ids if string(m) == string(spectral_method)])

for (i, mid) in enumerate(mids_sorted)
    entry = get_entry(spectral_table, mid)
    λm = entry.λ
    vm = entry.v_modes
    is_sel = string(mid) == string(spectral_method)
    base_c = method_palette[mod1(findfirst(==(mid), method_ids), length(method_palette))]

    # Alpha by (normalized) Koopman mode amplitude, as in Script 1's figure.
    vmin, vmax = extrema(vm)
    span = max(vmax - vmin, 1e-12)
    cols = [Makie.RGBAf(base_c.r, base_c.g, base_c.b,
                        0.25 + 0.75 * (v - vmin) / span) for v in vm]

    msize = is_sel ? 9 : 5
    scatter!(ax1, real.(λm), imag.(λm); color = cols, markersize = msize,
             marker = is_sel ? :diamond : :circle,
             strokecolor = is_sel ? :black : :transparent,
             strokewidth = is_sel ? 0.4 : 0.0,
             label = is_sel ? "$mid  (SELECTED)" : string(mid))

    μm = log.(λm) ./ dt_eff
    scatter!(ax2, real.(μm), imag.(μm); color = cols, markersize = msize,
             marker = is_sel ? :diamond : :circle,
             strokecolor = is_sel ? :black : :transparent,
             strokewidth = is_sel ? 0.4 : 0.0,
             label = is_sel ? "$mid  (SELECTED)" : string(mid))

    # Star the selected method's phase eigenvalue in both panels.
    if is_sel && get(entry, :j_phase, 0) > 0
        jp = entry.j_phase
        scatter!(ax1, [real(λm[jp])], [imag(λm[jp])];
                 color = COLOR_ACCENT, markersize = 18, marker = :star5,
                 strokecolor = :black, strokewidth = 0.6,
                 label = "phase mode ($mid)")
        scatter!(ax2, [real(μm[jp])], [imag(μm[jp])];
                 color = COLOR_ACCENT, markersize = 18, marker = :star5,
                 strokecolor = :black, strokewidth = 0.6,
                 label = "phase mode ($mid)")
    end
end

Legend(fig[1, 3], ax1; labelsize = 10, framevisible = true)

save(joinpath(save_dir, "Cross_Method_Spectrum.png"), fig)
println("Saved: Cross_Method_Spectrum.png")
fig = nothing
GC.gc()

# ============================================================================
# 8. FULL POD PROJECTION OF THE ZLS POINT CLOUD
# ============================================================================
# Stack ALL real-eigenvalue ZLS delay-state points, subsample (seeded) to the
# memory cap, center, and compute the POD basis via eigen(Symmetric(Xc*Xc'))
# (BLAS syrk — the SHARED_BRIEF memory rule; never svd's Vt).  The FULL
# projection Z = U' * Xc (all rows) is stored under pod_zls/ for Script 3.

println("\n" * "="^60)
println("POD of the stacked ZLS point cloud")
println("="^60)

# Stack all ZLS delay-state points (columns), eigenvalue by eigenvalue.
zls_blocks = [all_zls_results[j] for j in valid_js]
n_zls_total = sum(size(b, 2) for b in zls_blocks)
println("  Stacked ZLS point cloud: $n_zls_total points in $n_state-D delay space")

if n_zls_total == 0
    @warn "No ZLS points at all — skipping POD and phase color maps."
    jldopen(save_path_out, "a+") do f
        jld_write!(f, "pod_zls/U", Matrix{Float64}(undef, n_state, 0); overwrite=true)
        jld_write!(f, "pod_zls/σ", Float64[]; overwrite=true)
        jld_write!(f, "pod_zls/Z", Matrix{Float64}(undef, 0, 0); overwrite=true)
        jld_write!(f, "pod_zls/x_mean", zeros(n_state); overwrite=true)
        jld_write!(f, "pod_zls/subsample_idx", Int[]; overwrite=true)
        jld_write!(f, "j_phase", j_phase; overwrite=true)
        jld_write!(f, "phase_eigenvalue", j_phase > 0 ? λ[j_phase] : NaN + 0.0im; overwrite=true)
        jld_write!(f, "φ_phase_zls", ComplexF64[]; overwrite=true)
        jld_write!(f, "θ_zls", Float64[]; overwrite=true)
    end
else
    S_zls_all = reduce(hcat, zls_blocks)          # n_state × n_zls_total

    # Seeded subsample (rule ii): dedicated MersenneTwister stream, identical
    # for POD, phase colormaps and the POD-spectrum figure ("consistent").
    idx_pod = seeded_subsample_idx(n_zls_total, MAX_ZLS_POD, 0x30)
    S_zls = S_zls_all[:, idx_pod]                 # n_state × n_zls
    n_zls = size(S_zls, 2)
    println("  POD uses $n_zls / $n_zls_total points (seeded subsample, cap $MAX_ZLS_POD)")

    # --- Center + covariance POD (memory rule: eigen of syrk, no Vt) ---------
    x_mean = vec(mean(S_zls; dims = 2))
    Xc = S_zls .- x_mean
    G = Xc * Xc'                                  # n_state × n_state (BLAS syrk)
    print_mem("after ZLS covariance (syrk)")
    E = eigen(Symmetric(G))
    G = nothing
    p_pod = sortperm(E.values; rev = true)
    U_zls = E.vectors[:, p_pod]                   # n_state × n_state POD basis
    σ_zls = sqrt.(max.(E.values[p_pod], 0.0))     # singular values
    E = nothing

    # FULL projection (all rows) — required by the data contract.
    Z_full = U_zls' * Xc                          # n_state × n_zls
    Xc = nothing
    println("  POD: U $(size(U_zls)),  Z $(size(Z_full))  ($(fmt_bytes(sizeof(Z_full))))")

    # --- POD spectrum analysis -------------------------------------------------
    energy = σ_zls .^ 2
    total_energy = sum(energy)
    cum_energy = cumsum(energy) ./ total_energy
    r99  = findfirst(>=(0.99),  cum_energy)
    r999 = findfirst(>=(0.999), cum_energy)
    r_num = count(σ_zls .> eps(Float64) * max(σ_zls[1], eps()) * n_state)
    println("  Effective rank: r(99%) = $(r99),  r(99.9%) = $(r999),  " *
            "numerical rank ≈ $r_num  (of $(length(σ_zls)) modes)")
    println("  First singular values: ", round.(σ_zls[1:min(8, length(σ_zls))], sigdigits=4))

    fig = Figure(size = (1400, 550))
    Label(fig[0, :], "POD spectrum of the ZLS point cloud — $spectral_method",
          fontsize = 16, font = :bold)
    n_sv_show = min(100, length(σ_zls))
    ax1 = Axis(fig[1, 1];
        title = "Singular values (first $n_sv_show, log scale)",
        xlabel = "mode index", ylabel = "σ (log₁₀)", yscale = log10)
    scatter!(ax1, 1:n_sv_show, σ_zls[1:n_sv_show]; color = COLOR_PRIMARY, markersize = 7)
    lines!(ax1, 1:n_sv_show, σ_zls[1:n_sv_show]; color = COLOR_PRIMARY, linewidth = 1.2)

    ax2 = Axis(fig[1, 2];
        title = "Cumulative energy  (r₉₉ = $r99,  r₉₉.₉ = $r999)",
        xlabel = "mode index", ylabel = "Σσ² / Σσ²",
        limits = (nothing, nothing, 0, 1.05))
    lines!(ax2, 1:n_sv_show, cum_energy[1:n_sv_show]; color = COLOR_ACCENT, linewidth = 2)
    scatter!(ax2, 1:n_sv_show, cum_energy[1:n_sv_show]; color = COLOR_ACCENT, markersize = 5)
    hlines!(ax2, [0.99, 0.999]; color = :gray, linewidth = 1, linestyle = :dash)

    save(joinpath(save_dir, "POD_ZLS_spectrum.png"), fig)
    println("Saved: POD_ZLS_spectrum.png")
    fig = nothing
    GC.gc()

    # --- Save the POD block incrementally ---------------------------------------
    jldopen(save_path_out, "a+") do f
        jld_write!(f, "pod_zls/U", U_zls; overwrite=true)
        jld_write!(f, "pod_zls/σ", σ_zls; overwrite=true)
        jld_write!(f, "pod_zls/Z", Z_full; overwrite=true)
        jld_write!(f, "pod_zls/x_mean", x_mean; overwrite=true)
        jld_write!(f, "pod_zls/subsample_idx", idx_pod; overwrite=true)
    end
    println("  Saved pod_zls/{U, σ, Z, x_mean, subsample_idx}")
    print_mem("after POD of ZLS")

    # ============================================================================
    # 9. PHASE COLOR MAP ON THE ZLS POINTS
    # ============================================================================
    # Evaluate the phase eigenfunction φ_phase(s) = Ξ[:, j_phase]' · Ψ(s) on
    # EVERY stacked ZLS point (chunked lifting to bound memory), then
    # θ = angle(φ).  Two figures (phase.jl PHASE-1/PHASE-8 style):
    #   (a) ZLS points colored by θ in the PLOT dimensions (P_plot, 3-D);
    #   (b) the same points in the FULL POD projection (first 3 rows of Z).

    if j_phase <= 0
        @warn "No phase eigenvalue — skipping phase color maps."
        jldopen(save_path_out, "a+") do f
            jld_write!(f, "j_phase", j_phase; overwrite=true)
            jld_write!(f, "phase_eigenvalue", NaN + 0.0im; overwrite=true)
            jld_write!(f, "φ_phase_zls", ComplexF64[]; overwrite=true)
            jld_write!(f, "θ_zls", Float64[]; overwrite=true)
        end
    else
        println("\nEvaluating phase eigenfunction (j_phase=$j_phase) on $n_zls_total ZLS points...")
        ξ_phase = Ξ[:, j_phase]
        phase_eigenvalue = λ[j_phase]

        # Dictionary-space coordinates of ALL ZLS points (exact round trip
        # for SVD-reduced fits), lifted in chunks to bound Ψ memory.
        S_zls_dict = to_dict_space(S_zls_all)
        chunk_sz = 5000
        φ_phase_zls = Vector{ComplexF64}(undef, n_zls_total)
        p_eval = Progress(ceil(Int, n_zls_total / chunk_sz); desc="Evaluating φ_phase",
                          dt=1.0, barglyphs=BarGlyphs("[=> ]"), barlen=30)
        for cs in 1:chunk_sz:n_zls_total
            ce = min(cs + chunk_sz - 1, n_zls_total)
            Ψc = Psi_method(S_zls_dict[:, cs:ce])
            φ_phase_zls[cs:ce] .= vec(ξ_phase' * Ψc)
            Ψc = nothing
            next!(p_eval)
        end
        finish!(p_eval)
        S_zls_dict = nothing
        GC.gc()

        θ_zls = angle.(φ_phase_zls)
        println("  |φ_phase| on ZLS: median $(round(median(abs.(φ_phase_zls)), sigdigits=4))  " *
                "(should be ≈ constant on the limit cycle)")
        println("  θ range: [$(round(minimum(θ_zls), digits=3)), $(round(maximum(θ_zls), digits=3))] rad")

        jldopen(save_path_out, "a+") do f
            jld_write!(f, "j_phase", j_phase; overwrite=true)
            jld_write!(f, "phase_eigenvalue", phase_eigenvalue; overwrite=true)
            jld_write!(f, "φ_phase_zls", φ_phase_zls; overwrite=true)
            jld_write!(f, "θ_zls", θ_zls; overwrite=true)
        end

        # --- Figure (a): ZLS colored by θ in the PLOT dimensions ---------------
        # Seeded subsample for the scatter (rule ii) — the same POD subset is
        # reused so both phase figures show the SAME points.
        idx_fig = seeded_subsample_idx(n_zls_total, MAX_FIG_PTS, 0x30)
        ZLS_plot = plot_coords(S_zls_all[:, idx_fig])     # n_plot × n_fig

        fig = Figure(size = (1000, 800))
        Label(fig[0, :], "ZLS points colored by Koopman phase θ — $spectral_method",
              fontsize = 16, font = :bold)
        ax = Axis3(fig[1, 1];
            title = "Phase θ = arg(φ_phase),  j_phase=$j_phase,  " *
                    "λ = $(round(phase_eigenvalue, digits=5))",
            xlabel = plot_labels[1], ylabel = plot_labels[2],
            zlabel = n_plot >= 3 ? plot_labels[3] : "")
        hm = scatter!(ax, ZLS_plot[1, :], ZLS_plot[2, :],
                      n_plot >= 3 ? ZLS_plot[3, :] : zeros(length(idx_fig));
                      color = θ_zls[idx_fig], colormap = :thermal,
                      markersize = 6, colorrange = (-π, π))
        Colorbar(fig[1, 2], hm; label = "phase θ", width = 12)
        save(joinpath(save_dir, "ZLS_phase_colormap_plot.png"), fig)
        println("Saved: ZLS_phase_colormap_plot.png")
        fig = nothing
        ZLS_plot = nothing
        GC.gc()

        # --- Figure (b): ZLS colored by θ in the FULL POD projection ------------
        # First 3 POD coordinates of the SAME subsampled ZLS points.  Z_full's
        # column i corresponds to S_zls_all[:, idx_pod[i]]; locate the figure
        # subset inside the POD subset for exact correspondence.
        idx_in_pod = indexin(idx_fig, idx_pod)   # positions of idx_fig within idx_pod
        if any(isnothing, idx_in_pod)
            # idx_fig contains points outside the POD subsample (can only
            # happen if n_zls_total > MAX_ZLS_POD) — fall back to the POD
            # subsample itself for this figure.
            idx_in_pod = seeded_subsample_idx(n_zls, MAX_FIG_PTS, 0x31)
            θ_fig_b = θ_zls[idx_pod[idx_in_pod]]
        else
            idx_in_pod = Int.(idx_in_pod)
            θ_fig_b = θ_zls[idx_fig]
        end
        z1 = Z_full[1, idx_in_pod]
        z2 = size(Z_full, 1) >= 2 ? Z_full[2, idx_in_pod] : zeros(length(idx_in_pod))
        z3 = size(Z_full, 1) >= 3 ? Z_full[3, idx_in_pod] : zeros(length(idx_in_pod))

        fig = Figure(size = (1000, 800))
        Label(fig[0, :], "ZLS points in the full POD projection, colored by phase θ — $spectral_method",
              fontsize = 16, font = :bold)
        ax = Axis3(fig[1, 1];
            title = "First 3 POD coordinates of Z = U'·(S_zls − x̄)  " *
                    "(σ₁=$(round(σ_zls[1], sigdigits=3)), σ₂=$(round(σ_zls[min(2, length(σ_zls))], sigdigits=3)), " *
                    "σ₃=$(round(σ_zls[min(3, length(σ_zls))], sigdigits=3)))",
            xlabel = "z₁ (POD 1)", ylabel = "z₂ (POD 2)", zlabel = "z₃ (POD 3)")
        hm = scatter!(ax, z1, z2, z3;
                      color = θ_fig_b, colormap = :thermal,
                      markersize = 6, colorrange = (-π, π))
        Colorbar(fig[1, 2], hm; label = "phase θ", width = 12)
        save(joinpath(save_dir, "ZLS_phase_colormap_pod.png"), fig)
        println("Saved: ZLS_phase_colormap_pod.png")
        fig = nothing
        GC.gc()
        print_mem("after phase color maps")
    end

    S_zls_all = nothing
    S_zls = nothing
    Z_full = nothing
    GC.gc()
end

# ============================================================================
# FINALIZE + SUMMARY
# ============================================================================

jldopen(save_path_out, "a+") do f
    jld_write!(f, "script_complete", true; overwrite=true)
end
println("\nOutput dataset complete: $save_path_out")
print_mem("end of computation")

println("\n" * "="^70)
println("SUMMARY — spectral analysis ($regime_system $regime_name, method: $spectral_method)")
println("="^70)
if j_phase > 0
    ω_lc = abs(imag(log(λ[j_phase])) / dt_eff)
    T_lc = 2π / ω_lc
    println("  Phase eigenvalue:  j_phase = $j_phase,  λ = $(round(λ[j_phase], digits=6))")
    println("  Limit cycle:       ω = $(round(ω_lc, digits=4)) rad/s,  " *
            "T = $(round(T_lc, digits=4)) s,  f = $(round(1/T_lc, digits=4)) Hz")
else
    println("  Phase eigenvalue:  none identified")
end
println("  Real eigenvalues inside the unit circle: $(length(real_eig_indices))")
println("  ZLS points per eigenvalue:")
for j in valid_js
    println("    j=$j  (λ=$(round(real(λ[j]), digits=4))):  $(size(all_zls_results[j], 2)) points")
end
if !isempty(pair_angles)
    bp = pair_angles[1]
    println("  Best transverse pair: j=$(bp.j1) ∩ j=$(bp.j2),  " *
            "mean angle = $(round(bp.angle, digits=2))°")
else
    println("  Best transverse pair: none (need ≥ 2 eigenfunctions with ZLS)")
end
if n_zls_total > 0
    println("  POD of ZLS cloud ($n_zls_total points, $n_state-D):")
    println("    σ₁..σ₅ = ", round.(σ_zls[1:min(5, length(σ_zls))], sigdigits=4))
    println("    effective rank: r(99%) = $r99,  r(99.9%) = $r999,  numerical ≈ $r_num")
end
println("  Figures written to: $save_dir")
println("    ZLS_All_Real_Eigenvalues.png, ZLS_Transversality_Angle_Matrix.png,")
println("    Best_Pair_Intersection.png, Cross_Method_Spectrum.png,")
println("    POD_ZLS_spectrum.png, ZLS_phase_colormap_plot.png, ZLS_phase_colormap_pod.png")
println("="^70)
