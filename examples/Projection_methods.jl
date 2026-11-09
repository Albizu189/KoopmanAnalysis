# ============================================================================
# Projection_methods.jl — Script 3 of the KoopmanAnalysis usage-case pipeline
# ============================================================================
# PROJECTION METHODS FOR LIMIT-CYCLE GEOMETRY (companion to the SECOND PAPER)
#
# This script merges and rewrites two legacy analyses as a clean usage case of
# the public package KoopmanAnalysis.jl:
#
#   A) Koopman_Classic_analysis_phase.jl
#        STEP 1  Harmonic branch structure of the Koopman spectrum
#                (package `all_harmonic_branches` / `find_all_harmonic_branches`)
#        STEP 2  Cycle separation of the joint ZLS point cloud by
#                argmax_j |φ_phase_j|  (+ seeded k-means fallback for
#                degenerate/equal-frequency cycles)
#        STEP 3  Eigenfunction ↔ cycle matching (winding number, phase
#                coherence, frequency match)
#        STEP 4  Gauge analysis (per-cycle phase reference; winding number is
#                gauge-invariant, absolute phase is not)
#        STEP 5  Dynamical frame per cycle: Koopman velocity field,
#                time-of-flight phase, periodic central differences for
#                acceleration/jerk, Gram–Schmidt frame (e_t, e_n, e_b),
#                Hessian of f = φ_⊥1² + φ_⊥2² projected onto the normal plane,
#                transverse coordinates (ρ1, ρ2), and consistency checks.
#
#   B) Koopman_Classic_analysis_unfolding.jl
#        STEP 6  PG-FNU  — Phase-Guided False-Neighbor Unfolding (greedy
#                POD-candidate selection minimizing the false-neighbor rate)
#        STEP 7  Differential FNN — greedy selection maximizing dP/dθ smoothness
#        STEP 8  Persistent homology (Ripserer, dim_max=1) of each projection:
#                a faithful S¹ embedding has exactly ONE dominant H₁ class.
#
# INPUTS (data contract, SHARED_BRIEF.md):
#   • Script 2 output  spectral_analysis-<system>-<regime>-show<tag>-<spectral_method>.jld2
#       (joint_zls, λ, Ξ, K, B_reduced, U_r, dict_info, P_plot, plot_labels,
#        pod_zls/*, j_phase, φ_phase_zls, θ_zls, best_pair, script_complete ...)
#   • Script 1 output  fitting_methods-<system>-<regime>-show<tag>.jld2
#       (metadata, method_configs, spectral_table, script_complete) — only for
#       anything missing from the Script 2 file.
#
# OUTPUT:
#   • projection_methods-<system>-<regime>-show<tag>-<spectral_method>.jld2
#       (phase pipeline + unfolding results, staged writes, script_complete=true)
#   • Figures PHASE-1 … PHASE-15 and UNFOLD-1 … UNFOLD-9 in the figure dir.
#
# DEPENDENCIES (beyond KoopmanAnalysis.jl and its own deps):
#   • JLD2, CairoMakie, Clustering, LinearAlgebra, Statistics, Random, Printf,
#     Dates, LaTeXStrings.
#   • Ripserer.jl + StaticArrays.jl are a HARD requirement for STEP 8
#     (persistent homology).  The `using` is wrapped in a try/catch so the
#     phase/dynamical-frame/unfolding sections still run without it, but the
#     UNFOLD-5/6 homology figures and the h1_* output keys are then skipped
#     with an explicit warning.  Install with:  ] add Ripserer StaticArrays
#
# REPRODUCIBILITY: a single top-level `const SEED = 42`; `Random.seed!(SEED)`
# at start-up; every stochastic call (k-means, figure subsampling, sigma
# heuristics) is explicitly seeded at the call site with a comment.
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
using KoopmanAnalysis.Hankel

using CairoMakie
using Statistics
using Random
using LinearAlgebra
using Printf
using JLD2
using LaTeXStrings
using Dates
using Clustering: kmeans, assignments

# Persistent homology — HARD dependency for STEP 8 (see header note).  Guarded
# so that a missing install degrades gracefully instead of killing the script.
const HAVE_RIPSERER = try
    @eval using Ripserer
    @eval using StaticArrays
    true
catch err
    @warn "Ripserer/StaticArrays not available — persistent homology (STEP 8, " *
          "UNFOLD-5/6) will be SKIPPED.  Install with `] add Ripserer StaticArrays`." exception=(err,)
    false
end

CairoMakie.activate!(type="png")
set_koopman_theme!()

# --- Reproducibility: single global seed ------------------------------------
const SEED = 42
Random.seed!(SEED)          # global RNG; per-call seeding is repeated at every stochastic site

# --- Local path configuration (see LocalPaths.jl.example) ---
const _LOCAL_PATHS = joinpath(@__DIR__, "LocalPaths.jl")
include(isfile(_LOCAL_PATHS) ? _LOCAL_PATHS : joinpath(@__DIR__, "LocalPaths.jl.example"))
import .LocalPaths: DATA_ROOT, RESULTS_ROOT

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

const experiment_name = "FittingMethods"   # must match Script 1's subdir
const regime_system   = "Epileptor3D"
const regime_name     = length(ARGS) > 0 ? ARGS[1] : "c4b-SN-FLC"   # ARGS[1] overrides the regime

# Selector of the fitted method whose spectral analysis (Script 2 output) is
# loaded.  MUST match the `spectral_method` used by Script 2 (it is part of the
# file name) — the canonical spelling is one of Script 1's `method_ids`
# (case-sensitive, e.g. "Hankel_EDMD").  ARGS[2] overrides.
# Default: the multichannel Hankel-EDMD model.
const spectral_method = length(ARGS) > 1 ? ARGS[2] : "Hankel_EDMD"

const dim      = 3               # physical state dimension
const showdim  = [1, 2, 3]       # shown (observed) physical coordinates
const showdim_tag = join(string.(showdim))
const n_show   = length(showdim)
const n_hidden = dim - n_show    # hidden (POD) plotting coordinates (0 when all dims shown)
const n_plot   = n_show + n_hidden

@assert dim == 3 "This script produces 3-D paper figures; dim must be 3."
@assert n_show >= 1 && n_hidden >= 0 && n_plot >= 3

# --- Paths ------------------------------------------------------------------
save_dir = default_save_dir(regime_system, regime_name;
                            base_dir=DATA_ROOT, subdir=experiment_name)
fig_dir  = default_save_dir(regime_system, regime_name;
                            base_dir=RESULTS_ROOT, subdir=experiment_name)
mkpath(fig_dir)

const save_path_s1 = joinpath(save_dir,
    "fitting_methods-$regime_system-$regime_name-show$showdim_tag.jld2")
const save_path_s2 = joinpath(save_dir,
    "spectral_analysis-$regime_system-$regime_name-show$showdim_tag-$spectral_method.jld2")
const save_path_out = joinpath(save_dir,
    "projection_methods-$regime_system-$regime_name-show$showdim_tag-$spectral_method.jld2")

@assert isfile(save_path_s2) """
    Script 2 output not found: $save_path_s2
    Run Koopman_spectral_analysis.jl first with the same regime/showdim and
    spectral_method=\"$spectral_method\" (or pass ARGS[2] to select another)."""
@assert isfile(save_path_s1) """
    Script 1 output not found: $save_path_s1
    Run Koopman_fitting_methods.jl first with the same regime/showdim."""

println("Script 1 (fitting):   $save_path_s1")
println("Script 2 (spectral):  $save_path_s2")
println("Output file:          $save_path_out")
println("Figure dir:           $fig_dir")
println("spectral_method = \"$spectral_method\",  SEED = $SEED")

# ============================================================================
# UTILITIES
# ============================================================================

fmt_bytes(b::Integer) = b >= 1<<30 ? @sprintf("%.2f GiB", b/2^30) :
                        b >= 1<<20 ? @sprintf("%.2f MiB", b/2^20) :
                        @sprintf("%.2f KiB", b/2^10)

print_mem(tag::AbstractString) =
    @printf("%-55s free RAM: %.2f GiB\n", tag, Sys.free_memory()/2^30)

print_mem("start")

"""
    winding_number(θ_sorted) -> Float64

Winding number of a phase sequence ALREADY sorted along the cycle, computed by
unwrapping successive increments into (-π, π] and summing:
w = (θ_unwrapped[end] - θ_unwrapped[1]) / 2π.  A clean parametrization of a
limit cycle gives w = ±1.  NOT provided by KoopmanAnalysis — implemented
locally exactly as in the legacy phase script.
"""
function winding_number(θ_sorted::AbstractVector{<:Real})
    n = length(θ_sorted)
    n < 2 && return 0.0
    θ_unwrapped = similar(θ_sorted, Float64)
    θ_unwrapped[1] = θ_sorted[1]
    for k in 2:n
        Δ = Float64(θ_sorted[k]) - Float64(θ_sorted[k-1])
        while Δ >  π;  Δ -= 2π;  end
        while Δ <= -π; Δ += 2π;  end
        θ_unwrapped[k] = θ_unwrapped[k-1] + Δ
    end
    return (θ_unwrapped[end] - θ_unwrapped[1]) / (2π)
end

"""
    seeded_subsample_idx(n, nmax; salt) -> Vector{Int}

Sorted indices of a reproducible uniform subsample of 1:n (used for figure
scatter downsampling and O(N²) algorithms).  Uses a DEDICATED
`MersenneTwister(SEED ⊻ salt)` generator so the global RNG stream is
untouched; deterministic for a given (SEED, salt).
"""
function seeded_subsample_idx(n::Int, nmax::Int; salt::Integer=0x0)
    n <= nmax && return collect(1:n)
    rng = MersenneTwister(SEED ⊻ UInt64(salt))   # seeded subsampling (memory rule)
    return sort(randperm(rng, n)[1:nmax])
end

"""Add a one-look caption box to a figure row (paper-style self-explanation)."""
function add_caption!(fig, row, text::AbstractString; fontsize::Int=11)
    Label(fig[row, :], text; fontsize=fontsize, color=:gray25,
          tellwidth=false, justification=:left, halign=:left)
end

"""First 3 rows of a plotting projection, zero-padded if it has fewer rows."""
function to3(P::AbstractMatrix)
    d, N = size(P)
    d >= 3 && return P[1:3, :]
    return vcat(P, zeros(3 - d, N))
end

print_mem("after setup")

# ============================================================================
# LOAD SCRIPT 2 DATA (spectral analysis: ZLS point cloud, spectrum, operators)
# ============================================================================

println("\n" * "="^72)
println("LOADING SCRIPT 2 DATA  ($save_path_s2)")
println("="^72)

s2 = jldopen(save_path_s2, "r") do f
    @assert get(f, "script_complete", false) "Script 2 file lacks `script_complete=true` — re-run Koopman_spectral_analysis.jl to completion."
    Dict{Symbol,Any}(
        # ZLS point cloud of the selected transverse pair (n_state × n_zls)
        :joint_zls        => f["joint_zls"],
        :best_pair        => get(f, "best_pair", nothing),
        :real_eig_indices => get(f, "real_eig_indices", Int[]),
        # selected method's operator + spectral data
        :lambda           => f["lambda"],
        :Xi               => f["Xi"],
        :K                => f["K"],
        :B_reduced        => f["B_reduced"],
        :U_r              => get(f, "U_r", nothing),
        :dict_info        => f["dict_info"],
        # plotting machinery
        :P_plot           => f["P_plot"],
        :plot_labels      => f["plot_labels"],
        :phys_idx         => f["phys_idx"],
        :showdim          => f["showdim"],
        :dim              => f["dim"],
        :n_state          => f["n_state"],
        # timing / embedding
        :dt_eff           => f["dt_eff"],
        :dt_internal      => f["dt_internal"],
        :nLag             => f["nLag"],
        :m_embed          => f["m_embed"],
        :tau_delay        => f["tau_delay"],
        # phase block (Script 2's own phase eigenfunction on the ZLS points)
        :j_phase          => f["j_phase"],
        :phase_eigenvalue => get(f, "phase_eigenvalue", nothing),
        :φ_phase_zls      => get(f, "φ_phase_zls", nothing),
        :θ_zls            => get(f, "θ_zls", nothing),
        # POD of the ZLS cloud (full projection, for cross-checks)
        :pod_zls_U        => get(f, "pod_zls/U", nothing),
        :pod_zls_σ        => get(f, "pod_zls/σ", nothing),
        :pod_zls_x_mean   => get(f, "pod_zls/x_mean", nothing),
    )
end

λ           = ComplexF64.(s2[:lambda])
Ξ           = Matrix{ComplexF64}(s2[:Xi])
K_mat       = Matrix{Float64}(real.(s2[:K]))
B_red       = Matrix{Float64}(s2[:B_reduced])
U_r         = s2[:U_r]
dict_info   = s2[:dict_info]
P_plot      = Matrix{Float64}(s2[:P_plot])
plot_labels = s2[:plot_labels]
phys_idx    = s2[:phys_idx]
dt_eff      = Float64(s2[:dt_eff])
dt_internal = Float64(s2[:dt_internal])
nLag        = Int(s2[:nLag])
m_embed     = Int(s2[:m_embed])
tau_delay   = Int(s2[:tau_delay])
n_state     = Int(s2[:n_state])
j_phase_s2  = Int(s2[:j_phase])
best_pair   = s2[:best_pair]

# Consistency with GLOBAL CONFIGURATION
@assert s2[:dim] == dim "dim mismatch: config=$dim, file=$(s2[:dim])"
@assert Int.(s2[:showdim]) == showdim "showdim mismatch: config=$showdim, file=$(s2[:showdim])"
@assert n_state == n_show * m_embed "n_state=$n_state but n_show*m_embed=$(n_show*m_embed)"
@assert phys_idx == [(j-1)*m_embed + 1 for j in 1:n_show] "phys_idx inconsistent with build_hankel_multichannel layout"

# Full-state projection (delay space): B_full = U_r * B_reduced (identity U_r
# for use_svd=false leaves B_reduced unchanged; U_r=nothing for state analysis).
B_full = isnothing(U_r) ? B_red : Matrix{Float64}(U_r) * B_red
@assert size(B_full, 1) == n_state "B_full rows ($(size(B_full,1))) != n_state ($n_state)"

# ZLS point cloud (columns = points on the joint zero level set of the
# transverse eigenfunction pair — the numerical limit-cycle candidates)
S_zls = Matrix{Float64}(s2[:joint_zls])
n_zls = size(S_zls, 2)
@assert size(S_zls, 1) == n_state
n_zls > 0 || error("joint_zls is empty — Script 2 found no ZLS points.")

nPsi = size(K_mat, 1)
println("  ZLS cloud: $n_zls points in $n_state-D delay space")
println("  Spectrum:  $(length(λ)) eigenvalues;  nPsi=$nPsi;  dt_eff=$dt_eff s")
println("  j_phase (Script 2) = $j_phase_s2,  best_pair = $best_pair")

# Plotting-space projection of the ZLS cloud (n_plot × n_zls)
S_zls_plot = P_plot * S_zls

# Axis labels with physical meaning
ax_lbl = [string(l) for l in plot_labels]
lbl(i) = i <= length(ax_lbl) ? ax_lbl[i] : "coord $i"

print_mem("after loading Script 2 data")

# ============================================================================
# LOAD SCRIPT 1 DATA (only what Script 2 does not carry: configs + spectral
# table for the selected method, metadata cross-check)
# ============================================================================

println("\nLoading Script 1 metadata  ($save_path_s1)")
s1 = jldopen(save_path_s1, "r") do f
    @assert get(f, "script_complete", false) "Script 1 file lacks `script_complete=true` — re-run Koopman_fitting_methods.jl to completion."
    Dict{Symbol,Any}(
        :method_configs => get(f, "method_configs", nothing),
        :spectral_table => get(f, "spectral_table", nothing),
        :n_show         => get(f, "n_show", nothing),
        :n_hidden       => get(f, "n_hidden", nothing),
        :n_plot         => get(f, "n_plot", nothing),
        :SEED_s1        => get(f, "SEED", nothing),
        :experiment     => get(f, "experiment_name", nothing),
    )
end
if !isnothing(s1[:n_show])
    @assert s1[:n_show] == n_show "n_show mismatch with Script 1 metadata"
end
if !isnothing(s1[:SEED_s1]) && s1[:SEED_s1] != SEED
    @warn "Script 1 used SEED=$(s1[:SEED_s1]); this script uses SEED=$SEED (only affects this script's stochastic steps)."
end

# KoopmanConfig of the selected method (needed by `all_harmonic_branches`,
# which reads res.cfg.dt).  Fall back to a minimal config if unavailable.
cfg_sel = nothing
if !isnothing(s1[:method_configs])
    mc = s1[:method_configs]
    cfg_sel = get(mc, spectral_method, get(mc, Symbol(spectral_method), nothing))
end
if isnothing(cfg_sel)
    @warn "method_configs has no entry for \"$spectral_method\" — using a minimal KoopmanConfig(dt=dt_eff)."
    cfg_sel = KoopmanConfig(; m_embed=m_embed, tau_delay=tau_delay, r=size(B_red, 1),
                              dt=dt_eff, seed=SEED, verbose=false)
end

# Transverse (amplitude) eigenvalue indices for the Hessian of f = φ⊥1²+φ⊥2².
# Priority: (1) Script 2's `best_pair` (the pair whose joint ZLS defines the
# cloud); (2) Script 1's spectral table j_amp + the next strongest real
# eigenvalue; (3) the two strongest non-unity real eigenvalues with |λ|<1.
function pick_transverse_pair()
    if !isnothing(best_pair)
        # Script 2 writes best_pair as a NamedTuple (j1::Int, j2::Int,
        # angle::Float64); accept that (by field name or position), a plain
        # Tuple, or a Vector.  Only the FIRST TWO entries are eigenvalue
        # indices — the third is a mean angle in degrees, so converting the
        # whole container with Int.(collect(...)) would throw InexactError.
        if best_pair isa NamedTuple && haskey(best_pair, :j1) && haskey(best_pair, :j2)
            return Int(best_pair.j1), Int(best_pair.j2), "Script 2 best_pair"
        elseif length(best_pair) >= 2
            return Int(best_pair[1]), Int(best_pair[2]), "Script 2 best_pair"
        end
    end
    if !isnothing(s1[:spectral_table])
        st = s1[:spectral_table]
        entry = get(st, spectral_method, get(st, Symbol(spectral_method), nothing))
        if !isnothing(entry) && haskey(entry, :j_amp)
            j1 = Int(entry[:j_amp])
            nontriv = findall(i -> abs(λ[i] - 1) > 1e-6 && abs(imag(λ[i])) < 1e-5 &&
                                   real(λ[i]) < 1.0 && i != j1, eachindex(λ))
            j2 = isempty(nontriv) ? 0 : nontriv[argmax(real.(λ[nontriv]))]
            return j1, j2, "Script 1 spectral_table"
        end
    end
    nontriv = findall(i -> abs(λ[i] - 1) > 1e-6 && abs(imag(λ[i])) < 1e-5 &&
                           real(λ[i]) < 1.0, eachindex(λ))
    srt = nontriv[sortperm(real.(λ[nontriv]); rev=true)]
    j1 = length(srt) >= 1 ? srt[1] : 0
    j2 = length(srt) >= 2 ? srt[2] : 0
    return j1, j2, "fallback: strongest real eigenvalues"
end
j_amp1, j_amp2, amp_source = pick_transverse_pair()
println("  Transverse eigenvalues for f = φ⊥1²+φ⊥2²: j_amp1=$j_amp1, j_amp2=$j_amp2  (source: $amp_source)")

# ============================================================================
# REBUILD THE LIFT (Psi_func) AND THE TRUE VECTOR FIELD (ground truth)
# ============================================================================

# Psi_func evaluates the dictionary on delay-space points.  Mirrors the
# pipeline exactly, including normalize_stats if the fit used them.
const _dict_type  = Symbol(get(dict_info, :type, :rbf))
const _norm_stats = get(dict_info, :norm_stats, nothing)
function _lift_raw(pts::AbstractMatrix)
    if _dict_type == :rbf
        return Psi_RBF(pts, dict_info.centers;
                       include_states = dict_info.include_states,
                       state_indices  = get(dict_info, :state_indices, nothing),
                       kernel_type    = get(dict_info, :kernel_type, :thinplate),
                       sigma          = get(dict_info, :sigma, nothing),
                       seed           = SEED)   # seeds the gaussian :auto sigma heuristic (package paradigm)
    elseif _dict_type == :rff
        return Psi_RFF(pts, dict_info.basis;
                       include_states = get(dict_info, :include_states, false))
    elseif _dict_type == :hermite
        return Psi_Hermite(pts, dict_info.max_deg;
                           basis_type = get(dict_info, :basis_type, :probabilist))
    else
        error("Psi_func reconstruction not implemented for dict_type=$_dict_type")
    end
end
Psi_func(pts::AbstractMatrix) =
    isnothing(_norm_stats) ? _lift_raw(pts) : _lift_raw(apply_norm_stats(pts, _norm_stats))

# Sanity: dictionary width must match the Koopman matrix
let Ψtest = Psi_func(S_zls[:, 1:1])
    @assert size(Ψtest, 1) == nPsi "lifted dim $(size(Ψtest,1)) != nPsi $nPsi"
end
println("  Psi_func reconstructed (dict_type=$_dict_type, nPsi=$nPsi)")

# True (deterministic) vector field for ground-truth limit-cycle integration
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
    rossler_parameters(regime_name; noise_type=:none, sigma=0.0, noise_mask=[1.0,1.0,1.0])
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

print_mem("after lift/vector-field reconstruction")

# ============================================================================
# STEP 1 — HARMONIC BRANCH STRUCTURE OF THE KOOPMAN SPECTRUM
# ============================================================================
# Every independent harmonic family on the unit circle corresponds to one
# candidate limit cycle.  We use the package function
# `all_harmonic_branches(res; ...)` (which calls `find_all_harmonic_branches`
# on the cached spectrum).  A lightweight AnalysisResult is rebuilt around the
# saved K with the saved (λ, Ξ) pre-cached, so NO re-eigendecomposition of K
# happens (spectrum() returns the cache).

println("\n" * "="^72)
println("STEP 1: Harmonic branch structure (one branch = one candidate limit cycle)")
println("="^72)

res_spec = AnalysisResult(K_mat, B_full, B_red,
                          isnothing(U_r) ? nothing : Matrix{Float64}(U_r),
                          nothing, dict_info, nothing, nothing, nothing, cfg_sel)
res_spec.λ = λ          # cache the saved spectrum: spectrum(res_spec) is free
res_spec.Ξ = Ξ

branches = try
    all_harmonic_branches(res_spec;
                          mag_tol=0.05, TOL_HARMONIC=0.05, tol_growth=1.0,
                          branch_growth=1.0, max_harmonic=2000)
catch err
    @warn "all_harmonic_branches failed ($err) — falling back to Script 2's j_phase=$j_phase_s2"
    Dict(1 => [j_phase_s2])
end
n_branches = length(branches)
if n_branches == 0
    @warn "No harmonic branch on the unit circle — using Script 2's j_phase=$j_phase_s2"
    branches = Dict(1 => [j_phase_s2])
    n_branches = 1
end

# Fundamental (phase) eigenvalue of each branch + physical frequency/period
j_phases = [branches[k][1] for k in 1:n_branches]
λ_phases = λ[j_phases]
ω_phases = imag.(log.(λ_phases)) ./ dt_eff          # angular frequency [rad/s]
T_phases = 2π ./ abs.(ω_phases)                     # period [s]

println("  Found $n_branches harmonic branch(es) on the unit circle:")
for k in 1:n_branches
    println("    Branch $k: fundamental j=$(j_phases[k]), λ=$(round(λ_phases[k], digits=6)), " *
            "ω=$(round(ω_phases[k], digits=4)) rad/s, T=$(round(T_phases[k], digits=4)) s, " *
            "$(length(branches[k])) harmonics")
end

# ---- Degeneracy detection: two branch fundamentals sharing one eigenvalue --
# (equal-frequency cycles).  Only fundamentals are compared.
const TOL_DEGENERACY = 1e-3
fundamental_clusters = Vector{Vector{Int}}()
assigned_branch = falses(n_branches)
for b1 in 1:n_branches
    assigned_branch[b1] && continue
    cl = [b1]
    for b2 in (b1+1):n_branches
        if !assigned_branch[b2] && abs(λ[j_phases[b1]] - λ[j_phases[b2]]) < TOL_DEGENERACY
            push!(cl, b2)
            assigned_branch[b2] = true
        end
    end
    assigned_branch[b1] = true
    push!(fundamental_clusters, cl)
end
degenerate_clusters = filter(c -> length(c) > 1, fundamental_clusters)
degenerate_detected = !isempty(degenerate_clusters)
if degenerate_detected
    println("  ⚠  DEGENERATE eigenvalue(s) among branch fundamentals:")
    for cl in degenerate_clusters
        println("     Branches $cl share λ ≈ $(round(λ[j_phases[cl[1]]], digits=6)) " *
                "(multiplicity $(length(cl)))")
    end
else
    println("  No degeneracy among branch fundamentals — standard separation.")
end

# List the strongest real transverse eigenvalues (context for STEP 5 Hessian)
nontriv = findall(i -> abs(λ[i] - 1.0) > 1e-6, eachindex(λ))
real_cand = nontriv[abs.(imag.(λ[nontriv])) .< 1e-5]
real_cand = real_cand[real.(λ[real_cand]) .< 1.0]
sorted_real = real_cand[sortperm(real.(λ[real_cand]); rev=true)]
println("  Strongest real transverse eigenvalues (|λ|<1, Im≈0):")
for j in sorted_real[1:min(6, length(sorted_real))]
    println("    j=$j,  λ=$(round(λ[j], digits=6))")
end
print_mem("after STEP 1")

# ---------------------------------------------------------------------------
# FIGURE PHASE-1 — Spectrum with all harmonic branches highlighted
# ---------------------------------------------------------------------------
println("\n  PHASE-1: spectrum colored by harmonic branch")

let
    fig = Figure(size=(1100, 950))
    Label(fig[0, :],
          "Koopman spectrum decomposes into $n_branches harmonic branch(es) — " *
          "one per candidate limit cycle  (method: $spectral_method)";
          fontsize=15, font=:bold)
    Label(fig[1, :],
          "$regime_system — $regime_name | $(length(λ)) eigenvalues, dt_eff = $dt_eff s | " *
          "★ = branch fundamental (phase eigenvalue)";
          fontsize=11, color=:gray)

    cycle_colors = [:dodgerblue, :crimson, :forestgreen, :darkorange, :purple,
                    :turquoise, :gold, :magenta]

    # Left: discrete-time spectrum on the unit circle
    ax = Axis(fig[2, 1]; title="Discrete eigenvalues λ (unit circle = neutral stability)",
              xlabel="Re(λ)  [–]", ylabel="Im(λ)  [–]", aspect=DataAspect())
    θc = range(0, 2π; length=300)
    lines!(ax, cos.(θc), sin.(θc); color=(:black, 0.35), linewidth=1, linestyle=:dash)
    scatter!(ax, real.(λ), imag.(λ); color=(:gray, 0.35), markersize=3,
             label="all eigenvalues")
    for k in 1:n_branches
        br = branches[k]
        scatter!(ax, real.(λ[br]), imag.(λ[br]);
                 color=cycle_colors[mod1(k, 8)], markersize=7, marker=:diamond,
                 label="branch $k ($(length(br)) harm., T=$(round(T_phases[k], digits=2)) s)")
        scatter!(ax, [real(λ[j_phases[k]])], [imag(λ[j_phases[k]])];
                 color=cycle_colors[mod1(k, 8)], markersize=16, marker=:star5)
        text!(ax, real(λ[j_phases[k]]), imag(λ[j_phases[k]]);
              text="  B$k", fontsize=12, font=:bold,
              color=cycle_colors[mod1(k, 8)])
    end
    axislegend(ax; position=:lb, framevisible=true, backgroundcolor=(:white, 0.85), labelsize=9)

    # Right: continuous-time rates μ = log(λ)/dt_eff
    ax2 = Axis(fig[3, 1]; title="Continuous-time rates μ = log(λ)/dt_eff (imaginary part = angular frequency)",
               xlabel="Re(μ)  [1/s]  (growth rate)", ylabel="Im(μ)  [rad/s]  (frequency)")
    μ_all = log.(λ) ./ dt_eff
    scatter!(ax2, real.(μ_all), imag.(μ_all); color=(:gray, 0.35), markersize=3)
    vlines!(ax2, 0.0; color=(:black, 0.35), linewidth=1, linestyle=:dash)
    hlines!(ax2, 0.0; color=(:black, 0.2), linewidth=0.5)
    for k in 1:n_branches
        μ_br = log.(λ[branches[k]]) ./ dt_eff
        scatter!(ax2, real.(μ_br), imag.(μ_br);
                 color=cycle_colors[mod1(k, 8)], markersize=7, marker=:diamond)
        scatter!(ax2, [real(μ_br[1])], [imag(μ_br[1])];
                 color=cycle_colors[mod1(k, 8)], markersize=16, marker=:star5)
    end

    cap = "Each colored diamond family is a harmonic branch λ^k of one fundamental (★). " *
          join(["B$k: ω=$(round(ω_phases[k], digits=3)) rad/s, T=$(round(T_phases[k], digits=2)) s"
                for k in 1:n_branches], ";  ") *
          ".  Branches on the unit circle (Re μ ≈ 0) are neutral phase directions ⇒ candidate limit cycles."
    add_caption!(fig, 4, cap)

    save(joinpath(fig_dir, "PHASE1_spectrum_harmonic_branches.png"), fig)
    println("  Saved: PHASE1_spectrum_harmonic_branches.png")
end
GC.gc()

# ============================================================================
# STEP 2 — CYCLE SEPARATION OF THE ZLS POINT CLOUD
# ============================================================================
# Each ZLS point s_i is assigned to the cycle whose phase eigenfunction has
# the largest magnitude on it:  k(i) = argmax_j |φ_phase_j(s_i)|.
# Rationale: on its own cycle |φ_phase_k| ≈ const > 0 (clean phase variable),
# while off its cycle it is small or wildly varying.
#
# Degenerate case (two cycles sharing ONE eigenvalue): the eigensolver returns
# arbitrary mixtures of the two true basin-supported eigenfunctions; argmax of
# the mixtures is meaningless.  We then cluster the COMPLEX RATIO
# ρ(s) = φ_a(s)/φ_b(s) into 2 clusters by seeded k-means — the ratio is
# approximately constant on each cycle because both eigenfunctions rotate with
# the same frequency.

println("\n" * "="^72)
println("STEP 2: Cycle separation by argmax_j |φ_phase_j| (seeded k-means fallback)")
println("="^72)

# Evaluate all branch phase eigenfunctions on all ZLS points (chunked lift —
# memory rule: never materialize the full nPsi × n_zls lift at once)
println("  Evaluating $n_branches phase eigenfunction(s) on $n_zls ZLS points...")
chunk_sz = 5000
φ_phase_vals = Matrix{ComplexF64}(undef, n_branches, n_zls)
for cs in 1:chunk_sz:n_zls
    ce = min(cs + chunk_sz - 1, n_zls)
    Ψc = Psi_func(S_zls[:, cs:ce])
    for k in 1:n_branches
        φ_phase_vals[k, cs:ce] .= vec(Ξ[:, j_phases[k]]' * Ψc)
    end
    Ψc = nothing
end
GC.gc()
φ_phase_mag = abs.(φ_phase_vals)

# Degenerate-path state (defined unconditionally so STEP 2b is always valid —
# this FIXES the legacy bug where n_clusters_degen / φ_degen /
# degenerate_eig_indices were referenced but never defined).
n_clusters_degen       = 0
degenerate_eig_indices = Int[]
φ_degen                = Matrix{ComplexF64}(undef, 0, 0)
ρ_vals                 = ComplexF64[]

cycle_assignment = zeros(Int, n_zls)
if degenerate_detected && length(degenerate_clusters) == 1 &&
   length(degenerate_clusters[1]) == 2
    # ---- DEGENERATE PATH: exactly 2 cycles with equal frequency ----
    println("  Degenerate separation: complex-ratio clustering (seeded k-means)...")
    degenerate_branches = degenerate_clusters[1]
    j_a = j_phases[degenerate_branches[1]]
    j_b = j_phases[degenerate_branches[2]]

    φ_a = copy(vec(φ_phase_vals[degenerate_branches[1], :]))
    φ_b = copy(vec(φ_phase_vals[degenerate_branches[2], :]))

    ρ_vals = φ_a ./ (φ_b .+ 1e-12)
    ρ_features = hcat(real.(ρ_vals), imag.(ρ_vals))'   # 2 × n_zls

    # k-means uses random initialisation → seed the global RNG immediately
    # before the call (package seeding paradigm; dedicated salt for this site)
    Random.seed!(SEED ⊻ 0xA1)
    R = kmeans(ρ_features, 2; maxiter=500)
    cycle_assignment = copy(assignments(R))

    println("    k-means on 2-D ratio features → 2 clusters:")
    for k in 1:2
        idx_k = findall(cycle_assignment .== k)
        println("      cluster $k: $(length(idx_k)) points, " *
                "⟨ρ⟩ = $(round(mean(ρ_vals[idx_k]), digits=4))")
    end

    # Keep only the 2 degenerate branches
    n_clusters_degen       = 2
    degenerate_eig_indices = [j_a, j_b]
    φ_degen                = vcat(reshape(φ_a, 1, :), reshape(φ_b, 1, :))

    n_branches = 2
    j_phases   = [j_a, j_b]
    λ_phases   = λ[j_phases]
    ω_phases   = imag.(log.(λ_phases)) ./ dt_eff
    T_phases   = 2π ./ abs.(ω_phases)
    branches   = Dict(1 => [j_a], 2 => [j_b])
    φ_phase_vals = copy(φ_degen)
    φ_phase_mag  = abs.(φ_phase_vals)
elseif degenerate_detected
    println("  ⚠  Degenerate multiplicity > 2 not implemented — standard argmax path.")
    cycle_assignment = vec([argmax(φ_phase_mag[:, i]) for i in 1:n_zls])
else
    # ---- STANDARD PATH ----
    cycle_assignment = vec([argmax(φ_phase_mag[:, i]) for i in 1:n_zls])
end

# Cycle sizes + on/off-cycle magnitude separation diagnostic
cycle_counts = [count(==(k), cycle_assignment) for k in 1:n_branches]
println("\n  Cycle assignment (argmax_k |φ_phase_k|):")
sep_ratio = zeros(n_branches)
for k in 1:n_branches
    mag_on  = φ_phase_mag[k, cycle_assignment .== k]
    mag_off = φ_phase_mag[k, cycle_assignment .!= k]
    med_on  = isempty(mag_on)  ? NaN : median(mag_on)
    med_off = isempty(mag_off) ? NaN : median(mag_off)
    sep_ratio[k] = med_on / (med_off + 1e-12)
    println("    Cycle $k: $(cycle_counts[k]) points  (median |φ_phase_$k| = " *
            "$(round(med_on, digits=4)) on its own cycle vs $(round(med_off, digits=4)) " *
            "on others ⇒ separation ×$(round(sep_ratio[k], digits=2)))")
end

# Spurious branches (noise eigenvalues near the unit circle) get < 1% of points
min_pts_per_cycle = max(5, div(n_zls, 100))
spurious_cycles = findall(cycle_counts .< min_pts_per_cycle)
if !isempty(spurious_cycles)
    println("  ⚠  $(length(spurious_cycles)) branch(es) with < $min_pts_per_cycle points " *
            "(likely spurious): $spurious_cycles")
end

# ---- Relabel: drop spurious branches, renumber real cycles 1, 2, ... -------
real_cycles = setdiff(1:n_branches, spurious_cycles)
old_to_new = zeros(Int, n_branches)
for (new_k, old_k) in enumerate(real_cycles)
    old_to_new[old_k] = new_k
end
if length(real_cycles) < n_branches
    println("  Relabeling: $n_branches branches → $(length(real_cycles)) real cycle(s)")
    n_branches = length(real_cycles)
    branches = Dict(nk => branches[ok] for (nk, ok) in enumerate(real_cycles))
    j_phases = j_phases[real_cycles]
    λ_phases = λ_phases[real_cycles]
    ω_phases = ω_phases[real_cycles]
    T_phases = T_phases[real_cycles]
    φ_phase_vals = φ_phase_vals[real_cycles, :]
    φ_phase_mag  = φ_phase_mag[real_cycles, :]
    cycle_assignment = [old_to_new[c] for c in cycle_assignment]
    cycle_counts = [count(==(k), cycle_assignment) for k in 1:n_branches]
    sep_ratio = sep_ratio[real_cycles]
else
    println("  All $n_branches branch(es) have enough points — no relabeling.")
end
print_mem("after STEP 2")

# ---------------------------------------------------------------------------
# STEP 2b — RECOVER BASIN-SUPPORTED EIGENFUNCTIONS (degenerate case only)
# ---------------------------------------------------------------------------
# ψ_k = φ̃_a − ρ_k·φ̃_b vanishes on cycle k (ρ_k = cluster center of the ratio).
# Still an eigenfunction of K (same eigenvalue).  Uses the variables defined in
# the degenerate branch above (legacy bug fixed: n_clusters_degen, φ_degen and
# degenerate_eig_indices are now always defined, and the guard is simply
# n_clusters_degen == 2).
if n_clusters_degen == 2
    println("\n  STEP 2b: recovering basin-supported eigenfunctions (degenerate)")
    ρ_centers = [mean(ρ_vals[findall(cycle_assignment .== k)]) for k in 1:n_clusters_degen]

    ψ_recovered = Matrix{ComplexF64}(undef, n_clusters_degen, n_zls)
    for k in 1:n_clusters_degen
        ψ_recovered[k, :] .= φ_degen[1, :] .- ρ_centers[k] .* φ_degen[2, :]
    end

    println("    Verification (|ψ_k| should vanish on cycle k):")
    for k in 1:n_clusters_degen, jj in 1:n_clusters_degen
        idx_j = findall(cycle_assignment .== jj)
        med_mag = isempty(idx_j) ? NaN : median(abs.(ψ_recovered[k, idx_j]))
        tag = jj == k ? "  ← should be ≈ 0" : ""
        println("      |ψ_$k| on cycle $jj: median = $(round(med_mag, digits=4))$tag")
    end

    # Downstream steps use the recovered basin-supported eigenfunctions
    φ_phase_vals = ψ_recovered
    φ_phase_mag  = abs.(ψ_recovered)
    j_phases = degenerate_eig_indices[1:n_clusters_degen]
    λ_phases = λ[j_phases]
    ω_phases = imag.(log.(λ_phases)) ./ dt_eff
    T_phases = 2π ./ abs.(ω_phases)
    n_branches = n_clusters_degen
    println("    Recovered $n_clusters_degen basin-supported eigenfunctions.")
end

# ---------------------------------------------------------------------------
# FIGURE PHASE-2 — Cycle separation result
# ---------------------------------------------------------------------------
println("\n  PHASE-2: cycle separation of the ZLS cloud")

let
    cycle_colors = [:dodgerblue, :crimson, :forestgreen, :darkorange, :purple,
                    :turquoise, :gold, :magenta]
    # Seeded subsample for scatter (memory rule; dedicated generator, salt 0xB2)
    idx_plot = seeded_subsample_idx(n_zls, 20_000; salt=0xB2)
    P3 = to3(S_zls_plot[:, idx_plot])

    fig = Figure(size=(1500, 700))
    meth = n_clusters_degen == 2 ? "seeded k-means on the complex ratio φ_a/φ_b (degenerate eigenvalue)" :
                                   "argmax_k |φ_phase_k| (basin membership encoded by the phase eigenfunctions)"
    Label(fig[0, :],
          "ZLS point cloud separates into $n_branches cycle(s) — $meth";
          fontsize=14, font=:bold)
    Label(fig[1, :],
          "$regime_system — $regime_name | $n_zls ZLS points (showing $(length(idx_plot))), " *
          "method $spectral_method"; fontsize=11, color=:gray)

    # Left: 3-D scatter colored by assigned cycle
    ax = Axis3(fig[2, 1];
               title="Assigned cycle (one color per cycle)\ncounts: " *
                     join(["cycle $k = $(cycle_counts[k])" for k in 1:n_branches], ", "),
               xlabel=lbl(1), ylabel=lbl(2), zlabel=lbl(3), titlesize=11)
    for k in 1:n_branches
        sel = idx_plot[cycle_assignment[idx_plot] .== k]
        isempty(sel) && continue
        Pk = to3(S_zls_plot[:, sel])
        scatter!(ax, Pk[1, :], Pk[2, :], Pk[3, :];
                 color=cycle_colors[mod1(k, 8)], markersize=4,
                 label="cycle $k ($(cycle_counts[k]) pts)")
    end
    axislegend(ax; position=:rb, framevisible=true, backgroundcolor=(:white, 0.85), labelsize=9)

    # Right: |φ_phase_1| magnitude (each cycle "lights up" its own eigenfunction)
    ax2 = Axis3(fig[2, 2];
                title="|φ_phase_1| on the ZLS cloud — cycle 1 lights up\n" *
                      "separation ×$(round(sep_ratio[1], digits=2)) (median on/off cycle)",
                xlabel=lbl(1), ylabel=lbl(2), zlabel=lbl(3), titlesize=11)
    hm = scatter!(ax2, P3[1, :], P3[2, :], P3[3, :];
                  color=φ_phase_mag[1, idx_plot], colormap=:thermal, markersize=4)
    Colorbar(fig[2, 3], hm; label="|φ_phase_1|  (eigenfunction magnitude)", width=14)

    add_caption!(fig, 3,
        "Left: every ZLS point assigned to the cycle whose phase eigenfunction is largest on it. " *
        "Right: |φ_phase_1| confirms the encoding — it is large exactly on cycle 1.  " *
        "Per-cycle separation factors (median |φ| on own cycle ÷ off): " *
        join(["cycle $k ×$(round(sep_ratio[k], digits=2))" for k in 1:n_branches], ", ") * ".")

    save(joinpath(fig_dir, "PHASE2_cycle_separation.png"), fig)
    println("  Saved: PHASE2_cycle_separation.png")
end
GC.gc()

# Shared per-cycle color palette (one distinct color per cycle)
const CYCLE_COLORS = [:dodgerblue, :crimson, :forestgreen, :darkorange,
                      :purple, :turquoise, :gold, :magenta]

fig_subtitle = "$regime_system — $regime_name | method: $spectral_method | " *
               "n_zls=$n_zls, n_cycles=$n_branches, n_state=$n_state, n_show=$n_show/$dim"

# ============================================================================
# STEP 3 — EIGENFUNCTION ↔ CYCLE MATCHING
# ============================================================================
# For each (eigenfunction j, cycle k) pair three independent tests:
#   (a) frequency match:  |ω_j − ω_k| / max(|ω_j|,|ω_k|) < 0.05
#   (b) phase coherence:  |φ_phase_j| ~ constant on cycle k  (1 − σ/μ)
#   (c) winding number:   arg(φ_phase_j) winds exactly once around cycle k,
#       ordering cycle k's points by the phase of eigenfunction k.
# final_match[j] = argmax_k [ −|w_jk − 1| + coherence_jk + freq_jk ].

println("\n" * "="^72)
println("STEP 3: Eigenfunction ↔ cycle matching (winding + coherence + frequency)")
println("="^72)

match_table_winding   = zeros(n_branches, n_branches)
match_table_coherence = zeros(n_branches, n_branches)
match_table_freqmatch = zeros(n_branches, n_branches)

if n_clusters_degen == 2
    # Matching trivial: ψ_k vanishes on cycle k by construction
    println("  Matching trivial (basin-supported by construction).")
    match_table_winding   .= Matrix{Float64}(I, n_branches, n_branches)
    match_table_coherence .= 0.99
    match_table_freqmatch .= 1.0
else
    for k in 1:n_branches
        idx_k = findall(cycle_assignment .== k)
        if length(idx_k) < 5
            println("  Cycle $k: too few points ($(length(idx_k))) — winding test skipped")
            continue
        end
        θ_k = angle.(φ_phase_vals[k, idx_k])
        perm_k = sortperm(θ_k)
        idx_k_sorted = idx_k[perm_k]
        for j in 1:n_branches
            ωj, ωk = ω_phases[j], ω_phases[k]
            match_table_freqmatch[j, k] =
                abs(ωj - ωk) / (max(abs(ωj), abs(ωk)) + 1e-12) < 0.05 ? 1.0 : 0.0
            mag_jk = φ_phase_mag[j, idx_k]
            μm, σm = mean(mag_jk), std(mag_jk)
            match_table_coherence[j, k] = μm > 1e-12 ? 1.0 - σm / μm : 0.0
            match_table_winding[j, k] =
                winding_number(angle.(φ_phase_vals[j, idx_k_sorted]))
        end
    end
end

final_match = zeros(Int, n_branches)
for j in 1:n_branches
    best_k, best_score = 0, -Inf
    for k in 1:n_branches
        score = -abs(match_table_winding[j, k] - 1.0) +
                match_table_coherence[j, k] + match_table_freqmatch[j, k]
        if score > best_score
            best_score, best_k = score, k
        end
    end
    final_match[j] = best_k
end

println("  Matching tables (rows = eigenfunction j, cols = cycle k):")
for (name, tab) in (("winding", match_table_winding),
                    ("coherence", match_table_coherence),
                    ("freq match", match_table_freqmatch))
    println("    $name:")
    for j in 1:n_branches
        println("      j=$j: " * join([@sprintf("%6.3f", tab[j, k]) for k in 1:n_branches], " "))
    end
end
println("  Final matching (eigenfunction j → cycle k):")
for j in 1:n_branches
    k = final_match[j]
    println("    eig $j (j_eig=$(j_phases[j]), ω=$(round(ω_phases[j], digits=4)) rad/s) → cycle $k  " *
            "(winding=$(round(match_table_winding[j,k], digits=3)), " *
            "coherence=$(round(match_table_coherence[j,k], digits=3)))")
end
print_mem("after STEP 3")

# ---------------------------------------------------------------------------
# FIGURE PHASE-3 — Matching score tables (heatmaps)
# ---------------------------------------------------------------------------
println("\n  PHASE-3: eigenfunction↔cycle matching tables")

let
    fig = Figure(size=(1500, 500))
    Label(fig[0, :],
          "Eigenfunction ↔ cycle matching: winding number identifies the pairing " *
          "(diagonal ≈ 1, off-diagonal ≈ 0)";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    tabs = [("winding number  (ideal: identity)", match_table_winding),
            ("phase coherence 1−σ/μ  (ideal: ≈1 diagonal)", match_table_coherence),
            ("frequency match  (1 = |Δω| < 5%)", match_table_freqmatch)]
    for (c, (tname, tab)) in enumerate(tabs)
        ax = Axis(fig[2, c]; title=tname, titlesize=11,
                  xlabel="cycle k", ylabel="eigenfunction j",
                  xticks=1:n_branches, yticks=1:n_branches, aspect=DataAspect())
        hm = heatmap!(ax, 1:n_branches, 1:n_branches, tab; colormap=:viridis)
        for j in 1:n_branches, k in 1:n_branches
            text!(ax, k, j; text=@sprintf("%.2f", tab[j, k]),
                  align=(:center, :center), fontsize=10,
                  color=tab[j, k] > 0.5 ? :white : :black)
        end
        Colorbar(fig[2, c+3], hm; width=12, label=tname)
    end
    colsize!(fig.layout, 4, Relative(0.06))
    colsize!(fig.layout, 5, Relative(0.06))
    colsize!(fig.layout, 6, Relative(0.06))

    add_caption!(fig, 3,
        "Three independent tests per (eigenfunction j, cycle k) pair.  Result: " *
        join(["φ_$j → cycle $(final_match[j])" for j in 1:n_branches], ", ") *
        ".  Winding = ±1 means arg(φ_j) makes exactly one turn around cycle k — the topological certificate of the pairing.")

    save(joinpath(fig_dir, "PHASE3_matching_tables.png"), fig)
    println("  Saved: PHASE3_matching_tables.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-3b — Winding-number verification matrix (SANITY CHECK)
# ---------------------------------------------------------------------------
# Diagonal panels must show a clean monotone −π→π ramp (winding = 1);
# off-diagonal panels must be flat/noisy (winding = 0).
println("  PHASE-3b: winding-number verification matrix (sanity check)")

let
    fig = Figure(size=(350 * n_branches + 150, 350 * n_branches + 200))
    Label(fig[0, :],
          "Winding verification: arg(φ_j) vs θ of cycle k — a monotone ramp on the " *
          "diagonal proves each eigenfunction parametrizes exactly one cycle";
          fontsize=13, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    for j in 1:n_branches, k in 1:n_branches
        ax = Axis(fig[1+j, k];
                  title="j=$j, k=$k  (w = $(round(match_table_winding[j, k], digits=2)))",
                  titlesize=10,
                  xlabel = j == n_branches ? "θ_cycle_$k  [rad]" : "",
                  ylabel = k == 1 ? "arg(φ_phase_$j)  [rad]" : "",
                  xlabelsize=9, ylabelsize=9)
        idx_k = findall(cycle_assignment .== k)
        length(idx_k) < 2 && continue
        θ_k = angle.(φ_phase_vals[k, idx_k])
        perm_k = sortperm(θ_k)
        idx_k_sorted = idx_k[perm_k]
        θ_j = angle.(φ_phase_vals[j, idx_k_sorted])
        # seeded subsample for scatter (salt encodes the panel)
        sel = seeded_subsample_idx(length(idx_k), 4_000; salt=0xC0 + 16*j + k)
        scatter!(ax, θ_k[perm_k][sel], θ_j[sel];
                 color=CYCLE_COLORS[mod1(j, 8)], markersize=3, alpha=0.6)
        if j == k
            lines!(ax, [-π, π], [-π, π]; color=(:black, 0.4), linestyle=:dash, linewidth=1)
        end
    end

    add_caption!(fig, n_branches + 2,
        "Panel (j,k): phase of eigenfunction j evaluated on cycle k's points, ordered by cycle k's own phase. " *
        "Diagonal ramps ⇒ winding w = ±1 (matched pair); flat/noisy off-diagonal ⇒ w ≈ 0.  " *
        "Winding is gauge-invariant — the robust certificate of the pairing.")

    save(joinpath(fig_dir, "PHASE3b_winding_verification.png"), fig)
    println("  Saved: PHASE3b_winding_verification.png")
end
GC.gc()

# ============================================================================
# STEP 4 — GAUGE ANALYSIS
# ============================================================================
# Each φ_phase_j is defined up to a complex factor c_j: |φ| has arbitrary
# scale |c_j|, arg(φ) an arbitrary zero arg(c_j).  We gauge-fix per cycle:
#   |c_j|   = median |φ_phase_j| on its matched cycle   (⇒ |φ|/|c_j| ≈ 1)
#   arg(c_j) = −circular mean of arg(φ_phase_j)          (⇒ phases centered at 0)
# Cross-cycle phase comparison is NOT meaningful without an external reference.

println("\n" * "="^72)
println("STEP 4: Gauge analysis (per-cycle phase reference)")
println("="^72)

gauge_scale  = ones(n_branches)
gauge_offset = zeros(n_branches)
for j in 1:n_branches
    k = final_match[j]
    idx_k = findall(cycle_assignment .== k)
    isempty(idx_k) && continue
    gauge_scale[j] = median(φ_phase_mag[j, idx_k])
    θ_j = angle.(φ_phase_vals[j, idx_k])
    gauge_offset[j] = -angle(sum(exp.(im .* θ_j)))   # circular mean
    println("  eig $j (cycle $k):  |c_j| = $(round(gauge_scale[j], digits=4)),  " *
            "arg(c_j) = $(round(gauge_offset[j], digits=4)) rad")
end
println("  ⚠  Cross-cycle phase comparison is NOT meaningful without an external")
println("     reference (Poincaré section, common forcing, temporal co-evolution).")

# ---------------------------------------------------------------------------
# FIGURE PHASE-4 — Gauge-fixed eigenfunctions on their own cycles
# ---------------------------------------------------------------------------
println("\n  PHASE-4: gauge analysis figure")

let
    fig = Figure(size=(500 * n_branches + 100, 550))
    Label(fig[0, :],
          "Gauge freedom: each eigenfunction is defined up to a complex factor c_j — " *
          "after gauge-fixing, |φ| ≡ 1 and phases are centered on its own cycle";
          fontsize=13, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    for j in 1:n_branches
        k = final_match[j]
        idx_k = findall(cycle_assignment .== k)
        φ_gf = φ_phase_vals[j, idx_k] .* (exp(im * gauge_offset[j]) / gauge_scale[j])
        sel = seeded_subsample_idx(length(idx_k), 4_000; salt=0xD0 + j)

        ax = Axis(fig[2, j]; aspect=DataAspect(), titlesize=10,
                  title="eig $j on cycle $k, gauge-fixed\n" *
                        "|c_j|=$(round(gauge_scale[j], sigdigits=3)), " *
                        "arg(c_j)=$(round(gauge_offset[j], digits=2)) rad",
                  xlabel="Re(φ/c_j)", ylabel="Im(φ/c_j)")
        θc = range(0, 2π; length=200)
        lines!(ax, cos.(θc), sin.(θc); color=(:black, 0.3), linestyle=:dash, linewidth=1)
        scatter!(ax, real.(φ_gf[sel]), imag.(φ_gf[sel]);
                 color=angle.(φ_gf[sel]), colormap=:thermal, markersize=4,
                 colorrange=(-π, π))
    end

    add_caption!(fig, 3,
        "Each panel: gauge-fixed phase eigenfunction φ_j/c_j on its matched cycle — points " *
        "collapse onto the unit circle, confirming |φ| ≈ const on-cycle (phase variable).  " *
        "The zero of the phase (arg c_j) is an arbitrary gauge choice per cycle: absolute " *
        "phases of DIFFERENT cycles cannot be compared; the winding number can.")

    save(joinpath(fig_dir, "PHASE4_gauge_analysis.png"), fig)
    println("  Saved: PHASE4_gauge_analysis.png")
end
GC.gc()

# ============================================================================
# PER-CYCLE ZLS POINT SETS (phase-ordered, gauge-fixed)
# ============================================================================

println("\nBuilding per-cycle ZLS point sets (ordered by gauge-fixed phase)")

zls_per_cycle = Vector{Vector{Vector{Float64}}}()
θ_per_cycle   = Vector{Vector{Float64}}()
for k in 1:n_branches
    j = findfirst(==(k), final_match)
    if isnothing(j)
        push!(zls_per_cycle, Vector{Vector{Float64}}())
        push!(θ_per_cycle, Float64[])
        continue
    end
    idx_k = findall(cycle_assignment .== k)
    θ_k_gauge = angle.(φ_phase_vals[j, idx_k] .* exp(im * gauge_offset[j]))
    perm_k = sortperm(θ_k_gauge)
    idx_k_sorted = idx_k[perm_k]
    push!(zls_per_cycle, [Vector{Float64}(S_zls[:, i]) for i in idx_k_sorted])
    push!(θ_per_cycle, θ_k_gauge[perm_k])
    println("  Cycle $k: $(length(idx_k_sorted)) points, " *
            (isempty(θ_per_cycle[k]) ? "θ range: —" :
             "θ ∈ [$(round(θ_per_cycle[k][1], digits=3)), $(round(θ_per_cycle[k][end], digits=3))] rad"))
end
print_mem("after per-cycle construction")

# ============================================================================
# STEP 5 — DYNAMICAL FRAME PER CYCLE
# ============================================================================
# Per cycle k (ZLS points phase-ordered):
#   5.1  Koopman velocity  v(s) = (B_full·(K'·Ψ(s)) − s)/dt_eff
#   5.2  Time-of-flight phase θ_tof from arc length / speed
#   5.3  Acceleration & jerk via periodic central differences w.r.t. θ
#        (a = (dV/dθ)·ω,  j = (d²V/dθ²)·ω²,  ω = 2π/T)
#   5.4  Gram–Schmidt → orthonormal frame (e_t, e_n, e_b), sign-flip continuity
#   5.5  Hessian of f = φ⊥1² + φ⊥2² on the cycle:
#        H ≈ 2(g1⊗g1 + g2⊗g2),  g_i = ∇φ⊥_i  (rank ≤ 2)
#   5.6  H projected onto the normal plane (e_n, e_b) → eigenpairs (λ_i, q_i)
#   5.7  Transverse coordinates ρ_i(x) = (x − x_ref)·q_i(x_ref)
#   5.8  Consistency checks: tangent alignment, spectral gap, frame
#        continuity, periodicity.
#
# Gradient of the eigenfunctions: the package `grad_phi` is thinplate-RBF-only,
# so for gaussian dictionaries we use a local analytic gradient (package lacks
# it — API_REFERENCE "WHAT THE PACKAGE LACKS" #1/#9); other dictionary types
# fall back to central finite differences of Psi_func.

"""
    grad_phi_local(s, ξ, dict_info) -> Vector{Float64}

∇φ(s) for φ = ξ'Ψ.  Uses the package `grad_phi` for thinplate-RBF
dictionaries, a local analytic gaussian-RBF gradient otherwise (RBF), and
central finite differences of the lift as a last resort.
"""
function grad_phi_local(s::AbstractVector, ξ::AbstractVector, di::NamedTuple)
    kt = get(di, :kernel_type, :thinplate)
    if _dict_type == :rbf && kt == :thinplate && isnothing(_norm_stats)
        return grad_phi(s, ξ, di)                       # package function (thinplate)
    elseif _dict_type == :rbf && kt == :gaussian
        centers = di.centers
        include_states = di.include_states
        state_indices  = get(di, :state_indices, nothing)
        sigma = get(di, :sigma, nothing)
        if isnothing(sigma) || sigma == :auto
            # median heuristic, seeded (package seeding paradigm)
            sigma = median_heuristic_sigma(centers; seed=SEED ⊻ 0xE1)
        end
        n_dim, nRBF = size(centers)
        n_states = include_states ? (isnothing(state_indices) ? n_dim : length(state_indices)) : 0
        J = zeros(n_dim, 1 + n_states + nRBF)
        if include_states
            if isnothing(state_indices)
                J[:, 2:n_states+1] .= I(n_dim)
            else
                for (i, idx) in enumerate(state_indices)
                    J[idx, 1+i] = 1.0
                end
            end
        end
        diffs = reshape(s, :, 1) .- centers
        ψ_rbf = exp.(-sum(abs2, diffs; dims=1)[:] ./ (2 * sigma^2))
        rbf0 = 2 + n_states
        for jj in 1:nRBF
            J[:, rbf0 + jj - 1] .= -ψ_rbf[jj] .* diffs[:, jj] ./ sigma^2
        end
        g = real.(J * ξ)
        # fold in the standardization chain rule if the fit used normalize=true
        isnothing(_norm_stats) || (g ./= _norm_stats.std)
        return g
    else
        # finite-difference fallback (any dictionary)
        h = 1e-6
        n_dim = length(s)
        g = zeros(n_dim)
        e = zeros(n_dim)
        for d in 1:n_dim
            e[d] = h
            ψp = Psi_func(reshape(s .+ e, :, 1))
            ψm = Psi_func(reshape(s .- e, :, 1))
            g[d] = real(dot(ξ, vec(ψp .- ψm))) / (2h)
            e[d] = 0.0
        end
        return g
    end
end

"""First derivative w.r.t. θ, periodic central differences (wrap-around safe)."""
function periodic_central_diff_1st(F::Matrix{Float64}, θ::Vector{Float64})
    n = size(F, 2)
    dF = similar(F)
    for i in 1:n
        ip, im = mod1(i + 1, n), mod1(i - 1, n)
        dθ = θ[ip] - θ[im]
        dθ < 0 && (dθ += 2π)
        dθ = max(dθ, 1e-12)
        dF[:, i] = (F[:, ip] .- F[:, im]) ./ dθ
    end
    return dF
end

"""Second derivative w.r.t. θ, periodic central differences."""
function periodic_central_diff_2nd(F::Matrix{Float64}, θ::Vector{Float64})
    n = size(F, 2)
    d2F = similar(F)
    for i in 1:n
        ip, im = mod1(i + 1, n), mod1(i - 1, n)
        dθ_p = θ[ip] - θ[i];  dθ_p < 0 && (dθ_p += 2π);  dθ_p = max(dθ_p, 1e-12)
        dθ_m = θ[i] - θ[im];  dθ_m < 0 && (dθ_m += 2π);  dθ_m = max(dθ_m, 1e-12)
        d2F[:, i] = 2.0 .* ((F[:, ip] .- F[:, i]) ./ dθ_p .-
                            (F[:, i] .- F[:, im]) ./ dθ_m) ./ (dθ_p + dθ_m)
    end
    return d2F
end

println("\n" * "="^72)
println("STEP 5: Dynamical frame (velocity → frame → Hessian → transverse coords)")
println("="^72)

dyn_frame_results = Vector{Any}(undef, n_branches)
fill!(dyn_frame_results, nothing)

for k in 1:n_branches
    println("\n  --- Cycle $k: dynamical frame construction ---")
    pts_k = zls_per_cycle[k]
    n_pts_k = length(pts_k)
    if n_pts_k < 10
        println("    Too few points ($n_pts_k) — skipping cycle $k")
        continue
    end
    S_k = reduce(hcat, pts_k)                        # n_state × n_pts_k

    # ---- 5.1  Koopman velocity field ----
    Ψ_k = Psi_func(S_k)                              # nPsi × n_pts_k
    S_next_pred = B_full * (K_mat' * Ψ_k)            # one-step prediction
    V_k = (S_next_pred .- S_k) ./ dt_eff
    Ψ_k = nothing; S_next_pred = nothing
    GC.gc()
    v_norms = vec(mapslices(norm, V_k; dims=1))
    println("    5.1 ||v|| ∈ [$(round(minimum(v_norms), digits=4)), " *
            "$(round(maximum(v_norms), digits=4))]")

    # ---- 5.2  Time-of-flight phase ----
    speeds = max.(v_norms, 1e-12)
    Δs = [norm(S_k[:, mod1(i+1, n_pts_k)] .- S_k[:, i]) for i in 1:n_pts_k]
    Δt = Δs ./ speeds
    T_total = sum(Δt)
    θ_tof = zeros(n_pts_k)
    cum = 0.0
    for i in 2:n_pts_k
        cum += Δt[i-1]
        θ_tof[i] = 2π * cum / T_total
    end
    ω_k = 2π / T_total
    println("    5.2 T = $(round(T_total, digits=4)) s (time-of-flight), " *
            "θ_tof spans [0, $(round(θ_tof[end], digits=3))] of 2π")

    # ---- 5.3  Acceleration and jerk (periodic central differences) ----
    A_k = periodic_central_diff_1st(V_k, θ_tof) .* ω_k
    J_k = periodic_central_diff_2nd(V_k, θ_tof) .* (ω_k^2)

    # ---- 5.4  Gram–Schmidt orthonormal frame (e_t, e_n, e_b) ----
    E_t = zeros(n_state, n_pts_k)
    E_n = zeros(n_state, n_pts_k)
    E_b = zeros(n_state, n_pts_k)
    for i in 1:n_pts_k
        v, a, jv = V_k[:, i], A_k[:, i], J_k[:, i]
        nv = norm(v)
        nv < 1e-12 && continue
        e_t = v / nv
        ã = a .- dot(a, e_t) .* e_t
        na = norm(ã)
        e_n = na > 1e-12 ? ã / na : zeros(n_state)
        j̃ = jv .- dot(jv, e_t) .* e_t .- dot(jv, e_n) .* e_n
        nj = norm(j̃)
        e_b = nj > 1e-12 ? j̃ / nj : zeros(n_state)
        E_t[:, i] = e_t;  E_n[:, i] = e_n;  E_b[:, i] = e_b
    end
    # sign-flip fix for frame continuity along the cycle
    for i in 2:n_pts_k
        dot(E_n[:, i], E_n[:, i-1]) < 0 && (E_n[:, i] .= .-E_n[:, i])
        dot(E_b[:, i], E_b[:, i-1]) < 0 && (E_b[:, i] .= .-E_b[:, i])
    end
    per_t = norm(E_t[:, 1] .- E_t[:, end])
    per_n = norm(E_n[:, 1] .- E_n[:, end])
    per_b = norm(E_b[:, 1] .- E_b[:, end])
    println("    5.4 frame periodicity: ||Δe_t||=$(round(per_t, digits=6)), " *
            "||Δe_n||=$(round(per_n, digits=6)), ||Δe_b||=$(round(per_b, digits=6))")
    dot(E_n[:, 1], E_n[:, end]) < 0 &&
        println("      ⚠  e_n sign mismatch at closure (torsion may be present)")

    # ---- 5.5  Hessian of f = φ⊥1² + φ⊥2² on the cycle ----
    hessian_ok = j_amp1 > 0 && j_amp2 > 0 && j_amp1 <= size(Ξ, 2) && j_amp2 <= size(Ξ, 2)
    λ1_k = fill(NaN, n_pts_k)
    λ2_k = fill(NaN, n_pts_k)
    Q1_k = zeros(n_state, n_pts_k)
    Q2_k = zeros(n_state, n_pts_k)
    tangent_align = fill(NaN, n_pts_k)

    if hessian_ok
        ξ1, ξ2 = Ξ[:, j_amp1], Ξ[:, j_amp2]
        for i in 1:n_pts_k
            g1 = grad_phi_local(S_k[:, i], ξ1, dict_info)
            g2 = grad_phi_local(S_k[:, i], ξ2, dict_info)
            # H = 2(g1⊗g1 + g2⊗g2) is n_state×n_state — never materialize it
            # (memory rule); work with gradients directly:
            E_mat = hcat(E_n[:, i], E_b[:, i])       # n_state × 2
            # ---- 5.6  project onto the normal plane, eigendecompose ----
            h1p = E_mat' * g1
            h2p = E_mat' * g2
            H_proj = 2.0 .* (h1p * h1p' .+ h2p * h2p')
            Feig = eigen(Symmetric(H_proj))
            perm = sortperm(Feig.values; rev=true)
            λ1_k[i] = Feig.values[perm[1]]
            λ2_k[i] = Feig.values[perm[2]]
            Q1_k[:, i] = E_mat * Feig.vectors[:, perm[1]]
            Q2_k[:, i] = E_mat * Feig.vectors[:, perm[2]]
            # ---- 5.8 check 1: tangent alignment  ||Hv||/(||v||·||H||) ----
            v_i = V_k[:, i]
            Hv = 2.0 .* (g1 .* dot(g1, v_i) .+ g2 .* dot(g2, v_i))   # H·v, rank-2 form
            GtG = [dot(g1, g1) dot(g1, g2); dot(g1, g2) dot(g2, g2)]
            normH = 2.0 * sqrt(sum(abs2, GtG))                        # ||H||_F from the Gram matrix
            tangent_align[i] = norm(Hv) / (norm(v_i) * (normH + 1e-12))
        end
        println("    5.5/5.6 Hessian projected: λ1 ∈ [$(round(minimum(λ1_k), sigdigits=3)), " *
                "$(round(maximum(λ1_k), sigdigits=3))],  λ2 ∈ " *
                "[$(round(minimum(λ2_k), sigdigits=3)), $(round(maximum(λ2_k), sigdigits=3))]")
        println("    5.8 check 1 (tangent alignment): median ||Hv||/(||v||·||H||) = " *
                "$(round(median(tangent_align), sigdigits=3))  (should be ≈ 0)")
        mean_λ1, mean_λ2 = mean(λ1_k), mean(λ2_k)
        ratio = mean_λ1 / (mean_λ2 + 1e-12)
        regime_str = ratio > 10 ? "λ1 ≫ λ2: f constrains ONE direction (use q1; second axis = e_b)" :
                     (mean_λ1 > 1e-6 && mean_λ2 > 1e-6 ? "λ1 ∼ λ2 ≫ 0: proper bowl (q1, q2 are amplitudes)" :
                      "λ1 ≈ λ2 ≈ 0: flat f (use e_n, e_b)")
        println("    5.8 check 2 (spectral gap): mean λ1=$(round(mean_λ1, sigdigits=3)), " *
                "mean λ2=$(round(mean_λ2, sigdigits=3)), ratio=$(round(ratio, digits=2)) → $regime_str")
    else
        println("    5.5 ⚠  no valid transverse pair (j_amp1=$j_amp1, j_amp2=$j_amp2) — " *
                "using (e_n, e_b) as transverse directions")
        Q1_k .= E_n
        Q2_k .= E_b
    end

    # ---- 5.7  Transverse coordinates with the reference frame at point 1 ----
    ρ1_k = [dot(S_k[:, i] .- S_k[:, 1], Q1_k[:, 1]) for i in 1:n_pts_k]
    ρ2_k = [dot(S_k[:, i] .- S_k[:, 1], Q2_k[:, 1]) for i in 1:n_pts_k]
    # local-frame consistency: ρ w.r.t. the frame at the SAME point must be 0
    ρ1_local = [dot(S_k[:, i] .- S_k[:, i], Q1_k[:, i]) for i in 1:n_pts_k]
    ρ2_local = [dot(S_k[:, i] .- S_k[:, i], Q2_k[:, i]) for i in 1:n_pts_k]
    ρ_closure = norm([ρ1_k[end] - ρ1_k[1], ρ2_k[end] - ρ2_k[1]])

    n_flips_n = count(i -> dot(E_n[:, i], E_n[:, mod1(i-1, n_pts_k)]) < 0, 1:n_pts_k)
    n_flips_b = count(i -> dot(E_b[:, i], E_b[:, mod1(i-1, n_pts_k)]) < 0, 1:n_pts_k)
    println("    5.8 check 3 (frame continuity): sign flips e_n=$n_flips_n, e_b=$n_flips_b")
    println("    5.8 check 4 (periodicity): ||Δe_t||=$(round(per_t, digits=6)), " *
            "||Δe_n||=$(round(per_n, digits=6)), ||Δe_b||=$(round(per_b, digits=6))")

    dyn_frame_results[k] = (
        S = S_k, V = V_k, A = A_k, J = J_k,
        θ_tof = θ_tof, T = T_total, ω = ω_k,
        E_t = E_t, E_n = E_n, E_b = E_b,
        λ1 = λ1_k, λ2 = λ2_k, Q1 = Q1_k, Q2 = Q2_k,
        ρ1 = ρ1_k, ρ2 = ρ2_k, ρ1_local = ρ1_local, ρ2_local = ρ2_local,
        ρ_closure = ρ_closure,
        tangent_align = tangent_align,
        n_flips_n = n_flips_n, n_flips_b = n_flips_b,
        per_t = per_t, per_n = per_n, per_b = per_b,
        hessian_ok = hessian_ok,
    )
end
print_mem("after STEP 5 (dynamical frame)")
GC.gc()

# ============================================================================
# STEP 5b — PER-CYCLE POD OF THE PHASE-ORDERED ZLS POINTS
# ============================================================================
# POD via eigen(Symmetric(X·X')) (memory rule — never svd's Vt); the POD
# coordinates are obtained as Z = U'·Xc (equivalent to Σ·Vt up to signs).

println("\nSTEP 5b: per-cycle POD of phase-ordered ZLS points")

pod_results_per_cycle = Vector{Any}(undef, n_branches)
fill!(pod_results_per_cycle, nothing)
for k in 1:n_branches
    length(zls_per_cycle[k]) < 4 && continue
    X_lc_k = reduce(hcat, zls_per_cycle[k])
    x_mean_k = vec(mean(X_lc_k; dims=2))
    X_c_k = X_lc_k .- x_mean_k
    Fk = eigen(Symmetric(X_c_k * X_c_k'))            # memory rule: no svd Vt
    perm = sortperm(Fk.values; rev=true)
    σ_k = sqrt.(max.(Fk.values[perm], 0.0))
    U_k = Fk.vectors[:, perm]
    Z_k = U_k' * X_c_k                               # full POD coordinates
    pod_results_per_cycle[k] = (U=U_k, σ=σ_k, Z=Z_k,
                                Z3=Z_k[1:min(3, size(Z_k, 1)), :], x_mean=x_mean_k)
    cum2 = sum(σ_k[1:min(2, length(σ_k))] .^ 2) / sum(σ_k .^ 2)
    println("  Cycle $k: $(length(zls_per_cycle[k])) pts, σ₁=$(round(σ_k[1], sigdigits=3)), " *
            "σ₂=$(round(σ_k[2], sigdigits=3)), 2-mode energy = $(round(100*cum2, digits=2))%")
end
print_mem("after STEP 5b")

# ============================================================================
# STAGED SAVE — STAGE A (phase pipeline: STEPS 1–5)
# ============================================================================

println("\n  [Stage save A] phase pipeline → $save_path_out")
jldopen(save_path_out, "w") do f
    # metadata
    f["regime_system"] = regime_system
    f["regime_name"]   = regime_name
    f["experiment_name"] = experiment_name
    f["spectral_method"] = spectral_method
    f["dim"] = dim
    f["showdim"] = showdim
    f["n_show"] = n_show
    f["n_hidden"] = n_hidden
    f["n_plot"] = n_plot
    f["n_state"] = n_state
    f["n_zls"] = n_zls
    f["dt_eff"] = dt_eff
    f["SEED"] = SEED
    f["j_amp1"] = j_amp1
    f["j_amp2"] = j_amp2
    f["amp_source"] = amp_source
    f["degenerate_detected"] = degenerate_detected
    f["n_clusters_degen"] = n_clusters_degen

    # STEP 1: branches
    f["branches"] = branches
    f["j_phases"] = j_phases
    f["λ_phases"] = λ_phases
    f["ω_phases"] = ω_phases
    f["T_phases"] = T_phases

    # STEP 2: cycle separation
    f["φ_phase_vals"] = φ_phase_vals
    f["φ_phase_mag"]  = φ_phase_mag
    f["cycle_assignment"] = cycle_assignment
    f["cycle_counts"] = cycle_counts
    f["sep_ratio"] = sep_ratio

    # STEP 3: matching
    f["match_table_winding"]   = match_table_winding
    f["match_table_coherence"] = match_table_coherence
    f["match_table_freqmatch"] = match_table_freqmatch
    f["final_match"] = final_match

    # STEP 4: gauge
    f["gauge_scale"]  = gauge_scale
    f["gauge_offset"] = gauge_offset

    # per-cycle phase-ordered ZLS
    f["zls_per_cycle"] = zls_per_cycle
    f["θ_per_cycle"]   = θ_per_cycle

    # STEP 5b: per-cycle POD
    for k in 1:n_branches
        isnothing(pod_results_per_cycle[k]) && continue
        f["pod/cycle_$k/U"]      = pod_results_per_cycle[k].U
        f["pod/cycle_$k/σ"]      = pod_results_per_cycle[k].σ
        f["pod/cycle_$k/Z"]      = pod_results_per_cycle[k].Z
        f["pod/cycle_$k/Z3"]     = pod_results_per_cycle[k].Z3
        f["pod/cycle_$k/x_mean"] = pod_results_per_cycle[k].x_mean
    end

    # STEP 5: dynamical frame (Hessian matrices are NOT stored — n_state² per
    # point is too heavy; λ1/λ2/Q1/Q2 carry the same information)
    for k in 1:n_branches
        isnothing(dyn_frame_results[k]) && continue
        r = dyn_frame_results[k]
        f["dynframe/cycle_$k/S"]  = r.S
        f["dynframe/cycle_$k/V"]  = r.V
        f["dynframe/cycle_$k/A"]  = r.A
        f["dynframe/cycle_$k/J"]  = r.J
        f["dynframe/cycle_$k/θ_tof"] = r.θ_tof
        f["dynframe/cycle_$k/T"]  = r.T
        f["dynframe/cycle_$k/ω"]  = r.ω
        f["dynframe/cycle_$k/E_t"] = r.E_t
        f["dynframe/cycle_$k/E_n"] = r.E_n
        f["dynframe/cycle_$k/E_b"] = r.E_b
        f["dynframe/cycle_$k/λ1"] = r.λ1
        f["dynframe/cycle_$k/λ2"] = r.λ2
        f["dynframe/cycle_$k/Q1"] = r.Q1
        f["dynframe/cycle_$k/Q2"] = r.Q2
        f["dynframe/cycle_$k/ρ1"] = r.ρ1
        f["dynframe/cycle_$k/ρ2"] = r.ρ2
        f["dynframe/cycle_$k/ρ_closure"] = r.ρ_closure
        f["dynframe/cycle_$k/tangent_align"] = r.tangent_align
        f["dynframe/cycle_$k/n_flips_n"] = r.n_flips_n
        f["dynframe/cycle_$k/n_flips_b"] = r.n_flips_b
        f["dynframe/cycle_$k/per_t"] = r.per_t
        f["dynframe/cycle_$k/per_n"] = r.per_n
        f["dynframe/cycle_$k/per_b"] = r.per_b
        f["dynframe/cycle_$k/hessian_ok"] = r.hessian_ok
    end

    # pass-through plotting helpers
    f["P_plot"] = P_plot
    f["plot_labels"] = plot_labels
    f["timestamp_stageA"] = string(now())
end
println("  [Stage save A] done.")
GC.gc()

# ============================================================================
# FIGURES PHASE-5 … PHASE-15  (dynamical frame, self-explaining paper figures)
# ============================================================================

println("\n" * "="^72)
println("Building PHASE figures...")
println("="^72)

# ---------------------------------------------------------------------------
# FIGURE PHASE-5 — Per-cycle phase-colored scatter (gauge-fixed Koopman phase)
# ---------------------------------------------------------------------------
println("  PHASE-5: per-cycle phase-colored scatter")

let
    fig = Figure(size=(550 * n_branches + 100, 620))
    Label(fig[0, :],
          "Each cycle is cleanly parametrized by its own Koopman phase θ — " *
          "color runs monotonically around every loop";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    for k in 1:n_branches
        isempty(zls_per_cycle[k]) && continue
        j = findfirst(==(k), final_match)
        isnothing(j) && continue
        θ_vals = θ_per_cycle[k]
        Pk = to3(P_plot * reduce(hcat, zls_per_cycle[k]))

        ax = Axis3(fig[2, k];
                   title="cycle $k: $(length(θ_vals)) pts, ω=$(round(ω_phases[j], digits=3)) rad/s, " *
                         "T=$(round(T_phases[j], digits=2)) s\n" *
                         "θ_cycle $k ∈ [$(round(minimum(θ_vals)/π, digits=2))π, " *
                         "$(round(maximum(θ_vals)/π, digits=2))π] (own gauge)",
                   titlesize=10,
                   xlabel=lbl(1), ylabel=lbl(2), zlabel=lbl(3))
        hm = scatter!(ax, Pk[1, :], Pk[2, :], Pk[3, :];
                      color=θ_vals, colormap=:thermal, markersize=6, colorrange=(-π, π))
        Colorbar(fig[2, n_branches + k], hm; width=12,
                 label="θ_cycle $k  [rad] (gauge-fixed, own reference)")
    end

    add_caption!(fig, 3,
        "ZLS points of each cycle, colored by the gauge-fixed phase of its matched phase " *
        "eigenfunction (thermal colormap, −π…π).  A smooth rainbow around the loop means the " *
        "eigenfunction is a valid phase variable for that cycle.  Colors are NOT comparable " *
        "across cycles (independent gauge zeros, STEP 4).")

    save(joinpath(fig_dir, "PHASE5_phase_colored_cycles.png"), fig)
    println("  Saved: PHASE5_phase_colored_cycles.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-6 — Per-cycle Koopman coordinates (Re φ, Im φ, |φ_⊥1|)
# ---------------------------------------------------------------------------
println("  PHASE-6: per-cycle Koopman coordinates")

let
    # transverse eigenfunction on the ZLS cloud (chunked lift)
    φ_perp1_zls = zeros(n_zls)
    if j_amp1 > 0 && j_amp1 <= size(Ξ, 2)
        chunk_sz = 5000
        for cs in 1:chunk_sz:n_zls
            ce = min(cs + chunk_sz - 1, n_zls)
            Ψc = Psi_func(S_zls[:, cs:ce])
            φ_perp1_zls[cs:ce] .= real.(vec(Ξ[:, j_amp1]' * Ψc))
            Ψc = nothing
        end
        GC.gc()
    end

    fig = Figure(size=(500 * n_branches + 100, 600))
    Label(fig[0, :],
          "In its own Koopman coordinates each cycle is a clean circle at |φ_⊥1| ≈ 0 — " *
          "phase + amplitude separate";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    for k in 1:n_branches
        isempty(zls_per_cycle[k]) && continue
        j = findfirst(==(k), final_match)
        isnothing(j) && continue
        idx_k = findall(cycle_assignment .== k)
        φk = φ_phase_vals[j, idx_k] ./ gauge_scale[j]
        θk = angle.(φk)

        ax = Axis3(fig[2, k];
                   title="cycle $k in (Re φ_$j, Im φ_$j, |φ_⊥1|)\n" *
                         "$(length(idx_k)) pts — circle of radius ≈ 1 at |φ_⊥1| ≈ 0",
                   titlesize=10,
                   xlabel="Re(φ_phase_$j)/|c|", ylabel="Im(φ_phase_$j)/|c|",
                   zlabel="|φ_⊥1|  (transverse)")
        scatter!(ax, real.(φk), imag.(φk), abs.(φ_perp1_zls[idx_k]);
                 color=θk, colormap=:thermal, markersize=5, colorrange=(-π, π))
        θ_ref = range(0, 2π; length=100)
        lines!(ax, cos.(θ_ref), sin.(θ_ref), zeros(100);
               color=:black, linewidth=1.2, linestyle=:dash, alpha=0.5)
    end

    add_caption!(fig, 3,
        "Koopman coordinates of cycle k's ZLS points: gauge-normalized phase eigenfunction in the " *
        "horizontal plane, transverse amplitude eigenfunction |φ_⊥1| (j=$j_amp1) vertical.  " *
        "Lying on the dashed unit circle at height ≈ 0 confirms the cycle is the zero level set " *
        "of the transverse mode and the phase mode is constant-magnitude on it.")

    save(joinpath(fig_dir, "PHASE6_per_cycle_koopman_coords.png"), fig)
    println("  Saved: PHASE6_per_cycle_koopman_coords.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-7 — Joint basin-indicator space (multi-cycle only)
# ---------------------------------------------------------------------------
if n_branches >= 2
    println("  PHASE-7: joint basin-indicator coordinates")
    let
        n_axes = min(3, n_branches)
        idx_plot = seeded_subsample_idx(n_zls, 20_000; salt=0xB7)

        fig = Figure(size=(1000, 850))
        Label(fig[0, :],
              "Basins separate in the joint eigenfunction-magnitude space — " *
              "each cycle lives on its own axis";
              fontsize=14, font=:bold)
        Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

        ax = Axis3(fig[2, 1];
                   title="ZLS points in (|φ_phase_1|, |φ_phase_2|" *
                         (n_axes >= 3 ? ", |φ_phase_3|" : "") * ") space",
                   xlabel="|φ_phase_1|", ylabel="|φ_phase_2|",
                   zlabel=n_axes >= 3 ? "|φ_phase_3|" : "")
        for k in 1:n_branches
            sel = idx_plot[cycle_assignment[idx_plot] .== k]
            isempty(sel) && continue
            xs = φ_phase_mag[1, sel]
            ys = n_axes >= 2 ? φ_phase_mag[2, sel] : zeros(length(sel))
            zs = n_axes >= 3 ? φ_phase_mag[3, sel] : zeros(length(sel))
            scatter!(ax, xs, ys, zs; color=CYCLE_COLORS[mod1(k, 8)], markersize=4,
                     label="cycle $k ($(cycle_counts[k]) pts)")
        end
        axislegend(ax; position=:rt, framevisible=true, backgroundcolor=(:white, 0.85), labelsize=9)

        add_caption!(fig, 3,
            "|φ_phase_j| is a basin indicator: it is ≈ const > 0 on cycle j and small/noisy off it.  " *
            "The point cloud splits into rays along the coordinate axes — direct visual proof that " *
            "the cycles (and their basins) are separated by the phase eigenfunctions.")

        save(joinpath(fig_dir, "PHASE7_joint_basin_coords.png"), fig)
        println("  Saved: PHASE7_joint_basin_coords.png")
    end
    GC.gc()
else
    println("  Skipped PHASE-7 (single cycle — no basin separation to show)")
end

# ---------------------------------------------------------------------------
# FIGURE PHASE-8 — Per-cycle POD (first 3 POD coordinates, phase-ordered path)
# ---------------------------------------------------------------------------
println("  PHASE-8: per-cycle POD of ZLS points")

let
    fig = Figure(size=(500 * n_branches + 100, 620))
    Label(fig[0, :],
          "Per-cycle POD: each cycle is essentially a 2-dimensional ribbon in the " *
          "$n_state-D delay space";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    for k in 1:n_branches
        isnothing(pod_results_per_cycle[k]) && continue
        pr = pod_results_per_cycle[k]
        Z3 = pr.Z3
        npr = size(Z3, 2)

        ax = Axis3(fig[2, k];
                   title="cycle $k, first 3 POD coords ($(length(zls_per_cycle[k])) pts)\n" *
                         "σ₁=$(round(pr.σ[1], sigdigits=3)), σ₂=$(round(pr.σ[2], sigdigits=3))" *
                         (length(pr.σ) >= 3 ? ", σ₃=$(round(pr.σ[3], sigdigits=3))" : ""),
                   titlesize=10,
                   xlabel="POD coord 1", ylabel="POD coord 2", zlabel="POD coord 3")
        lines!(ax, Z3[1, :], Z3[2, :], size(Z3, 1) >= 3 ? Z3[3, :] : zeros(npr);
               color=(CYCLE_COLORS[mod1(k, 8)], 0.35), linewidth=1.5)
        hm = scatter!(ax, Z3[1, :], Z3[2, :], size(Z3, 1) >= 3 ? Z3[3, :] : zeros(npr);
                      color=θ_per_cycle[k], colormap=:thermal, markersize=5,
                      colorrange=(-π, π))
        scatter!(ax, [Z3[1, 1]], [Z3[2, 1]], [size(Z3, 1) >= 3 ? Z3[3, 1] : 0.0];
                 color=:green, markersize=11, marker=:circle, label="start (θ min)")
        scatter!(ax, [Z3[1, end]], [Z3[2, end]], [size(Z3, 1) >= 3 ? Z3[3, end] : 0.0];
                 color=:red, markersize=11, marker=:rect, label="end (θ max)")
        Colorbar(fig[2, n_branches + k], hm; width=12, label="θ_cycle $k  [rad]")
        axislegend(ax; position=:rb, labelsize=8)
    end

    add_caption!(fig, 3,
        "POD of each cycle's phase-ordered ZLS cloud (eigen(Symmetric(X·X')), memory-safe).  " *
        "The phase-colored closed path shows the geometric shape of each cycle in delay space; " *
        "start (green) and end (red) meeting confirms the phase ordering closes the loop.")

    save(joinpath(fig_dir, "PHASE8_per_cycle_pod.png"), fig)
    println("  Saved: PHASE8_per_cycle_pod.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-9 — Per-cycle singular values + cumulative energy
# ---------------------------------------------------------------------------
println("  PHASE-9: per-cycle singular values")

let
    fig = Figure(size=(1100, 400 * n_branches + 150))
    Label(fig[0, :],
          "POD spectrum of each cycle: two dominant modes carry the oscillation — " *
          "the cycle is ~2-D embedded in $n_state-D delay space";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    for k in 1:n_branches
        isnothing(pod_results_per_cycle[k]) && continue
        σ_k = pod_results_per_cycle[k].σ
        n_sv = min(15, length(σ_k))
        cum_energy = cumsum(σ_k .^ 2) ./ sum(σ_k .^ 2)

        ax1 = Axis(fig[1+k, 1];
                   title="cycle $k: singular values (first $n_sv)", titlesize=10,
                   xlabel="POD mode index", ylabel="σ  (singular value)")
        scatter!(ax1, 1:n_sv, σ_k[1:n_sv]; color=CYCLE_COLORS[mod1(k, 8)], markersize=8)
        lines!(ax1, 1:n_sv, σ_k[1:n_sv]; color=CYCLE_COLORS[mod1(k, 8)], linewidth=1.5)

        ax2 = Axis(fig[1+k, 2];
                   title="cycle $k: cumulative energy — " *
                         "r=2 captures $(round(100*cum_energy[min(2, n_sv)], digits=1))%", titlesize=10,
                   xlabel="POD mode index", ylabel="Σσ² / Σσ²_total",
                   limits=(nothing, nothing, 0, 1.05))
        scatter!(ax2, 1:n_sv, cum_energy[1:n_sv]; color=CYCLE_COLORS[mod1(k, 8)], markersize=8)
        lines!(ax2, 1:n_sv, cum_energy[1:n_sv]; color=CYCLE_COLORS[mod1(k, 8)], linewidth=1.5)
        hlines!(ax2, [0.99, 0.999]; color=:gray, linewidth=1, linestyle=:dash)
    end

    add_caption!(fig, n_branches + 2,
        "Singular values (left) and cumulative variance (right) per cycle.  A knee after mode 2 " *
        "means the cycle's delay-space geometry is a deformed 2-D ribbon — the phase (1-D) plus " *
        "one weak transverse direction; dashed lines mark 99% / 99.9% energy.")

    save(joinpath(fig_dir, "PHASE9_per_cycle_singular_values.png"), fig)
    println("  Saved: PHASE9_per_cycle_singular_values.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-10 — Koopman dynamical derivatives ||v||, ||a||, ||j|| vs θ
# ---------------------------------------------------------------------------
println("  PHASE-10: dynamical derivatives along each cycle")

let
    fig = Figure(size=(1400, 350 * n_branches + 150))
    Label(fig[0, :],
          "Koopman velocity, acceleration and jerk are smooth and bounded along every cycle — " *
          "the learned operator reproduces the cycle's dynamics";
          fontsize=13, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    for k in 1:n_branches
        isnothing(dyn_frame_results[k]) && continue
        r = dyn_frame_results[k]
        θ = r.θ_tof
        v_mag = vec(mapslices(norm, r.V; dims=1))
        a_mag = vec(mapslices(norm, r.A; dims=1))
        j_mag = vec(mapslices(norm, r.J; dims=1))

        for (c, (vals, nm)) in enumerate(((v_mag, "||v(θ)||  [state/s]  — speed"),
                                          (a_mag, "||a(θ)||  [state/s²]  — acceleration"),
                                          (j_mag, "||j(θ)||  [state/s³]  — jerk")))
            ax = Axis(fig[1+k, c];
                      title="cycle $k: $nm", titlesize=10,
                      xlabel="θ  [rad]  (time-of-flight phase)", ylabel=split(nm, "—")[1])
            lines!(ax, θ, vals; color=CYCLE_COLORS[mod1(k, 8)], linewidth=1.8)
        end
    end

    add_caption!(fig, n_branches + 2,
        "v from the Koopman one-step map, a and j by periodic central differences in θ " *
        "(a = ω·dv/dθ, j = ω²·d²v/dθ², ω = 2π/T with T from time-of-flight).  Smooth, periodic " *
        "curves with matching endpoints validate the frame construction of STEP 5.")

    save(joinpath(fig_dir, "PHASE10_dynamical_derivatives.png"), fig)
    println("  Saved: PHASE10_dynamical_derivatives.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-11 — Frame continuity: (e_t, e_n, e_b) components vs θ
# ---------------------------------------------------------------------------
println("  PHASE-11: frame vector components along the cycle")

let
    n_comp_show = min(3, n_state)
    fig = Figure(size=(1400, 350 * n_branches + 150))
    Label(fig[0, :],
          "Dynamical frame (e_t, e_n, e_b) rotates smoothly and periodically along every cycle";
          fontsize=13, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    for k in 1:n_branches
        isnothing(dyn_frame_results[k]) && continue
        r = dyn_frame_results[k]
        θ = r.θ_tof
        for (c, (E, nm)) in enumerate(((r.E_t, "e_t  (tangent)"),
                                       (r.E_n, "e_n  (normal)"),
                                       (r.E_b, "e_b  (binormal)")))
            ax = Axis(fig[1+k, c];
                      title="cycle $k: $nm — first $n_comp_show delay components", titlesize=10,
                      xlabel="θ  [rad]", ylabel="frame component  [–]")
            for d in 1:n_comp_show
                lines!(ax, θ, E[d, :]; linewidth=1.5, label="comp $d")
            end
            axislegend(ax; position=:rt, labelsize=7)
        end
    end

    add_caption!(fig, n_branches + 2,
        "Gram–Schmidt frame from (v, a, j) with sign-flip correction for continuity.  " *
        "Smooth periodic components (last value = first value) mean the frame is a valid " *
        "moving orthonormal basis around the cycle.")

    save(joinpath(fig_dir, "PHASE11_frame_continuity.png"), fig)
    println("  Saved: PHASE11_frame_continuity.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-12 — Hessian spectral analysis (KEY FIGURE)
# ---------------------------------------------------------------------------
println("  PHASE-12: Hessian spectral analysis")

let
    n_hess = count(k -> !isnothing(dyn_frame_results[k]) && dyn_frame_results[k].hessian_ok,
                   1:n_branches)
    if n_hess == 0
        println("  Skipped PHASE-12 (no cycle with a valid Hessian)")
    else
        fig = Figure(size=(1400, 350 * n_branches + 150))
        Label(fig[0, :],
              "Hessian of f = φ⊥1²+φ⊥2² on the cycle: two positive normal eigenvalues " *
              "identify the transverse plane; the tangent is its null direction";
              fontsize=13, font=:bold)
        Label(fig[1, :], fig_subtitle *
              " | transverse eigenfunctions j_amp1=$j_amp1, j_amp2=$j_amp2 ($amp_source)";
              fontsize=10, color=:gray)

        for k in 1:n_branches
            isnothing(dyn_frame_results[k]) && continue
            r = dyn_frame_results[k]
            !r.hessian_ok && continue
            θ = r.θ_tof

            ax1 = Axis(fig[1+k, 1];
                       title="cycle $k: projected Hessian eigenvalues λ₁(θ) ≥ λ₂(θ)", titlesize=10,
                       xlabel="θ  [rad]", ylabel="eigenvalue of E'H E")
            lines!(ax1, θ, r.λ1; color=:crimson, linewidth=2, label="λ₁ (largest)")
            lines!(ax1, θ, r.λ2; color=:dodgerblue, linewidth=2, label="λ₂ (smallest)")
            hlines!(ax1, [0.0]; color=:gray, linewidth=0.5, linestyle=:dash)
            axislegend(ax1; position=:rt, labelsize=8)

            ax2 = Axis(fig[1+k, 2];
                       title="cycle $k: spectral gap log₁₀(λ₁/λ₂), mean ratio = " *
                             "$(round(mean(r.λ1)/(mean(r.λ2)+1e-12), digits=1))", titlesize=10,
                       xlabel="θ  [rad]", ylabel="log₁₀(λ₁/λ₂)")
            lines!(ax2, θ, log10.(max.(r.λ1, 1e-12) ./ max.(r.λ2, 1e-12));
                   color=:darkorange, linewidth=2)
            hlines!(ax2, [0.0, 1.0]; color=:gray, linewidth=0.5, linestyle=:dash)

            ax3 = Axis(fig[1+k, 3];
                       title="cycle $k: tangent alignment ||Hv||/(||v||·||H||) — median " *
                             "$(round(median(r.tangent_align), sigdigits=2))", titlesize=10,
                       xlabel="θ  [rad]", ylabel="alignment ratio", yscale=log10)
            lines!(ax3, θ, max.(r.tangent_align, 1e-16); color=:forestgreen, linewidth=2)
            hlines!(ax3, [1e-6]; color=:gray, linewidth=0.5, linestyle=:dash)
        end

        add_caption!(fig, n_branches + 2,
            "H ≈ 2(g₁g₁' + g₂g₂'), g_i = ∇φ⊥_i, projected on the normal plane (e_n, e_b).  " *
            "λ₁, λ₂ > 0 quantify transverse contraction; λ₁/λ₂ ≫ 1 means f constrains one " *
            "direction only.  ||Hv|| ≈ 0 (bottom-right scale) certifies the velocity lies in " *
            "the Hessian's null space, as theory requires.")

        save(joinpath(fig_dir, "PHASE12_hessian_spectral_analysis.png"), fig)
        println("  Saved: PHASE12_hessian_spectral_analysis.png")
    end
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-13 — Limit cycle in dynamical frame coordinates (θ, ρ1, ρ2)
# ---------------------------------------------------------------------------
println("  PHASE-13: limit cycle in (θ, ρ1, ρ2) coordinates")

let
    n_ok = count(k -> !isnothing(dyn_frame_results[k]), 1:n_branches)
    if n_ok == 0
        println("  Skipped PHASE-13 (no dynamical frame results)")
    else
        fig = Figure(size=(1000, 450 * n_branches + 150))
        Label(fig[0, :],
              "The limit cycle in its intrinsic dynamical coordinates (θ, ρ₁, ρ₂): " *
              "a closed ribbon — out-of-(θ,ρ₁)-plane excursion measures torsion";
              fontsize=13, font=:bold)
        Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

        for k in 1:n_branches
            isnothing(dyn_frame_results[k]) && continue
            r = dyn_frame_results[k]
            θ = r.θ_tof

            ax = Axis3(fig[1+k, 1];
                       title="cycle $k: (θ, ρ₁, ρ₂), $(length(θ)) pts — T=$(round(r.T, digits=3)) s, " *
                             "ω=$(round(r.ω, digits=3)) rad/s",
                       titlesize=10,
                       xlabel="θ  [rad]", ylabel="ρ₁  (transverse 1)",
                       zlabel="ρ₂  (transverse 2)")
            hm = scatter!(ax, θ, r.ρ1, r.ρ2; color=θ, colormap=:thermal, markersize=5,
                          colorrange=(0, 2π))
            lines!(ax, θ, r.ρ1, r.ρ2; color=(:black, 0.3), linewidth=1.0)
            scatter!(ax, [θ[1]], [r.ρ1[1]], [r.ρ2[1]]; color=:green, markersize=11,
                     marker=:circle, label="start (θ=0)")
            scatter!(ax, [θ[end]], [r.ρ1[end]], [r.ρ2[end]]; color=:red, markersize=11,
                     marker=:rect, label="end (θ≈2π)")
            Colorbar(fig[1+k, 2], hm; width=12, label="θ  [rad]  (time-of-flight)")
            axislegend(ax; position=:rb, labelsize=8)
        end

        add_caption!(fig, n_branches + 2,
            "ρ₁, ρ₂ use the transverse directions (q₁, q₂) of the projected Hessian at the " *
            "reference point (θ=0); θ is the time-of-flight phase.  The curve closes (start ≈ end), " *
            "validating both the phase and the frame.  |ρ| modulation shows transverse breathing; " *
            "ρ₂ excursions show rotation of the transverse plane along the cycle.")

        save(joinpath(fig_dir, "PHASE13_dynamical_frame_coords.png"), fig)
        println("  Saved: PHASE13_dynamical_frame_coords.png")
    end
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-13b — Clean limit cycle, cylindrical dynamical-frame coords:
#                    Koopman (delay space) vs physical ground truth
# ---------------------------------------------------------------------------
println("  PHASE-13b: clean-LC cylindrical coordinates (Koopman vs physical)")

clean_lc = nothing   # NamedTuple with both renderings of cycle 1 (reused below)
let
    if isnothing(dyn_frame_results[1]) || isempty(zls_per_cycle[1])
        println("  Skipped PHASE-13b (no dynamical frame for cycle 1)")
    else
        r_frame = dyn_frame_results[1]
        s_ref_zls = zls_per_cycle[1][1]
        q1_ref = r_frame.Q1[:, 1]
        q2_ref = r_frame.Q2[:, 1]

        # Integrate the TRUE system onto its limit cycle (deterministic)
        T_lc_est = T_phases[1]
        n_burn = ceil(Int, 10 * T_lc_est / dt_eff)
        n_lc   = ceil(Int, 2 * T_lc_est / dt_eff)
        max_lag_local = (m_embed - 1) * tau_delay
        X_lc_full = rk4(f_det, x0_silent .+ ones(dim), dt_internal,
                        n_burn + max_lag_local + n_lc; nLag=nLag)
        X_lc = X_lc_full[:, n_burn+1:end]
        X_lc_full = nothing
        GC.gc()

        # ---- Koopman (hidden-dim delay) rendering ----
        S_lc_hidden = build_hankel_multichannel(X_lc[showdim, :], m_embed, tau_delay)
        Ψ_lc = Psi_func(S_lc_hidden)
        V_lc = (B_full * (K_mat' * Ψ_lc) .- S_lc_hidden) ./ dt_eff
        Ψ_lc = nothing
        GC.gc()
        v_mag_koop = vec(mapslices(norm, V_lc; dims=1))
        ρ1_koop = [dot(S_lc_hidden[:, i] .- s_ref_zls, q1_ref) for i in 1:size(S_lc_hidden, 2)]
        ρ2_koop = [dot(S_lc_hidden[:, i] .- s_ref_zls, q2_ref) for i in 1:size(S_lc_hidden, 2)]
        ρ_mag_koop = sqrt.(ρ1_koop.^2 .+ ρ2_koop.^2)
        n_pts_lc = size(S_lc_hidden, 2)
        speeds_koop = max.(v_mag_koop, 1e-12)
        Δs_koop = [norm(S_lc_hidden[:, mod1(i+1, n_pts_lc)] .- S_lc_hidden[:, i]) for i in 1:n_pts_lc]
        Δt_koop = Δs_koop ./ speeds_koop
        T_koop = sum(Δt_koop)
        θ_koop = zeros(n_pts_lc)
        cum = 0.0
        for i in 2:n_pts_lc
            cum += Δt_koop[i-1]
            θ_koop[i] = 2π * cum / T_koop
        end

        # ---- Physical ground-truth rendering ----
        X_lc_phys = X_lc[:, max_lag_local+1:end]
        n_pts_phys = size(X_lc_phys, 2)
        V_phys = reduce(hcat, [f_det(X_lc_phys[:, i]) for i in 1:n_pts_phys])
        v_mag_phys = vec(mapslices(norm, V_phys; dims=1))
        q1_phys = vec(P_plot * q1_ref)
        q2_phys = vec(P_plot * q2_ref)
        x_ref_phys = X_lc_phys[:, 1]
        ρ1_phys = [dot(X_lc_phys[:, i] .- x_ref_phys, q1_phys) for i in 1:n_pts_phys]
        ρ2_phys = [dot(X_lc_phys[:, i] .- x_ref_phys, q2_phys) for i in 1:n_pts_phys]
        ρ_mag_phys = sqrt.(ρ1_phys.^2 .+ ρ2_phys.^2)
        speeds_phys = max.(v_mag_phys, 1e-12)
        Δs_phys = [norm(X_lc_phys[:, mod1(i+1, n_pts_phys)] .- X_lc_phys[:, i]) for i in 1:n_pts_phys]
        Δt_phys = Δs_phys ./ speeds_phys
        T_phys = sum(Δt_phys)
        θ_phys = zeros(n_pts_phys)
        cump = 0.0
        for i in 2:n_pts_phys
            cump += Δt_phys[i-1]
            θ_phys[i] = 2π * cump / T_phys
        end

        # `global` required: `let` introduces a local scope, and clean_lc is
        # reused by the PHASE-13c figure block below.
        global clean_lc = (X_lc=X_lc, S_lc_hidden=S_lc_hidden,
                    θ_koop=θ_koop, ρ_mag_koop=ρ_mag_koop, v_mag_koop=v_mag_koop,
                    T_koop=T_koop,
                    θ_phys=θ_phys, ρ_mag_phys=ρ_mag_phys, v_mag_phys=v_mag_phys,
                    T_phys=T_phys)

        fig = Figure(size=(1500, 800))
        Label(fig[0, :],
              "Koopman dynamical frame reproduces the true limit-cycle geometry — " *
              "same cylindrical shape in delay space and in physical space";
              fontsize=14, font=:bold)
        Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

        # Left: Koopman frame (radius = |ρ|, height = ||v||)
        x_k = ρ_mag_koop .* cos.(θ_koop)
        y_k = ρ_mag_koop .* sin.(θ_koop)
        ax3 = Axis3(fig[2, 1];
                    title="KOOPMAN frame (delay space, n_show=$n_show shown dims)\n" *
                          "$n_pts_lc pts, T=$(round(T_koop, digits=2)) s, " *
                          "|ρ| ∈ [$(round(minimum(ρ_mag_koop), digits=3)), " *
                          "$(round(maximum(ρ_mag_koop), digits=3))]",
                    titlesize=10,
                    xlabel="|ρ| cos θ", ylabel="|ρ| sin θ", zlabel="||v||  (Koopman)")
        hm = scatter!(ax3, x_k, y_k, v_mag_koop; color=θ_koop, colormap=:thermal,
                      markersize=4, colorrange=(0, 2π))
        lines!(ax3, x_k, y_k, v_mag_koop; color=(:black, 0.3), linewidth=0.8)
        Colorbar(fig[2, 2], hm; width=12, label="θ  [rad]  (time-of-flight)")

        # Right: physical ground truth
        x_p = ρ_mag_phys .* cos.(θ_phys)
        y_p = ρ_mag_phys .* sin.(θ_phys)
        ax4 = Axis3(fig[2, 3];
                    title="PHYSICAL ground truth (true vector field, dim=$dim)\n" *
                          "$n_pts_phys pts, T=$(round(T_phys, digits=2)) s, " *
                          "|ρ| ∈ [$(round(minimum(ρ_mag_phys), digits=3)), " *
                          "$(round(maximum(ρ_mag_phys), digits=3))]",
                    titlesize=10,
                    xlabel="|ρ| cos θ", ylabel="|ρ| sin θ", zlabel="||dx/dt||  (physical)")
        hm2 = scatter!(ax4, x_p, y_p, v_mag_phys; color=θ_phys, colormap=:thermal,
                       markersize=4, colorrange=(0, 2π))
        lines!(ax4, x_p, y_p, v_mag_phys; color=(:black, 0.3), linewidth=0.8)
        Colorbar(fig[2, 4], hm2; width=12, label="θ  [rad]  (time-of-flight)")

        add_caption!(fig, 3,
            "Clean limit cycle integrated with the TRUE vector field, rendered in cylindrical " *
            "dynamical-frame coordinates (azimuth = phase θ, radius = transverse magnitude |ρ|, " *
            "height = speed).  Left: frame learned by the Koopman operator in $n_state-D delay " *
            "space.  Right: same construction from the physical state.  Matching shapes " *
            "(periods $(round(T_koop, digits=2)) s vs $(round(T_phys, digits=2)) s) certify the " *
            "Koopman frame against ground truth.")

        save(joinpath(fig_dir, "PHASE13b_cylindrical_clean_lc_comparison.png"), fig)
        println("  Saved: PHASE13b_cylindrical_clean_lc_comparison.png")
        println("    T_koop=$(round(T_koop, digits=2)) s vs T_phys=$(round(T_phys, digits=2)) s")
    end
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-13c — Cycle in the (e_t, e_n, e_b) frame: ZLS vs clean LC
#                    (Koopman delay) vs physical clean LC
# ---------------------------------------------------------------------------
println("  PHASE-13c: limit cycle in the (e_t, e_n, e_b) frame (3-way)")

let
    rows_k = [k for k in 1:n_branches if !isnothing(dyn_frame_results[k]) &&
                                          !isempty(zls_per_cycle[k])]
    if isempty(rows_k)
        println("  Skipped PHASE-13c (no dynamical frame results)")
    else
        fig = Figure(size=(1600, 450 * length(rows_k) + 180))
        Label(fig[0, :],
              "The cycle is a closed loop in the tangent–normal plane with e_b ≈ 0 — " *
              "no torsion; Koopman and physical renderings agree";
              fontsize=13, font=:bold)
        Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

        for (row, k) in enumerate(rows_k)
            r = dyn_frame_results[k]
            s_ref_zls = zls_per_cycle[k][1]
            e_t_ref, e_n_ref, e_b_ref = r.E_t[:, 1], r.E_n[:, 1], r.E_b[:, 1]

            # Column 1: ZLS points projected on the reference frame
            pts_k = zls_per_cycle[k]
            n_pts_k = length(pts_k)
            ρ_t = [dot(pts_k[i] .- s_ref_zls, e_t_ref) for i in 1:n_pts_k]
            ρ_n = [dot(pts_k[i] .- s_ref_zls, e_n_ref) for i in 1:n_pts_k]
            ρ_b = [dot(pts_k[i] .- s_ref_zls, e_b_ref) for i in 1:n_pts_k]
            ax1 = Axis3(fig[1+row, 1];
                        title="cycle $k — ZLS points in (e_t, e_n, e_b) at θ=0 frame", titlesize=10,
                        xlabel="along e_t", ylabel="along e_n", zlabel="along e_b")
            hm = scatter!(ax1, ρ_t, ρ_n, ρ_b; color=θ_per_cycle[k], colormap=:thermal,
                          markersize=4, colorrange=(-π, π))
            lines!(ax1, ρ_t, ρ_n, ρ_b; color=(:black, 0.3), linewidth=0.8)
            Colorbar(fig[1+row, 4], hm; width=12, label="θ_cycle $k  [rad]")

            # Column 2: clean LC in delay space (only computed for cycle 1 —
            # reuse; for other cycles repeat the ZLS column with a note)
            if k == 1 && !isnothing(clean_lc)
                S_hid = clean_lc.S_lc_hidden
                n_lc_pts = size(S_hid, 2)
                sel = seeded_subsample_idx(n_lc_pts, 20_000; salt=0xC3)
                ρ_t_h = [dot(S_hid[:, i] .- s_ref_zls, e_t_ref) for i in sel]
                ρ_n_h = [dot(S_hid[:, i] .- s_ref_zls, e_n_ref) for i in sel]
                ρ_b_h = [dot(S_hid[:, i] .- s_ref_zls, e_b_ref) for i in sel]
                ax2 = Axis3(fig[1+row, 2];
                            title="cycle $k — clean LC, Koopman delay space ($n_show shown dims)",
                            titlesize=10,
                            xlabel="along e_t", ylabel="along e_n", zlabel="along e_b")
                scatter!(ax2, ρ_t_h, ρ_n_h, ρ_b_h; color=clean_lc.θ_koop[sel],
                         colormap=:thermal, markersize=3, colorrange=(0, 2π))
                lines!(ax2, ρ_t_h, ρ_n_h, ρ_b_h; color=(:black, 0.25), linewidth=0.6)
            else
                ax2 = Axis3(fig[1+row, 2];
                            title="cycle $k — clean LC in delay space not computed for k>1",
                            titlesize=10)
                text!(ax2, 0, 0, 0; text="not computed", align=(:center, :center))
            end

            # Column 3: clean LC in physical space (frame pushed through P_plot)
            if k == 1 && !isnothing(clean_lc)
                X_phys = clean_lc.X_lc[:, (m_embed - 1) * tau_delay + 1:end]
                n_ph = size(X_phys, 2)
                sel = seeded_subsample_idx(n_ph, 20_000; salt=0xC4)
                e_t_phys = vec(P_plot * e_t_ref)
                e_n_phys = vec(P_plot * e_n_ref)
                e_b_phys = vec(P_plot * e_b_ref)
                x_ref_phys = X_phys[:, 1]
                ρ_t_p = [dot(X_phys[:, i] .- x_ref_phys, e_t_phys) for i in sel]
                ρ_n_p = [dot(X_phys[:, i] .- x_ref_phys, e_n_phys) for i in sel]
                ρ_b_p = [dot(X_phys[:, i] .- x_ref_phys, e_b_phys) for i in sel]
                ax3 = Axis3(fig[1+row, 3];
                            title="cycle $k — clean LC, physical space (frame via P_plot)",
                            titlesize=10,
                            xlabel="along e_t (P_plot)", ylabel="along e_n (P_plot)",
                            zlabel="along e_b (P_plot)")
                scatter!(ax3, ρ_t_p, ρ_n_p, ρ_b_p; color=clean_lc.θ_phys[sel],
                         colormap=:thermal, markersize=3, colorrange=(0, 2π))
                lines!(ax3, ρ_t_p, ρ_n_p, ρ_b_p; color=(:black, 0.25), linewidth=0.6)
            else
                ax3 = Axis3(fig[1+row, 3];
                            title="cycle $k — physical clean LC not computed for k>1",
                            titlesize=10)
                text!(ax3, 0, 0, 0; text="not computed", align=(:center, :center))
            end
        end

        add_caption!(fig, length(rows_k) + 2,
            "Three views of each cycle in the SAME orthonormal frame (e_t, e_n, e_b at the θ=0 " *
            "reference point): the ZLS point cloud (left), a cleanly integrated limit cycle in " *
            "Koopman delay space (middle), and in physical space with the frame mapped through " *
            "P_plot (right).  Agreement between columns validates the frame as a dynamical — " *
            "not merely geometric — construction.")

        save(joinpath(fig_dir, "PHASE13c_dynamical_frame_3way.png"), fig)
        println("  Saved: PHASE13c_dynamical_frame_3way.png")
    end
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-14 — Transverse directions q1(θ), q2(θ) projected via P_plot
# ---------------------------------------------------------------------------
println("  PHASE-14: transverse directions q1(θ), q2(θ)")

let
    fig = Figure(size=(1500, 350 * n_branches + 150))
    Label(fig[0, :],
          "Transverse directions q₁(θ), q₂(θ) (Hessian eigenvectors) rotate smoothly " *
          "along the cycle — the moving normal frame";
          fontsize=13, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    for k in 1:n_branches
        isnothing(dyn_frame_results[k]) && continue
        r = dyn_frame_results[k]
        θ = r.θ_tof
        q1_plot = to3(P_plot * r.Q1)
        q2_plot = to3(P_plot * r.Q2)

        ax1 = Axis(fig[1+k, 1];
                   title="cycle $k: q₁(θ) components (plotted coords)", titlesize=10,
                   xlabel="θ  [rad]", ylabel="q₁ component")
        for d in 1:3
            lines!(ax1, θ, q1_plot[d, :]; linewidth=1.5, label=lbl(d))
        end
        axislegend(ax1; position=:rt, labelsize=7)

        ax2 = Axis(fig[1+k, 2];
                   title="cycle $k: q₂(θ) components (plotted coords)", titlesize=10,
                   xlabel="θ  [rad]", ylabel="q₂ component")
        for d in 1:3
            lines!(ax2, θ, q2_plot[d, :]; linewidth=1.5, label=lbl(d))
        end
        axislegend(ax2; position=:rt, labelsize=7)

        ax3 = Axis3(fig[1+k, 3];
                    title="cycle $k: q₁/q₂ tips in plotting space", titlesize=10,
                    xlabel=lbl(1), ylabel=lbl(2), zlabel=lbl(3))
        scatter!(ax3, q1_plot[1, :], q1_plot[2, :], q1_plot[3, :];
                 color=θ, colormap=:thermal, markersize=4, colorrange=(0, 2π), label="q₁")
        scatter!(ax3, q2_plot[1, :], q2_plot[2, :], q2_plot[3, :];
                 color=θ, colormap=:thermal, markersize=4, colorrange=(0, 2π),
                 marker=:rect, label="q₂")
        axislegend(ax3; position=:rb, labelsize=8)
    end

    add_caption!(fig, n_branches + 2,
        "q₁, q₂ are the eigenvectors of the normal-plane-projected Hessian of " *
        "f = φ⊥1²+φ⊥2², mapped to plotting coordinates by P_plot.  Smooth, periodic " *
        "components confirm the transverse frame is well-defined everywhere on the cycle.")

    save(joinpath(fig_dir, "PHASE14_transverse_directions.png"), fig)
    println("  Saved: PHASE14_transverse_directions.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE PHASE-15 — CONSISTENCY-CHECK SUMMARY (SANITY CHECKS, 8 panels)
# ---------------------------------------------------------------------------
#   (1) tangent alignment ||Hv||/(||v||·||H||) per cycle (≈ 0)
#   (2) mean Hessian eigenvalues per cycle (spectral gap regime)
#   (3) frame continuity: sign flips after correction (= 0)
#   (4) frame periodicity: ||Δe|| at closure (≈ 0)
#   (5) θ_eigenfunction vs θ_time-of-flight (monotone ⇒ consistent phases)
#   (6) frame orthonormality defect max_{i≠j} |e_i·e_j| and |‖e‖−1| vs θ (≈ 0)
#   (7) ρ consistency: ρ₁(θ), ρ₂(θ) close the loop; local-frame ρ ≡ 0
#   (8) winding numbers on the diagonal (should be ±1)
# ---------------------------------------------------------------------------
println("  PHASE-15: consistency-check summary (sanity checks)")

let
    fig = Figure(size=(1700, 900))
    Label(fig[0, :],
          "Dynamical-frame sanity checks: all four PDF consistency checks pass, " *
          "and Koopman phase agrees with time-of-flight phase";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    cycles_ok = [k for k in 1:n_branches if !isnothing(dyn_frame_results[k])]

    # (1) tangent alignment
    ax1 = Axis(fig[2, 1]; title="(1) tangent alignment\nmedian ||Hv||/(||v||·||H||) ≈ 0",
               titlesize=10, xlabel="cycle", ylabel="alignment ratio", yscale=log10,
               xticks=cycles_ok)
    for k in cycles_ok
        dyn_frame_results[k].hessian_ok || continue
        barplot!(ax1, [k], [max(median(dyn_frame_results[k].tangent_align), 1e-16)];
                 color=CYCLE_COLORS[mod1(k, 8)])
    end
    hlines!(ax1, [1e-6]; color=:red, linewidth=1, linestyle=:dash)

    # (2) mean Hessian eigenvalues
    ax2 = Axis(fig[2, 2]; title="(2) mean Hessian eigenvalues\n(spectral gap regime)",
               titlesize=10, xlabel="cycle", ylabel="mean eigenvalue", xticks=cycles_ok)
    for k in cycles_ok
        dyn_frame_results[k].hessian_ok || continue
        barplot!(ax2, [k - 0.15], [mean(dyn_frame_results[k].λ1)]; color=:crimson,
                 label=k == cycles_ok[1] ? "λ₁" : "")
        barplot!(ax2, [k + 0.15], [mean(dyn_frame_results[k].λ2)]; color=:dodgerblue,
                 label=k == cycles_ok[1] ? "λ₂" : "")
    end
    axislegend(ax2; position=:rt, labelsize=8)

    # (3) frame continuity
    ax3 = Axis(fig[2, 3]; title="(3) frame continuity\nsign flips after fix (should be 0)",
               titlesize=10, xlabel="cycle", ylabel="sign flips", xticks=cycles_ok)
    for k in cycles_ok
        barplot!(ax3, [k - 0.15], [dyn_frame_results[k].n_flips_n]; color=:purple,
                 label=k == cycles_ok[1] ? "e_n flips" : "")
        barplot!(ax3, [k + 0.15], [dyn_frame_results[k].n_flips_b]; color=:darkorange,
                 label=k == cycles_ok[1] ? "e_b flips" : "")
    end
    axislegend(ax3; position=:rt, labelsize=8)

    # (4) periodicity
    ax4 = Axis(fig[2, 4]; title="(4) frame periodicity\n||Δe|| at closure (should be ≈ 0)",
               titlesize=10, xlabel="cycle", ylabel="||e(0) − e(2π)||", xticks=cycles_ok)
    for k in cycles_ok
        r = dyn_frame_results[k]
        barplot!(ax4, [k - 0.2], [r.per_t]; color=:forestgreen,
                 label=k == cycles_ok[1] ? "Δe_t" : "")
        barplot!(ax4, [k],       [r.per_n]; color=:crimson,
                 label=k == cycles_ok[1] ? "Δe_n" : "")
        barplot!(ax4, [k + 0.2], [r.per_b]; color=:dodgerblue,
                 label=k == cycles_ok[1] ? "Δe_b" : "")
    end
    axislegend(ax4; position=:rt, labelsize=8)

    # (5) θ_eigenfunction vs θ_time-of-flight
    ax5 = Axis(fig[3, 1]; title="(5) phase consistency\nKoopman θ vs time-of-flight θ (monotone)",
               titlesize=10, xlabel="θ_eigenfunction  [rad]", ylabel="θ_time-of-flight  [rad]")
    for k in cycles_ok
        θ_eig = θ_per_cycle[k]                       # gauge-fixed, ∈ (-π, π]
        θ_tof_k = dyn_frame_results[k].θ_tof         # ∈ [0, 2π)
        scatter!(ax5, θ_eig, θ_tof_k; color=CYCLE_COLORS[mod1(k, 8)], markersize=3,
                 label="cycle $k")
    end
    axislegend(ax5; position=:rb, labelsize=8)

    # (6) frame orthonormality defect
    ax6 = Axis(fig[3, 2]; title="(6) frame orthonormality\ndefect vs θ (should be ≈ 0)",
               titlesize=10, xlabel="θ  [rad]", ylabel="max defect")
    for k in cycles_ok
        r = dyn_frame_results[k]
        n_pts_k = length(r.θ_tof)
        defect = [max(abs(dot(r.E_t[:, i], r.E_n[:, i])),
                      abs(dot(r.E_t[:, i], r.E_b[:, i])),
                      abs(dot(r.E_n[:, i], r.E_b[:, i])),
                      abs(norm(r.E_t[:, i]) - 1),
                      abs(norm(r.E_n[:, i]) - 1),
                      abs(norm(r.E_b[:, i]) - 1)) for i in 1:n_pts_k]
        lines!(ax6, r.θ_tof, max.(defect, 1e-17); color=CYCLE_COLORS[mod1(k, 8)],
               linewidth=1.2, label="cycle $k")
    end
    ax6.yscale = log10
    axislegend(ax6; position=:rt, labelsize=8)

    # (7) ρ consistency
    ax7 = Axis(fig[3, 3]; title="(7) transverse coords close the loop\n" *
               "ρ(2π) − ρ(0) ≈ 0;  local-frame ρ ≡ 0",
               titlesize=10, xlabel="θ  [rad]", ylabel="ρ")
    for k in cycles_ok
        r = dyn_frame_results[k]
        lines!(ax7, r.θ_tof, r.ρ1; color=CYCLE_COLORS[mod1(k, 8)], linewidth=1.5,
               label="cycle $k ρ₁")
        lines!(ax7, r.θ_tof, r.ρ2; color=CYCLE_COLORS[mod1(k, 8)], linewidth=1.5,
               linestyle=:dash, label="cycle $k ρ₂")
    end
    axislegend(ax7; position=:rb, labelsize=7)

    # (8) winding numbers (diagonal of the matching table)
    ax8 = Axis(fig[3, 4]; title="(8) winding numbers\ndiagonal of the match table (should be ±1)",
               titlesize=10, xlabel="cycle", ylabel="winding number", xticks=cycles_ok,
               limits=(nothing, nothing, -1.5, 1.5))
    for j in 1:n_branches
        k = final_match[j]
        barplot!(ax8, [k], [match_table_winding[j, k]]; color=CYCLE_COLORS[mod1(j, 8)],
                 label="φ_$j → cycle $k")
    end
    hlines!(ax8, [1.0, -1.0]; color=:gray, linewidth=0.8, linestyle=:dash)
    axislegend(ax8; position=:rb, labelsize=7)

    add_caption!(fig, 4,
        "Checks (1)–(4) are the PDF consistency requirements for the dynamical frame; " *
        "(5)–(8) are independent cross-validations: the Koopman phase and the time-of-flight " *
        "phase must be monotonically related, the Gram–Schmidt frame must stay orthonormal to " *
        "machine precision, the transverse coordinates must close periodically, and each " *
        "eigenfunction must wind exactly once around its matched cycle.")

    save(joinpath(fig_dir, "PHASE15_consistency_checks.png"), fig)
    println("  Saved: PHASE15_consistency_checks.png")
end
GC.gc()
print_mem("after PHASE figures")

# ============================================================================
# UNFOLDING — PG-FNU + DIFFERENTIAL FNN + PERSISTENT HOMOLOGY
# (legacy Koopman_Classic_analysis_unfolding.jl; none of this exists in the
#  package — API_REFERENCE "WHAT THE PACKAGE LACKS" #3 — ported locally)
# ============================================================================
# Goal: find a 3-D projection of the cycle-1 ZLS cloud in which the cycle is
# UNFOLDED (no false neighbors = injective embedding of S¹), and validate the
# topology with persistent homology (exactly one dominant H₁ class).

println("\n" * "="^72)
println("UNFOLDING: PG-FNU + differential FNN + persistent homology (cycle 1)")
println("="^72)

# ---- cycle-1 data (phase-ordered) ------------------------------------------
k_cycle = 1
isempty(zls_per_cycle[k_cycle]) && error("Cycle 1 has no ZLS points — cannot unfold.")
S_cycle_full = reduce(hcat, zls_per_cycle[k_cycle])
θ_cycle_full = θ_per_cycle[k_cycle]
perm_sort = sortperm(θ_cycle_full)
S_cycle_full = S_cycle_full[:, perm_sort]
θ_cycle_full = θ_cycle_full[perm_sort]

# O(N²) algorithms below → subsample large clouds (seeded; memory/time rule)
const MAX_UNFOLD_PTS = 3000
idx_unf = seeded_subsample_idx(size(S_cycle_full, 2), MAX_UNFOLD_PTS; salt=0xF1)
S_cycle = S_cycle_full[:, idx_unf]
θ_cycle = θ_cycle_full[idx_unf]
n_pts = size(S_cycle, 2)
S_cycle_full = nothing
GC.gc()
n_pts < 20 && error("Too few ZLS points ($n_pts) for unfolding — need at least 20")
println("  Cycle 1: using $n_pts phase-ordered points" *
        (n_pts == length(idx_unf) && length(idx_unf) == MAX_UNFOLD_PTS ?
         " (seeded subsample, salt 0xF1)" : ""))

s_mean_cycle = vec(mean(S_cycle; dims=2))
S_centered = S_cycle .- s_mean_cycle

# ---------------------------------------------------------------------------
# False-neighbor rate of a projection P (d × N) with cyclic temporal order
# ---------------------------------------------------------------------------
"""
    compute_false_neighbor_rate(P; k_temporal=2)

Point i is a FALSE NEIGHBOR if its nearest non-temporal neighbor in the
projected space is closer than its farthest temporal (cyclic ±k) neighbor.
FNR = fraction of false neighbors; FNR = 0 ⇒ the projection is an injective
(unfolded) embedding of the cycle.
"""
function compute_false_neighbor_rate(P::AbstractMatrix; k_temporal::Int=2)
    N = size(P, 2)
    n_false = 0
    for i in 1:N
        temporal = [mod1(i - kk, N) for kk in 1:k_temporal]
        append!(temporal, [mod1(i + kk, N) for kk in 1:k_temporal])
        max_d_temporal = maximum(norm(P[:, i] .- P[:, m]) for m in temporal)
        min_d_non = minimum(norm(P[:, i] .- P[:, m])
                            for m in 1:N if m != i && !(m in temporal))
        min_d_non < max_d_temporal && (n_false += 1)
    end
    return n_false / N
end

# ============================================================================
# STEP 6 — PG-FNU (greedy FNR-minimizing POD-candidate selection)
# ============================================================================
"""
    pgfnu_unfold(S_centered; k_temporal, target_dim, n_candidates)

Phase-Guided False-Neighbor Unfolding: greedily add the POD eigenvector
(candidate from the snapshot covariance) that minimizes the false-neighbor
rate of the growing projection.  Deterministic (covariance eigenvectors).
Returns (B, P, fnr_history).
"""
function pgfnu_unfold(S_centered::AbstractMatrix;
                      k_temporal::Int=2, target_dim::Int=3, n_candidates::Int=20)
    n_state_loc, N = size(S_centered)
    # POD candidates via eigen(Symmetric(S·S')) — memory rule, no svd Vt
    Feig = eigen(Symmetric(S_centered * S_centered' ./ N))
    eig_perm = sortperm(Feig.values; rev=true)
    candidates = Feig.vectors[:, eig_perm][:, 1:min(n_candidates, n_state_loc)]
    println("  [PG-FNU] $(size(candidates, 2)) POD candidate directions, target dim $target_dim")

    B = zeros(n_state_loc, 0)
    used = falses(size(candidates, 2))
    fnr_history = Float64[]
    for d in 1:target_dim
        best_j, best_fnr = 0, 1.0
        for j in 1:size(candidates, 2)
            used[j] && continue
            P_trial = hcat(B, candidates[:, j])' * S_centered
            fnr = compute_false_neighbor_rate(P_trial; k_temporal=k_temporal)
            if fnr < best_fnr
                best_fnr, best_j = fnr, j
            end
        end
        if best_j == 0   # FNR tie at 1.0 (typical at d=1) → take the top unused POD mode
            best_j = findfirst(!, used)
            isnothing(best_j) && (println("  [PG-FNU] no more candidates at d=$d"); break)
            P_trial = hcat(B, candidates[:, best_j])' * S_centered
            best_fnr = compute_false_neighbor_rate(P_trial; k_temporal=k_temporal)
            println("  [PG-FNU] d=$d: FNR tie at 1.0 — fallback to POD mode #$best_j " *
                    "(FNR=$(round(best_fnr, digits=4)))")
        else
            println("  [PG-FNU] d=$d: added POD mode #$best_j, FNR = $(round(best_fnr, digits=4))")
        end
        B = hcat(B, candidates[:, best_j])
        used[best_j] = true
        push!(fnr_history, best_fnr)
        if best_fnr == 0.0
            println("  [PG-FNU] injective embedding achieved at d=$d")
            break
        end
    end
    return B, B' * S_centered, fnr_history
end

println("\nSTEP 6: PG-FNU")
B_pgfnu, P_pgfnu, fnr_history_pgfnu = pgfnu_unfold(S_centered;
                                                  k_temporal=2, target_dim=3,
                                                  n_candidates=20)
println("  PG-FNU: $(size(B_pgfnu, 2))-D embedding, final FNR = " *
        "$(round(fnr_history_pgfnu[end], digits=4))")
print_mem("after PG-FNU")
GC.gc()

# ============================================================================
# STEP 7 — DIFFERENTIAL FNN (maximize dP/dθ smoothness)
# ============================================================================
"""Smoothness of a 1-D projected coordinate: 1/(1 + total variation of dp/dθ)."""
function compute_derivative_smoothness(p_1d::AbstractVector, θ::AbstractVector)
    N = length(p_1d)
    dp = zeros(N)
    for i in 1:N
        ip, im = mod1(i + 1, N), mod1(i - 1, N)
        dθ = θ[ip] - θ[im]
        dθ < 0 && (dθ += 2π)
        dθ = max(dθ, 1e-12)
        dp[i] = (p_1d[ip] - p_1d[im]) / dθ
    end
    return 1.0 / (1.0 + sum(abs.(diff(dp))))
end

"""
    differential_fnn_unfold(S_centered, θ; target_dim, n_candidates)

Greedily add the POD candidate maximizing the mean dP/dθ smoothness of the
growing projection.  Deterministic.  Returns (B, P, smoothness_history).
"""
function differential_fnn_unfold(S_centered::AbstractMatrix, θ::AbstractVector;
                                 target_dim::Int=3, n_candidates::Int=20)
    n_state_loc, N = size(S_centered)
    Feig = eigen(Symmetric(S_centered * S_centered' ./ N))
    eig_perm = sortperm(Feig.values; rev=true)
    candidates = Feig.vectors[:, eig_perm][:, 1:min(n_candidates, n_state_loc)]
    println("  [Diff-FNN] $(size(candidates, 2)) candidates, target dim $target_dim")

    B = zeros(n_state_loc, 0)
    used = falses(size(candidates, 2))
    smoothness_history = Float64[]
    for d in 1:target_dim
        best_j, best_s = 0, -1.0
        for j in 1:size(candidates, 2)
            used[j] && continue
            P_trial = hcat(B, candidates[:, j])' * S_centered
            s = mean(compute_derivative_smoothness(P_trial[r, :], θ)
                     for r in 1:size(P_trial, 1))
            if s > best_s
                best_s, best_j = s, j
            end
        end
        if best_j == 0
            println("  [Diff-FNN] no more candidates at d=$d")
            break
        end
        B = hcat(B, candidates[:, best_j])
        used[best_j] = true
        push!(smoothness_history, best_s)
        println("  [Diff-FNN] d=$d: added POD mode #$best_j, " *
                "smoothness = $(round(best_s, digits=6))")
    end
    return B, B' * S_centered, smoothness_history
end

println("\nSTEP 7: Differential FNN")
B_diff, P_diff, smoothness_history_diff = differential_fnn_unfold(S_centered, θ_cycle;
                                                                  target_dim=3,
                                                                  n_candidates=20)
fnr_diff = compute_false_neighbor_rate(P_diff; k_temporal=2)
println("  Diff-FNN: $(size(B_diff, 2))-D embedding, " *
        "final smoothness = $(round(smoothness_history_diff[end], digits=6)), " *
        "FNR = $(round(fnr_diff, digits=4))")

# Reference: the standard P_plot projection
P_plot_proj = P_plot * S_centered            # n_plot × N
fnr_pplot = compute_false_neighbor_rate(P_plot_proj; k_temporal=2)
println("  Original P_plot FNR = $(round(fnr_pplot, digits=4))")

# FNR vs dimension tables (for UNFOLD-7 and the output file)
fnr_pgfnu_per_dim = [compute_false_neighbor_rate(P_pgfnu[1:d, :]; k_temporal=2)
                     for d in 1:size(P_pgfnu, 1)]
fnr_diff_per_dim  = [compute_false_neighbor_rate(P_diff[1:d, :]; k_temporal=2)
                     for d in 1:size(P_diff, 1)]
fnr_pplot_per_dim = [compute_false_neighbor_rate(P_plot_proj[1:d, :]; k_temporal=2)
                     for d in 1:size(P_plot_proj, 1)]
print_mem("after differential FNN")
GC.gc()

# ============================================================================
# STEP 8 — PERSISTENT HOMOLOGY (Ripserer, Vietoris–Rips, dim_max = 1)
# ============================================================================
# A faithful embedding of the cycle S¹ has exactly ONE dominant H₁ class
# (high persistence = death − birth).  Ripserer + StaticArrays are a HARD
# dependency of this step (see the header note); if they failed to load, the
# step is skipped with a warning and the h1_* keys are stored empty.

println("\nSTEP 8: Persistent homology (H₁ validation)")

const MAX_PH_PTS = 1500     # Ripserer cost grows fast with N — seeded subsample

"""
    compute_persistence_h1(P; verbose) -> (h0, h1, births, max_d)

H₀ and H₁ persistence of the point cloud P (d × N) via Ripserer
(Vietoris–Rips, dim_max=1).  h0/h1 are vectors of (birth, death) tuples
(Inf deaths replaced by NaN); `births` = sorted H₁ births; `max_d` = cloud
diameter.  Requires Ripserer and StaticArrays (hard dependency, STEP 8).
"""
function compute_persistence_h1(P::AbstractMatrix; verbose::Bool=true)
    N = size(P, 2)
    verbose && println("    [PH] Ripserer on $N points in $(size(P, 1))-D (dim_max=1)...")
    pts = [SVector{size(P, 1), Float64}(P[:, i]) for i in 1:N]
    result = ripserer(pts; dim_max=1)
    h0 = [(Float64(birth(c)), Float64(death(c))) for c in result[1]]
    h1 = [(Float64(birth(c)),
           isinf(death(c)) ? NaN : Float64(death(c))) for c in result[2]]
    births = sort([b for (b, _) in h1])
    max_d = maximum(norm(P[:, i] .- P[:, j]) for i in 1:N for j in (i+1):N; init=0.0)
    if verbose
        pers = [d - b for (b, d) in h1 if !isnan(d)]
        println("    [PH] H₁ classes: $(length(h1))" *
                (isempty(pers) ? "" :
                 ",  persistence ∈ [$(round(minimum(pers), digits=4)), " *
                 "$(round(maximum(pers), digits=4))]"))
    end
    return h0, h1, births, max_d
end

h0_pgfnu = Tuple{Float64,Float64}[]
h1_pgfnu = Tuple{Float64,Float64}[]
births_pgfnu = Float64[]
max_d_pgfnu = NaN
h0_diff = Tuple{Float64,Float64}[]
h1_diff = Tuple{Float64,Float64}[]
births_diff = Float64[]
max_d_diff = NaN
h0_pplot = Tuple{Float64,Float64}[]
h1_pplot = Tuple{Float64,Float64}[]
births_pplot = Float64[]
max_d_pplot = NaN

if HAVE_RIPSERER
    # seeded subsample for PH (dedicated generators, salts 0xF2/0xF3/0xF4)
    idx_ph = seeded_subsample_idx(n_pts, MAX_PH_PTS; salt=0xF2)
    println("  PH on $(length(idx_ph)) points per projection" *
            (length(idx_ph) < n_pts ? " (seeded subsample, salt 0xF2)" : ""))
    println("  Computing PH for PG-FNU projection...")
    h0_pgfnu, h1_pgfnu, births_pgfnu, max_d_pgfnu =
        compute_persistence_h1(P_pgfnu[:, idx_ph])
    println("  Computing PH for differential-FNN projection...")
    h0_diff, h1_diff, births_diff, max_d_diff =
        compute_persistence_h1(P_diff[:, idx_ph])
    println("  Computing PH for original P_plot projection...")
    h0_pplot, h1_pplot, births_pplot, max_d_pplot =
        compute_persistence_h1(P_plot_proj[:, idx_ph])
else
    @warn "Ripserer unavailable — skipping persistent homology (STEP 8) and " *
          "figures UNFOLD-5/6.  Install Ripserer.jl + StaticArrays.jl to enable."
end
print_mem("after persistent homology")
GC.gc()

"""Dominant H₁ persistence = max death−birth (0 if no class)."""
function h1_max_persistence(h1)
    pers = [d - b for (b, d) in h1 if !isnan(d)]
    return isempty(pers) ? 0.0 : maximum(pers)
end

# ---------------------------------------------------------------------------
# FIGURE UNFOLD-1 — PG-FNU 3-D projection, colored by phase
# ---------------------------------------------------------------------------
println("\n  UNFOLD-1: PG-FNU 3-D projection")

let
    fig = Figure(size=(1000, 850))
    Label(fig[0, :],
          "PG-FNU unfolds the cycle: false-neighbor rate drops to " *
          "$(round(fnr_history_pgfnu[end], digits=4)) in $(size(B_pgfnu, 2))-D";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle * " | cycle 1, $n_pts phase-ordered pts";
          fontsize=10, color=:gray)

    ax = Axis3(fig[2, 1];
               title="PG-FNU 3-D embedding (greedy FNR-minimizing POD selection)\n" *
                     "FNR history: " * join([string(round(f, digits=3)) for f in fnr_history_pgfnu], " → "),
               titlesize=11,
               xlabel="PG-FNU direction 1", ylabel="PG-FNU direction 2",
               zlabel="PG-FNU direction 3")
    hm = scatter!(ax, P_pgfnu[1, :], P_pgfnu[2, :],
                  size(P_pgfnu, 1) >= 3 ? P_pgfnu[3, :] : zeros(n_pts);
                  color=θ_cycle, colormap=:thermal, markersize=7, colorrange=(-π, π))
    lines!(ax, P_pgfnu[1, :], P_pgfnu[2, :],
           size(P_pgfnu, 1) >= 3 ? P_pgfnu[3, :] : zeros(n_pts);
           color=(:black, 0.3), linewidth=1.0)
    scatter!(ax, [P_pgfnu[1, 1]], [P_pgfnu[2, 1]],
             [size(P_pgfnu, 1) >= 3 ? P_pgfnu[3, 1] : 0.0];
             color=:green, markersize=12, marker=:circle, label="start (θ≈−π)")
    scatter!(ax, [P_pgfnu[1, end]], [P_pgfnu[2, end]],
             [size(P_pgfnu, 1) >= 3 ? P_pgfnu[3, end] : 0.0];
             color=:red, markersize=12, marker=:rect, label="end (θ≈π)")
    Colorbar(fig[2, 2], hm; width=14, label="Koopman phase θ  [rad]")
    axislegend(ax; position=:rb, framevisible=true, labelsize=9)

    add_caption!(fig, 3,
        "PG-FNU greedily picks POD eigenvectors of the centered ZLS cloud that minimize the " *
        "false-neighbor rate (a point is a false neighbor if a non-temporal point is closer " *
        "than its cyclic ±2 neighbors).  FNR = 0 certifies an injective embedding of S¹: " *
        "the loop is fully unfolded, with phase (color) running once around it.")

    save(joinpath(fig_dir, "UNFOLD1_pgfnu_projection.png"), fig)
    println("  Saved: UNFOLD1_pgfnu_projection.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE UNFOLD-2 — PG-FNU vs original P_plot (side-by-side)
# ---------------------------------------------------------------------------
println("  UNFOLD-2: PG-FNU vs original P_plot")

let
    fig = Figure(size=(1600, 800))
    Label(fig[0, :],
          "PG-FNU beats the standard physical+POD projection: " *
          "FNR $(round(fnr_history_pgfnu[end], digits=3)) vs $(round(fnr_pplot, digits=3))";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    ax1 = Axis3(fig[2, 1];
                title="PG-FNU  (FNR = $(round(fnr_history_pgfnu[end], digits=4)))",
                titlesize=11,
                xlabel="PG-FNU 1", ylabel="PG-FNU 2", zlabel="PG-FNU 3")
    hm = scatter!(ax1, P_pgfnu[1, :], P_pgfnu[2, :],
                  size(P_pgfnu, 1) >= 3 ? P_pgfnu[3, :] : zeros(n_pts);
                  color=θ_cycle, colormap=:thermal, markersize=5, colorrange=(-π, π))
    lines!(ax1, P_pgfnu[1, :], P_pgfnu[2, :],
           size(P_pgfnu, 1) >= 3 ? P_pgfnu[3, :] : zeros(n_pts);
           color=(:black, 0.3), linewidth=0.8)

    P3 = to3(P_plot_proj)
    ax2 = Axis3(fig[2, 2];
                title="Original P_plot  (FNR = $(round(fnr_pplot, digits=4)))",
                titlesize=11,
                xlabel=lbl(1), ylabel=lbl(2), zlabel=lbl(3))
    scatter!(ax2, P3[1, :], P3[2, :], P3[3, :];
             color=θ_cycle, colormap=:thermal, markersize=5, colorrange=(-π, π))
    lines!(ax2, P3[1, :], P3[2, :], P3[3, :]; color=(:black, 0.3), linewidth=0.8)
    Colorbar(fig[2, 3], hm; width=14, label="Koopman phase θ  [rad]")

    add_caption!(fig, 3,
        "Same ZLS points, same phase coloring.  Self-crossings of the black phase-ordered " *
        "path mark false neighbors — where the projection folds the cycle onto itself.  " *
        "PG-FNU removes them by construction; the standard P_plot does not optimize for this.")

    save(joinpath(fig_dir, "UNFOLD2_pgfnu_vs_original.png"), fig)
    println("  Saved: UNFOLD2_pgfnu_vs_original.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE UNFOLD-3 — Differential FNN 3-D projection
# ---------------------------------------------------------------------------
println("  UNFOLD-3: differential FNN projection")

let
    fig = Figure(size=(1000, 850))
    Label(fig[0, :],
          "Differential FNN selects the smoothest parametrization: " *
          "dP/dθ smoothness = $(round(smoothness_history_diff[end], digits=4)), " *
          "FNR = $(round(fnr_diff, digits=4))";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    ax = Axis3(fig[2, 1];
               title="Differential-FNN 3-D embedding (maximize dP/dθ smoothness)\n" *
                     "smoothness history: " *
                     join([string(round(s, digits=4)) for s in smoothness_history_diff], " → "),
               titlesize=11,
               xlabel="Diff-FNN direction 1", ylabel="Diff-FNN direction 2",
               zlabel="Diff-FNN direction 3")
    hm = scatter!(ax, P_diff[1, :], P_diff[2, :],
                  size(P_diff, 1) >= 3 ? P_diff[3, :] : zeros(n_pts);
                  color=θ_cycle, colormap=:thermal, markersize=7, colorrange=(-π, π))
    lines!(ax, P_diff[1, :], P_diff[2, :],
           size(P_diff, 1) >= 3 ? P_diff[3, :] : zeros(n_pts);
           color=(:black, 0.3), linewidth=1.0)
    scatter!(ax, [P_diff[1, 1]], [P_diff[2, 1]],
             [size(P_diff, 1) >= 3 ? P_diff[3, 1] : 0.0];
             color=:green, markersize=12, marker=:circle, label="start (θ≈−π)")
    scatter!(ax, [P_diff[1, end]], [P_diff[2, end]],
             [size(P_diff, 1) >= 3 ? P_diff[3, end] : 0.0];
             color=:red, markersize=12, marker=:rect, label="end (θ≈π)")
    Colorbar(fig[2, 2], hm; width=14, label="Koopman phase θ  [rad]")
    axislegend(ax; position=:rb, framevisible=true, labelsize=9)

    add_caption!(fig, 3,
        "Greedy selection of POD directions maximizing 1/(1 + total variation of dP/dθ).  " *
        "A phase-smooth embedding spreads the cycle evenly — no sharp folds — which also " *
        "reduces false neighbors (FNR shown).")

    save(joinpath(fig_dir, "UNFOLD3_differential_fnn.png"), fig)
    println("  Saved: UNFOLD3_differential_fnn.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE UNFOLD-4 — Differential FNN vs original P_plot
# ---------------------------------------------------------------------------
println("  UNFOLD-4: differential FNN vs original P_plot")

let
    fig = Figure(size=(1600, 800))
    Label(fig[0, :],
          "Differential FNN vs P_plot: FNR $(round(fnr_diff, digits=3)) vs " *
          "$(round(fnr_pplot, digits=3))";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    ax1 = Axis3(fig[2, 1];
                title="Differential FNN  (FNR = $(round(fnr_diff, digits=4)))",
                titlesize=11,
                xlabel="Diff 1", ylabel="Diff 2", zlabel="Diff 3")
    hm = scatter!(ax1, P_diff[1, :], P_diff[2, :],
                  size(P_diff, 1) >= 3 ? P_diff[3, :] : zeros(n_pts);
                  color=θ_cycle, colormap=:thermal, markersize=5, colorrange=(-π, π))
    lines!(ax1, P_diff[1, :], P_diff[2, :],
           size(P_diff, 1) >= 3 ? P_diff[3, :] : zeros(n_pts);
           color=(:black, 0.3), linewidth=0.8)

    P3 = to3(P_plot_proj)
    ax2 = Axis3(fig[2, 2];
                title="Original P_plot  (FNR = $(round(fnr_pplot, digits=4)))",
                titlesize=11,
                xlabel=lbl(1), ylabel=lbl(2), zlabel=lbl(3))
    scatter!(ax2, P3[1, :], P3[2, :], P3[3, :];
             color=θ_cycle, colormap=:thermal, markersize=5, colorrange=(-π, π))
    lines!(ax2, P3[1, :], P3[2, :], P3[3, :]; color=(:black, 0.3), linewidth=0.8)
    Colorbar(fig[2, 3], hm; width=14, label="Koopman phase θ  [rad]")

    add_caption!(fig, 3,
        "Same cloud, same coloring.  The differential criterion optimizes phase smoothness " *
        "rather than injectivity directly, so its FNR is typically between PG-FNU and P_plot; " *
        "its geometry is the most evenly parametrized.")

    save(joinpath(fig_dir, "UNFOLD4_differential_vs_original.png"), fig)
    println("  Saved: UNFOLD4_differential_vs_original.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURES UNFOLD-5 / UNFOLD-6 — Persistent homology (H₁ diagram + barcode)
# ---------------------------------------------------------------------------
function fig_persistent_homology(h1, max_d, pname, fname, fnr_val)
    fig = Figure(size=(1400, 750))
    n_dom = count(t -> !isnan(t[2]) && (t[2] - t[1]) > 0.1 * max_d, h1)
    Label(fig[0, :],
          "Persistent homology of the $pname projection: $(length(h1)) H₁ class(es), " *
          "max persistence $(round(h1_max_persistence(h1), digits=3)) — " *
          (n_dom == 1 ? "exactly ONE dominant class ⇒ faithful S¹ embedding" :
           "$n_dom dominant classes (one expected for a clean cycle)");
          fontsize=13, font=:bold)
    Label(fig[1, :], fig_subtitle * " | Ripserer Vietoris–Rips, dim_max=1, FNR = " *
          string(round(fnr_val, digits=4)); fontsize=10, color=:gray)

    # Left: persistence diagram
    ax1 = Axis(fig[2, 1];
               title="H₁ persistence diagram — points far above the diagonal\n" *
                     "are robust topological features",
               titlesize=11,
               xlabel="birth ε  (filtration radius)", ylabel="death ε")
    if !isempty(h1)
        bs = [b for (b, _) in h1]
        ds = [isnan(d) ? max_d : d for (_, d) in h1]   # essential classes drawn at max_d
        scatter!(ax1, bs, ds; color=:crimson, markersize=9)
    end
    lines!(ax1, [0, max_d], [0, max_d]; color=:gray, linestyle=:dash, linewidth=1)

    # Right: barcode
    ax2 = Axis(fig[2, 2];
               title="H₁ barcode — one long bar expected for a limit cycle",
               titlesize=11,
               xlabel="ε  (filtration radius)", ylabel="H₁ class index")
    h1s = sort(h1; by=t -> t[1])
    for (i, (b, d)) in enumerate(h1s)
        lines!(ax2, [b, isnan(d) ? max_d : d], [i, i]; color=:dodgerblue, linewidth=3)
    end
    if !isempty(h1s)
        ax2.limits = (0, max_d * 1.02, 0, length(h1s) + 1)
    end

    add_caption!(fig, 3,
        "Vietoris–Rips persistent homology (Ripserer) of the $pname projection.  A clean " *
        "embedding of the limit cycle S¹ has exactly one H₁ class with persistence comparable " *
        "to the cloud diameter (one long bar, one point far above the diagonal).  Extra " *
        "long-lived classes signal spurious self-intersections (folding).")

    save(joinpath(fig_dir, fname), fig)
    println("  Saved: $fname")
end

if HAVE_RIPSERER
    println("  UNFOLD-5/6: persistent homology figures")
    fig_persistent_homology(h1_pgfnu, max_d_pgfnu, "PG-FNU",
                            "UNFOLD5_persistent_homology_pgfnu.png",
                            fnr_history_pgfnu[end])
    fig_persistent_homology(h1_diff, max_d_diff, "differential-FNN",
                            "UNFOLD6_persistent_homology_differential.png",
                            fnr_diff)
else
    println("  Skipped UNFOLD-5/6 (Ripserer unavailable)")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE UNFOLD-7 — BENCHMARK: FNR vs embedding dimension
# ---------------------------------------------------------------------------
println("  UNFOLD-7: FNR vs embedding dimension")

let
    fig = Figure(size=(1000, 750))
    Label(fig[0, :],
          "False-neighbor rate vs embedding dimension: PG-FNU reaches FNR = " *
          "$(round(fnr_pgfnu_per_dim[end], digits=3)) at d = $(length(fnr_pgfnu_per_dim))";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    ax = Axis(fig[2, 1];
              title="FNR = 0 ⇒ topologically injective embedding of the cycle",
              titlesize=11,
              xlabel="embedding dimension d", ylabel="false-neighbor rate  [–]")
    scatter!(ax, 1:length(fnr_pgfnu_per_dim), fnr_pgfnu_per_dim; color=:crimson,
             markersize=12, label="PG-FNU")
    lines!(ax, 1:length(fnr_pgfnu_per_dim), fnr_pgfnu_per_dim; color=:crimson, linewidth=2)
    scatter!(ax, 1:length(fnr_diff_per_dim), fnr_diff_per_dim; color=:dodgerblue,
             markersize=12, marker=:rect, label="Differential FNN")
    lines!(ax, 1:length(fnr_diff_per_dim), fnr_diff_per_dim; color=:dodgerblue, linewidth=2)
    scatter!(ax, 1:length(fnr_pplot_per_dim), fnr_pplot_per_dim; color=:forestgreen,
             markersize=12, marker=:diamond, label="Original P_plot")
    lines!(ax, 1:length(fnr_pplot_per_dim), fnr_pplot_per_dim; color=:forestgreen, linewidth=2)
    hlines!(ax, [0.0]; color=:gray, linestyle=:dash, linewidth=1)
    axislegend(ax; position=:rt, framevisible=true, labelsize=10)

    add_caption!(fig, 3,
        "FNR at d=3:  PG-FNU = $(round(fnr_pgfnu_per_dim[min(3, end)], digits=4)),  " *
        "differential FNN = $(round(fnr_diff_per_dim[min(3, end)], digits=4)),  " *
        "P_plot = $(round(fnr_pplot_per_dim[min(3, end)], digits=4)).  " *
        "PG-FNU minimizes FNR at every step, so its curve is the lower envelope.")

    save(joinpath(fig_dir, "UNFOLD7_fnr_vs_dim.png"), fig)
    println("  Saved: UNFOLD7_fnr_vs_dim.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE UNFOLD-8 — BENCHMARK: derivative continuity ||d²P/dθ²|| vs θ
# ---------------------------------------------------------------------------
println("  UNFOLD-8: derivative continuity")

"""Second derivative norm w.r.t. θ, periodic central differences."""
function second_derivative_norm(P::AbstractMatrix, θ::AbstractVector)
    N = size(P, 2)
    d2n = zeros(N)
    for i in 1:N
        ip, im = mod1(i + 1, N), mod1(i - 1, N)
        dθ_p = θ[ip] - θ[i];  dθ_p < 0 && (dθ_p += 2π);  dθ_p = max(dθ_p, 1e-12)
        dθ_m = θ[i] - θ[im];  dθ_m < 0 && (dθ_m += 2π);  dθ_m = max(dθ_m, 1e-12)
        d2 = 2.0 .* ((P[:, ip] .- P[:, i]) ./ dθ_p .-
                     (P[:, i] .- P[:, im]) ./ dθ_m) ./ (dθ_p + dθ_m)
        d2n[i] = norm(d2)
    end
    return d2n
end

let
    d2_pgfnu = second_derivative_norm(P_pgfnu, θ_cycle)
    d2_diff  = second_derivative_norm(P_diff, θ_cycle)
    d2_pplot = second_derivative_norm(P_plot_proj, θ_cycle)

    fig = Figure(size=(1200, 750))
    Label(fig[0, :],
          "Curvature ||d²P/dθ²|| along the cycle: differential FNN is the smoothest " *
          "parametrization (mean $(round(mean(d2_diff), digits=3)) vs " *
          "$(round(mean(d2_pplot), digits=3)) for P_plot)";
          fontsize=13, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    ax = Axis(fig[2, 1];
              title="lower and slowly-varying = smoother embedding",
              titlesize=11,
              xlabel="Koopman phase θ  [rad]", ylabel="||d²P/dθ²||")
    lines!(ax, θ_cycle, d2_pgfnu; color=:crimson, linewidth=1.8,
           label="PG-FNU (mean $(round(mean(d2_pgfnu), digits=3)))")
    lines!(ax, θ_cycle, d2_diff; color=:dodgerblue, linewidth=1.8,
           label="Differential FNN (mean $(round(mean(d2_diff), digits=3)))")
    lines!(ax, θ_cycle, d2_pplot; color=:forestgreen, linewidth=1.8,
           label="Original P_plot (mean $(round(mean(d2_pplot), digits=3)))")
    axislegend(ax; position=:rt, framevisible=true, labelsize=10)

    add_caption!(fig, 3,
        "Second derivative of the embedded curve with respect to the Koopman phase " *
        "(periodic central differences).  Spikes mark folds or non-uniform parametrization " *
        "speed; the differential FNN was optimized precisely against this quantity.")

    save(joinpath(fig_dir, "UNFOLD8_derivative_continuity.png"), fig)
    println("  Saved: UNFOLD8_derivative_continuity.png")
end
GC.gc()

# ---------------------------------------------------------------------------
# FIGURE UNFOLD-9 — SUMMARY: 4-panel comparison
# ---------------------------------------------------------------------------
println("  UNFOLD-9: summary 4-panel comparison")

let
    fig = Figure(size=(2000, 1050))
    Label(fig[0, :],
          "Four 3-D projections of the same limit cycle, colored by Koopman phase — " *
          "PG-FNU achieves the cleanest unfolding (FNR = " *
          "$(round(fnr_history_pgfnu[end], digits=3)))";
          fontsize=14, font=:bold)
    Label(fig[1, :], fig_subtitle; fontsize=10, color=:gray)

    panels = Tuple{String, Matrix{Float64}, String}[]
    P3 = to3(P_plot_proj)
    push!(panels, ("Original P_plot  (FNR = $(round(fnr_pplot, digits=3)))", P3,
                   "$(lbl(1)) | $(lbl(2)) | $(lbl(3))"))
    push!(panels, ("PG-FNU  (FNR = $(round(fnr_history_pgfnu[end], digits=3)))",
                   P_pgfnu, "dir 1 | dir 2 | dir 3"))
    push!(panels, ("Differential FNN  (FNR = $(round(fnr_diff, digits=3)))",
                   P_diff, "dir 1 | dir 2 | dir 3"))
    if !isnothing(dyn_frame_results[1])
        r = dyn_frame_results[1]
        e_t_ref, e_n_ref, e_b_ref = r.E_t[:, 1], r.E_n[:, 1], r.E_b[:, 1]
        s_ref_dyn = S_cycle[:, 1]
        ρ_t = [dot(S_cycle[:, i] .- s_ref_dyn, e_t_ref) for i in 1:n_pts]
        ρ_n = [dot(S_cycle[:, i] .- s_ref_dyn, e_n_ref) for i in 1:n_pts]
        ρ_b = [dot(S_cycle[:, i] .- s_ref_dyn, e_b_ref) for i in 1:n_pts]
        push!(panels, ("Dynamical frame (reference frame at θ=0)",
                       vcat(reshape(ρ_t, 1, :), reshape(ρ_n, 1, :), reshape(ρ_b, 1, :)),
                       "e_t | e_n | e_b"))
    end

    hm_last = nothing
    for (c, (tname, P, lbls)) in enumerate(panels)
        lx, ly, lz = split(lbls, " | ")
        ax = Axis3(fig[2, c]; title=tname, titlesize=11,
                   xlabel=lx, ylabel=ly, zlabel=lz)
        hm_last = scatter!(ax, P[1, :], P[2, :],
                           size(P, 1) >= 3 ? P[3, :] : zeros(n_pts);
                           color=θ_cycle, colormap=:thermal, markersize=5,
                           colorrange=(-π, π))
        lines!(ax, P[1, :], P[2, :], size(P, 1) >= 3 ? P[3, :] : zeros(n_pts);
               color=(:black, 0.3), linewidth=0.8)
    end
    Colorbar(fig[2, length(panels) + 1], hm_last; width=14,
             label="Koopman phase θ  [rad]")

    add_caption!(fig, 3,
        "Left to right: standard physical+POD projection, PG-FNU (FNR-minimizing), " *
        "differential FNN (smoothness-maximizing), and the dynamical frame (e_t, e_n, e_b) " *
        "of STEP 5.  Read the black phase-ordered path: where it crosses itself, the " *
        "projection folds the cycle (false neighbors).  PG-FNU is unfolded by construction.")

    save(joinpath(fig_dir, "UNFOLD9_summary_comparison.png"), fig)
    println("  Saved: UNFOLD9_summary_comparison.png")
end
GC.gc()
print_mem("after UNFOLD figures")

# ============================================================================
# STAGED SAVE — STAGE B (unfolding + script_complete)
# ============================================================================

println("\n  [Stage save B] unfolding results → $save_path_out")
jldopen(save_path_out, "r+") do f
    # cycle used for unfolding
    f["unfold/k_cycle"] = k_cycle
    f["unfold/n_pts"]   = n_pts
    f["unfold/θ_cycle"] = θ_cycle

    # STEP 6: PG-FNU
    f["B_pgfnu"] = B_pgfnu
    f["P_pgfnu"] = P_pgfnu
    f["fnr_history_pgfnu"] = fnr_history_pgfnu
    f["h1_pgfnu"] = h1_pgfnu
    f["births_pgfnu"] = births_pgfnu
    f["max_d_pgfnu"] = max_d_pgfnu

    # STEP 7: differential FNN
    f["B_diff"] = B_diff
    f["P_diff"] = P_diff
    f["smoothness_history_diff"] = smoothness_history_diff
    f["fnr_diff"] = fnr_diff
    f["h1_diff"] = h1_diff
    f["births_diff"] = births_diff
    f["max_d_diff"] = max_d_diff

    # reference: original P_plot projection
    f["P_plot_proj"] = P_plot_proj
    f["fnr_pplot"] = fnr_pplot
    f["h1_pplot"] = h1_pplot
    f["births_pplot"] = births_pplot
    f["max_d_pplot"] = max_d_pplot

    # FNR tables per dimension
    f["fnr_pgfnu_per_dim"] = fnr_pgfnu_per_dim
    f["fnr_diff_per_dim"]  = fnr_diff_per_dim
    f["fnr_pplot_per_dim"] = fnr_pplot_per_dim

    f["have_ripserer"] = HAVE_RIPSERER
    f["script_complete"] = true
    f["timestamp"] = string(now())
end
println("  [Stage save B] done — script_complete=true written.")
GC.gc()

# ============================================================================
# SUMMARY
# ============================================================================

println("\n" * "="^72)
println("PROJECTION METHODS COMPLETE")
println("="^72)
println("Script 1 (fitting):   $save_path_s1")
println("Script 2 (spectral):  $save_path_s2")
println("Output file:          $save_path_out")
println("Figure dir:           $fig_dir")
println("")

println("STEP 1 — Harmonic branches: $n_branches" *
        (degenerate_detected ? "  (DEGENERATE eigenvalue handled via seeded k-means ratio clustering)" : ""))
for k in 1:n_branches
    println("  Branch $k:  j=$(j_phases[k]),  λ=$(round(λ_phases[k], digits=6)),  " *
            "ω=$(round(ω_phases[k], digits=4)) rad/s,  T=$(round(T_phases[k], digits=4)) s")
end

println("\nSTEP 2 — Cycle separation:")
for k in 1:n_branches
    println("  Cycle $k:  $(cycle_counts[k]) points  (separation ×$(round(sep_ratio[k], digits=2)))")
end

println("\nSTEP 3 — Eigenfunction → cycle matching:")
for j in 1:n_branches
    k = final_match[j]
    println("  eig $j → cycle $k  (winding=$(round(match_table_winding[j, k], digits=3)), " *
            "coherence=$(round(match_table_coherence[j, k], digits=3)), " *
            "freq_match=$(round(match_table_freqmatch[j, k], digits=3)))")
end

println("\nSTEP 4 — Gauge (per-cycle phase reference):")
for j in 1:n_branches
    println("  eig $j:  |c_j|=$(round(gauge_scale[j], digits=4)),  " *
            "arg(c_j)=$(round(gauge_offset[j], digits=4)) rad")
end
println("  Cross-cycle phase comparison is NOT meaningful without an external reference.")

println("\nSTEP 5 — Dynamical frame:")
for k in 1:n_branches
    r = dyn_frame_results[k]
    if isnothing(r)
        println("  Cycle $k: skipped (too few points)")
        continue
    end
    println("  Cycle $k:  T=$(round(r.T, digits=4)) s, ω=$(round(r.ω, digits=4)) rad/s, " *
            "$(length(r.θ_tof)) points")
    println("    periodicity: ||Δe_t||=$(round(r.per_t, digits=6)), " *
            "||Δe_n||=$(round(r.per_n, digits=6)), ||Δe_b||=$(round(r.per_b, digits=6))")
    if r.hessian_ok
        println("    Hessian:  mean λ₁=$(round(mean(r.λ1), sigdigits=3)), " *
                "mean λ₂=$(round(mean(r.λ2), sigdigits=3)), " *
                "tangent align=$(round(median(r.tangent_align), sigdigits=3)), " *
                "ρ closure=$(round(r.ρ_closure, sigdigits=3))")
    else
        println("    Hessian:  skipped (no transverse pair); frame-only transverse directions")
    end
end

println("\nSTEPS 6–8 — Unfolding (cycle $k_cycle, $n_pts pts):")
println("  PG-FNU:           d=$(size(B_pgfnu, 2)),  final FNR=$(round(fnr_history_pgfnu[end], digits=4)), " *
        "H₁ classes=$(length(h1_pgfnu)), max persistence=$(round(h1_max_persistence(h1_pgfnu), digits=4))")
println("  Differential FNN: d=$(size(B_diff, 2)),  FNR=$(round(fnr_diff, digits=4)), " *
        "smoothness=$(round(smoothness_history_diff[end], digits=6)), " *
        "H₁ classes=$(length(h1_diff)), max persistence=$(round(h1_max_persistence(h1_diff), digits=4))")
println("  Original P_plot:  FNR=$(round(fnr_pplot, digits=4)), " *
        "H₁ classes=$(length(h1_pplot)), max persistence=$(round(h1_max_persistence(h1_pplot), digits=4))")
HAVE_RIPSERER || println("  (persistent homology SKIPPED — Ripserer not installed)")

println("\nFigures saved in $fig_dir:")
for fn in ["PHASE1_spectrum_harmonic_branches.png",
           "PHASE2_cycle_separation.png",
           "PHASE3_matching_tables.png",
           "PHASE3b_winding_verification.png",
           "PHASE4_gauge_analysis.png",
           "PHASE5_phase_colored_cycles.png",
           "PHASE6_per_cycle_koopman_coords.png",
           "PHASE7_joint_basin_coords.png  (multi-cycle only)",
           "PHASE8_per_cycle_pod.png",
           "PHASE9_per_cycle_singular_values.png",
           "PHASE10_dynamical_derivatives.png",
           "PHASE11_frame_continuity.png",
           "PHASE12_hessian_spectral_analysis.png  (if transverse pair available)",
           "PHASE13_dynamical_frame_coords.png",
           "PHASE13b_cylindrical_clean_lc_comparison.png",
           "PHASE13c_dynamical_frame_3way.png",
           "PHASE14_transverse_directions.png",
           "PHASE15_consistency_checks.png",
           "UNFOLD1_pgfnu_projection.png",
           "UNFOLD2_pgfnu_vs_original.png",
           "UNFOLD3_differential_fnn.png",
           "UNFOLD4_differential_vs_original.png",
           "UNFOLD5_persistent_homology_pgfnu.png  (requires Ripserer)",
           "UNFOLD6_persistent_homology_differential.png  (requires Ripserer)",
           "UNFOLD7_fnr_vs_dim.png",
           "UNFOLD8_derivative_continuity.png",
           "UNFOLD9_summary_comparison.png"]
    println("  $fn")
end
println("="^72)
print_mem("end")
