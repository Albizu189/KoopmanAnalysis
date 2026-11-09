# ============================================================================
# Koopman fitting-methods benchmark — ALL methods on the SAME dataset
# (default: Epileptor3D, regime "c4b-SN-FLC", all 3 dims shown).
#
# Disk-backed rewrite of originals/Koopman_Experiments_Epileptor3D_disksaving.jl
# with the heading structure of originals/Koopman_Classic_analysis_main.jl
# (GLOBAL CONFIGURATION with experiment name / regime_system / regime_name /
# dim / showdim, plotting-dimension computation, per-system parameter
# dispatch, staged JLD2 saves, SUMMARY print block).
#
# THE SEVEN METHODS (method_ids), all fitted on the SHOWN-dim MULTI-TRAJECTORY
# pair data: delay_embed_pairs(X_shown_raw, Y_shown_raw, m_embed, tau_delay)
# builds ONE concatenated per-trajectory snapshot matrix S_all plus X/Y
# index views into it, and pair_analysis(pd::NamedTuple, cfg) fits every
# method on those pairs (see KoopmanAnalysis.Hankel / .Config):
#   1. "DMD"           — plain DMD on the raw shown state: m_embed=1,
#                        tau_delay=1, dict_type=:none, use_svd=false.
#                        NOTE (verified in Hankel.jl): hankel_dmd ALWAYS does
#                        an SVD of S and ignores use_svd; with r=nothing it
#                        keeps the full rank (= n_show here). The use_svd flag
#                        is therefore only descriptive for :none methods.
#   2. "Hankel_DMD"    — linear Hankel-DMD, full-rank SVD (r=nothing).
#   3. "HAVOK"         — continuous-time HAVOK (Brunton et al. 2017), rank
#                        chosen automatically (rank_method=:energy,
#                        energy=0.999). K is the diagnostics-only flow map
#                        exp([A B; 0 0]*dt); prediction uses RK4 on (A, B).
#   4. "Hankel_DMD_r"  — like Hankel_DMD but truncated to r = r_havok, the
#                        rank HAVOK selected (fitted AFTER HAVOK; r_havok is
#                        read from size(res_havok.U_r, 2), the number of POD
#                        modes actually kept).
#   5. "EDMD"          — RBF-EDMD on the raw shown state: m_embed=1,
#                        tau_delay=1, use_svd=false, thin-plate RBF dict.
#   6. "Hankel_EDMD"   — full Hankel RBF-EDMD (use_svd=false, dictionary on
#                        the full delay state), thin-plate RBF dict.
#   7. "Hankel_EDMD_r" — Hankel RBF-EDMD on r_havok POD modes (use_svd=true,
#                        r=r_havok), thin-plate RBF dict.
#
# MEMORY DESIGN (preserved-or-better from the disksaving original):
#   • one-method-at-a-time fitting with immediate offload_result! + GC.gc();
#   • heavy AnalysisResult fields (S, ΨX, ΨY, X_r) live in the JLD2 file
#     under fit/<mid>/, with shared-parent view deduplication;
#   • the fitting data is ONE concatenated per-trajectory snapshot matrix
#     S_all (delay_embed_pairs) — X/Y are index views of it, so the delay
#     data is stored ONCE (delay_embed_training_data would duplicate it);
#   • streaming per-trajectory prediction writes (delay_results/<mid>/traj_<j>)
#     — the full (n_state × n_pred × n_traj) array is never materialised;
#   • POD via eigen(Symmetric(S*S')) (BLAS syrk) — never svd's Vt;
#   • only n_plot-row projections are materialised for plotting.
#
# PACKAGE API ASSUMPTIONS VERIFIED AGAINST SOURCE (Config.jl / Hankel.jl /
# EDMD.jl) — PAIR versions of every entry point (single-trajectory limit of
# each reproduces its Hankel counterpart exactly):
#   • pair_analysis(pd, cfg) dispatches on the RAW cfg.dict_type against
#     :none/:kernel/:havok; :RBF falls through to edmd_pairs, which
#     lowercases it to :rbf. So dict_type=:RBF is safe here.
#   • dmd_pairs honours use_svd: use_svd=false → K = Y·pinv(X), U_r = I
#     (unlike hankel_dmd, which always SVDs); use_svd=true, r=nothing →
#     full-POD K = U_r(Y_r·pinv(X_r))U_r' == Y·pinv(X) (rotation-invariant).
#   • edmd_pairs(use_svd=false) returns U_r = I (full Matrix) and
#     X_r = view(S_all, :, pd.x) — X_r SHARES STORAGE with res.S, so
#     offload_result!'s objectid dedup writes it only once.
#   • ΨX/ΨY are INDEX views of ONE parent Ψall = ψ(Vc) (edmd_pairs lifts the
#     shared parent once) → offloaded once as fit/<mid>/ΨX_full.
#     HAVOK/:none results have ΨX=ΨY=nothing.
#   • havok_pairs takes the 4th-order central-difference derivatives INSIDE
#     each trajectory block (pd.cols) — the stencil never crosses a boundary;
#     POD basis and rank selection run on S_all (eigen(S·S'), as havok_dmd).
#   • spectrum(res) works for every method here: dense path when
#     size(K,1) < 3001, KrylovKit eigsolve otherwise. We pass
#     howmany ≤ krylovdim explicitly because KrylovKit requires it.
#     (:kernel results also work, but this script has no kernel method.)
#   • RBF centres returned in dict_info.centers live in the Vc coordinates:
#     REDUCED POD coordinates when use_svd=true (map back with U_r), delay
#     coordinates when use_svd=false (U_r = I, so U_r * centers is uniform).
# ============================================================================

using KoopmanAnalysis
using KoopmanAnalysis.Systems
using KoopmanAnalysis.Config
using KoopmanAnalysis.Plotting
using KoopmanAnalysis.Figures
using KoopmanAnalysis.Regimes
using KoopmanAnalysis.Utils
using KoopmanAnalysis.Spectral
using KoopmanAnalysis.Hankel
using KoopmanAnalysis.EDMD
using KoopmanAnalysis.Dictionaries
using KoopmanAnalysis.DataGeneration

using CairoMakie
using Statistics
using Random
using LinearAlgebra
using Printf
using JLD2
using LaTeXStrings
using Dates
using ProgressMeter

CairoMakie.activate!(type="png")
set_koopman_theme!()

# ── Reproducibility (seeding paradigm, brief rule 2) ──
# One top-level seed; the same value is threaded into every KoopmanConfig
# (cfg.seed → RBF centre clustering etc.) and into script-level sampling via
# dedicated generators seeded from it.
const SEED = 42
Random.seed!(SEED)

# ============================================================================
# DISK-BACKING UTILITIES  (verbatim from the disksaving original)
# ============================================================================

"""Human-readable byte count for logging."""
fmt_bytes(b::Integer) = b >= 1<<30 ? @sprintf("%.2f GiB", b/2^30) :
                        b >= 1<<20 ? @sprintf("%.2f MiB", b/2^20) :
                        @sprintf("%.2f KiB", b/2^10)

"""Print currently free system RAM (useful to verify the refactor)."""
print_mem(tag::AbstractString) =
    @printf("%-55s free RAM: %.2f GiB\n", tag, Sys.free_memory()/2^30)

# Field names of AnalysisResult considered "large". They are written to disk
# right after fitting and replaced by `nothing` in the in-RAM copy.
# (Verified against Config.jl: the heavy fields are S, ΨX, ΨY and X_r; for
#  HAVOK, X_r holds the r × n_snap POD coordinate matrix V_r.)
const HEAVY_RESULT_FIELDS = (:S, :ΨX, :ΨY, :X_r)

"""
    offload_result!(f, res, mid) -> NamedTuple

Write every heavy array field of `res` into the open JLD2 file `f` under
`fit/<mid>/<fieldname>`, release the in-RAM copy, and return a lightweight
NamedTuple with the same field names where heavy fields are `nothing`.

Since edmd_pairs returns ΨX/ΨY as INDEX views of ONE lifted matrix (and X_r as
a view of the coordinate matrix / of S_all itself when use_svd=false), shared
storage is written only once: views are stored via their PARENT array under
`fit/<mid>/<field>_full`, and later fields pointing to the same parent are
skipped. Read-back convention (pd indices are saved at stage 1):
    x_idx = read("pd_full/x");  y_idx = read("pd_full/y")
    ΨX = read("fit/<mid>/ΨX_full")[:, x_idx]
    ΨY = read("fit/<mid>/ΨX_full")[:, y_idx]
    X_r = read("fit/<mid>/X_r_full")[:, x_idx]   (or indexed columns of S_all)
"""
function offload_result!(f, res, mid; heavy_fields=HEAVY_RESULT_FIELDS)
    pairs = Pair{Symbol,Any}[]
    stored = Dict{UInt,String}()   # objectid(array) -> JLD2 path (dedup shared storage)
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

experiment_name = "StateSpace-LargeExp-multitraj-delays"           # experiment name (subdir of save dirs)
regime_system   = "FHN"
regime_name     = length(ARGS) > 0 ? ARGS[1] : "c4b-SN-FLC"

# ---------------------------------------------------------------------------
# Dimension-control parameters (as in Koopman_Classic_analysis_main.jl)
# ---------------------------------------------------------------------------
dim      = 2
showdim  = [1, 2]                 # which physical coords to TRAIN with
n_show   = length(showdim)
n_hidden = dim - n_show
n_plot   = dim                       # plotting directions: n_show phys + n_hidden POD

@assert issorted(showdim) "showdim must be sorted in increasing order"
@assert all(1 .<= showdim .<= dim) "showdim entries must lie in 1:dim ($dim)"
@assert n_show <= dim "n_show cannot exceed dim"

showdim_tag = join(string.(showdim))

# --- Local path configuration (see LocalPaths.jl.example) ---
const _LOCAL_PATHS = joinpath(@__DIR__, "LocalPaths.jl")
include(isfile(_LOCAL_PATHS) ? _LOCAL_PATHS : joinpath(@__DIR__, "LocalPaths.jl.example"))
import .LocalPaths: DATA_ROOT, RESULTS_ROOT

# Figures and light-weight outputs
save_dir = default_save_dir(regime_system, regime_name;
                            base_dir=RESULTS_ROOT, subdir=experiment_name)
mkpath(save_dir)
println("Output directory: ", save_dir)

# Heavy disk-backed datasets (JLD2), per the SHARED_BRIEF data contract
data_save_dir = default_save_dir(regime_system, regime_name;
                                 base_dir=DATA_ROOT, subdir=experiment_name)
mkpath(data_save_dir)
save_path = joinpath(data_save_dir,
                     "fitting_methods-$regime_system-$regime_name-show$showdim_tag.jld2")
println("Datasets directory: ", save_path)

# Start from a clean results file. MUST happen here, BEFORE the stage-1 save
# below (when this line lived at the fit loop it deleted the stage-1 backup).
isfile(save_path) && rm(save_path)

print_mem("start")

# ---------------------------------------------------------------------------
# System parameters (per-system dispatch, as in Koopman_Classic_analysis_main.jl)
# ---------------------------------------------------------------------------

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
println("Silent-state fixed point: x0 = ", x0_silent)

# Common figure subtitle strings
title_model = "$regime_system — $regime_name | shown coords: $(join(showdim, ",")) of $dim"

# ============================================================================
# DATA GENERATION — single long RK4 MASTER trajectory (reference / test only)
# ============================================================================
# The master trajectory is NO LONGER the fitting data: all seven methods are
# fitted on the MULTI-TRAJECTORY pair dataset generated further below.  The
# master still provides (i) the MI tau-search series, (ii) the test window /
# prediction ground truth, and (iii) the Figure-(a) reference portrait.
# Epileptor3D defaults from the disksaving original: dt_internal=0.01, nLag=10
# (so consecutive SAVED snapshots are dt_eff = 0.1 time units apart),
# n_train=100000, n_test=50000, n_pred=10000, n_traj_test=10.  A burn-in
# segment lets the trajectory leave the silent fixed point and reach the
# attracting regime before training data are taken (the original used
# burn_in=0; it is an explicit knob here).

dt_internal = 0.01          # internal RK4 step (time units)
nLag        = 10            # internal steps per saved snapshot
dt_eff      = dt_internal * nLag   # physical time BETWEEN SNAPSHOTS (cfg.dt)

n_train     = 1500000        # training snapshots (after burn-in)
n_test      = 50000         # test-window snapshots (after training)
n_pred      = 5000         # prediction horizon (snapshots)
n_traj_test = 5            # number of predicted test trajectories

burn_in     = 0          # discarded transient snapshots (knob; original: 0)
m_total     = burn_in + n_train + n_test
total_time  = m_total * dt_eff

println("dt_eff = ", dt_eff, "  |  total snapshots = ", m_total,
        "  (burn-in $burn_in + train $n_train + test $n_test)")

# rk4 is deterministic (no seed). X_master: dim × (m_total+1).
X_master = rk4(f_det, x0_silent, dt_internal, m_total; nLag=nLag)
println("Master trajectory shape: ", size(X_master))
print_mem("after master trajectory")

# ---------------------------------------------------------------------------
# Split into train / test blocks (SHOWN coordinates for fitting, full dim for
# the record).  Column layout of X_master:
#   [1 : burn_in]                        burn-in (discarded from training)
#   [burn_in+1 : burn_in+n_train]        training block
#   [burn_in+n_train+1 : +n_test]        test window
# ---------------------------------------------------------------------------

X_shown_master = X_master[showdim, :]                      # n_show × (m_total+1)

train_cols = (burn_in + 1):(burn_in + n_train)
test_cols  = (burn_in + n_train + 1):(burn_in + n_train + n_test)

X_train = X_shown_master[:, train_cols]                    # n_show × n_train
@assert size(X_train) == (n_show, n_train)

sub_model_data = "RK4 method, integration step = $dt_internal, save delay = $nLag, " *
                 "train = $n_train, test = $n_test, burn-in = $burn_in"

# ---------------------------------------------------------------------------
# Figure (a) — MASTER trajectory in physical (shown) coordinates (reference /
# test context; the FITTING data is the multi-trajectory dataset below).
# NO subsampling (brief requirement); 3D when n_show >= 3, else 2D.
# ---------------------------------------------------------------------------
if n_show >= 3
    fig = fig_training_data(X_train;
        dim=3, projection_dims=(1, 2, 3),
        title="Master trajectory (reference / test data)",
        subtitle=title_model, param_str=sub_model_data)
elseif n_show == 2
    fig = fig_training_data(X_train;
        dim=2, projection_dims=(1, 2),
        title="Master trajectory (reference / test data)",
        subtitle=title_model, param_str=sub_model_data)
else
    # single shown coordinate: time series instead of a phase portrait
    fig = fig_time_series(vec(X_train[1, :]), dt_eff;
        label="x$(showdim[1])(t)",
        title="Master trajectory, shown coordinate (reference / test data)",
        subtitle=title_model, param_str=sub_model_data)
end
save(joinpath(save_dir, "01_original_training_data.png"), fig)
println("Saved: 01_original_training_data.png")
fig = nothing; GC.gc()

# ============================================================================
# TAU_DELAY SELECTION — mutual information on the first shown component
# ============================================================================
# Knob: set tau_delay_manual to an Int to skip the automatic search.
# Auto: tau_delay = τ of the first local minimum of the MI curve; if the
# search fails (no true local minimum), fall back to the default 10.

tau_delay_manual = 10          # ← tau_delay = nothing → auto

println("Finding optimal τ using mutual information (MI) measurement...")

τ_range = 1:100
I_vals  = [mutual_information(vec(X_train[1, :]), τ; n_bins=50) for τ in τ_range]

tau_opt = let idx = find_first_minimum(I_vals)
    # find_first_minimum falls back to argmin when NO local minimum exists;
    # detect that fallback and treat it as a failed search (→ default 10).
    is_local_min = 1 < idx < length(I_vals) &&
                   I_vals[idx] < I_vals[idx-1] && I_vals[idx] <= I_vals[idx+1]
    is_local_min ? τ_range[idx] : nothing
end

tau_delay = if !isnothing(tau_delay_manual)
    println("tau_delay set manually: $tau_delay_manual (MI search skipped for the choice)")
    tau_delay_manual
elseif !isnothing(tau_opt)
    println("Optimal τ = $tau_opt snapshots (first MI minimum)")
    tau_opt
else
    @warn "MI minimum search failed (no local minimum) — using default tau_delay = 10"
    10
end

# Figure (c) — mutual-information result
fig = Figure(size=(800, 400))
ax  = Axis(fig[1, 1], xlabel="τ (snapshots)", ylabel="I(τ) (nats)",
           title="Mutual Information | $regime_name (dt_eff = $dt_eff)", titlesize=14)
lines!(ax, collect(τ_range), I_vals, color=COLOR_PRIMARY, linewidth=2)
vlines!(ax, tau_delay, color=:orange, linewidth=2, linestyle=:dash)
τ_idx = findfirst(==(tau_delay), collect(τ_range))
if !isnothing(tau_opt) && !isnothing(τ_idx)
    text!(ax, tau_delay + 2, I_vals[τ_idx] + 0.05,
          text="τ_delay = $tau_delay", color=:orange, fontsize=12)
end
save(joinpath(save_dir, "03_tau_delay_mutual_information.png"), fig)
println("Saved: 03_tau_delay_mutual_information.png")
fig = nothing; GC.gc()

# ============================================================================
# DELAY EMBEDDING — multichannel delay PAIRS of the SHOWN coordinates
# ============================================================================
# m_embed is the per-channel embedding depth: with n_show channels the delay
# state has n_state = n_show * m_embed rows.  TWO PAIR DATASETS are needed,
# both built from the multi-trajectory raw pairs: the main one
# (m_embed, tau_delay) for the Hankel methods and the trivial one
# (m_embed=1, tau_delay=1 — the raw shown state) for the DMD / EDMD reference
# methods.  Each is a delay_embed_pairs dataset: ONE concatenated
# per-trajectory snapshot matrix S_all plus X/Y index views into it (the
# delay data is stored ONCE — X and Y share the parent, so the library can
# lift/project it a single time).

m_embed  = 20                       # per-channel embedding depth (main)
max_lag  = (m_embed - 1) * tau_delay  # samples spanned by one delay vector
n_state  = n_show * m_embed           # delay-state dimension (main embedding)
n_snap   = n_train - max_lag          # Hankel snapshot count (main embedding)

println("Delay embedding: m_embed=$m_embed, tau_delay=$tau_delay " *
        "→ n_state = $n_show × $m_embed = $n_state (single-traj n_snap = $n_snap)")

n_traj    = 2000
window_ic = 2.0

m_traj = div(n_train, n_traj) + (m_embed - 1) * tau_delay + 1
n_pairs_per_traj = m_traj - max_lag

println("\nMulti-trajectory training: $n_traj trajectories × $m_traj steps each")

# ---------------------------------------------------------------------------
# Generate multi-trajectory data
# ---------------------------------------------------------------------------
println("\nGenerating multi-trajectory training data (full $dim-D) ...")

X_train_raw, Y_train_raw, _, _, _ = edmd_training_data(
    regime_system, regime_name;
    m_train        = m_traj,
    n_trajectories = n_traj,
    window         = window_ic,
    dt             = dt_internal,
    nLag           = nLag,
    center         = x0_silent,
    seed           = SEED
)
print_mem("after training-data generation")

# ---------------------------------------------------------------------------
# Delay-embed the shown coordinates only
# ---------------------------------------------------------------------------
println("\nBuilding delay-embedded training pairs (shown coords only) ...")

X_shown_raw = @view X_train_raw[showdim, :]
Y_shown_raw = @view Y_train_raw[showdim, :]

# THE FITTING DATA — multi-trajectory delay pairs as index views of ONE
# concatenated per-trajectory snapshot matrix (trajectory boundaries never
# enter a delay vector, and X/Y pairs never cross a boundary).
pd_full  = delay_embed_pairs(X_shown_raw, Y_shown_raw, m_embed, tau_delay;
                             n_trajectories = n_traj)
# m_embed=1, tau_delay=1 degenerates to the state-space concatenation — the
# pair dataset for the DMD / EDMD reference methods (same machinery).
pd_small = delay_embed_pairs(X_shown_raw, Y_shown_raw, 1, 1;
                             n_trajectories = n_traj)

@assert pd_full.n_pairs == n_pairs_per_traj "delay_embed_pairs pair-count mismatch"
@assert size(pd_full.S) == (n_state, (n_pairs_per_traj + 1) * n_traj)

n_pairs_total = length(pd_full.x)
println("Delay-embedded pairs (shown only): X = Y = ($n_state, $n_pairs_total) " *
        "views of S_all ", size(pd_full.S),
        " (", fmt_bytes(sizeof(pd_full.S)), ")")
println("State-space reference pairs (m_embed=1): ", length(pd_small.x),
        " pairs, S_all ", size(pd_small.S))

# NOTE: X_train_raw / Y_train_raw stay alive on purpose — method_plot_projection
# re-embeds them at foreign (m_embed, tau_delay) when n_hidden > 0.  They are
# tiny (n × m_traj·n_traj) compared with the delay matrices, so keeping them
# costs nothing; releasing Y_train_raw here would break that helper.
print_mem("after delay-embedded pair datasets (S_all shared by X/Y views)")

# ---------------------------------------------------------------------------
# Computation of plotting dimensions (POD of the shown-only delay matrix)
# ---------------------------------------------------------------------------
# U_pod is used ONLY for plotting directions when n_hidden > 0.  Computed via
# eigen(Symmetric(S*S')) (BLAS syrk, n_state × n_state) — never svd's Vt,
# which would be an n_state × n_snap matrix (several GiB) that is never used.
# NOTE: eigenvector signs may differ from svd()'s — irrelevant for a plotting
# basis.

println("\nPerforming POD analysis on the multitrajectory delay matrix (plotting only) ...")
G_pod = pd_full.S * pd_full.S'
E_pod = eigen(Symmetric(G_pod))
p_pod = sortperm(E_pod.values; rev=true)
U_pod = E_pod.vectors[:, p_pod]                    # n_state × n_state
σ_pod = sqrt.(max.(E_pod.values[p_pod], 0.0))
G_pod = nothing; E_pod = nothing
println("  POD singular values (first $(min(10, n_state))): ",
        round.(σ_pod[1:min(10, n_state)], digits=4))
print_mem("after POD analysis")

# ---------------------------------------------------------------------------
# phys_idx — current-time shown coords in the delay state.
# VERIFIED against Hankel.jl: build_hankel puts the LATEST sample in row 1 of
# each block (S[d+1, t] = v[t + max_lag - d*τ]), and build_hankel_multichannel
# stacks one such block per channel, CHANNEL-MAJOR: channel j occupies rows
# (j-1)*m_embed+1 : j*m_embed.  Hence the current-time value of shown coord j
# sits at row (j-1)*m_embed + 1.
# ---------------------------------------------------------------------------
phys_idx = [(j - 1) * m_embed + 1 for j in 1:n_show]
println("phys_idx (current-time shown coords in delay state): ", phys_idx)

# Physical-coordinate labels (per system) — needed for plot_labels
function sys_phys_label(system::AbstractString, d::Int)
    labels = if system == "VanderPol"
        ["x", "y", "z"]
    elseif system == "Lorenz"
        ["x", "y", "z"]
    elseif system == "Epileptor3D"
        ["x", "y", "z"]
    elseif system == "FHN"
        ["v", "w"]
    elseif system == "Duffing"
        ["x", "ẋ"]
    else
        ["x$k" for k in 1:dim]
    end
    @assert 1 <= d <= length(labels) "sys_phys_label: d=$d out of range for system $system"
    return labels[d]
end

# P_plot:  n_plot × n_state  such that  P_plot * s → n_plot plotting coords
#   rows 1..n_show:      unit vectors at phys_idx positions (shown coords)
#   rows n_show+1..n_plot: U_pod[:, 1:n_hidden]' (POD directions)
P_plot = zeros(n_plot, n_state)
for j in 1:n_show
    P_plot[j, phys_idx[j]] = 1.0
end
if n_hidden > 0
    P_plot[n_show+1:n_plot, :] = U_pod[:, 1:n_hidden]'
end

plot_labels = vcat(
    [sys_phys_label(regime_system, showdim[j]) for j in 1:n_show],
    ["POD$h" for h in 1:n_hidden]
)
println("P_plot: $(size(P_plot))  (n_plot × n_state)")
println("plot_labels: ", plot_labels)

# Helper for projecting delay states to plotting coords
plot_coords(S::AbstractMatrix) = P_plot * S
plot_coords(s::AbstractVector) = P_plot * s

# Small training-data projection for the figures: the delay-embedded training
# trajectory projected back to PHYSICAL shown coords via phys_idx (plus POD
# rows when n_hidden > 0).  Only n_plot rows are materialised (memory rule).
X_train_plot3 = P_plot * pd_full.S                   # n_plot × n_pairs_total
println("X_train_plot3 (delay-conformed training data, plot space): ",
        size(X_train_plot3))

# ---------------------------------------------------------------------------
# Figure (b) — training data in PHYSICAL dimensions after the Hankel matrix
# is conformed (delay-embedded trajectory projected back via phys_idx).
# ---------------------------------------------------------------------------
if n_plot >= 3
    fig = fig_training_data(X_train_plot3;
        dim=3, projection_dims=(1, 2, 3), dim_labels=plot_labels,
        title="Training data — delay-conformed, physical (shown) coordinates",
        subtitle=title_model,
        param_str="m_embed=$m_embed, tau_delay=$tau_delay, n_pairs=$n_pairs_total, no subsampling")
else
    fig = fig_training_data(X_train_plot3;
        dim=2, projection_dims=(1, 2), dim_labels=plot_labels,
        title="Training data — delay-conformed, physical (shown) coordinates",
        subtitle=title_model,
        param_str="m_embed=$m_embed, tau_delay=$tau_delay, n_pairs=$n_pairs_total, no subsampling")
end
save(joinpath(save_dir, "02_training_data_delay_conformed.png"), fig)
println("Saved: 02_training_data_delay_conformed.png")
fig = nothing; GC.gc()

# ============================================================================
# TEST INITIAL CONDITIONS + raw test data (seeded sampling, brief rule 2)
# ============================================================================
# Test trajectories start inside the test window with the full delay history
# available.  Sampling uses a dedicated generator seeded from SEED
# (MersenneTwister(SEED ⊻ 0x1)) so it is reproducible independently of the
# global RNG stream.

min_start = burn_in + n_train + 1 + max_lag          # delay history available
max_start = size(X_master, 2) - n_pred               # full horizon available
@assert min_start ≤ max_start "Not enough data for m_embed=$m_embed test predictions"

test_starts = rand(MersenneTwister(SEED ⊻ 0x1), min_start:max_start, n_traj_test)
println("Test start indices in X_master: $test_starts")

# Test window INCLUDING the lag buffer (dim × (n_test + max_lag))
X_test_raw = X_master[:, (burn_in + n_train + 1 - max_lag):(burn_in + n_train + n_test)]
S_test     = build_hankel_multichannel(X_test_raw[showdim, :], m_embed, tau_delay)
@assert size(S_test) == (n_state, n_test)
println("Test delay matrix (shown only): ", size(S_test), "  (", fmt_bytes(sizeof(S_test)), ")")

# True delay embedding of the FIRST prediction window (for the record)
idx1  = test_starts[1]
S_true = build_hankel_multichannel(X_shown_master[:, (idx1 - max_lag):(idx1 + n_pred)],
                                   m_embed, tau_delay)
@assert size(S_true) == (n_state, n_pred + 1)

# First-shown-component ground truth, identical for every method (tiny).
X_true_ref = zeros(Float64, 1, n_pred + 1, n_traj_test)
for j in 1:n_traj_test
    idx = test_starts[j]
    X_true_ref[1, :, j] = X_shown_master[1, idx:(idx + n_pred)]
end

# ============================================================================
# STAGE-1 BACKUP — metadata, raw series, optimizer results
# (the file was reset right after `save_path` was defined)
# ============================================================================
println("\n" * "="^60)
println("Stage-1 save: metadata + raw data...")
println("="^60)

jldopen(save_path, "a+") do f
    # Metadata (data contract)
    f["regime_system"]   = regime_system
    f["regime_name"]     = regime_name
    f["experiment_name"] = experiment_name
    f["dim"]             = dim
    f["showdim"]         = showdim
    f["n_show"]          = n_show
    f["n_hidden"]        = n_hidden
    f["n_state"]         = n_state
    f["n_plot"]          = n_plot
    f["phys_idx"]        = phys_idx
    f["P_plot"]          = P_plot
    f["plot_labels"]     = plot_labels
    f["dt_internal"]     = dt_internal
    f["nLag"]            = nLag
    f["dt_eff"]          = dt_eff
    f["m_embed"]         = m_embed
    f["tau_delay"]       = tau_delay
    f["n_train"]         = n_train
    f["n_test"]          = n_test
    f["n_pred"]          = n_pred
    f["n_traj_test"]     = n_traj_test
    # Multi-trajectory FITTING-data bookkeeping (the fitting paradigm)
    f["n_traj"]           = n_traj           # training trajectories
    f["m_traj"]           = m_traj           # snapshots per training trajectory
    f["window_ic"]        = window_ic        # IC sampling half-width around x0
    f["n_pairs_per_traj"] = n_pairs_per_traj # delay pairs per trajectory
    f["n_pairs_total"]    = n_pairs_total    # total delay pairs (fitting data)
    f["pd_full/x"]        = pd_full.x        # X-pair column indices into S_all
    f["pd_full/y"]        = pd_full.y        # Y-pair column indices into S_all
    f["pd_full/cols"]     = pd_full.cols     # per-trajectory column ranges
    f["SEED"]            = SEED
    f["test_starts"]     = test_starts
    f["tau_opt"]         = tau_opt
    f["I_vals"]          = I_vals
    # extra (additive) bookkeeping
    f["burn_in"]         = burn_in
    f["m_total"]         = m_total

    # Raw data (data contract)
    f["X_master"]  = X_master
    f["X_train"]   = X_train
    f["X_test_raw"] = X_test_raw
    f["S_test"]    = S_test
    f["S_true"]    = S_true
end
println("Stage-1 saved: $save_path")

# S_test / S_true now live on disk; they are not needed again by this script.
S_test = nothing
S_true = nothing
GC.gc()
print_mem("after stage-1 save (S_test/S_true released)")

# ============================================================================
# METHOD SPECIFICATIONS — the seven fitting methods
# ============================================================================
# Shared thin-plate RBF dictionary (brief default): nRBF=1000 centres chosen
# by k-means, include_states=true, kernel_type=:thinplate, seed=SEED (the
# explicit :seed in dict_params takes precedence over cfg.seed in
# Config.hankel_analysis — both are SEED here).

const RBF_DICT_PARAMS = (nRBF=1000, include_states=true,
                         kernel_type=:thinplate, seed=SEED)

method_ids = ["DMD", "Hankel_DMD", "HAVOK", "Hankel_DMD_r",
              "EDMD", "Hankel_EDMD", "Hankel_EDMD_r"]

method_titles = Dict(
    "DMD"           => "DMD (no embedding)",
    "Hankel_DMD"    => "Hankel-DMD (full rank)",
    "HAVOK"         => "HAVOK (continuous, RK4)",
    "Hankel_DMD_r"  => "Hankel-DMD (r = HAVOK rank)",
    "EDMD"          => "RBF-EDMD (no embedding)",
    "Hankel_EDMD"   => "Hankel RBF-EDMD (full)",
    "Hankel_EDMD_r" => "Hankel RBF-EDMD (r = HAVOK rank)",
)

# Shared config parameters (as the disksaving original's `base`).
base_cfg = (
    dt          = dt_eff,
    edmd_method = :ridge,
    edmd_alpha  = 1e-1,
    proj_alpha  = 1e-6,
    return_ψ    = true,   # NOTE: Ψ is offloaded to disk right after fitting
    verbose     = true,
    seed        = SEED,   # config-level seed → dictionary stochastic steps
)

"""
    build_method_config(mid, r_havok) -> KoopmanConfig

Build the KoopmanConfig of method `mid`.  The `_r` variants need `r_havok`,
the rank HAVOK selected (number of POD modes kept); HAVOK is fitted third,
so `r_havok` is known by the time those configs are built.
"""
function build_method_config(mid::AbstractString, r_havok)
    cfg = if mid == "DMD"
        # No embedding: m_embed=1, tau_delay=1 → the "Hankel" matrix is the
        # raw shown state.  hankel_dmd ignores use_svd (verified in
        # Hankel.jl) and r=nothing keeps the full rank (= n_show).
        KoopmanConfig(; m_embed=1, tau_delay=1, r=nothing, use_svd=false,
                      dict_type=:none, dict_params=NamedTuple(), base_cfg...)
    elseif mid == "Hankel_DMD"
        KoopmanConfig(; m_embed=m_embed, tau_delay=tau_delay, r=nothing,
                      use_svd=true, dict_type=:none, dict_params=NamedTuple(),
                      base_cfg...)
    elseif mid == "HAVOK"
        KoopmanConfig(; m_embed=m_embed, tau_delay=tau_delay, r=nothing,
                      use_svd=true, dict_type=:havok,
                      dict_params=(rank_method=:energy, energy=0.999,
                                   ridge_alpha=0.0),
                      base_cfg...)
    elseif mid == "Hankel_DMD_r"
        isnothing(r_havok) && error("Hankel_DMD_r requires r_havok — fit HAVOK first")
        KoopmanConfig(; m_embed=m_embed, tau_delay=tau_delay, r=r_havok,
                      use_svd=true, dict_type=:none, dict_params=NamedTuple(),
                      base_cfg...)
    elseif mid == "EDMD"
        KoopmanConfig(; m_embed=1, tau_delay=1, r=nothing, use_svd=false,
                      dict_type=:RBF, dict_params=RBF_DICT_PARAMS, base_cfg...)
    elseif mid == "Hankel_EDMD"
        KoopmanConfig(; m_embed=m_embed, tau_delay=tau_delay, r=nothing,
                      use_svd=false, dict_type=:RBF, dict_params=RBF_DICT_PARAMS,
                      base_cfg...)
    elseif mid == "Hankel_EDMD_r"
        isnothing(r_havok) && error("Hankel_EDMD_r requires r_havok — fit HAVOK first")
        KoopmanConfig(; m_embed=m_embed, tau_delay=tau_delay, r=r_havok,
                      use_svd=true, dict_type=:RBF, dict_params=RBF_DICT_PARAMS,
                      base_cfg...)
    else
        error("Unknown method id: $mid")
    end
    validate!(cfg)
    return cfg
end

"""Short human-readable one-liner describing a method config (for panels)."""
function method_subtitle(mid::AbstractString, cfg::KoopmanConfig)
    embed_str = "m=$(cfg.m_embed), τ=$(cfg.tau_delay)"
    pod_str   = if cfg.dict_type == :havok
        "r=auto"
    elseif cfg.dict_type == :none
        isnothing(cfg.r) ? "full rank" : "r=$(cfg.r)"
    else
        cfg.use_svd ? (isnothing(cfg.r) ? "full POD" : "r=$(cfg.r)") : "no POD"
    end
    dict_str = if cfg.dict_type == :none
        "linear"
    elseif cfg.dict_type == :havok
        "rank=$(cfg.dict_params.rank_method)($(cfg.dict_params.energy))"
    else
        "nRBF=$(cfg.dict_params.nRBF), $(cfg.dict_params.kernel_type)"
    end
    return "$embed_str, $pod_str, $dict_str"
end

# Shared method → colour map (used by EVERY cross-method figure).
const METHOD_PALETTE = Makie.wong_colors()
method_colors = Dict(mid => METHOD_PALETTE[mod1(i, length(METHOD_PALETTE))]
                     for (i, mid) in enumerate(method_ids))

# ============================================================================
# SPECTRAL ENTRY (computed while the real AnalysisResult is alive)
# ============================================================================
# Adapted from the disksaving original's compute_spectral_entry, using the
# package machinery: spectrum(res) + select_phase_amplitude_modes.  The print
# block (phase eigenvalue, computed phase, limit-cycle frequency and
# harmonics) follows the original's spectral printout.

"""
    compute_spectral_entry(res, dt_eff, mid) -> NamedTuple

Run `spectrum(res)` and the phase/amplitude mode selection, print the
per-method spectral summary, and return the lightweight `spectral_table`
entry.  Must be called BEFORE the result is offloaded, because `spectrum`
dispatches on the AnalysisResult type.

NOTE on spectrum size: for size(K,1) < 3001 `spectrum` uses a dense
eigendecomposition (all eigenvalues).  Otherwise it uses KrylovKit's
eigsolve, which requires howmany ≤ krylovdim — we pass both explicitly
(howmany=1000, krylovdim=1250) so the large Hankel spectra are the 1000
largest-magnitude eigenvalues (the physically relevant unit-circle region).
"""
function compute_spectral_entry(res, dt_eff, mid)
    nK = size(res.K, 1)
    λ, Ξ = spectrum(res; howmany=min(1000, nK), krylovdim=min(nK, 1250))

    # ---- Koopman mode amplitudes for the first shown observable ----
    # Row 1 of the delay state is the latest sample of shown channel 1, so
    # B_full[1, :] maps eigenfunctions to that observable.
    if !isnothing(res.B_full) && size(res.B_full, 2) == size(Ξ, 1)
        v_modes = vec(abs.(transpose(res.B_full[1, :]) * Ξ))
    else
        # Fallback (e.g. :none methods with r < n_state, HAVOK): the
        # eigenvectors already live in the full delay / POD space.
        v_modes = abs.(Ξ[1, :])
    end

    entry = try
        j_phase, mu_phase, j_amp, mu_amp, branch_idx =
            select_phase_amplitude_modes(λ, dt_eff; TOL_UNITY=1e-6,
                                         TOL_IMAG=1e-5, TOL_HARMONIC=0.05)

        freq_est   = abs(imag(mu_phase)) / (2π)
        period_est = iszero(freq_est) ? Inf : 1.0 / freq_est

        # ---- Print block (model: disksaving original / main.jl) ----
        println("  --- Spectral summary [$mid] ---")
        @printf("  Phase eigenvalue:   λ = %.6f %+.6fim  (j = %d)\n",
                real(λ[j_phase]), imag(λ[j_phase]), j_phase)
        @printf("  Computed phase:     ∠λ = %.6f rad  (%.4f cycles/snapshot)\n",
                angle(λ[j_phase]), angle(λ[j_phase]) / (2π))
        @printf("  μ_phase = log(λ)/dt_eff = %.6f %+.6fim\n",
                real(mu_phase), imag(mu_phase))
        @printf("  Limit cycle:  ω = %.6f rad/s,  T = %.4f s,  f = %.6f Hz\n",
                imag(mu_phase), period_est, freq_est)
        println("  Harmonic branch ($(length(branch_idx)) modes):")
        for (k, j) in enumerate(branch_idx)
            μ_k = log(λ[j]) / dt_eff
            f_k = abs(imag(μ_k)) / (2π)
            @printf("    k=%-3d j=%-5d λ = %.6f %+.6fim   f_k = %.6f Hz  (f_k/f₁ = %.2f)\n",
                    k, j, real(λ[j]), imag(λ[j]), f_k,
                    iszero(freq_est) ? NaN : f_k / freq_est)
        end
        if j_amp > 0
            @printf("  Amplitude mode:     λ = %.6f %+.6fim  (j = %d)\n",
                    real(λ[j_amp]), imag(λ[j_amp]), j_amp)
        end

        (λ = λ, Ξ = Ξ, v_modes = v_modes,
         j_phase = j_phase, j_amp = j_amp, branch = branch_idx,
         n_harmonics = length(branch_idx),
         period = period_est, frequency = freq_est)
    catch e
        @warn "no clear limit-cycle modes found for $mid ($e)"
        (λ = λ, Ξ = Ξ, v_modes = v_modes,
         j_phase = 0, j_amp = 0, branch = Int[],
         n_harmonics = 0, period = NaN, frequency = NaN)
    end
    return entry
end

# ============================================================================
# FITTING METHODS — one at a time, with immediate disk offload
# ============================================================================
# `results` holds lightweight NamedTuples (same field names as AnalysisResult,
# heavy fields = nothing).  The heavy arrays live at fit/<mid>/<field> inside
# `save_path`.  Everything that dispatches on the AnalysisResult type
# (spectrum) happens INSIDE the per-method block, while the full object is
# still alive.

results        = Dict{String, NamedTuple}()   # lightweight, disk-backed
fit_times      = Dict{String, Float64}()
spectral_table = Dict{String, NamedTuple}()
methods        = Dict{String, KoopmanConfig}()
# HAVOK's selected rank, set during the fit loop.  A Ref is used because the
# loop runs inside a jldopen closure: plain assignment would create a
# closure-local variable (shadowing), while mutating a Ref is scope-safe.
const r_havok_ref = Base.RefValue{Int}(0)     # 0 = not set yet

jldopen(save_path, "a+") do f
    @showprogress "Fitting methods..." for (i, mid) in enumerate(method_ids)
        println("\n" * "="^50)
        println("\n[$i/$(length(method_ids))] Fitting $mid — $(method_titles[mid])...")
        cfg = build_method_config(mid, r_havok_ref[] > 0 ? r_havok_ref[] : nothing)
        methods[mid] = cfg
        println("  Config: ", cfg)

        # Select the pair dataset matching this method's embedding
        # (m_embed=1, tau_delay=1 → state-space pairs; otherwise the delay one).
        pd_used = (cfg.m_embed == 1 && cfg.tau_delay == 1) ? pd_small : pd_full

        t0 = time()
        res = pair_analysis(pd_used, cfg)    # peak RAM: S_all + Ψ, library-internal
        t1 = time()
        fit_times[mid] = t1 - t0
        println("$mid fitted in $(round(fit_times[mid], digits=2)) s")
        println("  Koopman matrix size: $(size(res.K))")
        println("  Dictionary type:     $(res.cfg.dict_type)")
        print_mem("after fit $mid")

        # HAVOK's automatic rank choice is only known after fitting; read the
        # number of POD modes actually kept from U_r (n_state × r_havok).
        if mid == "HAVOK"
            r_havok_ref[] = size(res.U_r, 2)
            println("  HAVOK selected rank r_havok = $(r_havok_ref[]) " *
                    "(rank_method=$(cfg.dict_params.rank_method), " *
                    "energy=$(cfg.dict_params.energy))")
        end

        # ---- Spectrum & mode selection (needs the real AnalysisResult) ----
        println("  Spectrum $mid...")
        spectral_table[mid] = compute_spectral_entry(res, dt_eff, mid)

        # ---- Offload heavy fields to disk, keep a light handle, free RAM ----
        results[mid] = offload_result!(f, res, mid)
        # Make sure the light handle carries the spectrum (belt-and-braces:
        # λ/Ξ are not heavy fields, so offload_result! already keeps them).
        results[mid] = merge(results[mid],
                             (λ = spectral_table[mid].λ, Ξ = spectral_table[mid].Ξ))

        res = nothing
        GC.gc()
        print_mem("after offload $mid")
    end
end

# HAVOK must have set the rank used by the _r variants (fit order guarantees it).
@assert r_havok_ref[] > 0 "HAVOK fit did not set r_havok"
havok_rank = r_havok_ref[]
println("\nHAVOK rank used by Hankel_DMD_r / Hankel_EDMD_r: r_havok = $havok_rank")

# ============================================================================
# RBF CENTRE ARTIFACTS + POD GROUP (small, kept for the data contract)
# ============================================================================
# dict_info survives the offload (not a heavy field), and U_r is kept in RAM,
# so the RBF centres can be mapped back to delay coordinates now:
#   • use_svd=true  → centres live in the reduced POD coordinates Vc = Σ Vt;
#                     map back to delay coords with U_r (verified in
#                     Hankel._build_dict_and_project: clustering runs on Vc).
#   • use_svd=false → U_r = I, centres already live in delay/state coords.
# The uniform map is therefore  centers_delay = U_r * centers.

"""
    method_plot_projection(m_e, tau_e) -> Matrix (n_plot × n_show*m_e)

Plotting projection for a method whose delay embedding differs from the main
one (m_embed, tau_delay).  Rows 1..n_show are unit vectors at that embedding's
phys_idx; if n_hidden > 0 the remaining rows are the first POD modes of the
corresponding (small) Hankel matrix, computed via eigen(S*S') as for U_pod.
For the main embedding this returns P_plot itself.
"""
function method_plot_projection(m_e::Int, tau_e::Int)
    n_st = n_show * m_e
    (m_e == m_embed && tau_e == tau_delay) && return P_plot
    Pm = zeros(n_plot, n_st)
    for j in 1:n_show
        Pm[j, (j - 1) * m_e + 1] = 1.0     # same block ordering as phys_idx
    end
    if n_hidden > 0
        # POD rows from the SAME multitrajectory paradigm: the m_embed=1
        # dataset is pd_small; foreign embeddings re-embed the raw pairs
        # (X_train_raw / Y_train_raw are still alive — see the note at the
        # delay-embedding stage).
        Ss = (m_e == 1 && tau_e == 1) ? pd_small.S :
             delay_embed_pairs(X_shown_raw, Y_shown_raw, m_e, tau_e;
                              n_trajectories = n_traj).S
        Es = eigen(Symmetric(Ss * Ss'))                        # n_st × n_st (small)
        ps = sortperm(Es.values; rev=true)
        Pm[n_show+1:n_plot, :] = Es.vectors[:, ps][:, 1:n_hidden]'
    end
    return Pm
end

rbf_method_ids = [mid for mid in method_ids if methods[mid].dict_type == :RBF]
centers_delay = Dict{String, Matrix{Float64}}()   # full delay/state coords
centers_plot3 = Dict{String, Matrix{Float64}}()   # n_plot × nRBF plot coords

for mid in rbf_method_ids
    res_l = results[mid]
    cd = res_l.U_r * res_l.dict_info.centers               # back to delay coords
    Pm = method_plot_projection(res_l.cfg.m_embed, res_l.cfg.tau_delay)
    centers_delay[mid] = cd
    centers_plot3[mid] = Pm * cd
    println("  $mid centres: $(size(cd)) delay coords → $(size(centers_plot3[mid])) plot coords")
end

# ============================================================================
# PREDICTION ENGINE — streamed per trajectory (from the disksaving original,
# adapted to the multichannel Hankel layout)
# ============================================================================
# Instead of building S_pred (n_state × n_pred+1 × n_traj) in RAM and then
# dumping it, each trajectory is written to disk as soon as it is computed,
# reusing the same preallocated buffer.  All buffers (state vectors, ψ,
# k_vec) are allocated ONCE per method.  This local engine OVERRIDES the
# package predict (brief rule 1: memory pattern wins).
#
# Per-method closures preserved from the original: :none (linear DMD
# iteration), :kernel (kept for completeness — this script fits no kernel
# method), :havok (RK4 with true/frozen forcing), and dictionary EDMD
# (lift → K'ψ → project back, with optional POD reconstruction).

"""
    predict_trajectory!(res, X_show, idx, n_pred, S_buf, x_pred, s_curr, s_next)

Fill `S_buf` (n_state_m × n_pred+1) with the predicted delay trajectory
starting at master column `idx`, and `x_pred` (length n_pred+1) with the
predicted FIRST SHOWN component (row 1 of the delay state = latest sample of
channel 1).  `res` may be the lightweight NamedTuple — only K, U_r,
B_reduced, dict_info and cfg are accessed.  `X_show` is the n_show × m_total+1
shown master series.  Uses mul! everywhere: zero per-step allocation.
"""
function predict_trajectory!(res, X_show::AbstractMatrix, idx::Int, n_pred::Int,
                             S_buf::AbstractMatrix{Float64}, x_pred::AbstractVector{Float64},
                             s_curr::Vector{Float64}, s_next::Vector{Float64})
    m_e = res.cfg.m_embed
    τ   = res.cfg.tau_delay
    n_sh = size(X_show, 1)

    # Initial delay vector — same channel-major layout as
    # build_hankel_multichannel (latest sample first within each block).
    @inbounds for j in 1:n_sh, d in 0:(m_e - 1)
        s_curr[(j - 1) * m_e + d + 1] = X_show[j, idx - d * τ]
    end
    x_pred[1]     = s_curr[1]
    S_buf[:, 1] .= s_curr

    # ── Linear DMD (dict_type == :none): s_next = K s_curr ──
    # (K acts directly on delay vectors — see API conventions.)
    if res.cfg.dict_type == :none
        @inbounds for k in 1:n_pred
            mul!(s_next, res.K, s_curr)
            x_pred[k+1]     = s_next[1]
            S_buf[:, k+1] .= s_next
            s_curr, s_next  = s_next, s_curr   # swap buffers (local rebinding)
        end
        return nothing
    end

    # ── Kernel EDMD (kept verbatim from the original; unused here) ──
    if !isnothing(res.dict_info) && get(res.dict_info, :type, :none) == :kernel
        X_dict = res.dict_info.X_dict
        σ      = res.dict_info.sigma
        N      = size(X_dict, 2)
        k_vec  = Vector{Float64}(undef, N)
        ψ_next = Vector{Float64}(undef, size(res.K, 1))
        x_r    = isnothing(res.U_r) ? Vector{Float64}(undef, m_e * n_sh) :
                                      Vector{Float64}(undef, size(res.U_r, 2))
        x_next = Vector{Float64}(undef, size(res.B_reduced, 1))

        @inbounds for k in 1:n_pred
            if isnothing(res.U_r)
                copyto!(x_r, s_curr)
            else
                mul!(x_r, res.U_r', s_curr)
            end
            for i in 1:N
                k_vec[i] = rbf_kernel(x_r, view(X_dict, :, i), σ)
            end
            mul!(ψ_next, res.K', k_vec)
            mul!(x_next, res.B_reduced, ψ_next)
            if isnothing(res.U_r)
                copyto!(s_next, x_next)
            else
                mul!(s_next, res.U_r, x_next)
            end
            x_pred[k+1]     = s_next[1]
            S_buf[:, k+1] .= s_next
            s_curr, s_next  = s_next, s_curr
        end
        return nothing
    end

    # ── HAVOK — RK4 integration of dv/dt = A v + B v_input ──
    if !isnothing(res.dict_info) && get(res.dict_info, :type, :none) == :havok
        A   = res.dict_info.A
        Bv  = vec(res.dict_info.B)
        U_r = res.U_r
        r     = size(U_r, 2)
        r_lin = r - 1
        dt_h  = res.cfg.dt
        use_true_forcing = get(res.cfg.dict_params, :use_true_forcing, true)
        ur = U_r[:, r]                    # forcing direction in delay space

        v_full = U_r' * s_curr
        v = collect(v_full[1:r_lin])
        u_curr = v_full[r]

        s_true = similar(s_curr)          # scratch delay vector (true forcing)

        @inbounds for k in 1:n_pred
            if use_true_forcing
                # True forcing at master index idx+k:  u = ur' * s(idx+k)
                for j in 1:n_sh, d in 0:(m_e - 1)
                    s_true[(j - 1) * m_e + d + 1] = X_show[j, idx + k - d * τ]
                end
                u_next = dot(ur, s_true)
            else
                u_next = u_curr           # frozen forcing (autonomous mode)
            end
            u_mid = 0.5 * (u_curr + u_next)

            # RK4 step over one snapshot interval
            k1 = A * v                     .+ Bv .* u_curr
            k2 = A * (v .+ 0.5 * dt_h .* k1) .+ Bv .* u_mid
            k3 = A * (v .+ 0.5 * dt_h .* k2) .+ Bv .* u_mid
            k4 = A * (v .+ dt_h .* k3)       .+ Bv .* u_next
            v .+= (dt_h / 6) .* (k1 .+ 2 .* (k2 .+ k3) .+ k4)

            u_curr = u_next

            # Reconstruct delay state; store observable and full delay vector
            s_next .= U_r * vcat(v, u_curr)
            x_pred[k+1]     = s_next[1]
            S_buf[:, k+1] .= s_next
        end
        return nothing
    end

    # ── Explicit dictionary EDMD (RBF here): lift → K'ψ → project back ──
    # For use_svd=false results U_r = I (never nothing from hankel_edmd), so
    # the isnothing fallbacks are defensive only.
    x_r    = isnothing(res.U_r) ? copy(s_curr) : res.U_r' * s_curr
    ψ      = lift_state(x_r, res.dict_info)
    ψ_next = similar(ψ)
    x_next = Vector{Float64}(undef, size(res.B_reduced, 1))

    @inbounds for k in 1:n_pred
        mul!(ψ_next, res.K', ψ)
        mul!(x_next, res.B_reduced, ψ_next)
        if isnothing(res.U_r)
            copyto!(s_next, x_next)
        else
            mul!(s_next, res.U_r, x_next)
        end
        x_pred[k+1]     = s_next[1]
        S_buf[:, k+1] .= s_next
        s_curr, s_next  = s_next, s_curr
        ψ, ψ_next       = ψ_next, ψ
    end
    return nothing
end

println("\nRunning predictions (streaming one trajectory at a time to disk)...")
pred_results = Dict{String, Tuple{Array{Float64,3}, Array{Float64,3}}}()

jldopen(save_path, "a+") do f
    @showprogress "Predicting & writing delay vectors..." for mid in method_ids
        res     = results[mid]          # lightweight: K, U_r, B_reduced, dict_info only
        n_st    = res.cfg.m_embed * n_show

        S_buf  = zeros(Float64, n_st, n_pred + 1)   # ONE trajectory at a time
        X_pred = zeros(Float64, 1, n_pred + 1, n_traj_test)
        s_curr = zeros(Float64, n_st)
        s_next = zeros(Float64, n_st)

        for j in 1:n_traj_test
            predict_trajectory!(res, X_shown_master, test_starts[j], n_pred,
                                S_buf, view(X_pred, 1, :, j), s_curr, s_next)
            # JLD2 serializes immediately → safe to reuse S_buf next iteration
            f["delay_results/$mid/traj_$j"] = S_buf
        end
        pred_results[mid] = (X_true_ref, X_pred)

        S_buf = nothing
        GC.gc()
        print_mem("after predictions $mid")
    end
end

# ============================================================================
# STAGE-2 BACKUP — configs, lightweight results, spectra, predictions, POD
# ============================================================================
println("\n" * "="^60)
println("Stage-2 save: configs + results + spectra + predictions + POD...")
println("="^60)

jldopen(save_path, "a+") do f
    # Metadata that only exists after the fits
    f["havok_rank"] = havok_rank
    f["method_ids"] = method_ids

    # Methods & fitting (lightweight NamedTuples — heavy fields on disk only)
    f["method_configs"]   = methods
    f["analysis_results"] = results
    f["fit_times"]        = fit_times
    f["spectral_table"]   = spectral_table

    # Predictions (first shown component only — lightweight tuples)
    f["predictions"] = pred_results

    # POD plotting artifacts (data contract)
    f["pod/U_pod"]        = U_pod
    f["pod/σ_pod"]        = σ_pod
    f["pod/X_train_plot3"] = X_train_plot3
    for mid in rbf_method_ids
        f["pod/centers_$mid"]        = centers_delay[mid]   # full delay coords
        f["pod/centers_$(mid)_plot3"] = centers_plot3[mid]  # plot-space coords
    end
end
println("Stage-2 saved: $save_path")
print_mem("after stage-2 save")

# ============================================================================
# FIGURE 1: TRAJECTORY COMPARISON (first shown component only) + RMSE panel
# ============================================================================
# Like the disksaving original's Figure 1, but restricted to the FIRST SHOWN
# component even when all dims are shown (the prediction arrays are already
# (1 × n_pred+1 × n_traj)).

println("\nBuilding Figure 1: Trajectory comparison...")

obs_label = plot_labels[1]

fig1 = Figure(size=(1800, 200 + 500 * (ceil(Int, (length(method_ids) + 1) / 2) + 1)))

row_start = 0
Label(fig1[row_start, :], "Prediction accuracy comparison — $(obs_label)(t) only",
      fontsize=24, font=:bold, tellwidth=false, halign=:center)
row_start += 1
Label(fig1[row_start, :],
      "$regime_system $regime_name — $n_traj_test trajectories, $n_pred-step horizon",
      fontsize=18, tellwidth=false, halign=:center)
row_start += 1

ax_real = Axis(fig1[row_start, 1],
    title="Real trajectories",
    xlabel="Step", ylabel="$(obs_label)(t)",
    titlesize=18, titlefont=:bold)

cols_real = blue_tones(n_traj_test)
cols_pred = orange_tones(n_traj_test)
for j in 1:n_traj_test
    lines!(ax_real, 0:n_pred, X_true_ref[1, :, j],
           color=cols_real[j], linewidth=1.5)
end

# row_start += 1
n_method_rows = ceil(Int, (length(method_ids) + 1) / 2)

for (i, mid) in enumerate(method_ids)
    row = row_start + i ÷ 2
    col = i % 2 + 1

    sub = method_subtitle(mid, methods[mid])
    if methods[mid].dict_type == :havok
        sub = replace(sub, "r=auto" => "r=$havok_rank")
    end
    panel_title = "$(mid): $(method_titles[mid]) | $sub"

    ax = Axis(fig1[row, col],
        title=panel_title,
        xlabel="Step", ylabel="$(obs_label)(t)",
        titlesize=18, titlefont=:bold)

    X_pred = pred_results[mid][2]

    for j in 1:n_traj_test
        lines!(ax, 0:n_pred, X_pred[1, :, j],
               color=cols_pred[j], linewidth=1.2)
    end
    for j in 1:n_traj_test
        lines!(ax, 0:n_pred, X_true_ref[1, :, j],
               color=(cols_real[j], 0.5), linewidth=1.0, linestyle=:dash)
    end
end

err_row = row_start + n_method_rows
ax_err = Axis(fig1[err_row, :],
    title="Average RMSE — $(obs_label)(t)", xlabel="Step", ylabel="RMSE",
    titlesize=18, titlefont=:bold)

for mid in method_ids
    X_pred = pred_results[mid][2]
    rmse = sqrt.(mean((X_true_ref .- X_pred).^2, dims=3))[1, :, 1]
    lines!(ax_err, 0:n_pred, rmse, label=mid, color=method_colors[mid], linewidth=2)
end
axislegend(ax_err, position=:rt, framevisible=true)

save(joinpath(save_dir, "10_prediction_trajectory_comparison.png"), fig1)
println("Saved: 10_prediction_trajectory_comparison.png")
fig1 = nothing; GC.gc()

# ============================================================================
# FIGURE (d): CROSS-METHOD SPECTRUM COMPARISON — one figure, two panels
# ============================================================================
# Left:  discrete Koopman eigenvalues λ with the unit circle.
# Right: continuous-time analogue μ = log(λ)/dt_eff.
# All seven methods overlaid with the shared method → colour map.

println("\nBuilding spectrum comparison figure...")

fig_spec = Figure(size=(1800, 800))
Label(fig_spec[0, :], "Koopman spectrum comparison — all methods",
      fontsize=22, font=:bold, tellwidth=false, halign=:center)

# ── Panel 1: discrete eigenvalues + unit circle ──
ax1 = Axis(fig_spec[1, 1],
    title="Discrete eigenvalues λ",
    xlabel="Re(λ)", ylabel="Im(λ)",
    aspect=DataAspect(), titlesize=16, titlefont=:bold)
θ = range(0.0, 2π, length=200)
lines!(ax1, cos.(θ), sin.(θ), color=(:black, 0.4), linewidth=1, linestyle=:dash)
hlines!(ax1, 0.0, color=(:black, 0.2), linewidth=0.5)
vlines!(ax1, 0.0, color=(:black, 0.2), linewidth=0.5)
for mid in method_ids
    λ = spectral_table[mid].λ
    scatter!(ax1, real.(λ), imag.(λ), color=(method_colors[mid], 0.6),
             markersize=6, label=mid)
end

# ── Panel 2: continuous-time analogue μ = log(λ)/dt_eff ──
ax2 = Axis(fig_spec[1, 2],
    title="Continuous-time analogue  μ = log(λ)/dt_eff",
    xlabel="Re(μ)  (growth rate, 1/s)", ylabel="Im(μ)  (angular frequency, rad/s)",
    titlesize=16, titlefont=:bold)
hlines!(ax2, 0.0, color=(:black, 0.2), linewidth=0.5)
vlines!(ax2, 0.0, color=(:black, 0.4), linewidth=1, linestyle=:dash)
for mid in method_ids
    μ = log.(spectral_table[mid].λ) ./ dt_eff
    scatter!(ax2, real.(μ), imag.(μ), color=(method_colors[mid], 0.6),
             markersize=6, label=mid)
end

Legend(fig_spec[1, 3],
       [MarkerElement(color=method_colors[mid], marker=:circle) for mid in method_ids],
       method_ids, "Method")

save(joinpath(save_dir, "11_spectrum_comparison.png"), fig_spec)
println("Saved: 11_spectrum_comparison.png")
fig_spec = nothing; GC.gc()

# ============================================================================
# FIGURES (e): RBF centres in the plot space (P_plot projection), per method
# ============================================================================
# One figure per RBF method: thin-plate kernel centres projected to the
# n_plot-D plot space via the method's own plotting projection (centres of
# use_svd=true methods were first mapped back to delay coords via U_r).

for mid in rbf_method_ids
    res_l   = results[mid]
    # Background cloud in the SAME plot space as the centres: for the main
    # embedding that is X_train_plot3 (= P_plot * pd_full.S); for the m_embed=1
    # methods it is the method's own projection of pd_small.S (identical to
    # the physical coords when n_hidden = 0).
    Pm = method_plot_projection(res_l.cfg.m_embed, res_l.cfg.tau_delay)
    bg = (res_l.cfg.m_embed == 1 && res_l.cfg.tau_delay == 1) ?
         Pm * pd_small.S : X_train_plot3
    c_plot  = centers_plot3[mid]
    ktype   = res_l.dict_info.kernel_type

    if n_plot >= 3
        fig = fig_clusterized_data(bg, c_plot;
            dim=3, projection_dims=(1, 2, 3), dim_labels=plot_labels,
            title="$mid — $(ktype) RBF centres in plot space",
            subtitle=title_model,
            param_str="nRBF=$(size(c_plot, 2)), $(method_subtitle(mid, res_l.cfg))")
    else
        fig = fig_clusterized_data(bg, c_plot;
            dim=2, projection_dims=(1, 2), dim_labels=plot_labels,
            title="$mid — $(ktype) RBF centres in plot space",
            subtitle=title_model,
            param_str="nRBF=$(size(c_plot, 2)), $(method_subtitle(mid, res_l.cfg))")
    end
    save(joinpath(save_dir, "12_rbf_centers_$mid.png"), fig)
    println("Saved: 12_rbf_centers_$mid.png")
    fig = nothing
end
GC.gc()

# ============================================================================
# STAGE-3 (FINAL) BACKUP — metadata aggregate + timestamp
# ============================================================================
# Individual metadata keys were written in stage 1 (matching the legacy
# top-level layout); the aggregate NamedTuple below mirrors the data
# contract's metadata list in one place for convenience.

metadata = (regime_system=regime_system, regime_name=regime_name,
            experiment_name=experiment_name, dim=dim, showdim=showdim,
            n_show=n_show, n_hidden=n_hidden, n_state=n_state, n_plot=n_plot,
            phys_idx=phys_idx, P_plot=P_plot, plot_labels=plot_labels,
            dt_internal=dt_internal, nLag=nLag, dt_eff=dt_eff,
            m_embed=m_embed, tau_delay=tau_delay, n_train=n_train,
            n_test=n_test, n_pred=n_pred, n_traj_test=n_traj_test,
            n_traj=n_traj, m_traj=m_traj, window_ic=window_ic,
            n_pairs_per_traj=n_pairs_per_traj, n_pairs_total=n_pairs_total,
            SEED=SEED, test_starts=test_starts, tau_opt=tau_opt,
            I_vals=I_vals, havok_rank=havok_rank, method_ids=method_ids)

jldopen(save_path, "a+") do f
    f["metadata"]  = metadata
    f["timestamp"] = string(now())
    # Completion marker (data contract): downstream scripts assert this key
    # before consuming the file, so it must be the LAST key written.
    f["script_complete"] = true
end
println("Stage-3 saved: $save_path  (script_complete=true written)")
print_mem("end of script")

# ============================================================================
# SUMMARY
# ============================================================================
println("\n" * "="^70)
println("FITTING-METHODS BENCHMARK COMPLETE")
println("="^70)
println("Regime:        $regime_system — $regime_name")
println("Dimensions:    dim=$dim, showdim=$showdim, n_show=$n_show, n_hidden=$n_hidden")
println("Delay state:   n_state = $n_show × $m_embed = $n_state  (tau_delay=$tau_delay, tau_opt=$tau_opt)")
println("Fitting data:  $n_traj trajectories × $n_pairs_per_traj delay pairs = $n_pairs_total pairs (X/Y views of S_all)")
println("Plotting:      $n_show phys_idx + $n_hidden POD = $n_plot directions")
println("HAVOK rank:    r_havok = $havok_rank (used by Hankel_DMD_r, Hankel_EDMD_r)")
println("Output dir:    $save_dir")
println("Results file:  $save_path")
println("Heavy fields:  fit/<mid>/{S,ΨX,ΨY,X_r}[_full] + delay_results/<mid>/traj_<j>")
println("")
println("Methods fitted (rank / K size / fit time / limit-cycle frequency):")
for mid in method_ids
    cfg  = methods[mid]
    rstr = cfg.dict_type == :havok ? "r=$havok_rank (auto)" :
           isnothing(cfg.r) ? (cfg.use_svd ? "full" : "no POD") : "r=$(cfg.r)"
    info = spectral_table[mid]
    @printf("  %-14s %-16s K=%-12s fit=%8.2f s   f = %.6f Hz  (%d harmonics)\n",
            mid, rstr, "$(size(results[mid].K,1))²", fit_times[mid],
            info.frequency, info.n_harmonics)
end
println("")
println("Figures: 01_original_training_data, 02_training_data_delay_conformed,")
println("         03_tau_delay_mutual_information, 10_prediction_trajectory_comparison,")
println("         11_spectrum_comparison, 12_rbf_centers_<mid> ($(join(rbf_method_ids, ", ")))")
println("="^70)