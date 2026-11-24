# ============================================================================
# Koopman_spectral_analysis.jl  —  Post-processing of Koopman_fitting_methods.jl
#
# Uses the window-integrated backward-sampling paradigm:
#   random ICs in a physical window, with delay coordinates filled by
#   backward integration of the true dynamics.  NaN trajectories are
#   discarded and resampled.
#
# OUTPUT figures:
#   01 — Starting points (2D scatter + density grid)
#   20 — ZLS of all real eigenfunctions (huge grid)
#   21.. — Progressive joint interceptions (top-2, top-3, ..., top-k)
#   23.. — Phase-eigenfunction-colored progressive joint interceptions
# ============================================================================

using KoopmanAnalysis
using KoopmanAnalysis.Utils: find_zls_gradient_descent, grad_phi, normalize_vector
using KoopmanAnalysis.Dictionaries: Psi_slice
using KoopmanAnalysis.DataGeneration: sample_seeds_window_integrated
using KoopmanAnalysis.Systems
using KoopmanAnalysis.Hankel
using KoopmanAnalysis.Plotting
using KoopmanAnalysis.Spectral: select_phase_amplitude_modes, koopman_eigendecomposition
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

# --- Local path configuration ---
const _LOCAL_PATHS = joinpath(@__DIR__, "LocalPaths.jl")
include(isfile(_LOCAL_PATHS) ? _LOCAL_PATHS : joinpath(@__DIR__, "LocalPaths.jl.example"))
import .LocalPaths: DATA_ROOT, RESULTS_ROOT

const SEED = 42
Random.seed!(SEED)
BLAS.set_num_threads(1)
println("Running with $(nthreads()) Julia threads.")

const TOL_IMAG_REAL = 1e-4
const TOL_UNITY = 1e-4
const TOL_ZERO = 1e-3

# ============================================================================
# UTILITIES
# ============================================================================

fmt_bytes(b::Integer) = b >= 1<<30 ? @sprintf("%.2f GiB", b/2^30) :
                        b >= 1<<20 ? @sprintf("%.2f MiB", b/2^20) :
                                     @sprintf("%.2f KiB", b/2^10)

print_mem(tag::AbstractString) =
    @printf("%-55s free RAM: %.2f GiB\n", tag, Sys.free_memory()/2^30)

function jld_write!(f, key::AbstractString, value; overwrite::Bool=false)
    if haskey(f, key)
        overwrite || return false
        delete!(f, key)
    end
    f[key] = value
    return true
end

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

experiment_name = "HiddenDim-LargeExp-multitraj-delays"
regime_system   = "FHN"
regime_name     = length(ARGS) > 0 ? ARGS[1] : "stable-limit-cycle"

spectral_method = length(ARGS) > 1 ? ARGS[2] : "Hankel_EDMD"

showdim = [1]
n_show  = length(showdim)

save_dir = default_save_dir(regime_system, regime_name;
                            base_dir=RESULTS_ROOT, subdir=experiment_name)
mkpath(save_dir)

data_save_dir = default_save_dir(regime_system, regime_name;
                                 base_dir=DATA_ROOT, subdir=experiment_name)
mkpath(data_save_dir)

println("Output directory (figures): ", save_dir)
println("Data directory (JLD2):      ", data_save_dir)
println("spectral_method = $spectral_method")

# --- Locate Script 1 dataset ------------------------------------------------
showdim_tag = join(string.(showdim))
fit_filename = "prediction_accuracy_results-$regime_system-$regime_name.jld2"
save_path_fit = joinpath(data_save_dir, fit_filename)

if !isfile(save_path_fit)
    hits = String[]
    isdir(DATA_ROOT) && for (root, _, files) in walkdir(DATA_ROOT)
        fit_filename in files && push!(hits, joinpath(root, fit_filename))
    end
    isempty(hits) && error("Script 1 output not found: $save_path_fit")
    save_path_fit = sort!(hits)[1]
    data_save_dir = dirname(save_path_fit)
    @warn "Using Script 1 file found at $save_path_fit"
end
println("Script 1 dataset: $save_path_fit")

save_path_out = joinpath(data_save_dir,
    "spectral_analysis-$regime_system-$regime_name-show$showdim_tag-$spectral_method.jld2")
println("Output dataset:   $save_path_out")
print_mem("start")

# ============================================================================
# 1. LOAD SCRIPT 1 DATA
# ============================================================================

println("\nLoading Script 1 metadata and '$spectral_method' operator...")

s1 = jldopen(save_path_fit, "r") do f
    Dict(
        :dt_internal     => f["metadata/dt_internal"],
        :dt_eff          => f["metadata/dt_eff"],
        :nLag            => f["metadata/nLag"],
        :m_embed         => f["metadata/m_embed"],
        :tau_delay       => f["metadata/tau_delay"],
        :n_train         => f["metadata/n_train"],
        :n_traj          => f["metadata/n_traj"],
        :method_list     => f["metadata/method_list"],
        :avg_rmse        => f["metadata/avg_rmse"],
    )
end

dt_internal = s1[:dt_internal]
dt_eff      = s1[:dt_eff]
nLag        = s1[:nLag]
m_embed     = s1[:m_embed]
tau_delay   = s1[:tau_delay]
n_train     = s1[:n_train]
n_traj      = s1[:n_traj]

# Load the selected method's result
res_data = jldopen(save_path_fit, "r") do f
    Dict(
        :K         => f["results/$spectral_method/res"].K,
        :B_reduced => f["results/$spectral_method/res"].B_reduced,
        :U_r       => f["results/$spectral_method/res"].U_r,
        :dict_info => f["results/$spectral_method/res"].dict_info,
        :cfg       => f["results/$spectral_method/res"].cfg,
    )
end

K         = res_data[:K]
B_reduced = res_data[:B_reduced]
U_r       = res_data[:U_r]
dict_info = res_data[:dict_info]
cfg_fit   = res_data[:cfg]

# Compute spectrum
nK = size(K, 1)
λ, Ξ = koopman_eigendecomposition(K)

println("  Fit config: dict_type=$(cfg_fit.dict_type), r=$(cfg_fit.r)")
println("  Operator: K $(size(K)), $(length(λ)) eigenvalues")

# Plotting machinery (reconstruct from fit config)
n_state = cfg_fit.m_embed * n_show
phys_idx = [(j - 1) * cfg_fit.m_embed + 1 for j in 1:n_show]
P_plot = Matrix{Float64}(I, n_state, n_state)  # identity for now
plot_labels = ["v(t)", "w(t)"]

plot_coords(S::AbstractMatrix) = P_plot * S
plot_coords(s::AbstractVector) = P_plot * s

print_mem("after loading Script 1 data")

# ============================================================================
# DICTIONARY CHECKS
# ============================================================================

dict_info.type == :rbf || error(
    "spectral_method='$spectral_method' requires thin-plate RBF dictionary.")

n_dict = size(dict_info.centers, 1)
println("  Dictionary space dimension: n_dict = $n_dict")

Psi_method(X::AbstractMatrix) = Psi_slice(X, dict_info)

to_dict_space(S::AbstractMatrix) =
    size(S, 1) == n_dict ? Matrix(S) :
    (isnothing(U_r) ? error("Cannot map: U_r is nothing") : Matrix(U_r)' * S)

to_delay_space(X::AbstractMatrix) =
    size(X, 1) == n_state ? Matrix(X) :
    (isnothing(U_r) ? error("Cannot map: U_r is nothing") : Matrix(U_r) * X)

# ============================================================================
# RECONSTRUCT SYSTEM PARAMETERS
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
# PHASE/AMPLITUDE MODE SELECTION
# ============================================================================

j_phase = 0
mu_phase = NaN + 0.0im
j_amp = 0
mu_amp = NaN + 0.0im
branch_idx = Int[]

try
    global j_phase, mu_phase, j_amp, mu_amp, branch_idx =
        select_phase_amplitude_modes(λ, dt_eff;
            TOL_UNITY=TOL_UNITY, TOL_IMAG=TOL_IMAG_REAL, TOL_HARMONIC=0.05, tol_growth=1.0)
    println("Phase mode:      j=$j_phase, λ=$(round(λ[j_phase], digits=6))")
    println("Amplitude mode:  j=$j_amp, λ=$(round(λ[j_amp], digits=6))")
catch e
    @warn "select_phase_amplitude_modes failed: $e"
end

# ============================================================================
# 2. WINDOW-INTEGRATED SEED SAMPLING AND GRADIENT DESCENT SETUP
# ============================================================================

window_ic = 2.0
x0_center = [0.0,0.5]        # x0_silen by default
fill_direction = :backward
n_starts_modes = 1000
n_starts_joint = 1000

println("\n" * "="^60)
println("Sampling seeds (window-integrated, $fill_direction)...")
println("="^60)

S0_modes = sample_seeds_window_integrated(
    f_det, x0_center, window_ic, length(x0_center), showdim,
    dt_internal, nLag, m_embed, tau_delay;
    n_starts=n_starts_modes, direction=fill_direction,
    rng=MersenneTwister(SEED), max_consecutive_failures=10)

S0_joint = sample_seeds_window_integrated(
    f_det, x0_center, window_ic, length(x0_center), showdim,
    dt_internal, nLag, m_embed, tau_delay;
    n_starts=n_starts_joint, direction=fill_direction,
    rng=MersenneTwister(SEED), max_consecutive_failures=10)

println("  S0_modes: $(size(S0_modes))")
println("  S0_joint: $(size(S0_joint))")

# Map to dictionary space for GD
S0_modes_dict = to_dict_space(S0_modes)
S0_joint_dict = to_dict_space(S0_joint)

print_mem("after seed sampling")

n_joint_modes_max = 7

# Gradient-descent settings for all ZLS runs (recorded in the output JLD2)
const GD_N_ITER    = 10000
const GD_LR        = 0.1
const GD_TOL_MODES = 1e-20
const GD_TOL_JOINT = 1e-20

# ============================================================================
# 3. ZLS FOR ALL REAL EIGENVALUES
# ============================================================================

real_eig_indices = findall(x -> abs(imag(x)) < TOL_IMAG_REAL &&
                                abs(real(x) - 1.0) > TOL_UNITY &&
                                abs(real(x)) < 1.0 &&
                                abs(real(x)) > TOL_ZERO, λ)

println("\nFound $(length(real_eig_indices)) real eigenvalues inside the unit circle:")
for j in real_eig_indices
    println("  j=$j, λ=$(round(real(λ[j]), digits=6))")
end

# Initialize output JLD2
if !isfile(save_path_out)
    jldopen(save_path_out, "w") do f
        f["regime_system"]   = regime_system
        f["regime_name"]     = regime_name
        f["spectral_method"] = spectral_method
        f["SEED"]            = SEED
        f["GD_N_ITER"]       = GD_N_ITER
        f["GD_LR"]           = GD_LR
        f["GD_TOL_MODES"]    = GD_TOL_MODES
        f["GD_TOL_JOINT"]    = GD_TOL_JOINT
        f["S0_modes"]        = S0_modes
        f["S0_joint"]        = S0_joint
        f["lambda"]          = λ
        f["Xi"]              = Ξ
        f["dict_info"]       = dict_info
        f["script_complete"] = false
    end
end

all_zls_results = Dict{Int, Matrix{Float64}}()

println("\nComputing ZLS for all real eigenvalues...")
p_zls = Progress(length(real_eig_indices); desc="ZLS per eigenvalue", dt=1.0)

for j in real_eig_indices
    key = "zls/j_$j"
    resumed = jldopen(save_path_out, "r") do f
        haskey(f, key) ? read(f, key) : nothing
    end

    if !isnothing(resumed)
        all_zls_results[j] = resumed
        next!(p_zls; showvalues=[("j", j), ("status", "loaded")])
        continue
    end

    zls_pts_dict = find_zls_gradient_descent(
        S0_modes_dict, Psi_method, Ξ, j, dict_info;
        starts = S0_modes_dict,   # use our integrated seeds
        n_iter = GD_N_ITER,
        lr     = GD_LR,
        tol    = GD_TOL_MODES,
        verbose = false,
        seed   = SEED,
    )

    zls_matrix = isempty(zls_pts_dict) ?
                 Matrix{Float64}(undef, n_state, 0) :
                 to_delay_space(reduce(hcat, zls_pts_dict))
    all_zls_results[j] = zls_matrix

    jldopen(save_path_out, "a+") do f
        f[key] = zls_matrix
    end

    next!(p_zls; showvalues=[("j", j), ("pts", size(zls_matrix, 2))])
end
finish!(p_zls)
GC.gc()
print_mem("after ZLS extraction")

# ============================================================================
# 4. CLEAN LIMIT CYCLE
# ============================================================================

println("\n--- Computing clean limit cycle ---")

if j_phase > 0
    ω_lc = abs(imag(log(λ[j_phase])) / dt_eff)
    T_lc = 2π / ω_lc
    println("  Period T = $(round(T_lc, digits=4)) s")
else
    T_lc = 20.0
    ω_lc = 2π / T_lc
    println("  Fallback period T = $T_lc s")
end

n_burn_steps  = ceil(Int, 10 * T_lc / dt_eff)
n_lc_steps    = ceil(Int, 2 * T_lc / dt_eff)
max_lag_local = (m_embed - 1) * tau_delay

x0_lc = x0_silent .+ ones(length(x0_silent))
X_lc_full = rk4(f_det, x0_lc, dt_internal,
                n_burn_steps + max_lag_local + n_lc_steps; nLag=nLag)
X_lc = X_lc_full[:, n_burn_steps+1:end]
X_lc_full = nothing

S_lc_shown = build_hankel_multichannel(X_lc[showdim, :], m_embed, tau_delay)
lc_plot = plot_coords(S_lc_shown)
lc_len  = size(lc_plot, 2)
lc_x = lc_plot[1, :]
lc_y = n_show >= 2 ? lc_plot[2, :] : zeros(lc_len)
lc_z = n_show >= 3 ? lc_plot[3, :] : zeros(lc_len)

println("  Delay-embedded LC: $lc_len points")

jldopen(save_path_out, "a+") do f
    jld_write!(f, "S_lc_shown", S_lc_shown; overwrite=true)
    jld_write!(f, "T_lc",       T_lc;       overwrite=true)
    jld_write!(f, "omega_lc",   ω_lc;       overwrite=true)
end

GC.gc()
print_mem("after limit cycle")

# ============================================================================
# 5. FIGURE: STARTING POINTS
# ============================================================================

println("\nGenerating starting-points figure...")

P_modes = plot_coords(S0_modes)
P_joint = plot_coords(S0_joint)

fig = Figure(size=(1300, 560))
for (k, (P, name)) in enumerate([(P_modes, "S0_modes $(size(S0_modes, 2))"),
                                 (P_joint, "S0_joint $(size(S0_joint, 2))")])
    ax = Axis(fig[1, k]; title=name, xlabel="v", ylabel="w", aspect=DataAspect())
    n_draw = min(size(P, 2), 40000)
    idx = round.(Int, range(1, size(P, 2), length=n_draw))
    scatter!(ax, P[1, idx], P[2, idx]; color=(:steelblue, 0.25), markersize=3)
    lines!(ax, lc_x, lc_y; color=:red, linewidth=1.5)
end
save(joinpath(save_dir, "01_ZLS_window_integrated_seeds.png"), fig)
println("Saved: 01_ZLS_window_integrated_seeds.png")
fig = nothing; GC.gc()

# ============================================================================
# 6. HUGE FIGURE: ZLS OF ALL REAL EIGENVALUES
# ============================================================================

println("\nGenerating huge figure (all real eigenfunctions)...")

valid_js = sort!(filter(j -> !isempty(all_zls_results[j]) &&
                             size(all_zls_results[j], 2) > 0,
                        collect(keys(all_zls_results))))
n_plots = length(valid_js)

if n_plots == 0
    println("No ZLS points found!")
else
    n_cols = ceil(Int, sqrt(n_plots))
    n_rows = ceil(Int, n_plots / n_cols)

    fig = Figure(size=(n_cols * 450, n_rows * 450))
    Label(fig[0, 1:n_cols],
          "Zero Level Sets — $spectral_method (window-integrated seeds)",
          fontsize=24, font=:bold)

    for (idx, j) in enumerate(valid_js)
        row = ceil(Int, idx / n_cols)
        col = ((idx - 1) % n_cols) + 1
        n_pts_j = size(all_zls_results[j], 2)
        title_j = "j=$j, λ=$(round(real(λ[j]), digits=4))\n$n_pts_j pts"

        pts_plot = plot_coords(all_zls_results[j])
        n_draw = min(size(pts_plot, 2), 20000)
        idx_draw = round.(Int, range(1, size(pts_plot, 2), length=n_draw))
        pts_draw = pts_plot[:, idx_draw]

        if n_show >= 3
            ax = Axis3(fig[row, col]; title=title_j, titlegap=5,
                xlabel="v", ylabel="w", zlabel="z",
                xlabelsize=10, ylabelsize=10, zlabelsize=10, titlesize=18)
            scatter!(ax, pts_draw[1, :], pts_draw[2, :], pts_draw[3, :];
                     color=(:orange, 0.6), markersize=4)
            lines!(ax, lc_x, lc_y, lc_z; color=(COLOR_PRIMARY, 1.0), linewidth=1.5)
        else
            ax = Axis(fig[row, col]; title=title_j, titlegap=5,
                xlabel="v", ylabel="w",
                xlabelsize=10, ylabelsize=10, titlesize=18)
            scatter!(ax, pts_draw[1, :], pts_draw[2, :];
                     color=COLOR_ACCENT, strokecolor=:black, strokewidth=0.5,
                     markersize=6)
            lines!(ax, lc_x, lc_y; color=(:black, 1.0), linestyle=:dash, linewidth=1.5)
        end
    end

    save(joinpath(save_dir, "20_ZLS_all_real_eigenfunctions.png"), fig)
    println("Saved: 20_ZLS_all_real_eigenfunctions.png")
    fig = nothing; GC.gc()
end

# ============================================================================
# 7. PROGRESSIVE JOINT INTERCEPTIONS (top-k real eigenvalues)
# ============================================================================
# Sort real eigenvalues by |λ| (largest first) and run joint GD for
# top-2, top-3, ..., top-k modes.

println("\n" * "="^60)
println("Progressive joint interceptions (top-2 to top-$n_joint_modes_max)")
println("="^60)

# Sort real eigenvalues by magnitude (descending)
real_js_sorted = sort(real_eig_indices, by=j -> abs(λ[j]), rev=true)

if length(real_js_sorted) < 2
    @warn "Fewer than 2 real eigenvalues — skipping progressive interceptions."
else
    n_joint_max_actual = min(n_joint_modes_max, length(real_js_sorted))

    for k in 1:n_joint_max_actual
        modes_k = real_js_sorted[1:k]
        println("\n--- Joint ZLS for top-$k modes: $modes_k ---")

        key_joint = "joint_zls/top_$(k)"
        resumed = jldopen(save_path_out, "r") do f
            haskey(f, key_joint) ? read(f, key_joint) : nothing
        end

        if !isnothing(resumed)
            println("  Loaded from disk ($(size(resumed, 2)) pts)")
            joint_zls = resumed
        else
            joint_pts_dict = find_zls_gradient_descent(
                S0_joint_dict, Psi_method, Ξ, modes_k, dict_info;
                starts   = S0_joint_dict,
                n_iter   = GD_N_ITER,
                lr       = GD_LR,
                tol      = GD_TOL_JOINT,
                verbose  = false,
                seed     = SEED,
            )

            joint_zls = isempty(joint_pts_dict) ?
                        Matrix{Float64}(undef, n_state, 0) :
                        to_delay_space(reduce(hcat, joint_pts_dict))

            jldopen(save_path_out, "a+") do f
                f[key_joint] = joint_zls
            end
            println("  Computed: $(size(joint_zls, 2)) pts")
        end

        # ---- Figure: progressive interception ---------------------------
        if size(joint_zls, 2) > 0
            pts_plot = plot_coords(joint_zls)
            n_draw = min(size(pts_plot, 2), 20000)
            idx_draw = round.(Int, range(1, size(pts_plot, 2), length=n_draw))
            pts_draw = pts_plot[:, idx_draw]

            fig = Figure(size=(800, 600))
            modes_str = join(["j=$j (λ=$(round(real(λ[j]), digits=4)))" for j in modes_k], ",\n")

            if n_show >= 3
                ax = Axis3(fig[1, 1];
                    title="Progressive interception (top-$k real modes)\n$modes_str\n$(size(joint_zls, 2)) pts",
                    xlabel="v", ylabel="w", zlabel="z")
                scatter!(ax, pts_draw[1, :], pts_draw[2, :], pts_draw[3, :];
                         color=:orange, markersize=6, label="Intersection")
                lines!(ax, lc_x, lc_y, lc_z; color=COLOR_PRIMARY, linewidth=2,
                       label="Limit cycle")
                axislegend(ax)
            else
                ax = Axis(fig[1, 1];
                    title="Progressive interception (top-$k real modes)\n$modes_str\n$(size(joint_zls, 2)) pts",
                    xlabel="v", ylabel="w")
                scatter!(ax, pts_draw[1, :], pts_draw[2, :];
                         color=:orange, markersize=6, label="Intersection")
                lines!(ax, lc_x, lc_y; color=COLOR_PRIMARY, linewidth=2,
                       label="Limit cycle")
                axislegend(ax)
            end

            save(joinpath(save_dir, "21_joint_interception_top$(k).png"), fig)
            println("  Saved: 21_joint_interception_top$(k).png")

            # ---- Figure 23: phase-eigenfunction-colored progressive interception ----
            if j_phase > 0
                S_joint_dict = to_dict_space(joint_zls)
                Psi_joint    = Psi_slice(S_joint_dict, dict_info)
                φ_vals       = vec(Ξ[:, j_phase]' * Psi_joint)
                θ_vals       = angle.(φ_vals)

                fig_phase = Figure(size=(950, 600))
                modes_str = join(["j=$j (λ=$(round(real(λ[j]), digits=4)))" for j in modes_k], ",\n")

                if n_show >= 3
                    ax_phase = Axis3(fig_phase[1, 1];
                        title="Progressive interception (top-$k) — colored by phase eigenfunction\n$modes_str\n$(size(joint_zls, 2)) pts",
                        xlabel="v", ylabel="w", zlabel="z")
                    hm = scatter!(ax_phase, pts_draw[1, :], pts_draw[2, :], pts_draw[3, :];
                             color=θ_vals[idx_draw], colormap=:thermal, markersize=6,
                             colorrange=(-π, π))
                    lines!(ax_phase, lc_x, lc_y, lc_z; color=(:black, 0.5), linewidth=1.5)
                else
                    ax_phase = Axis(fig_phase[1, 1];
                        title="Progressive interception (top-$k) — colored by phase eigenfunction\n$modes_str\n$(size(joint_zls, 2)) pts",
                        xlabel="v", ylabel="w")
                    hm = scatter!(ax_phase, pts_draw[1, :], pts_draw[2, :];
                             color=θ_vals[idx_draw], colormap=:thermal, markersize=6,
                             colorrange=(-π, π))
                    lines!(ax_phase, lc_x, lc_y; color=(:black, 0.5), linewidth=1.5)
                end
                Colorbar(fig_phase[1, 2], hm; label="arg(φ_phase) [rad]", width=14)

                save(joinpath(save_dir, "23_joint_interception_top$(k)_phase.png"), fig_phase)
                println("  Saved: 23_joint_interception_top$(k)_phase.png")
                fig_phase = nothing; GC.gc()
            end

            fig = nothing; pts_plot = nothing; pts_draw = nothing; GC.gc()
        end
    end
end

# ============================================================================
# 8. SPECTRUM COMPARISON FIGURE
# ============================================================================

println("\nBuilding spectrum comparison figure...")

fig_spec = Figure(size=(1200, 500))
Label(fig_spec[0, :], "Koopman spectrum — $spectral_method",
      fontsize=24, font=:bold)

ax1 = Axis(fig_spec[1, 1], title="Discrete eigenvalues",
    xlabel="Re(λ)", ylabel="Im(λ)", aspect=DataAspect())
θ = range(0.0, 2π, length=200)
lines!(ax1, cos.(θ), sin.(θ), color=(:black, 0.3), linestyle=:dash)
scatter!(ax1, real.(λ), imag.(λ); color=(:steelblue, 0.5), markersize=4)
# Highlight real eigenvalues
scatter!(ax1, real.(λ[real_eig_indices]), imag.(λ[real_eig_indices]);
         color=:red, markersize=6, marker=:x, label="real")
axislegend(ax1)

ax2 = Axis(fig_spec[1, 2], title="Continuous analogue μ = log(λ)/dt_eff",
    xlabel="Re(μ)", ylabel="Im(μ)")
μ_all = log.(λ) ./ dt_eff
scatter!(ax2, real.(μ_all), imag.(μ_all); color=(:steelblue, 0.5), markersize=4)
scatter!(ax2, real.(μ_all[real_eig_indices]), imag.(μ_all[real_eig_indices]);
         color=:red, markersize=6, marker=:x, label="real")
axislegend(ax2)

save(joinpath(save_dir, "22_spectrum.png"), fig_spec)
println("Saved: 22_spectrum.png")
fig_spec = nothing; GC.gc()

# ============================================================================
# FINALIZE
# ============================================================================

jldopen(save_path_out, "a+") do f
    jld_write!(f, "script_complete", true; overwrite=true)
end

println("\n" * "="^60)
println("SPECTRAL ANALYSIS COMPLETE")
println("="^60)
println("Output JLD2: $save_path_out")
println("Figures:     $save_dir")
print_mem("done")