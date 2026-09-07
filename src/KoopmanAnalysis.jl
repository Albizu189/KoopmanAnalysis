"""
    KoopmanAnalysis

A reusable Julia toolbox for learning Koopman operators from low-dimensional
dynamical systems using Extended Dynamic Mode Decomposition (EDMD), Hankel-EDMD,
and kernel EDMD.

The module is organised into sub-modules:

- `Systems`: vector fields (FHN, Duffing, 3D Epileptor, Lorenz, third-order
  Van der Pol) and RK4 integration.
- `Dictionaries`: Hermite, RBF and Random Fourier Feature dictionaries.
- `EDMD`: EDMD operator estimation and projection operators.
- `Hankel`: delay-embedding matrices and Hankel-EDMD.
- `DataGeneration`: ready-made training/test data builders.
- `Spectral`: Koopman spectrum, eigenfunctions, phase/amplitude analysis.
- `Plotting`: shared colours and plotting helpers.
- `Figures`: ready-made figure routines for the EDMD pipeline.
- `Regimes`: regime metadata and output directories.
- `Utils`: numerical helpers.
- `Config`: methods organizer for the Koopman pipeline

# Example

```julia
using KoopmanAnalysis

cfg = regime_config("FHN", "stable-limit-cycle")
rhs = fhn_rhs(cfg.params)
X = rk4(rhs, [0.1, 0.0], 0.01, 1000)

# Split into current/next state matrices
Xc = X[:, 1:end-1]
Y = X[:, 2:end]

# Hermite EDMD
K, ΨX, ΨY = compute_koopman_operator(Xc, Y, X -> Psi_Hermite(X, 4))
λ, Ξ = koopman_eigendecomposition(K)
```
"""
module KoopmanAnalysis

# Standard library packages used across the module.
using LinearAlgebra
using Statistics
using Random
using Printf

# External dependencies declared in Project.toml.
using CairoMakie
using ColorSchemes
using Clustering
using DynamicPolynomials
using MultivariateBases: maxdegree_basis, ProbabilistsHermite, PhysicistsHermite
using LaTeXStrings
using Roots
using ProgressMeter

include("Utils.jl")
include("Systems.jl")
include("Dictionaries.jl")
include("EDMD.jl")
include("Hankel.jl")
include("Regimes.jl")
include("DataGeneration.jl")
include("Spectral.jl")
include("Plotting.jl")
include("Figures.jl")
include("Config.jl")

# Re-export a convenient subset for interactive/notebook use.
using .Utils
using .Systems
using .Dictionaries
using .EDMD
using .Hankel
using .DataGeneration
using .Spectral
using .Plotting
using .Figures
using .Regimes
using .Config

export
    # Utils
    normalize_vector, signed_area, meshgrid_2d, classify_fixed_point_2d,
    finite_difference_jacobian, mutual_information, find_first_minimum,
    false_nearest_neighbors,
    grad_Psi_RBF, grad_phi, find_zls_gradient_descent,
    # Systems
    euler_maruyama, rk4, generate_trajectories, detect_limit_cycle,
    fhn_rhs, fhn_drift, fhn_diffusion, fhn_parameters,
    duffing_rhs, duffing_drift, duffing_diffusion, duffing_parameters,
    epileptor3d_rhs, epileptor3d_drift, epileptor3d_diffusion,
    epileptor3d_parameters, path_parameters, compute_xs,
    lorenz_rhs, lorenz_drift, lorenz_diffusion, lorenz_parameters,
    vanderpol_rhs, vanderpol_drift, vanderpol_diffusion, vanderpol_parameters,
    rossler_rhs, rossler_drift, rossler_diffusion, rossler_parameters,
    find_fixed_points_fhn, find_fixed_points_duffing, find_fixed_points_lorenz,
    find_fixed_points_vanderpol, find_fixed_points_rossler,
    fixed_point, generate_test_trajectories,
    # Dictionaries
    get_dim_psi, hermite_basis, Psi_Hermite,
    cluster_data, Psi_RBF, construct_projection_operator_hermite,
    RFFBasis, build_rff_basis, Psi_RFF, lift_state,
    # EDMD
    compute_koopman_operator, construct_projection_operator,
    rbf_kernel, median_heuristic_sigma, kernel_feature_vector, kernel_edmd_rbf,
    edmd_predict_from_psi,
    # Hankel
    build_hankel, hankel_dmd, hankel_edmd, delay_space_edmd_prediction,
    havok_dmd, havok_predict,
    # DataGeneration
    edmd_training_data, hankel_training_data, state_space_predictions,
    delay_space_predictions, state_space_predict_from_psi,
    # Spectral
    koopman_eigendecomposition, detect_limit_cycle_modes,
    build_harmonic_branch, find_all_harmonic_branches, select_phase_amplitude_modes,
    evaluate_eigenfunction_grid, evaluate_eigenfunction_slice,
    # Plotting
    COLOR_PRIMARY, COLOR_ACCENT, COLOR_SECONDARY, COLOR_BACKGROUND,
    blue_tones, orange_tones, set_koopman_theme!,
    # Figures
    fig_phase_portrait, fig_phase_portrait_3d, fig_phase_portrait_all_views,
    fig_training_data, fig_clusterized_data,
    fig_prediction, fig_eigenvalues, fig_eigenfunctions, fig_phase_amplitude,
    fig_time_series,
    # Regimes
    RegimeConfig, regime_config, default_save_dir, list_regimes,
    # Config
    KoopmanConfig, AnalysisResult, hankel_analysis, state_analysis,
    predict, spectrum, embed, all_harmonic_branches

function __init__()
    # Apply the shared figure theme when the module is loaded.
    set_koopman_theme!()
end

end # module
