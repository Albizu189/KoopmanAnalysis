# KoopmanAnalysis — Julia Package for Koopman Operator Analysis

A reusable Julia toolbox for learning Koopman operators from low-dimensional
dynamical systems using Extended Dynamic Mode Decomposition (EDMD), Hankel-EDMD,
kernel EDMD, and HAVOK (Hankel Alternative View of Koopman).

The package provides vector-field definitions for classic nonlinear systems,
parallelized dictionary constructions (Hermite / RBF / Random Fourier Features),
edmd operator estimation, spectral analysis, delay-embedding pipelines, and
ready-made figure routines.

---

## Repository structure

```
KoopmanAnalysis
├── docs
│   ├── README.md
│   └── deps.txt
├── examples/          # Runnable example scripts
├── scripts/           # User analysis scripts
└── src
    ├── KoopmanAnalysis.jl    # Umbrella module
    ├── Systems.jl            # Vector fields + RK4 + Euler–Maruyama + fixed points
    ├── Dictionaries.jl       # Hermite / RBF / RFF dictionaries
    ├── EDMD.jl               # EDMD + kernel EDMD + prediction
    ├── Hankel.jl             # Delay-embedding / Hankel-EDMD / HAVOK
    ├── DataGeneration.jl     # Ready-made training/test data builders
    ├── Spectral.jl           # Spectrum, eigenfunctions, phase/amplitude
    ├── Plotting.jl           # Shared colours and plotting helpers
    ├── Figures.jl            # Reusable EDMD figure routines
    ├── Regimes.jl            # Regime metadata and output directories
    ├── Utils.jl              # Numerical helpers + zero-level-set solvers
    └── Config.jl             # High-level KoopmanConfig + AnalysisResult API
```

---

## Installation

Clone the repository and activate the Julia environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()   # installs all dependencies
```

Or from a terminal:

```bash
cd KoopmanAnalysis
julia --project=.
```

Then load the package:

```julia
using KoopmanAnalysis
```

**Julia version:** 1.11+  
**Dependencies** are listed in `docs/deps.txt` and pinned in `Manifest.toml`.

---

## Supported dynamical systems

| System | Regimes |
|--------|---------|
| **FHN** (FitzHugh–Nagumo) | `"stable-limit-cycle"`, `"stable-node"`, `"three-equilibrium-regime"` |
| **Duffing** | `"stable-node"`, `"three-equilibrium-attractors"`, `"three-equilibrium-centers"` |
| **Epileptor3D** | `"c2s-SN-SH"`, `"c3s-SN-supH"`, `"c10s-supH-SH"`, `"c11s-supH-supH"`, `"c2b-SN-SH"`, `"c4b-SN-FLC"`, `"c14b-subH-SH"`, `"c16b-subH-FLC"` |
| **Lorenz** | `"chaotic"` |
| **Van der Pol (3rd order)** | `"stable-focus"`, `"stable-limit-cycle"`, `"chaotic"` |
| **Rössler** | `"stable-limit-cycle"`, `"chaotic"` |

All systems support deterministic and stochastic (additive / state-dependent
noise) dynamics via `euler_maruyama`.

---

## Quick start

### Generate training data automatically

```julia
using KoopmanAnalysis

# EDMD training data: 10 trajectories of 1000 snapshots each
X_train, Y_train, cfg, rhs, X_fixed = edmd_training_data(
    "FHN", "stable-limit-cycle";
    m_train=1000, n_trajectories=10, window=1.0, dt=0.01, nLag=1
)

# Hankel-EDMD training data from a scalar trace
v_train, S, cfg_h, rhs_h = hankel_training_data(
    "FHN", "stable-limit-cycle";
    m_embed=10, tau_delay=2, m_train=2000, window=1.0, dt=0.01, nLag=1
)
```

### EDMD with Hermite dictionary

```julia
cfg = regime_config("FHN", "stable-limit-cycle")
rhs = fhn_rhs(cfg.params)
X   = rk4(rhs, [0.1, 0.0], 0.01, 2000; nLag=1)

Xc = X[:, 1:end-1]
Y  = X[:, 2:end]

K, ΨX, ΨY = compute_koopman_operator(Xc, Y, Z -> Psi_Hermite(Z, 4))
λ, Ξ = koopman_eigendecomposition(K)
```

### RBF dictionary

```julia
nRBF = 50
centers = cluster_data(Xc, nRBF)

K, ΨX, ΨY = compute_koopman_operator(
    Xc, Y,
    Z -> Psi_RBF(Z, centers; include_states=true),
    method=:ridge, alpha=1e-3
)
```

### Hankel-EDMD from a scalar trace

```julia
v = X[1, :]           # scalar observable
m_embed   = 10
tau_delay = 5

K, ΨX, ΨY, B_proj = hankel_edmd(
    v, m_embed, tau_delay,
    S -> Psi_Hermite(S, 3),
    method=:pinv
)
```

### High-level pipeline (Config.jl)

```julia
# Configure the analysis
cfg = KoopmanConfig(
    m_embed=500, tau_delay=5, r=50,
    dict_type=:rbf, dict_params=(nRBF=200, include_states=true),
    edmd_method=:ridge, edmd_alpha=1e-2, dt=0.01
)

# Build Hankel matrix and run analysis
S = embed(v_train, cfg)
res = hankel_analysis(S, cfg)

# Compute spectrum (dense or iterative KrylovKit)
λ, Ξ = spectrum(res)

# Predict
X_true, X_pred = predict(res, [1, 500, 1000], 200)

# Extract harmonic branches
branches = all_harmonic_branches(res)
```

---

## Module reference

### `Systems`

| Function | Purpose |
|----------|---------|
| `rk4(rhs, x0, dt, m; nLag=1)` | Explicit RK4 integrator. Works for any dimension `n`. |
| `euler_maruyama(drift, diffusion, x0, dt, m; nLag=1, seed=nothing)` | Euler–Maruyama for SDEs. |
| `generate_trajectories(rhs, n, dt, m; nLag=1, center, scale)` | Generate an ensemble of trajectories. |
| `detect_limit_cycle(rhs, dt, transient, cycle; ...)` | Drop transient and return periodic tail. |
| `fhn_rhs(p)`, `fhn_drift(p)`, `fhn_diffusion(p)`, `fhn_parameters(regime; ...)` | FHN vector field / parameters. |
| `duffing_rhs(p)`, `duffing_drift(p)`, `duffing_diffusion(p)`, `duffing_parameters(regime; ...)` | Duffing vector field / parameters. |
| `epileptor3d_rhs(p)`, `epileptor3d_drift(p)`, `epileptor3d_diffusion(p)`, `epileptor3d_parameters(regime; ...)` | 3D Epileptor vector field / parameters. |
| `lorenz_rhs(p)`, `lorenz_drift(p)`, `lorenz_diffusion(p)`, `lorenz_parameters(regime; ...)` | Lorenz vector field / parameters. |
| `vanderpol_rhs(p)`, `vanderpol_drift(p)`, `vanderpol_diffusion(p)`, `vanderpol_parameters(regime; ...)` | 3rd-order Van der Pol vector field / parameters. |
| `rossler_rhs(p)`, `rossler_drift(p)`, `rossler_diffusion(p)`, `rossler_parameters(regime; ...)` | Rössler vector field / parameters. |
| `find_fixed_points_fhn(p)`, `find_fixed_points_duffing(p)`, `find_fixed_points_lorenz(p)`, `find_fixed_points_vanderpol(p)`, `find_fixed_points_rossler(p)` | Fixed-point lists per system. |
| `fixed_point(system, p)` | Single representative fixed point. |
| `generate_test_trajectories(rhs, n, dt, top_pred_step; ...)` | Test trajectories with prediction horizon. |
| `path_parameters(z, A, B, R)` | Great-circle parametrisation for Epileptor3D path. |
| `compute_xs(mu2, mu1)` | Upper-branch fixed point of the fast cubic subsystem. |

### `Dictionaries`

| Function | Purpose |
|----------|---------|
| `get_dim_psi(n, max_deg)` | Size of the Hermite dictionary. |
| `hermite_basis(n, max_deg; basis_type=:probabilist)` | Build Hermite basis (`:probabilist` or `:physicist`). |
| `Psi_Hermite(X, max_deg; basis_type=:probabilist)` | Hermite basis evaluation, thread-parallel over columns. |
| `cluster_data(X, nRBF; max_points)` | k-means centres for RBFs (optional subsampling). |
| `Psi_RBF(X, centers; include_states=true, state_indices)` | Thin-plate RBF dictionary, thread-parallel over columns. |
| `build_rff_basis(n, D, sigma)` | Random Fourier Feature basis. |
| `Psi_RFF(X, basis; include_states=false)` | RFF dictionary evaluation. |
| `construct_projection_operator_hermite(state_dim, max_deg; basis_type)` | Analytic projection extracting linear monomials. |
| `lift_state(x, dict_info)` | Lift a single state vector into dictionary space. |
| `Psi_slice(X_slice, dict_info; full_dim, active_dims, fixed_values)` | Lift a partial-state matrix (for eigenfunction slices). |

### `EDMD`

| Function | Purpose |
|----------|---------|
| `compute_koopman_operator(ΨX, ΨY; method=:ridge, alpha=1e-3)` | EDMD from lifted snapshots (`:pinv` or `:ridge`). |
| `compute_koopman_operator(X, Y, Psi_func; ...)` | Convenience wrapper with lifting. |
| `construct_projection_operator(state_dim, ΨX, X_train; alpha=1e-6)` | Learn `B` such that `x ≈ B Ψ(x)`. |
| `rbf_kernel(x, y, sigma)` | Gaussian RBF kernel. |
| `median_heuristic_sigma(X; n_sample=1000)` | Median heuristic for bandwidth (thread-parallel). |
| `kernel_feature_vector(x, X_dict, sigma)` | RBF feature vector against landmarks. |
| `kernel_edmd_rbf(X, Y, X_dict, sigma; alpha=1e-6)` | Kernel EDMD with Gaussian RBF landmarks (thread-parallel column stripes). |
| `edmd_predict_from_psi(ΨX, K_edmd, B_full, S, start_indices, n_pred)` | Multi-step prediction iterating purely in ψ-space (thread-parallel over trajectories). |

### `Hankel`

| Function | Purpose |
|----------|---------|
| `build_hankel(v, m_embed, tau_delay)` | Delay-embedding matrix. |
| `build_hankel_multichannel(X, m_embed, tau_delay)` | Multi-channel Hankel from n-dimensional state trajectory. |
| `hankel_dmd(S; r=nothing, dt=1.0)` | Linear Hankel-DMD with optional SVD truncation. |
| `hankel_edmd(S; r, dt, dict_type, dict_params, edmd_method, edmd_alpha, proj_alpha, return_ψ, use_svd)` | Hankel-EDMD with explicit dictionary (memory-efficient single-lift). |
| `hankel_kernel_edmd(S; r, dt, sigma, alpha, use_svd, N_subsample)` | Hankel-Kernel-EDMD with Gaussian RBF. |
| `havok_dmd(S; r, dt, rank_method, energy, ridge_alpha)` | HAVOK continuous-time analysis (Brunton et al. 2017). |
| `havok_predict(A, B, U_r, S, start_indices, n_pred; dt, use_true_forcing)` | Predict by RK4 integration of the HAVOK linear model. |
| `select_svd_rank(σ; method=:energy, energy=0.999, aspect=1.0)` | Automatic rank selection (energy / Gavish–Donoho / gap). |
| `delay_embed_training_data(X_train, Y_train, m_embed, tau_delay; n_trajectories)` | Build delay-embedded EDMD pairs from state-space training data. |
| `delay_space_edmd_prediction(S, K_edmd, B_full, U_r, start_indices, n_pred, dict_info)` | EDMD prediction for new ICs with re-lifting at each step. |

### `DataGeneration`

| Function | Purpose |
|----------|---------|
| `edmd_training_data(system, regime; m_train, n_trajectories, window, dt, nLag, center, save_dir)` | Build `(X_train, Y_train, cfg, rhs, X_fixed)`. |
| `hankel_training_data(system, regime; m_embed, tau_delay, m_train, window, dt, nLag, observed_state, center, save_dir)` | Build `(v_train, S, cfg, rhs)`. |
| `state_space_predictions(rhs, K, x0s, dt, n_pred; Psi_func, B_proj, nLag)` | Predict in full state space. |
| `state_space_predict_from_psi(X_test, K, Psi_func, B_proj, start_indices, n_pred)` | State-space analogue of `edmd_predict_from_psi`. |
| `delay_space_predictions(S, K, start_indices, n_pred)` | Pure delay-space linear prediction. |

### `Spectral`

| Function | Purpose |
|----------|---------|
| `koopman_eigendecomposition(K; sort_by=abs)` | Sorted eigenvalues / right eigenvectors. |
| `detect_limit_cycle_modes(λ, dt; mag_tol, imag_tol, exclude_unity)` | Find unit-circle oscillatory modes. |
| `build_harmonic_branch(λ, j1; TOL_HARMONIC, max_harmonic, tol_growth)` | Build harmonic family of a fundamental eigenvalue. |
| `find_all_harmonic_branches(λ, dt; ...)` | Iteratively extract all independent harmonic families. |
| `select_phase_amplitude_modes(λ, τ; ...)` | Pick phase and amplitude eigenvalues. |
| `evaluate_eigenfunction_grid(Ξ, Psi_func, grids...; n_modes)` | Eigenfunctions on a full Cartesian grid. |
| `evaluate_eigenfunction_slice(Ξ, Psi_func, grids...; slice_dims, slice_values, n_modes)` | Eigenfunctions on a k-D slice. |

### `Figures`

| Function | Purpose |
|----------|---------|
| `fig_phase_portrait(data; projection_dims, fixed_points, xlim, ylim, title, subtitle, param_str)` | 2-D phase portrait (or projection). |
| `fig_phase_portrait_3d(data; fixed_points, title, subtitle, param_str)` | 3-D phase portrait. |
| `fig_phase_portrait_all_views(data; fixed_points, title, subtitle, param_str)` | 3-D view plus three coordinate planes. |
| `fig_training_data(X_train; dim, projection_dims, title, subtitle, param_str)` | Training snapshots scatter (2D or 3D). |
| `fig_clusterized_data(X, centers; dim, projection_dims, title, subtitle, param_str)` | Data + RBF centres (2D or 3D). |
| `fig_time_series(X, dt; dim_labels, title, subtitle, param_str, highlight_region)` | Time-series panels per dimension. |
| `fig_prediction(X_true, X_pred; dt, dim_labels, title, subtitle, param_str, max_trajectories)` | True vs predicted trajectory (error panel + per-observable panels). |
| `fig_eigenvalues(λ; title, subtitle, param_str, highlight_indices)` | Unit-circle spectrum with tight data limits. |
| `fig_eigenfunctions(Ξ, Psi_func, plane_dims, grid_range, indices; λ, fixed_points, normalize, title, subtitle, param_str, slice_values)` | Grid of eigenfunction heatmaps with zero-level contours. |
| `fig_phase_amplitude(Q1_grid, Qr_grid, grid_ranges; title, subtitle, param_str)` | Phase and log-amplitude panels. |

### `Plotting`

| Constant / Function | Purpose |
|---------------------|---------|
| `COLOR_PRIMARY`, `COLOR_ACCENT`, `COLOR_SECONDARY`, `COLOR_BACKGROUND` | Shared palette (deep teal / burnt orange / light blue / near-white). |
| `blue_tones(n)`, `orange_tones(n)` | Distinct colour vectors. |
| `set_koopman_theme!()` | Apply the shared Makie theme (called automatically on `using KoopmanAnalysis`). |

### `Regimes`

| Function | Purpose |
|----------|---------|
| `RegimeConfig` | Struct holding system, regime, parameters, and save directory. |
| `regime_config(system, regime; save_dir)` | Full metadata for a regime. |
| `list_regimes(system)` | Supported regimes for a system. |
| `default_save_dir(system, regime; base_dir, subdir)` | Default output directory. |

### `Utils`

| Function | Purpose |
|----------|---------|
| `normalize_vector(x; lb=0.35)` | Normalise for colour mapping. |
| `signed_area(x, y)` | Signed area of a closed polygon. |
| `meshgrid_2d(xrange, yrange)` | 2-D meshgrid matrices. |
| `classify_fixed_point_2d(eigenvalues)` | Classify planar fixed points (saddle, node, spiral, center). |
| `finite_difference_jacobian(f, x; h=1e-6)` | Central-difference Jacobian. |
| `mutual_information(x, τ; n_bins=50)` | Histogram-based mutual information (for delay selection). |
| `find_first_minimum(v)` | First local minimum index. |
| `false_nearest_neighbors(x, m, τ; rtol, atol)` | False-nearest-neighbour fraction (for embedding dimension). |
| `grad_Psi_RBF(s, centers; ...)` | Analytic Jacobian of the thin-plate RBF dictionary. |
| `grad_phi(s, ξ, dict_info)` | Gradient of a single Koopman eigenfunction `φ(s) = ξ'Ψ(s)`. |
| `find_zls_gradient_descent(S, Psi_func, Ξ, j_modes, dict_info; ...)` | Multi-start gradient descent on zero level sets (thread-parallel). |

### `Config` — High-level API

| Type / Function | Purpose |
|-----------------|---------|
| `KoopmanConfig` | Keyword-argument struct controlling the full pipeline (`m_embed`, `tau_delay`, `r`, `dict_type`, `dict_params`, `edmd_method`, `edmd_alpha`, `proj_alpha`, `dt`, `return_ψ`, `verbose`). |
| `AnalysisResult` | Mutable container for `K`, `B_full`, `B_reduced`, `U_r`, `X_r`, `dict_info`, `ΨX`, `ΨY`, `S`, `λ`, `Ξ`, and the config. Supports iteration over the 9 primary fields. |
| `embed(v, cfg)` | Build Hankel matrix from a scalar trace. |
| `hankel_analysis(v, cfg)` / `hankel_analysis(S, cfg)` | Run the full Hankel-EDMD / Hankel-DMD / HAVOK / Kernel-EDMD pipeline depending on `cfg.dict_type`. |
| `state_analysis(X, Y, cfg)` | Run state-space EDMD with an explicit dictionary. |
| `predict(res, start_indices, n_pred)` | Dispatch to the correct predictor (ψ-space, kernel, HAVOK, or linear delay-space). |
| `spectrum(res; howmany, krylovdim, tol)` | Dense eigendecomposition for small matrices; iterative KrylovKit `eigsolve` for large matrices. |
| `all_harmonic_branches(res; ...)` | Convenience wrapper extracting all harmonic families from the spectrum. |

---

## Threading and performance

Several heavy routines are thread-parallelized via `Threads.@spawn`:

- **`Psi_Hermite`** — column-block parallel evaluation.
- **`Psi_RBF`** — column-block parallel evaluation.
- **`median_heuristic_sigma`** — parallel pairwise-distance fill.
- **`kernel_edmd_rbf`** — parallel column-stripe Gram-matrix construction.
- **`edmd_predict_from_psi`** — parallel over trajectories.
- **`find_zls_gradient_descent`** — parallel multi-start gradient descent.

BLAS/LAPACK stages (`pinv`, `\`, `*`) are **not** hand-threaded; control core
usage explicitly when mixing task and BLAS parallelism:

```julia
using LinearAlgebra
BLAS.set_num_threads(4)   # BLAS parallelism
# Launch Julia with: julia -t auto --project=.
```

---

## Adding a new dynamical system

1. Define a parameter function `my_system_parameters(regime; noise_type, sigma, noise_mask)`.
2. Write `my_system_drift(p)` and optionally `my_system_diffusion(p)`.
3. Provide a backward-compatible alias `my_system_rhs(p) = my_system_drift(p)`.
4. Add fixed-point helpers if desired.
5. Register the regime in `Regimes.jl` for `regime_config` / `edmd_training_data` support.

Because the dictionaries and EDMD routines are dimension-agnostic, no module
changes are required for systems of dimension `n ≥ 1`.

---

## License and citation

This is research code. If you reuse the modules, please cite this repository and
the relevant publications on Koopman operator theory.
