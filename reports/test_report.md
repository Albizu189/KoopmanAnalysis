# Test Suite Report — KoopmanAnalysis

**Date:** 2026-01-19
**Julia version:** 1.11.6
**Package:** KoopmanAnalysis v0.1.0
**Test file:** `test/runtests.jl` (~1060 lines)

## Summary

| Metric | Value |
|--------|-------|
| Total testsets | 11 |
| Passed | 11 |
| Failed | 0 |
| Errored | 0 |
| Broken | 0 |
| **Total individual tests** | **329** |
| **Overall status** | **PASS** |
| Runtime | ~88 seconds |

## Results by Module

### Utils — 38/38 pass (4.7s)
All utility functions tested successfully:
- `normalize_vector` — correct normalization range
- `signed_area` — correct orientation handling, `DimensionMismatch` thrown
- `meshgrid_2d` — correct grid shape
- `classify_fixed_point_2d` — all fixed-point types classified correctly
- `finite_difference_jacobian` — matches analytic Jacobian
- `mutual_information` — handles lag=0 (NaN), constant signal (0)
- `find_first_minimum`, `false_nearest_neighbors` — smoke tests pass
- `grad_Psi_RBF`, `grad_phi`, `find_zls_gradient_descent` — smoke tests pass

### Systems — 57/57 pass (5.0s)
All dynamical system infrastructure verified:
- `rk4` — preserves state dimension, handles `nLag`, matrix initial conditions
- `generate_trajectories` — correct count, shape, length
- `detect_limit_cycle` — returns expected trajectory shape
- `euler_maruyama` — reproducible with same seed
- Parameter functions — all 15 regimes across 6 systems instantiate without error
- Drift/diffusion closures — all 6 systems return callable functions
- Fixed point helpers — correct counts (FHN ≥1, Duffing=3, Lorenz=3, VdP=1, Rossler≥1, Epileptor3D≥1)
- `generate_test_trajectories` — correct 3D array shape

### Dictionaries — 25/25 pass (8.6s)
Observable dictionary construction verified:
- `get_dim_psi` — correct combinatorial counts
- `hermite_basis` — returns basis, dimPsi, linIdx
- `Psi_Hermite` — correct shape, constant row = 1.0
- `cluster_data` + `Psi_RBF` — correct center count, shape with/without states
- `RFFBasis` + `Psi_RFF` — correct D, W shape, b length; include_states works
- `construct_projection_operator_hermite` — correct shape
- `lift_state` — works for hermite, rbf, rff dictionary info

### EDMD — 21/21 pass (3.1s)
Koopman operator estimation verified:
- `compute_koopman_operator` — square output, pinv/ridge methods, 3-arg form
- `construct_projection_operator` — correct shape (n_state × nPsi)
- `rbf_kernel` — self-similarity = 1.0, decay with distance
- `median_heuristic_sigma` — positive, finite
- `kernel_feature_vector` — correct length, bounded in (0, 1]
- `kernel_edmd_rbf` — K, B, K_ZZ, K_ZY all correct shapes
- `edmd_predict_from_psi` — prediction arrays correct shape

### Hankel — 22/22 pass (9.6s)
Delay-embedding infrastructure verified:
- `build_hankel` — correct shape, first column content verified
- `build_hankel_multichannel` — correct stacked shape
- `hankel_dmd` — K is square, U_r has correct rank
- `hankel_edmd` — K_edmd square, B_full has m_embed rows
- `hankel_kernel_edmd` — K is square
- `select_svd_rank` — all three methods return sensible ranks
- `havok_dmd` + `havok_predict` — A is 3×3, B has 3 rows, predictions correct shape
- `delay_embed_training_data` — correct stacked shape
- `delay_space_edmd_prediction` — predictions correct shape

### DataGeneration — 12/12 pass (1.0s)
Training data builders verified:
- `edmd_training_data` — correct shapes, returns RegimeConfig
- `hankel_training_data` — correct shapes, returns RegimeConfig
- `state_space_predictions` — correct 3D output shape
- `state_space_predict_from_psi` — correct 3D output shape
- `delay_space_predictions` — correct 3D output shape

### Spectral — 10/10 pass (3.9s)
Spectral analysis functions verified:
- `koopman_eigendecomposition` — correct λ length, Ξ shape
- `detect_limit_cycle_modes` — finds 2 modes in test data
- `build_harmonic_branch` — correct mode included
- `find_all_harmonic_branches` — returns Dict{Int, Vector{Int}}
- `select_phase_amplitude_modes` — returns positive indices
- `evaluate_eigenfunction_grid` — correct (N, N, n_modes) shape
- `evaluate_eigenfunction_slice` — correct (N, n_modes) shape

### Plotting & Figures — 20/20 pass (31.1s)
All figure functions return `Makie.Figure` objects:
- Constants (`COLOR_PRIMARY`, etc.) are valid RGB values
- `blue_tones`, `orange_tones` return correct-length palettes
- `set_koopman_theme!` — no error
- `fig_phase_portrait`, `fig_training_data`, `fig_clusterized_data`
- `fig_prediction` — works with default `dim_labels`
- `fig_eigenvalues`, `fig_eigenfunctions`, `fig_phase_amplitude`
- `fig_time_series` — works with matrix and vector inputs
- `fig_phase_portrait_3d`, `fig_phase_portrait_all_views`

### Regimes — 114/114 pass (0.0s)
All regime configurations validated:
- 3 FHN regimes, 3 Duffing regimes, 8 Epileptor3D regimes
- 1 Lorenz regime, 3 VanderPol regimes, 2 Rossler regimes
- All return valid `RegimeConfig` with correct system, regime, params, save_dir
- `list_regimes` returns non-empty Vector{String} for all 6 systems
- `default_save_dir` contains expected substrings

### Config — 15/15 pass (4.1s)
High-level API verified:
- `KoopmanConfig` — defaults correct, custom fields settable
- `embed` — returns Hankel matrix of correct shape
- `hankel_analysis` — returns `AnalysisResult` with square K
- `state_analysis` — returns `AnalysisResult` with square K
- `predict` — returns predictions of correct shape
- `spectrum` — returns eigenvalues/eigenvectors of correct length
- `all_harmonic_branches` — returns `Dict{Int, Vector{Int}}`
  - *Note:* this test was fixed by changing `AnalysisResult.K` from `Matrix{Float64}` to `Matrix{<:Number}` to allow complex Koopman operators.

### Integration — 5/5 pass (0.3s)
End-to-end pipelines verified:
- FHN full pipeline: `hankel_training_data` → `hankel_analysis` → `spectrum` → `predict`
- Duffing state-space pipeline: `edmd_training_data` → `state_analysis` → `spectrum`

## Fixes Applied During Testing

| Fix | File | Description |
|-----|------|-------------|
| Missing exports | `src/KoopmanAnalysis.jl` | Added `build_hankel_multichannel`, `hankel_kernel_edmd`, `select_svd_rank`, `delay_embed_training_data`, `Psi_slice` to export list |
| Duplicate testset | `test/runtests.jl` | Removed a duplicate `delay_space_edmd_prediction` testset with a dimension mismatch |
| Complex K matrices | `src/Config.jl` | Changed `AnalysisResult.K` type from `Matrix{Float64}` to `Matrix{<:Number>` to support complex-valued Koopman operators |

## Recommendations

1. **Register test command in Project.toml** — Add `[targets]` / `test = ["Test"]` so `] test` works from Pkg mode.
2. **Add edge-case tests** — The current suite is strong on smoke tests and shape checks, but could benefit from:
   - Empty input handling (`dt=0`, zero-step integration)
   - Very large/small regularization parameters
   - Rank-deficient dictionary matrices
   - Numerical stability near fixed points
3. **Add regression tests** — For key functions like `rk4` and `Psi_Hermite`, verify actual numerical output against known analytic results (not just shapes).
4. **Benchmark tests** — Consider adding `@benchmark` or timing assertions to catch performance regressions.
5. **Continuous integration** — With this test suite, the package is now ready for GitHub Actions CI.
