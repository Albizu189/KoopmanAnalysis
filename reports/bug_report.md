# Bug Report — KoopmanAnalysis

**Date:** 2026-01-19
**Package:** KoopmanAnalysis v0.1.0
**Julia:** 1.11.6

## Summary

| Severity | Count | Issues |
|----------|-------|--------|
| Critical | 0 | — |
| High | 0 | — |
| Medium | 1 | `KoopmanConfig` accepts invalid parameter values silently |
| Low | 3 | `rk4(dt=0)` silent pass; `ridge(alpha=1e-20)` silent pass; `Psi_slice` no dimension validation |
| OK / Expected | 5 | API changes, missing-input errors, exports working |

---

## Bug 1: KoopmanConfig accepts invalid values without validation [MEDIUM]

- **Module:** Config
- **Function:** `KoopmanConfig` constructor
- **Severity:** Medium
- **Reproduction:**
  ```julia
  bad_cfg = KoopmanConfig(m_embed=-5, dt=0.0, edmd_alpha=-1.0)
  # No error is thrown; bad_cfg is created successfully
  ```
- **Expected:** The constructor should validate that:
  - `m_embed > 0`
  - `tau_delay > 0`
  - `dt > 0`
  - `edmd_alpha >= 0`
  - `proj_alpha >= 0`
  - `r === nothing || r > 0`
  - `dict_type` is one of `:hermite`, `:rbf`, `:rff`, `:none`, `:kernel`, `:havok`
  - `edmd_method` is one of `:pinv`, `:ridge`
- **Actual:** Any values are accepted, including negative dimensions and zero time step.
- **Impact:** Silent propagation of bad config into downstream functions, causing hard-to-debug errors later in the pipeline.
- **Suggested Fix:** Add validation logic to the `KoopmanConfig` constructor (or a separate `validate` function) that throws `ArgumentError` with descriptive messages.

---

## Bug 2: rk4 with dt=0 silently returns a constant trajectory [LOW]

- **Module:** Systems
- **Function:** `rk4`
- **Severity:** Low
- **Reproduction:**
  ```julia
  rhs = fhn_rhs(fhn_parameters("stable-node"))
  X = rk4(rhs, [0.1, 0.0], 0.0, 10)  # no error
  ```
- **Expected:** Either an `ArgumentError` (dt must be positive) or documented behavior.
- **Actual:** Returns a constant trajectory (all columns equal to initial condition).
- **Impact:** Users may not realize their integration did nothing.
- **Suggested Fix:** Add `dt > 0 || throw(ArgumentError("dt must be positive"))` at the start of `rk4`.

---

## Bug 3: compute_koopman_operator with extremely small ridge alpha [LOW]

- **Module:** EDMD
- **Function:** `compute_koopman_operator`
- **Severity:** Low
- **Reproduction:**
  ```julia
  K = compute_koopman_operator(randn(5,3), randn(5,3); method=:ridge, alpha=1e-20)
  ```
- **Expected:** A warning or error that the regularization is numerically unstable.
- **Actual:** Computes without complaint; results may be garbage due to floating-point underflow.
- **Impact:** Silent numerical instability in scientific results.
- **Suggested Fix:** Add a warning if `alpha < eps(Float64)` (≈ 2e-16).

---

## Bug 4: Psi_slice does not validate dimension compatibility [LOW]

- **Module:** Dictionaries
- **Function:** `Psi_slice`
- **Severity:** Low
- **Reproduction:**
  ```julia
  X_slice = randn(2, 5)  # 2D slice
  centers = randn(3, 3)   # 3D centers
  dict_info = (type=:rbf, centers=centers, include_states=true, state_indices=nothing)
  Psi = Psi_slice(X_slice, dict_info; full_dim=3, active_dims=[1,2], fixed_values=[0.0])
  # No error
  ```
- **Expected:** An error when the slice dimensions don't match the dictionary's state dimension.
- **Actual:** Silently produces output that may be meaningless.
- **Impact:** Hard-to-detect dimension mismatches in analysis code.
- **Suggested Fix:** Add `@assert` or explicit validation that `size(X_slice,1) == length(active_dims)` and `length(active_dims) + length(fixed_values) == full_dim`.

---

## Verified-OK Behaviors

The following were tested and behave correctly:

1. **`fig_prediction` rejects old `dim` keyword** — CORRECT. The API was updated to use `dim_labels`; passing `dim` now raises `MethodError`.
2. **`predict` with missing `S` matrix errors** — CORRECT. Falls through to `delay_space_predictions(::Nothing, ...)` which raises `MethodError`.
3. **New exports work correctly** — `build_hankel_multichannel`, `hankel_kernel_edmd`, `select_svd_rank`, `delay_embed_training_data`, `Psi_slice` all load and execute correctly after the export fix.
4. **RFF default sigma** — `KoopmanConfig(dict_type=:rff, dict_params=(D=10,))` correctly uses default `sigma=1.0`.

---

## Pre-Existing Known Issues (from optimization report, not new bugs)

These are design/performance issues rather than functional bugs:

- `epileptor3d_drift` triple-calls `path_parameters` — inefficient but produces correct output.
- `_thread_chunks` duplicated across modules — code smell, not a runtime bug.
- Missing docstrings on many public functions.
- `examples/prediction_example.jl` still uses old `dim` keyword (documented in AGENTS.md).
