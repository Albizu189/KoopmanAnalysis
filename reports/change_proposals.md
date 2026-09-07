# Change Proposals — KoopmanAnalysis

**Synthesized from:** bug_report.md, optimization_report.md, test_report.md  
**Date:** 2026-01-19  
**Package:** KoopmanAnalysis v0.1.0  
**Julia:** 1.11.6

---

## Executive Summary

The `KoopmanAnalysis` package is in solid functional shape: all 329 tests across 11 testsets pass, the core EDMD/Hankel/kernel pipelines are numerically correct, and the module architecture is clean. The audit confirmed that recent fixes (export additions, `AnalysisResult.K` type widening, and duplicate-testset cleanup) resolved several pre-existing issues.

However, three themes require attention before the package can be considered production-ready:

1. **Crash bugs and API drift** — `fixed_point` can crash on parameter regimes with no real equilibria, and the primary example script `prediction_example.jl` fails because it uses an obsolete keyword argument. These are user-facing breakages.
2. **Performance at scale** — The RK4 and Euler–Maruyama integrators allocate heavily in their hot loops, and the heaviest prediction paths (`_kernel_delay_predict`, trajectory generators, HAVOK predict) remain serial despite working parallel implementations existing in `Mod vParallel`.
3. **Code quality and maintainability** — `_thread_chunks` is duplicated, the export list has duplicate blocks, dictionary-building logic is copied between `Hankel.jl` and `Config.jl`, and `KoopmanConfig` silently accepts nonsensical values.

This document contains **19 actionable proposals**: 2 Critical, 8 High, 6 Medium, and 4 Low. Most High items are small-to-medium effort and deliver outsized value.

---

## Proposed Changes (Prioritized)

### Critical — Must Fix

#### 1. `fixed_point` crashes with `BoundsError` when no real fixed points exist
- **One-line:** `Systems.fixed_point` calls `fps[1]` without checking `isempty(fps)`, crashing for regimes like Rossler with `disc < 0`.
- **Source reports:** optimization_report (Section E.1)
- **Files affected:** `src/Systems.jl` (~line 686)
- **Effort:** Small
- **Rationale:** A parameter change can silently turn a working notebook into a crash. All six systems delegate to this single function.
- **Suggested implementation:**
  ```julia
  isempty(fps) && error("No real fixed points found for system=$system with given parameters.")
  return fps[1]
  ```

#### 2. `examples/prediction_example.jl` fails with `MethodError` on obsolete `dim` keyword
- **One-line:** The example script calls `fig_prediction(...; dim=1, ...)` but `fig_prediction` expects `dim_labels`.
- **Source reports:** bug_report (pre-existing), optimization_report (Section D.1, E.4), AGENTS.md
- **Files affected:** `examples/prediction_example.jl` (line 76), `src/Figures.jl` (line 253)
- **Effort:** Small
- **Rationale:** The first entry point for new users is a broken script. This degrades onboarding and trust.
- **Suggested implementation:** Either (a) change the example to `dim_labels=["x[1]"]` and `dt=dt`, or (b) add a backward-compatible `dim` positional fallback inside `fig_prediction` that auto-generates labels.

---

### High — Should Fix Soon

#### 3. `KoopmanConfig` accepts invalid values without validation
- **One-line:** Negative `m_embed`, zero `dt`, invalid `dict_type`/`edmd_method` symbols, and negative regularization parameters are accepted silently.
- **Source reports:** bug_report (Bug 1), optimization_report (Section D.4, E.16)
- **Files affected:** `src/Config.jl` (lines 23–36)
- **Effort:** Medium
- **Rationale:** Bad configs propagate deep into the pipeline before surfacing as inscrutable linear-algebra errors. Catching them at construction saves debugging time and prevents incorrect scientific results.
- **Suggested implementation:** Add a `validate!(cfg::KoopmanConfig)` function (or inline constructor checks) that throws `ArgumentError` for:
  - `m_embed > 0`, `tau_delay > 0`, `dt > 0`
  - `edmd_alpha >= 0`, `proj_alpha >= 0`
  - `r === nothing || r > 0`
  - `dict_type ∈ (:hermite, :rbf, :rff, :none, :kernel, :havok)`
  - `edmd_method ∈ (:pinv, :ridge)`

#### 4. `epileptor3d_drift` triple-calls `path_parameters` per RHS evaluation
- **One-line:** The Epileptor3D drift calls `path_parameters` three times inside the closure; each call does `sqrt`, `acos`, `atan`.
- **Source reports:** optimization_report (Section C.3, E.5)
- **Files affected:** `src/Systems.jl` (lines 339–341)
- **Effort:** Small
- **Rationale:** A one-line fix yields ~2–3× speedup for every Epileptor3D integration, with no change in numerical output.
- **Suggested implementation:**
  ```julia
  mu2, mu1, nu, theta, phi = path_parameters(x[3], p.A, p.B, p.R)
  xs = compute_xs(mu2, mu1)
  ```

#### 5. `rk4` allocates temporary vectors in its hot loop
- **One-line:** Column copies (`x[:, k]`), intermediate broadcast temporaries (`xk .+ 0.5 * dt * k1`), and compound-update temporaries allocate on every internal step.
- **Source reports:** optimization_report (Section C.1, E.6)
- **Files affected:** `src/Systems.jl` (lines 41–56)
- **Effort:** Medium
- **Rationale:** For 10 000 trajectories of 2000 steps, this is ~160 M transient allocations — heavy GC pressure that becomes the bottleneck before the physics does.
- **Suggested implementation:** Hoist `k1…k4` and `tmp` buffers outside the loop; replace `x[:, k]` with `@view x[:, k]`; use `@.` for the final update. If preserving the closure-based RHS signature, this still cuts allocations ~80 %.

#### 6. `euler_maruyama` has the same allocation pattern
- **One-line:** Copies `x[:, k]`, allocates `dx_det` and `dx_stoch` on every step.
- **Source reports:** optimization_report (Section C.2, E.7)
- **Files affected:** `src/Systems.jl` (lines 130–145)
- **Effort:** Small
- **Rationale:** Same GC-pressure issue as `rk4`, but for stochastic ensembles.
- **Suggested implementation:** Hoist `@view x[:, k]`, pre-allocate `dx_det`/`dx_stoch`, update with `@. x[:, k+1] = xk + dx_det + dx_stoch`.

#### 7. Heavy prediction and training paths remain serial
- **One-line:** `edmd_training_data`, `state_space_predictions`, `delay_space_predictions`, `havok_predict`, `delay_space_edmd_prediction`, and `_kernel_delay_predict` loop serially over `n_trajectories`.
- **Source reports:** optimization_report (Section B.2, E.8–11)
- **Files affected:** `src/DataGeneration.jl`, `src/Hankel.jl`, `src/Config.jl`
- **Effort:** Large
- **Rationale:** These are the dominant runtime costs when scaling to large ensembles. `Mod vParallel` contains working, signature-compatible `@spawn` implementations that give near-linear speedup.
- **Suggested implementation:** Port the `@spawn` + `_thread_chunks` pattern from `Mod vParallel` for each function. Ensure reproducibility by giving each trajectory/task its own RNG stream.

#### 8. Export list in `KoopmanAnalysis.jl` contains duplicate blocks
- **One-line:** The Dictionaries and Hankel re-export blocks are duplicated (e.g. `get_dim_psi` exported twice), and the overall list is inconsistent with submodule public APIs.
- **Source reports:** optimization_report (Section A.5), test_report (fixes applied)
- **Files affected:** `src/KoopmanAnalysis.jl` (lines 83–137)
- **Effort:** Small
- **Rationale:** Duplicate exports generate warnings and make backward-compatibility guarantees harder. Some missing exports were already added during testing, but the list still needs a full audit.
- **Suggested implementation:** Deduplicate the export block. Optionally, stop re-exporting entirely and let users qualify names (`KoopmanAnalysis.Dictionaries.Psi_slice`) — cleaner as the package grows.

#### 9. Dense eigensolve threshold is too aggressive
- **One-line:** `spectrum` uses dense `eigvals`/`eigvecs` for matrices up to 10 000 × 10 000.
- **Source reports:** optimization_report (Section C.7, E.19)
- **Files affected:** `src/Config.jl` (line 295)
- **Effort:** Small
- **Rationale:** A 5000×5000 real nonsymmetric dense solve is ~125× slower than 1000×1000 and already stresses memory. The threshold should be conservative.
- **Suggested implementation:** Lower to `n < 2000` or add a `dense_threshold` field to `KoopmanConfig`.

#### 10. Inconsistent keyword naming across figure routines
- **One-line:** `fig_training_data` and `fig_clusterized_data` use `dim`; `fig_prediction` and `fig_time_series` use `dim_labels`.
- **Source reports:** optimization_report (Section D.1)
- **Files affected:** `src/Figures.jl` (lines 143, 195, 253, 532)
- **Effort:** Small
- **Rationale:** Inconsistent API surface forces users to read source code to know which keyword to pass.
- **Suggested implementation:** Standardize on `dim_labels` everywhere, or add deprecation fallbacks so `dim` still works with a warning.

---

### Medium — Nice to Have

#### 11. Deduplicate `_thread_chunks` across modules
- **One-line:** The same chunking helper exists verbatim in `Dictionaries.jl` and `EDMD.jl`.
- **Source reports:** optimization_report (Section A.1, E.12)
- **Files affected:** `src/Dictionaries.jl` (lines 23–37), `src/EDMD.jl` (lines 17–31), `src/Utils.jl`
- **Effort:** Small
- **Rationale:** Any change to oversubscription policy must be edited in N places. `Utils.jl` is loaded first, so it can own the canonical copy.
- **Suggested implementation:** Move `_thread_chunks` to `Utils.jl`; replace the inline definitions with `using ..Utils: _thread_chunks`.

#### 12. Introduce generic `make_diffusion` factory
- **One-line:** Six systems (`fhn`, `duffing`, `lorenz`, `vanderpol`, `rossler`, `epileptor3d`) contain near-identical diffusion closures.
- **Source reports:** optimization_report (Section A.3, E.13)
- **Files affected:** `src/Systems.jl` (lines 181–558)
- **Effort:** Medium
- **Rationale:** Removes ~40 lines of duplicated boilerplate and makes adding a seventh system a one-liner.
- **Suggested implementation:**
  ```julia
  function make_diffusion(sigma, noise_mask, noise_type; state_dep_fn=nothing)
      # generic branch logic
  end
  ```
  Per-system `*_diffusion(p)` becomes a one-line delegation with the appropriate `state_dep_fn` closure.

#### 13. Unify dictionary-building logic between `Hankel.jl` and `Config.jl`
- **One-line:** `_build_dict_and_project` and `state_analysis` both contain near-identical `if dict_type == :hermite / :rbf / :rff` branches.
- **Source reports:** optimization_report (Section A.2, E.14)
- **Files affected:** `src/Hankel.jl` (~lines 502–552), `src/Config.jl` (~lines 180–226)
- **Effort:** Medium
- **Rationale:** Adding a new dictionary type requires edits in two places. A single entry point in `Dictionaries.jl` eliminates this drift.
- **Suggested implementation:** Refactor `state_analysis` to call a public `build_dictionary_pair(X, Y, dict_type, dict_params)` that lives in `Dictionaries.jl`.

#### 14. Chunked eigenfunction evaluation to cut transient RAM
- **One-line:** `evaluate_eigenfunction_grid` and `slice` build the full `Ψ_pts` matrix at once; for a 500×500 grid with nΨ=200 this is ~400 MB.
- **Source reports:** optimization_report (Section C.6, E.15)
- **Files affected:** `src/Spectral.jl` (lines 215–287)
- **Effort:** Medium
- **Rationale:** Transient RAM spikes crash notebooks on laptops. `Mod vParallel2` achieved ~92 % RAM reduction by chunking.
- **Suggested implementation:** Evaluate/project in column blocks instead of all-at-once.

#### 15. Add thread-control and seed fields to `KoopmanConfig`
- **One-line:** There is no way to request serial execution or a reproducible RNG seed through the high-level config API.
- **Source reports:** optimization_report (Section D.4, E.16)
- **Files affected:** `src/Config.jl` (lines 23–36)
- **Effort:** Small
- **Rationale:** Reproducibility in published work requires deterministic execution; benchmarking requires the ability to disable task parallelism.
- **Suggested implementation:** Add `parallel::Bool = true` and `seed::Union{Nothing,Int} = nothing` to the struct, then thread them through `generate_trajectories` and prediction loops.

#### 16. Register `] test` command in `Project.toml`
- **One-line:** The package has a working test suite but no `[targets]` entry, so `] test` fails from Pkg mode.
- **Source reports:** test_report (Recommendation 1)
- **Files affected:** `Project.toml`
- **Effort:** Small
- **Rationale:** Standard Julia package hygiene; enables CI and `Pkg.test()`.
- **Suggested implementation:**
  ```toml
  [targets]
  test = ["Test"]
  ```

---

### Low — Cosmetic / Optional

#### 17. `rk4` with `dt=0` silently returns a constant trajectory
- **One-line:** Passing `dt=0` produces no error; every snapshot equals the initial condition.
- **Source reports:** bug_report (Bug 2)
- **Files affected:** `src/Systems.jl` (line 41)
- **Effort:** Small
- **Rationale:** A silent no-op integration is easy to miss in long pipelines.
- **Suggested implementation:** Add `dt > 0 || throw(ArgumentError("dt must be positive"))` at the top of `rk4`.

#### 18. `compute_koopman_operator` with extremely small ridge alpha
- **One-line:** `alpha=1e-20` is accepted without complaint, but `float(alpha) * I` underflows to zero, silently disabling regularization.
- **Source reports:** bug_report (Bug 3)
- **Files affected:** `src/EDMD.jl` (lines 44–55)
- **Effort:** Small
- **Rationale:** Users may think they are regularizing when they are not, leading to ill-conditioned solves.
- **Suggested implementation:** Emit a `@warn` if `alpha < eps(Float64)` (≈ 2e-16).

#### 19. Missing docstrings on public/semi-public functions
- **One-line:** `get_dim_psi`, `hermite_basis`, `build_rff_basis`, `RFFBasis`, `select_svd_rank`, `delay_embed_training_data`, `state_space_predict_from_psi`, and `_build_dict_and_project` lack docstrings.
- **Source reports:** optimization_report (Section D.2, E.18)
- **Files affected:** `src/Dictionaries.jl`, `src/Hankel.jl`, `src/DataGeneration.jl`
- **Effort:** Medium
- **Rationale:** Docstrings are the primary documentation for notebook users who do not read source files.
- **Suggested implementation:** Add short docstrings describing inputs, outputs, and a one-line usage example.

#### 20. Add `configure_threads!` BLAS helper
- **One-line:** The codebase contains correct commentary about BLAS oversubscription but no runtime enforcement or convenience helper.
- **Source reports:** optimization_report (Section B.4, E.20)
- **Files affected:** `src/Utils.jl`
- **Effort:** Small
- **Rationale:** Users must manually remember `BLAS.set_num_threads(1)` to avoid oversubscription when using task parallelism.
- **Suggested implementation:**
  ```julia
  function configure_threads!(; blas_threads=1, julia_threads=Threads.nthreads())
      BLAS.set_num_threads(blas_threads)
      @info "Threading config: BLAS=$blas_threads, Julia tasks=$julia_threads"
  end
  ```
  Expose it so notebooks can call one function instead of the incantation.

---

## Consolidated Recommendations

These are the 10 items the author should tackle first, ordered by impact-to-effort ratio:

1. **Guard `fixed_point` against empty `fps`** (`Systems.jl`, Small) — prevents crashes for legitimate parameter choices.
2. **Fix `examples/prediction_example.jl` keyword** (`examples/`, Small) — the first impression for new users must work.
3. **Deduplicate exports in `KoopmanAnalysis.jl`** (`KoopmanAnalysis.jl`, Small) — clean namespace, no warnings.
4. **Add `KoopmanConfig` validation** (`Config.jl`, Medium) — catches user errors before they become debugging sessions.
5. **Cache `path_parameters` in `epileptor3d_drift`** (`Systems.jl`, Small) — 2–3× speedup for the heaviest RHS with one line.
6. **Lower dense eigensolve threshold** (`Config.jl`, Small) — prevents accidental O(n³) memory and time blowups.
7. **Hoist temporaries in `rk4` and `euler_maruyama`** (`Systems.jl`, Medium) — removes the dominant allocation source at scale.
8. **Register `] test` in `Project.toml`** (`Project.toml`, Small) — unlocks CI and standard Pkg workflows.
9. **Standardize figure keyword naming** (`Figures.jl`, Small) — consistent API reduces user friction.
10. **Port prediction/training parallelization from `Mod vParallel`** (`DataGeneration.jl`, `Hankel.jl`, `Config.jl`, Large) — the single biggest throughput improvement for ensemble workflows.

---

## Appendix: Changes Already Applied

The following fixes were made during the audit process (confirmed in current `src/`):

| Fix | File | Description |
|-----|------|-------------|
| Missing exports | `src/KoopmanAnalysis.jl` | Added `build_hankel_multichannel`, `hankel_kernel_edmd`, `select_svd_rank`, `delay_embed_training_data`, `Psi_slice` to the export list. |
| Duplicate testset | `test/runtests.jl` | Removed a duplicate `delay_space_edmd_prediction` testset with a dimension mismatch. |
| Complex K matrices | `src/Config.jl` | Widened `AnalysisResult.K` from `Matrix{Float64}` to `Matrix{<:Number}` to support complex-valued Koopman operators. |
| Rossler in DataGeneration | `src/DataGeneration.jl` | **Verified present** — both `edmd_training_data` and `hankel_training_data` already contain `elseif system == "Rossler"` branches. The optimization report flagged this as missing, but the current source is complete. |
| `dict_params.sigma` safety | `src/Config.jl` | **Verified present** — the `:kernel` branch already uses `get(cfg.dict_params, :sigma, 1.0)`. The optimization report's `KeyError` concern appears to have been addressed or was a false positive against an earlier version. |
| `Psi_slice` validation | `src/Dictionaries.jl` | **Verified present** — the function already checks `length(active_dims) == n_slice`, `maximum(active_dims) <= full_dim`, and `length(fixed_values) == length(inactive)`. The bug report's dimension-mismatch concern is already handled. |

### What the audits confirmed is working well

- **Numerical correctness:** All 329 tests pass, including shape checks, eigendecomposition correctness, limit-cycle detection, and end-to-end FHN/Duffing pipelines.
- **Parallel infrastructure:** `Psi_Hermite`, `Psi_RBF`, `median_heuristic_sigma`, `kernel_edmd_rbf` Gram fill, and `edmd_predict_from_psi` are correctly threaded with `@spawn` and produce bit-identical results.
- **API evolution:** The `fig_prediction` API change from `dim` to `dim_labels` was correctly enforced — passing the old keyword raises `MethodError` as intended.
- **New features load:** `Psi_slice`, `build_hankel_multichannel`, `delay_embed_training_data`, and `hankel_kernel_edmd` all execute correctly after the export fix.
