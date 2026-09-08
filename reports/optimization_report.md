# KoopmanAnalysis Package — Optimization Report

**Date:** 2026-08-27  
**Scope:** `src/*.jl` (canonical source) vs. `Mod vParallel/` and `Mod vParallel2/` parallelization proposals  
**Analyst:** Optimization Analyst (sub-agent)  

---

## Executive Summary

The `KoopmanAnalysis` package is a well-structured Julia research toolbox with sensible module boundaries and a clear pipeline. The canonical `src/` already incorporates the **Dictionaries.jl** and **EDMD.jl** parallelizations from `Mod vParallel`, but it is **missing** the parallelization of **Utils.false_nearest_neighbors**, **DataGeneration** trajectory generators, **Hankel.havok_predict / delay_space_edmd_prediction**, and **Config._kernel_delay_predict**. Beyond threading gaps, there are several **redundancies** (six copies of `_thread_chunks`, duplicated dictionary-selection logic), **allocation hotspots** in `rk4` and `euler_maruyama`, and **API inconsistencies** (export list out of sync with sub-modules, a known `fig_prediction` keyword mismatch). This report details each issue and provides a prioritized, actionable remediation list.

---

## Section A — Redundancies & Overlaps

### A.1 `_thread_chunks` duplicated in every threaded module

**Finding:** The exact same chunking helper exists in:

| File | Lines |
|---|---|
| `src/Dictionaries.jl` | 23–37 |
| `src/EDMD.jl` | 17–31 |
| `src/Utils.jl` | *not present in canonical src* (but present in Mod vParallel) |
| `src/Hankel.jl` | *not present in canonical src* (but present in Mod vParallel) |
| `src/DataGeneration.jl` | *not present in canonical src* (but present in Mod vParallel) |
| `src/Config.jl` | *not present in canonical src* (but present in Mod vParallel) |

**Impact:** Violates DRY. Any change to chunking policy (e.g. adjusting the `4T` oversubscription factor) must be edited in N places.

**Recommendation:** Move `_thread_chunks` to `Utils.jl` and have all threaded modules import it. `Utils` is included first in `KoopmanAnalysis.jl`, so it is available to every downstream module via `..Utils`.

---

### A.2 Dictionary-selection logic duplicated between `Hankel.jl` and `Config.jl`

**Finding:** `Hankel._build_dict_and_project` (lines 502–552) and `Config.state_analysis` (lines 180–226) both contain near-identical `if dict_type == :hermite / :rbf / :rff` branches that build the dictionary, compute `ΨX`/`ΨY`, and form `dict_info`. `Config.state_analysis` does not reuse `_build_dict_and_project`.

**Impact:** Any change to dictionary defaults (e.g. adding a new `dict_type`) must be applied in two places.

**Recommendation:** Refactor `state_analysis` to call a public version of `_build_dict_and_project` (or move the logic entirely into `Dictionaries.jl` as a single entry-point `build_dictionary_pair(X, Y, dict_type, dict_params)`).

---

### A.3 Diffusion-function boilerplate repeated for every dynamical system

**Finding:** `fhn_diffusion`, `duffing_diffusion`, `epileptor3d_diffusion`, `lorenz_diffusion`, `vanderpol_diffusion`, and `rossler_diffusion` are structurally identical:

```julia
function xxx_diffusion(p)
    n = length(p.noise_mask)
    function g(x)
        if p.noise_type == :none
            return zeros(n)
        elseif p.noise_type == :additive
            return p.sigma .* p.noise_mask
        elseif p.noise_type == :state_dependent
            return p.sigma .* abs.(x) .* p.noise_mask   # or abs(x[1]) etc.
        else
            error("Unknown noise_type: $(p.noise_type)")
        end
    end
    return g
end
```

**Impact:** Six copies of the same noise-branching logic.

**Recommendation:** Introduce a generic `make_diffusion(sigma, noise_mask, noise_type; state_dep_fn=nothing)` factory in `Systems.jl` and have each `*_diffusion` delegate to it. The only per-system variation is the `state_dependent` amplitude rule (`abs.(x)`, `abs(x[1])`, `abs(x[1]-xs)`, etc.), which can be passed as a closure.

---

### A.4 Dead / notebook-only exports

**Finding:** The following are exported from `KoopmanAnalysis.jl` but have **no internal call sites** inside the package:

- `signed_area` — used only in notebooks (PCA orientation).
- `meshgrid_2d` — convenience wrapper; notebooks can define it locally.
- `kernel_feature_vector` — internal helper for `kernel_edmd_rbf`; not part of the public API.
- `finite_difference_jacobian` — only used for fixed-point classification in notebooks.

**Impact:** Bloats the namespace and complicates backward-compatibility guarantees.

**Recommendation:** Remove `kernel_feature_vector` from exports. Keep the others but move them to a `KoopmanAnalysis.NotebookUtils` submodule or document them explicitly as "notebook helpers".

---

### A.5 Export list out of sync with sub-module exports

**Finding:** Several functions are exported from sub-modules but **not** re-exported from `KoopmanAnalysis.jl`:

| Sub-module exports it | Missing from `KoopmanAnalysis` exports |
|---|---|
| `Dictionaries.Psi_slice` | ✗ |
| `Hankel.delay_embed_training_data` | ✗ |
| `Hankel.select_svd_rank` | ✗ (exported from Config but not Hankel in main module) |
| `Hankel.build_hankel_multichannel` | ✗ |

Conversely, `KoopmanAnalysis.jl` exports `select_svd_rank` (from Config) while `Hankel.jl` also exports it, creating a duplicate-export risk if both modules are used directly.

**Recommendation:** Audit the export list. Either (a) re-export everything that is public, or (b) stop re-exporting and let users qualify names (`KoopmanAnalysis.Dictionaries.Psi_slice`). Option (b) is cleaner for a growing package.

---

### A.6 Include order is correct but not minimal

**Finding:** `KoopmanAnalysis.jl` includes modules in this order:

```
Utils → Systems → Dictionaries → EDMD → Hankel → Regimes → DataGeneration → Spectral → Plotting → Figures → Config
```

This respects the dependency DAG (Hankel needs EDMD & Dictionaries; DataGeneration needs Regimes, Systems, Hankel; Config needs almost everything). **No issue here.**

However, `Plotting` and `Figures` are loaded before `Config`, which means `Config.jl` can use them, but `Config.jl` currently does not. If `Config` ever needs plotting helpers, the order is already fine.

---

## Section B — Further Parallelization Opportunities

### B.1 What `Mod vParallel` already parallelized (present in `src/`)

| Function | Module | Status in `src/` | Mechanism |
|---|---|---|---|
| `Psi_Hermite` | Dictionaries | ✅ Parallel | `@spawn` over column chunks |
| `Psi_RBF` | Dictionaries | ✅ Parallel | `@spawn` over column chunks |
| `median_heuristic_sigma` | EDMD | ✅ Parallel | `@spawn` over distance stripes |
| `kernel_edmd_rbf` Gram fill | EDMD | ✅ Parallel | `@spawn` over column stripes |
| `edmd_predict_from_psi` | EDMD | ✅ Parallel | `@spawn` over trajectories |
| `find_zls_gradient_descent` | Utils | ✅ Parallel | `@spawn` over starts (atomics + lock for progress) |

### B.2 What `Mod vParallel` parallelized but `src/` is **missing**

| Function | Module | Gap | Expected benefit |
|---|---|---|---|
| `false_nearest_neighbors` | Utils | ❌ Still serial, and uses the old high-memory algorithm (`Y_mp1` matrix) | Near-linear speedup + ~5–20× memory reduction (see PARALLELIZATION_REPORT.md) |
| `mutual_information_curve` | Utils | ❌ Does not exist in `src/` | Batched MI sweep for delay embedding; trivial to add |
| `edmd_training_data` | DataGeneration | ❌ Serial | Near-linear over `n_trajectories` with reproducible seeded RNG |
| `state_space_predictions` | DataGeneration | ❌ Serial | Near-linear over `n_traj` |
| `state_space_predict_from_psi` | DataGeneration | ❌ Serial | Near-linear over `n_traj` |
| `delay_space_predictions` | DataGeneration | ❌ Serial | Near-linear over `n_traj` |
| `havok_predict` | Hankel | ❌ Serial | Near-linear over `n_traj` |
| `delay_space_edmd_prediction` | Hankel | ❌ Serial | Near-linear over `n_traj` |
| `_kernel_delay_predict` | Config | ❌ Serial | Near-linear over `n_traj`; this is the heaviest predict path |

### B.3 What `Mod vParallel2` added on top

`Mod vParallel2/PARALLELIZATION_REPORT.md` describes additional work on `Systems.txt` and `Spectral.txt`:

| Function | Action | Rationale |
|---|---|---|
| `generate_trajectories` | `@threads` over trajectories + per-trajectory RNG streams | Ensemble members are independent |
| `generate_test_trajectories` | Same | Same |
| `rk4` | Allocation refactor (`@view`, hoisted scalars) — **not parallel** | 20–40 % fewer allocations |
| `euler_maruyama` | Same allocation refactor | Reproducibility-preserving |
| `epileptor3d_drift` | Eliminated triple `path_parameters` call | ~3× cheaper per RHS eval |
| `evaluate_eigenfunction_grid` | `@threads` over column chunks with fused projection | User callback dominates |
| `evaluate_eigenfunction_slice` | Same | Same |

**None of these `Mod vParallel2` improvements are in canonical `src/`.**

### B.4 BLAS vs. task-threading conflicts

**Finding:** The codebase already contains extensive, correct commentary about BLAS/LAPACK oversubscription (e.g. EDMD.jl lines 37–43, Utils.jl lines 253–258). However, **no runtime enforcement exists**. Users must manually call `BLAS.set_num_threads(1)`.

**Recommendation:** Add a lightweight helper to `Utils.jl`:

```julia
function configure_threads!(; blas_threads=1, julia_threads=Threads.nthreads())
    BLAS.set_num_threads(blas_threads)
    @info "Threading config: BLAS=$blas_threads, Julia tasks=$julia_threads"
end
```

Call it from `KoopmanAnalysis.__init__` with a sensible default, or at least expose it so notebooks can call one function instead of remembering the incantation.

### B.5 `@threads` vs. `@spawn`

**Finding:** The existing parallel code (and `Mod vParallel`) uses `@spawn` with manual chunking. `Mod vParallel2` uses `@threads` for `generate_trajectories` and `evaluate_eigenfunction_grid`.

**Assessment:**
- `@spawn` + chunking is **superior** for **uneven workloads** (gradient-descent starts, prediction rollouts of varying length).
- `@threads` is **simpler** and **better** for **even, deterministic** loops (eigenfunction grid evaluation, trajectory generation where each task does the same amount of work).

**Recommendation:** Keep `@spawn` for prediction paths and FNN. Adopt `@threads` for `evaluate_eigenfunction_grid/slice` and `generate_trajectories` because the work per iteration is uniform. This matches the `Mod vParallel2` design.

---

## Section C — Performance & Allocation Hotspots

### C.1 `rk4` — temporary vectors in hot loop (Systems.jl, lines 41–56)

**Code:**
```julia
for k in 1:n_internal
    xk = x[:, k]              # ← copies a column
    k1 = rhs(xk)              # ← allocates
    k2 = rhs(xk .+ 0.5 * dt * k1)   # ← allocates temp + result
    ...
    x[:, k+1] .= xk .+ (dt / 6) * (k1 .+ 2 .* (k2 .+ k3) .+ k4)
end
```

**Problems:**
1. `xk = x[:, k]` copies instead of viewing.
2. `xk .+ 0.5 * dt * k1` allocates an intermediate vector.
3. The final update expression creates **multiple** temporaries.

**Impact:** For a 2000-step RK4 with n=3, this allocates ~16 000 vectors per trajectory. With 10 000 trajectories, that's 160 M transient allocations — heavy GC pressure.

**Fix (allocation-free, bit-identical):**
```julia
function rk4(rhs, x0, dt, m; nLag=1)
    x0_vec = vec(x0)
    n = length(x0_vec)
    n_internal = nLag * m
    x = zeros(n, n_internal + 1)
    x[:, 1] .= x0_vec
    
    k1 = Vector{Float64}(undef, n)
    k2 = Vector{Float64}(undef, n)
    k3 = Vector{Float64}(undef, n)
    k4 = Vector{Float64}(undef, n)
    tmp = Vector{Float64}(undef, n)
    
    for k in 1:n_internal
        xk = @view x[:, k]
        rhs!(k1, xk)          # in-place RHS (or k1 .= rhs(xk) if closure)
        @. tmp = xk + 0.5 * dt * k1
        rhs!(k2, tmp)
        @. tmp = xk + 0.5 * dt * k2
        rhs!(k3, tmp)
        @. tmp = xk + dt * k3
        rhs!(k4, tmp)
        @. x[:, k+1] = xk + (dt / 6) * (k1 + 2*(k2 + k3) + k4)
    end
    return x[:, 1:nLag:end]
end
```

> **Note:** This requires an in-place RHS signature `rhs!(out, x)`. If the existing closure-based API must be preserved, hoisting temporaries and using `@.` still cuts allocations by ~80 % without changing the function signature.

---

### C.2 `euler_maruyama` — same allocation pattern (Systems.jl, lines 130–145)

**Problems:** `xk = x[:, k]` copies; `drift(xk)` and `diffusion(xk)` allocate; `dx_det` and `dx_stoch` allocate.

**Fix:** Hoist temporaries and use `@.` broadcasting:
```julia
xk = @view x[:, k]
dx_det .= drift(xk) .* dt          # or in-place drift!(dx_det, xk)
dx_stoch .= diffusion(xk) .* (sqrt_dt .* randn(n))
@. x[:, k+1] = xk + dx_det + dx_stoch
```

---

### C.3 `epileptor3d_drift` — triple `path_parameters` call (Systems.jl, lines 337–349)

**Code:**
```julia
xs = compute_xs(path_parameters(x[3], p.A, p.B, p.R)[1],
                path_parameters(x[3], p.A, p.B, p.R)[2])
mu2, mu1, nu = path_parameters(x[3], p.A, p.B, p.R)[1:3]
```

**Problem:** `path_parameters` is called **three times** per RHS evaluation. It contains `sqrt`, `acos`, `atan` — not cheap.

**Fix:**
```julia
mu2, mu1, nu, theta, phi = path_parameters(x[3], p.A, p.B, p.R)
xs = compute_xs(mu2, mu1)
```

This is exactly what `Mod vParallel2` did. The fix is **one line** and yields ~2–3× speedup for Epileptor integrations.

---

### C.4 `finite_difference_jacobian` — copies inside loop (Utils.jl, lines 96–108)

**Problem:** `xh = copy(x)` and `xh_m = copy(x)` on every column iteration.

**Fix:** Hoist one copy outside and mutate in-place:
```julia
xh = copy(x)
xh_m = copy(x)
for j in 1:n
    xh[j] = x[j] + h
    xh_m[j] = x[j] - h
    J[:, j] .= (f(xh) .- f(xh_m)) ./ (2h)
    xh[j] = x[j]
    xh_m[j] = x[j]
end
```

---

### C.5 `Psi_RFF` — bias broadcast and `vcat` allocation (Dictionaries.jl, lines 230–237)

**Problem:** `Z = basis.W * X .+ basis.b` allocates `Z`, then `Ψ = sqrt(2/D) .* cos.(Z)` allocates `Ψ`, then `vcat(X, Ψ)` allocates a third matrix if `include_states=true`.

**Fix (in-place):**
```julia
function Psi_RFF!(Ψ, X, basis)
    mul!(Ψ, basis.W, X)
    Ψ .+= basis.b
    Ψ .= sqrt(2.0 / basis.D) .* cos.(Ψ)
    return Ψ
end
```

For the `include_states=true` path, pre-allocate `Ψ = zeros(n + D, m)` and write directly into the bottom block.

---

### C.6 `evaluate_eigenfunction_grid` / `slice` — all-at-once Psi evaluation (Spectral.jl, lines 215–287)

**Problem:** The functions build **all** grid points into an `n × N` matrix (`N = prod(sizes)`) and call `Psi_func(pts)` once. For a 500×500 grid with nΨ=200, `Ψ_pts` is 200 × 250 000 = **400 MB**. The subsequent projection `Ξ[:, k]' * Ψ_pts` creates another large temporary.

**Fix:** Chunk the grid into column blocks and evaluate/project incrementally. This is exactly what `Mod vParallel2` did, achieving **92 % RAM reduction** on a 500×500 slice (from 120 MB extra down to ~9 MB).

---

### C.7 `spectrum` — dense threshold possibly too high (Config.jl, lines 287–337)

**Code:**
```julia
if n < 10000
    λ, Ξ = koopman_eigendecomposition(res.K)
```

**Problem:** Dense `eigvals`/`eigvecs` on a 5000×5000 real nonsymmetric matrix is ~O(n³) ≈ 125× slower than 1000×1000 and already stresses memory. The threshold of 10 000 may be too aggressive.

**Recommendation:** Lower the dense threshold to `n < 2000` or `n < 3000`, or make it user-configurable in `KoopmanConfig`.

---

## Section D — API & Usability Improvements

### D.1 Inconsistent keyword arguments

**Known bug (documented in AGENTS.md):**
- `examples/prediction_example.jl` calls `fig_prediction(...; dim=1, ...)`.
- `Figures.fig_prediction` signature uses `dim_labels` **not** `dim`.

**Other inconsistencies:**
- `fig_training_data` uses `dim::Int=2` and `projection_dims`.
- `fig_clusterized_data` uses the same pattern.
- `fig_prediction` uses `dim_labels`.

**Recommendation:** Align on `dim_labels` everywhere, or support `dim` as a positional fallback.

### D.2 Missing docstrings

The following **public or semi-public** functions lack docstrings in `src/`:

| Function | Module | Why it matters |
|---|---|---|
| `get_dim_psi` | Dictionaries | Used to size pre-allocations |
| `hermite_basis` | Dictionaries | Core API for Hermite dictionaries |
| `build_rff_basis` | Dictionaries | Needed to construct RFF dictionaries manually |
| `RFFBasis` | Dictionaries | Exported struct with no documentation |
| `_build_dict_and_project` | Hankel | Internal but complex; deserves a docstring |
| `select_svd_rank` | Hankel | Public utility for HAVOK rank selection |
| `delay_embed_training_data` | Hankel | Public utility for multi-channel embedding |
| `state_space_predict_from_psi` | DataGeneration | Public prediction API |

### D.3 Export list out of sync

**Already detailed in Section A.5.** Key action: decide on a single source of truth for exports.

### D.4 `KoopmanConfig` could be simplified/extended

**Current pain points:**

1. **No validation.** `edmd_method` accepts any Symbol; invalid values error deep inside `compute_koopman_operator`.
2. **Ambiguous defaults.** `m_embed=5000` is huge for a default; most notebooks use 10–100.
3. **Direct field access vs. `get()`.** `Config.hankel_analysis` accesses `cfg.dict_params.sigma` directly (line 135) but uses `get()` for `alpha` and `N_subsample`. This crashes if the user omits `:sigma` from `dict_params`.
4. **Missing fields for threading control.** No way to request `parallel=false` in `KoopmanConfig` for reproducibility.

**Recommendations:**
- Add a `validate!(cfg)` function that checks `edmd_method ∈ (:ridge, :pinv)`, `dict_type ∈ (:none, :hermite, :rbf, :rff, :kernel, :havok)`, etc.
- Use `get(cfg.dict_params, :sigma, 1.0)` consistently everywhere.
- Add `parallel::Bool = true` and `seed::Union{Nothing,Int} = nothing` fields.
- Reduce `m_embed` default to something conservative (e.g. 100) and let notebooks override.

### D.5 Missing systems in `DataGeneration`

**Finding:** `edmd_training_data` and `hankel_training_data` handle `FHN`, `Duffing`, `Epileptor3D`, `Lorenz`, `VanderPol` but **omit `Rossler`**.

**Fix:** Add `elseif system == "Rossler"` branches.

### D.6 `fixed_point` can error on Rossler with no real fixed points

**Finding:** `Systems.fixed_point` (line 668) calls `fps[1]` without checking emptiness. For `Rossler` with `disc < 0`, `find_fixed_points_rossler` returns `[]`, causing a `BoundsError`.

**Fix:**
```julia
isempty(fps) && error("No real fixed points found for system=$system with given parameters.")
return fps[1]
```

---

## Section E — Recommended Changes (prioritized)

### Critical (fixes bugs or crashes)

| # | File | Line(s) | Change | Expected benefit |
|---|---|---|---|---|
| 1 | `Systems.jl` | 668 | Add `isempty(fps)` guard before `fps[1]` in `fixed_point` | Prevents `BoundsError` for parameter regimes with no real fixed points |
| 2 | `DataGeneration.jl` | 28–40, 84–96 | Add `Rossler` branches to `edmd_training_data` and `hankel_training_data` | Completes API parity for all supported systems |
| 3 | `Config.jl` | 135 | Replace `cfg.dict_params.sigma` with `get(cfg.dict_params, :sigma, 1.0)` | Prevents `KeyError` when user omits `:sigma` |
| 4 | `Figures.jl` / `examples/prediction_example.jl` | 253, 76 | Rename `dim` kwarg in example to `dim_labels`, or add `dim` fallback in `fig_prediction` | Fixes known crash in example script |

### High (major performance gains, low risk)

| # | File | Line(s) | Change | Expected benefit |
|---|---|---|---|---|
| 5 | `Systems.jl` | 337–349 | Cache `path_parameters` result in `epileptor3d_drift` (single call instead of 3) | ~2–3× speedup for Epileptor3D integrations |
| 6 | `Systems.jl` | 41–56 | Hoist temporaries in `rk4`; replace `x[:, k]` copy with `@view` | ~20–40 % fewer allocations; less GC pressure |
| 7 | `Systems.jl` | 130–145 | Same allocation refactor for `euler_maruyama` | ~20–40 % fewer allocations |
| 8 | `Utils.jl` | 185–232 | Port `false_nearest_neighbors` parallelization + memory optimization from `Mod vParallel` | Near-linear speedup + ~5–20× memory reduction |
| 9 | `DataGeneration.jl` | — | Port trajectory/prediction parallelizations from `Mod vParallel` (`edmd_training_data`, `state_space_predictions`, `state_space_predict_from_psi`, `delay_space_predictions`) | Near-linear speedup over `n_trajectories` |
| 10 | `Hankel.jl` | 400–404, 542–564 | Port `havok_predict` and `delay_space_edmd_prediction` parallelization from `Mod vParallel` | Near-linear speedup over `n_traj` |
| 11 | `Config.jl` | 284–300 | Port `_kernel_delay_predict` parallelization from `Mod vParallel` | Near-linear speedup on heaviest predict path |

### Medium (code quality & maintainability)

| # | File | Line(s) | Change | Expected benefit |
|---|---|---|---|---|
| 12 | `Utils.jl` | — | Add `_thread_chunks` (or import from a single location) and remove duplicates from Dictionaries, EDMD, etc. | DRY; one place to tune chunking policy |
| 13 | `Systems.jl` | 181–558 | Introduce generic `make_diffusion` factory; collapse six nearly-identical diffusion functions | ~40 lines removed; easier to add new systems |
| 14 | `Hankel.jl` / `Config.jl` | 502–552, 180–226 | Unify dictionary-building logic into one callable used by both `_build_dict_and_project` and `state_analysis` | DRY; single source of truth for dictionary defaults |
| 15 | `Spectral.jl` | 215–287 | Port chunked eigenfunction evaluation from `Mod vParallel2` (or implement chunked `Psi_func` + fused projection) | −90 % transient RAM on large grids |
| 16 | `Config.jl` | 23–36 | Add `validate!(cfg::KoopmanConfig)` and thread-control fields (`parallel`, `seed`) | Better UX; catches config errors early |
| 17 | `KoopmanAnalysis.jl` | 84–132 | Audit and reconcile export list with sub-module exports | Consistent public API |

### Low (polish & documentation)

| # | File | Line(s) | Change | Expected benefit |
|---|---|---|---|---|
| 18 | `Dictionaries.jl` | 43–59 | Add docstrings to `get_dim_psi`, `hermite_basis`, `build_rff_basis`, `RFFBasis` | Better discoverability |
| 19 | `Config.jl` | 354 | Lower dense-eigensolve threshold from `n < 10000` to `n < 2000` or make configurable | Avoids accidental O(n³) slowdowns |
| 20 | `Utils.jl` | — | Add `configure_threads!(; blas_threads=1)` helper and mention in `__init__` docstring | Removes manual BLAS-tuning burden from users |

---

## Appendix — Quick-reference: what to port from `Mod vParallel`

If you want to bring `src/` up to the `Mod vParallel` level with minimal effort, copy these files verbatim (they are signature-compatible):

1. `Mod vParallel/Utils.jl` → `src/Utils.jl`  
   (adds `mutual_information_curve`, parallel `false_nearest_neighbors`, keeps existing `find_zls_gradient_descent`)
2. `Mod vParallel/DataGeneration.jl` → `src/DataGeneration.jl`  
   (parallel training/prediction, reproducible seeds)
3. `Mod vParallel/Hankel.jl` → `src/Hankel.jl`  
   (parallel `havok_predict`, `delay_space_edmd_prediction`)
4. `Mod vParallel/Config.jl` → `src/Config.jl`  
   (parallel `_kernel_delay_predict`)

Then apply the **Critical** and **High** fixes from Section E on top (especially the Epileptor RHS deduplication and RK4 allocation refactor).

---

*End of report.*
