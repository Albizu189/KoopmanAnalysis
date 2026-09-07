module Spectral

using LinearAlgebra
using Statistics

export koopman_eigendecomposition,
       detect_limit_cycle_modes, build_harmonic_branch, find_all_harmonic_branches, select_phase_amplitude_modes,
       evaluate_eigenfunction_grid, evaluate_eigenfunction_slice

# ---------------------------------------------------------------------------
# Basic eigendecomposition
# ---------------------------------------------------------------------------

"""
    koopman_eigendecomposition(K; sort_by=abs)

Compute eigenvalues and right eigenvectors of the Koopman matrix `K` and sort
them by `sort_by(λ)`.

Returns `(λ, Ξ)` with eigenvectors as columns of `Ξ`.
"""
function koopman_eigendecomposition(K::AbstractMatrix; sort_by=abs)
    λ = eigvals(K)
    Ξ = eigvecs(K)
    idx = reverse(sortperm(λ; by=sort_by))
    return λ[idx], Ξ[:, idx]
end

# ---------------------------------------------------------------------------
# Limit-cycle / oscillatory mode detection
# ---------------------------------------------------------------------------

"""
    detect_limit_cycle_modes(λ, dt; mag_tol=0.05, imag_tol=1e-6, exclude_unity=true)

Detect eigenvalues of `λ` that lie near the unit circle and have non-zero
imaginary part.  Returns a vector of `(index, λ)` tuples.
"""
function detect_limit_cycle_modes(λ, dt; mag_tol::Real=0.05, imag_tol::Real=1e-6,
                                  exclude_unity::Bool=true)
    candidates = Tuple{Int, Complex{Float64}}[]
    for (j, lj) in enumerate(λ)
        mag = abs(lj)
        is_on_circle = abs(mag - 1.0) <= mag_tol
        has_phase = abs(imag(lj)) > imag_tol
        is_not_unity = !(exclude_unity && abs(lj - 1.0) < 1e-8)
        if is_on_circle && has_phase && is_not_unity
            push!(candidates, (j, lj))
        end
    end
    return candidates
end

# ---------------------------------------------------------------------------
# Adaptive multi-branch harmonic extraction
# ---------------------------------------------------------------------------

"""
    build_harmonic_branch(λ, j1; TOL_HARMONIC=0.05, max_harmonic=5, tol_growth=1.5)

Build the harmonic family of eigenvalue `λ[j1]` with tolerance that grows
with harmonic order `k`:
    tol(k) = TOL_HARMONIC * tol_growth^(k-1)
"""
function build_harmonic_branch(λ::AbstractVector{<:Complex}, j1::Int;
                               TOL_HARMONIC::Real=0.05,
                               max_harmonic::Int=5,
                               tol_growth::Real=1.5)
    branch_idx = Int[j1]

    # Conjugate pair (fundamental, k=1)
    jc = findfirst(abs.(λ .- conj(λ[j1])) .< TOL_HARMONIC)
    !isnothing(jc) && push!(branch_idx, jc)

    # Integer harmonics k = 2..max_harmonic
    for k in 2:max_harmonic
        tol_k = TOL_HARMONIC * tol_growth^(k - 1)
        for target in (λ[j1]^k, conj(λ[j1])^k)
            d = abs.(λ .- target)
            j_match = argmin(d)
            if d[j_match] < tol_k
                push!(branch_idx, j_match)
            end
        end
    end

    return unique(branch_idx)
end


# ---------------------------------------------------------------------------
# Adaptive multi-branch harmonic extraction
# ---------------------------------------------------------------------------

"""
    find_all_harmonic_branches(λ::AbstractVector{<:Complex}, dt::Real; kwargs...)

Iteratively extract all independent harmonic families from the unit circle.

Returns `Dict{Int, Vector{Int}}` mapping branch number → original eigenvalue indices.
"""
function find_all_harmonic_branches(λ::AbstractVector{<:Complex}, dt::Real;
                                    mag_tol::Real=0.05,
                                    TOL_HARMONIC::Real=0.05,
                                    tol_growth::Real=1.5,
                                    branch_growth::Real=1.0,
                                    max_harmonic::Int=5)
    # Unit-circle candidates (excluding λ ≈ 1)
    on_circle = abs.(abs.(λ) .- 1.0) .<= mag_tol
    has_phase = abs.(imag.(λ)) .> 1e-6
    not_unity = abs.(λ .- 1.0) .> 1e-8
    candidates = findall(on_circle .&& has_phase .&& not_unity)

    branches = Dict{Int, Vector{Int}}()
    branch_count = 0
    current_base_tol = TOL_HARMONIC

    # Use select_phase_amplitude_modes to identify the dominant fundamental as branch #1
    try
        j_phase, _, _, _, branch0 = select_phase_amplitude_modes(λ, dt;
                                                                  TOL_HARMONIC=TOL_HARMONIC)
        if j_phase > 0 && !isempty(branch0)
            branch_count += 1
            branches[branch_count] = sort(branch0)
            candidates = setdiff(candidates, branch0)
            current_base_tol *= branch_growth
        end
    catch
        # no clear fundamental found; proceed from scratch
    end

    # Iteratively peel off remaining branches
    while !isempty(candidates)
        pos_mask = imag.(λ[candidates]) .> 0
        pos_cand = candidates[pos_mask]
        isempty(pos_cand) && break

        j1 = pos_cand[argmin(abs.(imag.(λ[pos_cand])))]

        branch_idx = build_harmonic_branch(λ, j1;
                                           TOL_HARMONIC=current_base_tol,
                                           max_harmonic=max_harmonic,
                                           tol_growth=tol_growth)

        branch_count += 1
        branches[branch_count] = sort(branch_idx)
        candidates = setdiff(candidates, branch_idx)
        current_base_tol *= branch_growth
    end

    return branches
end

"""
    select_phase_amplitude_modes(λ, τ; TOL_UNITY=1e-3, TOL_IMAG=1e-4, TOL_HARMONIC=0.05)

Select a fundamental phase eigenvalue (oscillatory, near unit circle) and a
real amplitude eigenvalue for phase/amplitude reconstruction.

Returns `(j_phase, mu_phase, j_amp, mu_amp)`.
"""
function select_phase_amplitude_modes(λ, τ; TOL_UNITY::Real=1e-3,
                                      TOL_IMAG::Real=1e-4,
                                      TOL_HARMONIC::Real=0.05, 
                                      tol_growth::Real=1.0)
    nontriv = findall(abs.(λ .- 1.0) .> TOL_UNITY)
    isempty(nontriv) && error("No non-trivial eigenvalues found.")

    dist_to_circle = abs.(abs.(λ[nontriv]) .- 1.0)
    has_imag = abs.(imag.(λ[nontriv])) .> TOL_IMAG
    close_to_1 = dist_to_circle .< TOL_UNITY
    cand_mask = has_imag .&& close_to_1
    complex_cand = nontriv[cand_mask]

    isempty(complex_cand) && error("No oscillatory eigenvalue near unit circle found.")
    sorted_cand = complex_cand[sortperm(abs.(imag.(λ[complex_cand])))]
    j1_pos = sorted_cand[imag.(λ[sorted_cand]) .> 0]
    isempty(j1_pos) && error("No oscillatory eigenvalue with positive imaginary part found.")
    j_phase = j1_pos[1]
    mu_phase = log(λ[j_phase]) / τ

    branch_idx = build_harmonic_branch(λ, j_phase;
                                       TOL_HARMONIC=TOL_HARMONIC,
                                       tol_growth=tol_growth)
    remaining = setdiff(1:length(λ), branch_idx)
    remaining = remaining[abs.(λ[remaining] .- 1.0) .> TOL_UNITY]
    real_rem = remaining[abs.(imag.(λ[remaining])) .< TOL_IMAG]
    if isempty(real_rem)
        @warn "No purely real eigenvalue found for amplitude; using closest-to-1 regardless."
        real_rem = remaining
    end
    j_amp = real_rem[argmax(real.(λ[real_rem]))]
    mu_amp = log(λ[j_amp]) / τ

    return j_phase, mu_phase, j_amp, mu_amp, branch_idx
end

# ---------------------------------------------------------------------------
# Eigenfunction evaluation
# ---------------------------------------------------------------------------

"""
    evaluate_eigenfunction_grid(Ξ, Psi_func, grids...)

Evaluate the first `n_modes` eigenfunctions on a Cartesian grid.

- `Ξ`: right eigenvector matrix (`nPsi × n_modes`).
- `Psi_func`: function that accepts an `n × N` matrix of grid points and returns
  an `nPsi × N` lifted matrix.
- `grids`: one `AbstractVector` per state dimension.

For high dimensions this becomes expensive; use `evaluate_eigenfunction_slice`
to evaluate on a 2-D slice instead.
"""
function evaluate_eigenfunction_grid(Ξ::AbstractMatrix, Psi_func::Function, grids::AbstractVector...;
                                     n_modes::Int=size(Ξ, 2))
    n = length(grids)
    sizes = length.(grids)
    n_modes = min(n_modes, size(Ξ, 2))
    φ = zeros(ComplexF64, sizes..., n_modes)

    # Build all grid points as columns of an n × N matrix.
    it = Iterators.product(grids...)
    N = prod(sizes)
    pts = zeros(n, N)
    for (i, x) in enumerate(it)
        pts[:, i] .= collect(x)
    end
    Ψ_pts = Psi_func(pts)

    # Reshape back to grid shape.
    for k in 1:n_modes
        vals = (Ξ[:, k]' * Ψ_pts)[:]
        φ[(Colon() for _ in 1:n)..., k] .= reshape(vals, sizes...)
    end
    return φ
end

"""
    evaluate_eigenfunction_slice(Ξ, Psi_func, grids...; slice_dims, slice_values, n_modes)

Evaluate eigenfunctions on a k-D slice of state space (k = 1, 2, 3, ...).

- `grids`: one `AbstractVector` per slice dimension.
- `slice_dims`: which state dimensions form the grid axes. Defaults to `(1, 2, ..., k)`.
- `slice_values`: values of the remaining state dimensions.
- `n_modes`: number of eigenfunctions to evaluate.

Returns an array of shape `(length(grids[1]), ..., length(grids[k]), n_modes)`.
"""
function evaluate_eigenfunction_slice(Ξ::AbstractMatrix, Psi_func::Function,
                                        grids::AbstractVector...;
                                        slice_dims::Tuple=ntuple(i->i, length(grids)),
                                        slice_values::AbstractVector=Float64[],
                                        n_modes::Int=size(Ξ, 2))
    n_dims = length(grids)
    @assert length(slice_dims) == n_dims "slice_dims must match number of grids"

    n_modes = min(n_modes, size(Ξ, 2))
    n = max(maximum(slice_dims), length(slice_values) + n_dims)

    sizes = length.(grids)
    N = prod(sizes)

    # Build all grid points as columns of an n × N matrix
    it = Iterators.product(grids...)
    pts = zeros(n, N)
    for (i, x) in enumerate(it)
        for (j, d) in enumerate(slice_dims)
            pts[d, i] = x[j]
        end
        other_dims = setdiff(1:n, slice_dims)
        for (k, d) in enumerate(other_dims)
            pts[d, i] = slice_values[k]
        end
    end

    Ψ_pts = Psi_func(pts)

    # Reshape back to grid shape
    φ = zeros(ComplexF64, sizes..., n_modes)
    for k in 1:n_modes
        vals = (Ξ[:, k]' * Ψ_pts)[:]
        φ[(Colon() for _ in 1:n_dims)..., k] .= reshape(vals, sizes...)
    end
    return φ
end

end # module
