module Systems

using LinearAlgebra
using Random
using Roots: find_zeros, find_zero

export euler_maruyama,
       rk4, generate_trajectories, detect_limit_cycle,
       fhn_rhs, fhn_drift, fhn_diffusion, fhn_parameters,
       duffing_rhs, duffing_drift, duffing_diffusion, duffing_parameters,
       epileptor3d_rhs, epileptor3d_drift, epileptor3d_diffusion, epileptor3d_parameters,
       path_parameters, compute_xs,
       lorenz_rhs, lorenz_drift, lorenz_diffusion, lorenz_parameters,
       vanderpol_rhs, vanderpol_drift, vanderpol_diffusion, vanderpol_parameters,
       rossler_rhs, rossler_drift, rossler_diffusion, rossler_parameters,
       find_fixed_points_fhn, find_fixed_points_duffing, find_fixed_points_lorenz,
       find_fixed_points_vanderpol, find_fixed_points_rossler,
       fixed_point, generate_test_trajectories

# ---------------------------------------------------------------------------
# Generic explicit RK4 integrator
# ---------------------------------------------------------------------------

"""
    rk4(rhs, x0, dt, m; nLag=1)

Integrate the ODE ``\\dot{x} = rhs(x)`` with a fixed-step explicit Runge–Kutta 4
scheme.

# Arguments
- `rhs`: function `x -> xdot` with `x` a vector of length `n`.
- `x0`: initial condition, either an `n`-vector or an `n×1` matrix.
- `dt`: time step.
- `m`: number of *output* snapshots requested.
- `nLag`: sub-sampling factor; the integrator takes `nLag` internal steps
  between each saved snapshot.

# Returns
An `n×(m+1)` matrix whose columns are `x(0), x(dt*nLag), ..., x(dt*nLag*m)`.
"""
function rk4(rhs, x0, dt, m; nLag=1)
    x0_vec = vec(x0)
    n = length(x0_vec)
    n_internal = nLag * m
    x = zeros(n, n_internal + 1)
    x[:, 1] .= x0_vec
    tmp = Vector{Float64}(undef, n)
    for k in 1:n_internal
        xk = @view x[:, k]
        k1 = rhs(xk)
        @. tmp = xk + 0.5 * dt * k1
        k2 = rhs(tmp)
        @. tmp = xk + 0.5 * dt * k2
        k3 = rhs(tmp)
        @. tmp = xk + dt * k3
        k4 = rhs(tmp)
        @. x[:, k+1] = xk + (dt / 6) * (k1 + 2*(k2 + k3) + k4)
    end
    return x[:, 1:nLag:end]
end

"""
    generate_trajectories(rhs, n_trajectories, dt, m; nLag=1, center=zeros(n), scale=1.0)

Generate `n_trajectories` initial conditions around `center` with the given
`scale` and integrate each one with `rk4`.

`center` may be supplied as a vector or an `n×1` matrix.  `scale` is either a
scalar or a vector of length `n`.
"""
function generate_trajectories(rhs, n_trajectories, dt, m; nLag=1, center=zeros(0), scale=1.0)
    # Use a single short trajectory to infer dimension if center is empty.
    if isempty(center)
        x0_test = randn(2)
        try
            _ = rhs(x0_test)
            center = zeros(length(x0_test))
        catch
            error("Could not infer state dimension. Please provide `center`.")
        end
    end
    c = vec(center)
    n = length(c)
    s = length(scale) == n ? scale : fill(float(scale), n)
    trajectories = Vector{Matrix{Float64}}(undef, n_trajectories)
    for i in 1:n_trajectories
        x0 = c .+ s .* (2 .* rand(n) .- 1)
        trajectories[i] = rk4(rhs, x0, dt, m; nLag=nLag)
    end
    return trajectories
end

"""
    detect_limit_cycle(rhs, dt, transient_steps, cycle_steps; nLag=1, x0=nothing)

Integrate a long trajectory, drop the transient, and return the tail.  The tail
is intended to approximate a limit cycle / bursting orbit.
"""
function detect_limit_cycle(rhs, dt, transient_steps, cycle_steps; nLag=1, x0=nothing)
    # Infer dimension
    if isnothing(x0)
        x0 = zeros(2)
        try
            _ = rhs(x0)
        catch
            error("Could not infer state dimension. Please provide `x0`.")
        end
    end
    total_steps = transient_steps + cycle_steps
    X = rk4(rhs, x0, dt, total_steps; nLag=nLag)
    return X[:, transient_steps:end]
end


# ---------------------------------------------------------------------------
# Stochastic integrator (Euler-Maruyama)
# ---------------------------------------------------------------------------

"""
    euler_maruyama(drift, diffusion, x0, dt, m; nLag=1, seed=nothing)

Integrate the SDE  dx = drift(x)dt + diffusion(x)dW  using the Euler-Maruyama
scheme with fixed step size `dt`.  `nLag` internal steps are taken between each
saved snapshot.

# Arguments
- `drift`:  deterministic vector field `x -> xdot`.
- `diffusion`: noise amplitude function `x -> g(x)` (same units as drift).
- `x0`: initial condition.
- `dt`: time step.
- `m`: number of output snapshots.
- `seed`: optional random seed for reproducibility.
"""
function euler_maruyama(drift, diffusion, x0, dt, m; nLag=1, seed=nothing)
    isnothing(seed) || Random.seed!(seed)
    x0_vec = vec(x0)
    n = length(x0_vec)
    total_steps = nLag * m
    x = zeros(n, total_steps + 1)
    x[:, 1] .= x0_vec
    sqrt_dt = sqrt(dt)
    dx_det = Vector{Float64}(undef, n)
    dx_stoch = Vector{Float64}(undef, n)
    dW = Vector{Float64}(undef, n)
    for k in 1:total_steps
        xk = @view x[:, k]
        dx_det .= drift(xk) .* dt
        randn!(dW)
        dx_stoch .= diffusion(xk) .* (sqrt_dt .* dW)
        @. x[:, k+1] = xk + dx_det + dx_stoch
    end
    return x[:, 1:nLag:end]
end


# ---------------------------------------------------------------------------
# FitzHugh–Nagumo
# ---------------------------------------------------------------------------

"""
    fhn_parameters(regime::String)

Return a named tuple of FHN parameters for the requested regime.
Supported regimes: `"stable-limit-cycle"`, `"stable-node"`, `"three-equilibrium-regime"`.
"""
function fhn_parameters(regime::String;
                        noise_type::Symbol=:none,
                        sigma::Real=0.0,
                        noise_mask::Union{Nothing,AbstractVector}=nothing)
    base = if regime == "stable-limit-cycle"
        (a=0.7, b=0.0, epsilon=0.08, I_ext=0.5)
    elseif regime == "stable-node"
        (a=0.7, b=0.8, epsilon=0.08, I_ext=-0.4)
    elseif regime == "three-equilibrium-regime"
        (a=0.7, b=0.8, epsilon=0.08, I_ext=0.32)
    else
        error("Unknown FHN regime: $regime")
    end
    mask = isnothing(noise_mask) ? [1.0, 0.0] : vec(float.(noise_mask))
    return merge(base, (noise_type=noise_type, sigma=float(sigma), noise_mask=mask))
end

function fhn_drift(p)
    rhs(x) = [x[1] - x[1]^3 / 3 - x[2] + p.I_ext;
              p.epsilon * (x[1] + p.a - p.b * x[2])]
    return rhs
end

function fhn_diffusion(p)
    n = length(p.noise_mask)
    function g(x)
        if p.noise_type == :none
            return zeros(n)
        elseif p.noise_type == :additive
            return p.sigma .* p.noise_mask
        elseif p.noise_type == :state_dependent
            return p.sigma .* abs(x[1]) .* p.noise_mask
        else
            error("Unknown noise_type: $(p.noise_type)")
        end
    end
    return g
end

# Backward-compatible alias
fhn_rhs(p) = fhn_drift(p)

# ---------------------------------------------------------------------------
# Duffing oscillator
# ---------------------------------------------------------------------------

"""
    duffing_parameters(regime::String)

Return a named tuple of Duffing parameters for the requested regime.
Supported regimes: `"stable-node"`, `"three-equilibrium-attractors"`,
`"three-equilibrium-centers"`.
"""
function duffing_parameters(regime::String;
                            noise_type::Symbol=:none,
                            sigma::Real=0.0,
                            noise_mask::Union{Nothing,AbstractVector}=nothing)
    base = if regime == "stable-node"
        (delta=0.3, beta=1.0, alpha=-1.0)
    elseif regime == "three-equilibrium-attractors"
        (delta=0.2, beta=-1.0, alpha=1.0)
    elseif regime == "three-equilibrium-centers"
        (delta=0.0, beta=-1.0, alpha=1.0)
    else
        error("Unknown Duffing regime: $regime")
    end
    mask = isnothing(noise_mask) ? [1.0, 0.0] : vec(float.(noise_mask))
    return merge(base, (noise_type=noise_type, sigma=float(sigma), noise_mask=mask))
end

"""
    duffing_rhs(p)

Return the Duffing vector field as a closure `x -> xdot` for the parameter tuple
`p`.
"""
function duffing_drift(p)
    rhs(x) = [x[2];
              -p.delta * x[2] - p.beta * x[1] - p.alpha * x[1]^3]
    return rhs
end

function duffing_diffusion(p)
    n = length(p.noise_mask)
    function g(x)
        if p.noise_type == :none
            return zeros(n)
        elseif p.noise_type == :additive
            return p.sigma .* p.noise_mask
        elseif p.noise_type == :state_dependent
            return p.sigma .* abs(x[1]) .* p.noise_mask
        else
            error("Unknown noise_type: $(p.noise_type)")
        end
    end
    return g
end

# Backward-compatible alias
duffing_rhs(p) = duffing_drift(p)

# ---------------------------------------------------------------------------
# 3D Epileptor
# ---------------------------------------------------------------------------

"""
    path_parameters(z, A, B, R)

Great-circle parametrisation of the unfolding path on the sphere used by the
3D Epileptor.  Returns `(mu2, mu1, nu, theta, phi)`.
"""
function path_parameters(z, A, B, R)
    mu2_A, neg_mu1_A, nu_A = A
    mu2_B, neg_mu1_B, nu_B = B

    mu2_line = mu2_A + (mu2_B - mu2_A) * z
    neg_mu1_line = neg_mu1_A + (neg_mu1_B - neg_mu1_A) * z
    nu_line = nu_A + (nu_B - nu_A) * z

    r_line = sqrt(mu2_line^2 + neg_mu1_line^2 + nu_line^2)
    if r_line > 1e-12
        scale = R / r_line
        mu2_sph = mu2_line * scale
        neg_mu1_sph = neg_mu1_line * scale
        nu_sph = nu_line * scale
    else
        mu2_sph = mu2_line
        neg_mu1_sph = neg_mu1_line
        nu_sph = nu_line
    end

    mu1_sph = -neg_mu1_sph
    theta = acos(clamp(nu_sph / R, -1.0, 1.0))
    phi = atan(neg_mu1_sph, mu2_sph)

    return mu2_sph, mu1_sph, nu_sph, theta, phi
end

"""
    compute_xs(mu2, mu1)

Upper-branch fixed point of the fast cubic subsystem `x^3 - mu2*x - mu1 = 0`.
"""
function compute_xs(mu2, mu1)
    f(x) = x^3 - mu2*x - mu1
    roots = find_zeros(f, -5.0, 5.0)
    if isempty(roots)
        root = find_zero(f, 2.0)
        return root
    end
    return maximum(roots)
end

function epileptor3d_parameters(regime::String;
                                 noise_type::Symbol=:none,
                                 sigma::Real=0.0,
                                 noise_mask::Union{Nothing,AbstractVector}=nothing)
    regimes = Dict(
        "c2s-SN-SH"   => (A=[0.0, 0.0, -0.5],       B=[0.2400, 0.0900, -0.3400], R=0.4, c=0.01, d_star=0.3, tau0=1.0),
        "c3s-SN-supH" => (A=[0.0, 0.0, -0.5],       B=[0.2636, 0.0909, -0.3182], R=0.4, c=0.01, d_star=0.3, tau0=1.0),
        "c10s-supH-SH"=> (A=[0.0727, 0.0, -0.4273], B=[0.3273, 0.0, -0.2273],    R=0.4, c=0.01, d_star=0.3, tau0=1.0),
        "c11s-supH-supH"=>(A=[0.1091, 0.0, -0.3909],B=[0.3636, 0.0, -0.1909],    R=0.4, c=0.01, d_star=0.3, tau0=1.0),
        "c2b-SN-SH"   => (A=[0.0, 0.0, -0.5],       B=[0.1600, 0.1000, -0.3800], R=0.4, c=0.02, d_star=0.3, tau0=1.0),
        "c4b-SN-FLC"  => (A=[0.1871, -0.0251, -0.3526], B=[0.3072, 0.0655, -0.2476], R=0.4, c=0.02, d_star=0.3, tau0=1.0),
        "c14b-subH-SH"=> (A=[0.0873, -0.0524, -0.4413], B=[0.2400, 0.0300, -0.3300], R=0.4, c=0.02, d_star=0.3, tau0=1.0),
        "c16b-subH-FLC"=>(A=[0.1091, -0.0300, -0.4182], B=[0.2727, 0.0500, -0.3000], R=0.4, c=0.02, d_star=0.3, tau0=1.0),
    )
    haskey(regimes, regime) || error("Unknown Epileptor regime: $regime")
    base = regimes[regime]
    mask = isnothing(noise_mask) ? [1.0, 0.0, 0.0] : vec(float.(noise_mask))
    return merge(base, (noise_type=noise_type, sigma=float(sigma), noise_mask=mask))
end

"""
    epileptor3d_rhs(p)

Return the 3D Epileptor vector field as a closure `x -> xdot` for the parameter
tuple `p`.
"""
function epileptor3d_drift(p)
    function f(x)
        xs = compute_xs(path_parameters(x[3], p.A, p.B, p.R)[1],
                        path_parameters(x[3], p.A, p.B, p.R)[2])
        mu2, mu1, nu = path_parameters(x[3], p.A, p.B, p.R)[1:3]

        dx = -x[2]
        dy = x[1]^3 - mu2 * x[1] - mu1 - x[2] * (nu + x[1] + x[1]^2)
        dz = -p.c * (sqrt((x[1] - xs)^2 + x[2]^2) - p.d_star)
        return [dx; dy; dz]
    end
    return f
end

function epileptor3d_diffusion(p)
    n = length(p.noise_mask)
    function g(x)
        if p.noise_type == :none
            return zeros(n)
        elseif p.noise_type == :additive
            return p.sigma .* p.noise_mask
        elseif p.noise_type == :state_dependent
            mu2, mu1 = path_parameters(x[3], p.A, p.B, p.R)[1:2]
            xs = compute_xs(mu2, mu1)
            return p.sigma .* abs(x[1] - xs) .* p.noise_mask
        else
            error("Unknown noise_type: $(p.noise_type)")
        end
    end
    return g
end

epileptor3d_rhs(p) = epileptor3d_drift(p)

# ---------------------------------------------------------------------------
# Lorenz system
# ---------------------------------------------------------------------------

"""
    lorenz_parameters(regime::String)

Return a named tuple of Lorenz parameters for the requested regime.
Supported regimes: `"chaotic"` (standard parameters from Brunton et al. 2017,
arXiv:1608.05306 / Nat. Commun. 8, 2017).
"""
function lorenz_parameters(regime::String;
                            noise_type::Symbol=:none,
                            sigma::Real=0.0,
                            noise_mask::Union{Nothing,AbstractVector}=nothing)
    base = if regime == "chaotic"
        (σ=10.0, ρ=28.0, β=8.0/3.0)
    else
        error("Unknown Lorenz regime: $regime")
    end
    mask = isnothing(noise_mask) ? [1.0, 1.0, 1.0] : vec(float.(noise_mask))
    return merge(base, (noise_type=noise_type, sigma=float(sigma), noise_mask=mask))
end

function lorenz_drift(p)
    rhs(x) = [p.σ * (x[2] - x[1]);
              x[1] * (p.ρ - x[3]) - x[2];
              x[1] * x[2] - p.β * x[3]]
    return rhs
end

function lorenz_diffusion(p)
    n = length(p.noise_mask)
    function g(x)
        if p.noise_type == :none
            return zeros(n)
        elseif p.noise_type == :additive
            return p.sigma .* p.noise_mask
        elseif p.noise_type == :state_dependent
            return p.sigma .* abs.(x) .* p.noise_mask
        else
            error("Unknown noise_type: $(p.noise_type)")
        end
    end
    return g
end

# Backward-compatible alias
lorenz_rhs(p) = lorenz_drift(p)

# ---------------------------------------------------------------------------
# Van der Pol oscillator (third-order / jerk form)
# ---------------------------------------------------------------------------
#
#   ẋ = y
#   ẏ = z
#   ż = μ(1 - x²)z - x - y
#
# Equivalently x''' - μ(1 - x²)x'' + x' + x = 0.  The origin is the unique
# fixed point; its characteristic polynomial is λ³ - μλ² + λ + 1 = 0, so by
# Routh-Hurwitz it is asymptotically stable iff μ < -1 (a Hopf pair crosses
# the imaginary axis at μ = -1).  Distinct regimes identified numerically:
#
#   * μ < -1        : stable focus (damped spirals into the origin).  The
#                     basin is bounded — keep initial conditions well inside
#                     the unit ball (|x0| ≲ 0.5); large ICs blow up in finite
#                     time because μ(1 - x²)z pumps energy for |x| > 1.
#   * -1 < μ ≲ 0    : origin is a saddle-focus and generic trajectories
#                     diverge (no bounded attractor).
#   * 0 < μ ≲ 0.5   : large-amplitude relaxation limit cycle born in a fold
#                     of cycles near μ = 0 (period grows from ≈ 11 at μ = 0.2
#                     to ≈ 50 at μ = 0.4).  Explicit RK4 with dt = 0.01 is
#                     adequate here.
#   * 0.5 ≲ μ ≲ 0.65: period-doubling cascade into a weakly chaotic attractor
#                     (λ₁ ≈ +0.07 at μ = 0.58, near-1D return map of maxima).
#                     Stiff during the large excursions — use dt ≲ 5e-4 with
#                     explicit RK4 (e.g. dt = 1e-4, or nLag sub-stepping).
#   * μ ≳ 0.7       : boundary crisis — the attractor is destroyed and
#                     trajectories diverge.

"""
    vanderpol_parameters(regime::String)

Return a named tuple of parameters for the third-order Van der Pol oscillator
`ẋ = y, ẏ = z, ż = μ(1 - x²)z - x - y`.

Supported regimes: `"stable-focus"` (μ = -2.0), `"stable-limit-cycle"`
(μ = 0.3), `"chaotic"` (μ = 0.58).  See the module comment above for the
stability structure and integration caveats (bounded basin for μ < -1;
small `dt` required in the chaotic regime).
"""
function vanderpol_parameters(regime::String;
                              noise_type::Symbol=:none,
                              sigma::Real=0.0,
                              noise_mask::Union{Nothing,AbstractVector}=nothing)
    base = if regime == "stable-focus"
        (mu=-2.0,)
    elseif regime == "stable-limit-cycle"
        (mu=0.3,)
    elseif regime == "chaotic"
        (mu=0.58,)
    else
        error("Unknown Van der Pol regime: $regime")
    end
    mask = isnothing(noise_mask) ? [1.0, 0.0, 0.0] : vec(float.(noise_mask))
    return merge(base, (noise_type=noise_type, sigma=float(sigma), noise_mask=mask))
end

"""
    vanderpol_rhs(p)

Return the third-order Van der Pol vector field as a closure `x -> xdot` for
the parameter tuple `p`.
"""
function vanderpol_drift(p)
    rhs(x) = [x[2];
              x[3];
              p.mu * (1.0 - x[1]^2) * x[3] - x[1] - x[2]]
    return rhs
end

function vanderpol_diffusion(p)
    n = length(p.noise_mask)
    function g(x)
        if p.noise_type == :none
            return zeros(n)
        elseif p.noise_type == :additive
            return p.sigma .* p.noise_mask
        elseif p.noise_type == :state_dependent
            return p.sigma .* abs(x[1]) .* p.noise_mask
        else
            error("Unknown noise_type: $(p.noise_type)")
        end
    end
    return g
end

# Backward-compatible alias
vanderpol_rhs(p) = vanderpol_drift(p)

# ---------------------------------------------------------------------------
# Rössler system
# ---------------------------------------------------------------------------

"""
    rossler_parameters(regime::String)

Return a named tuple of Rössler parameters for the requested regime.
Supported regimes: `"stable-limit-cycle"` (c=2.5, periodic),
`"chaotic"` (c=5.7, classic chaotic attractor).
"""
function rossler_parameters(regime::String;
                            noise_type::Symbol=:none,
                            sigma::Real=0.0,
                            noise_mask::Union{Nothing,AbstractVector}=nothing)
    base = if regime == "stable-limit-cycle"
        (a=0.2, b=0.2, c=2.5)
    elseif regime == "chaotic"
        (a=0.2, b=0.2, c=5.7)
    else
        error("Unknown Rössler regime: $regime")
    end
    mask = isnothing(noise_mask) ? [1.0, 1.0, 1.0] : vec(float.(noise_mask))
    return merge(base, (noise_type=noise_type, sigma=float(sigma), noise_mask=mask))
end

function rossler_drift(p)
    rhs(x) = [-x[2] - x[3];
               x[1] + p.a * x[2];
               p.b + x[3] * (x[1] - p.c)]
    return rhs
end

function rossler_diffusion(p)
    n = length(p.noise_mask)
    function g(x)
        if p.noise_type == :none
            return zeros(n)
        elseif p.noise_type == :additive
            return p.sigma .* p.noise_mask
        elseif p.noise_type == :state_dependent
            return p.sigma .* abs.(x) .* p.noise_mask
        else
            error("Unknown noise_type: $(p.noise_type)")
        end
    end
    return g
end

# Backward-compatible alias
rossler_rhs(p) = rossler_drift(p)

# ---------------------------------------------------------------------------
# Fixed-point helpers
# ---------------------------------------------------------------------------

"""
    find_fixed_points_fhn(p)

Return a list of fixed points `[v, w]` for the FHN model with parameters `p`.
"""
function find_fixed_points_fhn(p)
    if p.b == 0
        v_fixed = -p.a
        w_fixed = v_fixed - v_fixed^3 / 3 + p.I_ext
        return [[v_fixed, w_fixed]]
    else
        c1 = -3 * (1 - 1 / p.b)
        c0 = 3 * (p.a / p.b - p.I_ext)
        f(v) = v^3 + c1 * v + c0
        roots = find_zeros(f, -3.0, 3.0)
        fps = Vector{Float64}[]
        for v in roots
            w = (v + p.a) / p.b
            push!(fps, [v, w])
        end
        return fps
    end
end

"""
    find_fixed_points_duffing(p)

Return a list of fixed points `[x1, x2]` for the Duffing oscillator with
parameters `p`.
"""
function find_fixed_points_duffing(p)
    fps = Vector{Float64}[[0.0, 0.0]]
    if p.beta * p.alpha < 0
        x1 = sqrt(-p.beta / p.alpha)
        push!(fps, [x1, 0.0])
        push!(fps, [-x1, 0.0])
    end
    return fps
end

"""
    find_fixed_points_lorenz(p)

Return the list of fixed points `[x, y, z]` for the Lorenz system with parameters `p`.
"""
function find_fixed_points_lorenz(p)
    fps = Vector{Float64}[]
    push!(fps, [0.0, 0.0, 0.0])          # origin
    if p.ρ > 1.0
        xyz = sqrt(p.β * (p.ρ - 1.0))
        z   = p.ρ - 1.0
        push!(fps, [ xyz,  xyz, z])
        push!(fps, [-xyz, -xyz, z])
    end
    return fps
end

"""
    find_fixed_points_vanderpol(p)

Return the unique fixed point `[x, y, z]` of the third-order Van der Pol
system.  `y = 0, z = 0 ⇒ x = 0`, so the origin is the only equilibrium for
every value of μ.
"""
function find_fixed_points_vanderpol(p)
    return [[0.0, 0.0, 0.0]]
end


"""
    find_fixed_points_rossler(p)

Return the list of fixed points [x, y, z] for the Rössler system.
Solves the quadratic a·y² + c·y + b = 0 arising from the fixed-point equations.
"""
function find_fixed_points_rossler(p)
    fps = Vector{Float64}[]
    disc = p.c^2 - 4 * p.a * p.b
    if disc < 0
        # Complex roots — no real fixed points (not expected for standard params)
        return fps
    end
    sqrt_disc = sqrt(disc)
    y_vals = [(-p.c + sqrt_disc) / (2 * p.a),
              (-p.c - sqrt_disc) / (2 * p.a)]
    for y in y_vals
        x = -p.a * y
        z = -y
        push!(fps, [x, y, z])
    end
    return fps
end

"""
    fixed_point(system::String, p)

Return a single representative fixed point for the requested `system`.
For FHN/Duffing this is the first fixed point found; for Epileptor3D it is the
silent state computed from the fast subsystem at `z = 0`; for VanderPol it is
the origin.
"""
function fixed_point(system::String, p)
    if system == "FHN"
        fps = find_fixed_points_fhn(p)
    elseif system == "Duffing"
        fps = find_fixed_points_duffing(p)
    elseif system == "Epileptor3D"
        mu2, mu1 = path_parameters(0.0, p.A, p.B, p.R)[1:2]
        xs = compute_xs(mu2, mu1)
        return [xs, 0.0, 0.0]
    elseif system == "Lorenz"
        fps = find_fixed_points_lorenz(p)
    elseif system == "VanderPol"
        fps = find_fixed_points_vanderpol(p)
    elseif system == "Rossler"
        fps = find_fixed_points_rossler(p)
    else
        error("fixed_point not implemented for system: $system")
    end
    return fps[1]
end

# ---------------------------------------------------------------------------
# Training / test data builders
# ---------------------------------------------------------------------------

"""
    generate_test_trajectories(rhs, n_trajectories::Int, dt::Real, top_pred_step::Int;
                               nLag::Int=1,
                               center::Union{Nothing,AbstractVector}=nothing,
                               window::Real=1.0)

Generate `n_trajectories` test trajectories of length `top_pred_step + 1`.

# Returns
`(X_test, X_init)` where `X_test` is `n × (top_pred_step+1) × n_trajectories` and
`X_init` is `n × n_trajectories`.
"""
function generate_test_trajectories(rhs, n_trajectories::Int, dt::Real, top_pred_step::Int;
                                    nLag::Int=1,
                                    center::Union{Nothing,AbstractVector}=nothing,
                                    window::Real=1.0)
    # Infer dimension
    if isnothing(center)
        x0_test = zeros(2)
        try
            _ = rhs(x0_test)
            center = zeros(2)
        catch
            error("Could not infer state dimension. Please provide `center`.")
        end
    end
    c = vec(center)
    n = length(c)

    X_test = zeros(n, top_pred_step + 1, n_trajectories)
    X_init = zeros(n, n_trajectories)

    for i in 1:n_trajectories
        x0 = c .+ window .* (2 .* rand(n) .- 1)
        X_init[:, i] .= x0
        X_test[:, :, i] .= rk4(rhs, x0, dt, top_pred_step; nLag=nLag)
    end

    return X_test, X_init
end

end # module
