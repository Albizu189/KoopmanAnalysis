using Test
using KoopmanAnalysis
using LinearAlgebra
using Statistics
using Random
using CairoMakie
using KoopmanAnalysis
using LinearAlgebra
using Statistics
using Random

# ============================================================================
# Global test helpers
# ============================================================================
const SEED = 1234

function seeded_rand(args...)
    Random.seed!(SEED)
    return rand(args...)
end

function seeded_randn(args...)
    Random.seed!(SEED)
    return randn(args...)
end

# Small reproducible deterministic trajectory for 2D tests
function tiny_trajectory_2d()
    p = fhn_parameters("stable-node")
    rhs = fhn_rhs(p)
    return rk4(rhs, [0.1, 0.0], 0.01, 30)
end

# Small reproducible deterministic trajectory for 3D tests
function tiny_trajectory_3d()
    p = lorenz_parameters("chaotic")
    rhs = lorenz_rhs(p)
    return rk4(rhs, [1.0, 1.0, 1.0], 0.01, 30)
end

# ============================================================================
# Utils
# ============================================================================
@testset "Utils" begin
    # --- normalize_vector ---
    @testset "normalize_vector" begin
        v = [1.0, 2.0, 3.0, 4.0, 5.0]
        nv = normalize_vector(v)
        @test length(nv) == length(v)
        @test minimum(nv) >= 0.0
        @test maximum(nv) <= 1.0
        @test isapprox(minimum(nv), 0.35 / 1.35, atol=1e-10)
        @test isapprox(maximum(nv), 1.0, atol=1e-10)

        # Constant vector
        cv = [2.0, 2.0, 2.0]
        ncv = normalize_vector(cv)
        @test all(ncv .≈ 0.35 / 1.35)
    end

    # --- signed_area ---
    @testset "signed_area" begin
        # Unit square: (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
        x = [0.0, 1.0, 1.0, 0.0]
        y = [0.0, 0.0, 1.0, 1.0]
        @test isapprox(signed_area(x, y), 1.0, atol=1e-10)

        # Same square reversed orientation
        xr = [0.0, 0.0, 1.0, 1.0]
        yr = [0.0, 1.0, 1.0, 0.0]
        @test isapprox(signed_area(xr, yr), -1.0, atol=1e-10)

        @test_throws DimensionMismatch signed_area([1.0, 2.0], [1.0])
    end

    # --- meshgrid_2d ---
    @testset "meshgrid_2d" begin
        xr = 1:3
        yr = 4:5
        X, Y = meshgrid_2d(xr, yr)
        @test size(X) == (2, 3)
        @test size(Y) == (2, 3)
        @test X[1, :] ≈ [1, 2, 3]
        @test X[2, :] ≈ [1, 2, 3]
        @test Y[:, 1] ≈ [4, 5]
        @test Y[:, 2] ≈ [4, 5]
    end

    # --- classify_fixed_point_2d ---
    @testset "classify_fixed_point_2d" begin
        @test classify_fixed_point_2d([-1.0, -2.0]) == "stable node"
        @test classify_fixed_point_2d([1.0, 2.0]) == "unstable node"
        @test classify_fixed_point_2d([-1.0, 1.0]) == "saddle"
        @test classify_fixed_point_2d([-1.0 + 0.5im, -1.0 - 0.5im]) == "stable spiral/focus"
        @test classify_fixed_point_2d([0.5 + 0.5im, 0.5 - 0.5im]) == "unstable spiral/focus"
        @test classify_fixed_point_2d([0.0 + 1.0im, 0.0 - 1.0im]) == "center"
        @test classify_fixed_point_2d([-1.0, -1.0]) == "stable node (star/degenerate)"
    end

    # --- finite_difference_jacobian ---
    @testset "finite_difference_jacobian" begin
        f(x) = [x[1]^2 + x[2]; x[1] * x[2]]
        x0 = [2.0, 3.0]
        J = finite_difference_jacobian(f, x0)
        @test size(J) == (2, 2)
        # Analytic: [2x1  1; x2  x1]
        @test isapprox(J[1, 1], 4.0, atol=1e-4)
        @test isapprox(J[1, 2], 1.0, atol=1e-4)
        @test isapprox(J[2, 1], 3.0, atol=1e-4)
        @test isapprox(J[2, 2], 2.0, atol=1e-4)
    end

    # --- mutual_information ---
    @testset "mutual_information" begin
        x = sin.(0:0.1:10)
        mi0 = mutual_information(x, 0)
        @test isnan(mi0)
        mi1 = mutual_information(x, 1)
        @test mi1 >= 0.0 || isnan(mi1)
        # Constant signal
        @test mutual_information(ones(10), 1) == 0.0
    end

    # --- find_first_minimum ---
    @testset "find_first_minimum" begin
        @test find_first_minimum([3.0, 2.0, 1.0, 2.0, 3.0]) == 3
        @test find_first_minimum([1.0, 2.0, 3.0]) == 1  # global min, no local min
    end

    # --- false_nearest_neighbors ---
    @testset "false_nearest_neighbors" begin
        x = sin.(0:0.1:20)
        fnn = false_nearest_neighbors(x, 2, 1)
        @test 0.0 <= fnn <= 1.0
    end

    # --- grad_Psi_RBF ---
    @testset "grad_Psi_RBF" begin
        centers = randn(2, 3)
        s = [0.5, -0.5]
        J = grad_Psi_RBF(s, centers; include_states=true)
        @test size(J, 1) == 2
        @test size(J, 2) == 1 + 2 + 3  # constant + states + RBFs

        J2 = grad_Psi_RBF(s, centers; include_states=false)
        @test size(J2, 2) == 3
    end

    # --- grad_phi ---
    @testset "grad_phi" begin
        centers = randn(2, 3)
        dict_info = (centers=centers, include_states=true, state_indices=[1, 2])
        s = [0.5, -0.5]
        ξ = ones(6)
        g = grad_phi(s, ξ, dict_info)
        @test length(g) == 2
    end

    # --- find_zls_gradient_descent (smoke test) ---
    @testset "find_zls_gradient_descent" begin
        S = randn(2, 20)
        centers = randn(2, 3)
        Psi_func(X) = Psi_RBF(X, centers; include_states=true)
        Ξ = randn(6, 4)
        dict_info = (centers=centers, include_states=true, state_indices=[1, 2])
        pts = find_zls_gradient_descent(S, Psi_func, Ξ, 1, dict_info;
                                        n_starts=5, n_iter=5, verbose=false)
        @test pts isa Vector{Vector{Float64}}
    end
end

# ============================================================================
# Systems
# ============================================================================
@testset "Systems" begin
    # --- rk4 ---
    @testset "rk4 preserves state dimension" begin
        p = fhn_parameters("stable-node")
        rhs = fhn_rhs(p)
        x0 = [0.1, 0.0]
        dt = 0.01
        m = 20
        X = rk4(rhs, x0, dt, m)
        @test size(X, 1) == 2
        @test size(X, 2) == m + 1

        # With nLag
        X2 = rk4(rhs, x0, dt, m; nLag=2)
        @test size(X2, 1) == 2
        @test size(X2, 2) == m + 1

        # Matrix initial condition
        X3 = rk4(rhs, reshape(x0, :, 1), dt, m)
        @test size(X3, 1) == 2
        @test size(X3, 2) == m + 1
    end

    # --- generate_trajectories ---
    @testset "generate_trajectories" begin
        p = fhn_parameters("stable-node")
        rhs = fhn_rhs(p)
        trajs = generate_trajectories(rhs, 2, 0.01, 10; center=[0.0, 0.0], scale=0.5)
        @test length(trajs) == 2
        @test all(size(t, 1) == 2 for t in trajs)
        @test all(size(t, 2) == 11 for t in trajs)
    end

    # --- detect_limit_cycle ---
    @testset "detect_limit_cycle" begin
        p = fhn_parameters("stable-limit-cycle")
        rhs = fhn_rhs(p)
        X = detect_limit_cycle(rhs, 0.01, 10, 20)
        @test size(X, 1) == 2
        @test size(X, 2) == 22  # cycle_steps + 2 (rk4 returns m+1 cols, slice from transient_steps)
    end

    # --- euler_maruyama regression ---
    @testset "euler_maruyama reproducibility" begin
        p = fhn_parameters("stable-node"; noise_type=:additive, sigma=0.1)
        drift = fhn_drift(p)
        diffu = fhn_diffusion(p)
        x0 = [0.1, 0.0]
        dt = 0.01
        m = 10

        X1 = euler_maruyama(drift, diffu, x0, dt, m; seed=SEED)
        X2 = euler_maruyama(drift, diffu, x0, dt, m; seed=SEED)
        @test X1 ≈ X2

        @test size(X1, 1) == 2
        @test size(X1, 2) == m + 1
    end

    # --- parameter smoke tests ---
    @testset "parameter functions" begin
        @test_nowarn fhn_parameters("stable-limit-cycle")
        @test_nowarn fhn_parameters("stable-node")
        @test_nowarn fhn_parameters("three-equilibrium-regime")

        @test_nowarn duffing_parameters("stable-node")
        @test_nowarn duffing_parameters("three-equilibrium-attractors")
        @test_nowarn duffing_parameters("three-equilibrium-centers")

        @test_nowarn epileptor3d_parameters("c2s-SN-SH")
        @test_nowarn epileptor3d_parameters("c4b-SN-FLC")

        @test_nowarn lorenz_parameters("chaotic")

        @test_nowarn vanderpol_parameters("stable-focus")
        @test_nowarn vanderpol_parameters("stable-limit-cycle")
        @test_nowarn vanderpol_parameters("chaotic")

        @test_nowarn rossler_parameters("stable-limit-cycle")
        @test_nowarn rossler_parameters("chaotic")
    end

    # --- drift/diffusion smoke tests ---
    @testset "drift and diffusion closures" begin
        p_fhn = fhn_parameters("stable-node")
        @test_nowarn fhn_drift(p_fhn)
        @test_nowarn fhn_diffusion(p_fhn)
        @test fhn_rhs(p_fhn) isa Function

        p_duff = duffing_parameters("stable-node")
        @test_nowarn duffing_drift(p_duff)
        @test_nowarn duffing_diffusion(p_duff)

        p_epi = epileptor3d_parameters("c2s-SN-SH")
        @test_nowarn epileptor3d_drift(p_epi)
        @test_nowarn epileptor3d_diffusion(p_epi)

        p_lor = lorenz_parameters("chaotic")
        @test_nowarn lorenz_drift(p_lor)
        @test_nowarn lorenz_diffusion(p_lor)

        p_vdp = vanderpol_parameters("stable-focus")
        @test_nowarn vanderpol_drift(p_vdp)
        @test_nowarn vanderpol_diffusion(p_vdp)

        p_ros = rossler_parameters("chaotic")
        @test_nowarn rossler_drift(p_ros)
        @test_nowarn rossler_diffusion(p_ros)
    end

    # --- fixed point helpers ---
    @testset "fixed point helpers" begin
        p_fhn = fhn_parameters("stable-node")
        fps = find_fixed_points_fhn(p_fhn)
        @test fps isa Vector{Vector{Float64}}
        @test length(fps) >= 1

        p_duff = duffing_parameters("three-equilibrium-attractors")
        fps_d = find_fixed_points_duffing(p_duff)
        @test length(fps_d) == 3

        p_lor = lorenz_parameters("chaotic")
        fps_l = find_fixed_points_lorenz(p_lor)
        @test length(fps_l) == 3

        p_vdp = vanderpol_parameters("stable-focus")
        fps_v = find_fixed_points_vanderpol(p_vdp)
        @test fps_v == [[0.0, 0.0, 0.0]]

        p_ros = rossler_parameters("chaotic")
        fps_r = find_fixed_points_rossler(p_ros)
        @test fps_r isa Vector{Vector{Float64}}

        # fixed_point wrapper
        @test fixed_point("FHN", p_fhn) isa Vector{Float64}
        @test fixed_point("Duffing", p_duff) isa Vector{Float64}
        @test fixed_point("Lorenz", p_lor) isa Vector{Float64}
        @test fixed_point("VanderPol", p_vdp) isa Vector{Float64}
        @test fixed_point("Rossler", p_ros) isa Vector{Float64}

        p_epi = epileptor3d_parameters("c2s-SN-SH")
        @test fixed_point("Epileptor3D", p_epi) isa Vector{Float64}
    end

    # --- generate_test_trajectories ---
    @testset "generate_test_trajectories" begin
        p = fhn_parameters("stable-node")
        rhs = fhn_rhs(p)
        X_test, X_init = generate_test_trajectories(rhs, 2, 0.01, 5; center=[0.0, 0.0], window=0.5)
        @test size(X_test, 1) == 2
        @test size(X_test, 2) == 6
        @test size(X_test, 3) == 2
        @test size(X_init) == (2, 2)
    end
end

# ============================================================================
# Dictionaries
# ============================================================================
@testset "Dictionaries" begin
    # --- get_dim_psi ---
    @testset "get_dim_psi" begin
        # n=2, max_deg=1: 1 + 2 = 3
        @test get_dim_psi(2, 1) == 3
        # n=2, max_deg=2: 1 + 2 + 3 = 6
        @test get_dim_psi(2, 2) == 6
        # n=3, max_deg=1: 1 + 3 = 4
        @test get_dim_psi(3, 1) == 4
    end

    # --- hermite_basis ---
    @testset "hermite_basis" begin
        basis, dimPsi, linIdx = hermite_basis(2, 2)
        @test dimPsi == get_dim_psi(2, 2)
        @test length(linIdx) == 2
    end

    # --- Psi_Hermite ---
    @testset "Psi_Hermite" begin
        X = randn(2, 10)
        max_deg = 2
        Ψ = Psi_Hermite(X, max_deg)
        expected_rows = get_dim_psi(2, max_deg)
        @test size(Ψ, 1) == expected_rows
        @test size(Ψ, 2) == 10

        # First row should be all ones (constant term)
        @test all(Ψ[1, :] .≈ 1.0)
    end

    # --- cluster_data + Psi_RBF ---
    @testset "Psi_RBF" begin
        X = randn(2, 50)
        nRBF = 5
        centers = cluster_data(X, nRBF)
        @test size(centers, 1) == 2
        @test size(centers, 2) == nRBF

        Ψ = Psi_RBF(X, centers; include_states=true)
        expected_rows = nRBF + 1 + 2  # RBF + constant + states
        @test size(Ψ, 1) == expected_rows
        @test size(Ψ, 2) == 50
        @test all(Ψ[1, :] .≈ 1.0)

        Ψ2 = Psi_RBF(X, centers; include_states=false)
        @test size(Ψ2, 1) == nRBF
    end

    # --- RFFBasis + Psi_RFF ---
    @testset "Psi_RFF" begin
        n = 2
        D = 10
        sigma = 1.0
        basis = build_rff_basis(n, D, sigma)
        @test basis.D == D
        @test size(basis.W) == (D, n)
        @test length(basis.b) == D

        X = randn(2, 10)
        Ψ = Psi_RFF(X, basis)
        @test size(Ψ, 1) == D
        @test size(Ψ, 2) == 10

        Ψ2 = Psi_RFF(X, basis; include_states=true)
        @test size(Ψ2, 1) == n + D
    end

    # --- construct_projection_operator_hermite ---
    @testset "construct_projection_operator_hermite" begin
        B, nPsi = construct_projection_operator_hermite(2, 2)
        @test size(B) == (2, nPsi)
        @test nPsi == get_dim_psi(2, 2)
    end

    # --- lift_state ---
    @testset "lift_state" begin
        x = [0.5, -0.5]

        dict_info_h = (type=:hermite, max_deg=2, basis_type=:probabilist)
        ψh = lift_state(x, dict_info_h)
        @test length(ψh) == get_dim_psi(2, 2)

        X = randn(2, 20)
        centers = cluster_data(X, 3)
        dict_info_r = (type=:rbf, centers=centers, include_states=true, state_indices=nothing)
        ψr = lift_state(x, dict_info_r)
        @test length(ψr) == 3 + 1 + 2

        basis = build_rff_basis(2, 10, 1.0)
        dict_info_f = (type=:rff, basis=basis, include_states=false)
        ψf = lift_state(x, dict_info_f)
        @test length(ψf) == 10
    end

    # --- Psi_slice ---
    # NOTE: Psi_slice is defined in Dictionaries but not re-exported from KoopmanAnalysis.
    # It is tested internally if needed.
    # @testset "Psi_slice" begin
    #     X_slice = randn(1, 5)
    #     centers = randn(2, 3)
    #     dict_info = (type=:rbf, centers=centers, include_states=true, state_indices=nothing)
    #     Ψ = Psi_slice(X_slice, dict_info; full_dim=2, active_dims=[1], fixed_values=[0.0])
    #     @test size(Ψ, 1) == 3 + 1 + 2
    #     @test size(Ψ, 2) == 5
    # end
end

# ============================================================================
# EDMD
# ============================================================================
@testset "EDMD" begin
    # --- compute_koopman_operator ---
    @testset "compute_koopman_operator size and squareness" begin
        nPsi = 5
        m = 20
        ΨX = randn(nPsi, m)
        ΨY = randn(nPsi, m)

        K = compute_koopman_operator(ΨX, ΨY; method=:pinv)
        @test size(K, 1) == nPsi
        @test size(K, 2) == nPsi
        @test size(K, 1) == size(K, 2)

        K2 = compute_koopman_operator(ΨX, ΨY; method=:ridge, alpha=1e-3)
        @test size(K2) == (nPsi, nPsi)

        # Three-argument form
        X = randn(2, m)
        Y = randn(2, m)
        Psi_func(Xm) = Psi_Hermite(Xm, 2)
        K3, ΨX3, ΨY3 = compute_koopman_operator(X, Y, Psi_func)
        @test size(K3, 1) == size(ΨX3, 1)
        @test size(K3, 2) == size(ΨX3, 1)
    end

    # --- construct_projection_operator ---
    @testset "construct_projection_operator" begin
        nPsi = 5
        m = 20
        ΨX = randn(nPsi, m)
        X_train = randn(2, m)
        B = construct_projection_operator(2, ΨX, X_train)
        @test size(B, 1) == 2
        @test size(B, 2) == nPsi
    end

    # --- rbf_kernel ---
    @testset "rbf_kernel" begin
        x = [1.0, 2.0]
        y = [1.0, 2.0]
        @test rbf_kernel(x, y, 1.0) ≈ 1.0
        @test 0.0 < rbf_kernel(x, [2.0, 3.0], 1.0) < 1.0
    end

    # --- median_heuristic_sigma ---
    @testset "median_heuristic_sigma" begin
        X = randn(2, 30)
        sigma = median_heuristic_sigma(X; n_sample=20)
        @test sigma > 0.0
        @test isfinite(sigma)
    end

    # --- kernel_feature_vector ---
    @testset "kernel_feature_vector" begin
        x = [0.5, 0.5]
        X_dict = randn(2, 5)
        kv = kernel_feature_vector(x, X_dict, 1.0)
        @test length(kv) == 5
        @test all(kv .> 0.0)
        @test all(kv .<= 1.0)
    end

    # --- kernel_edmd_rbf ---
    @testset "kernel_edmd_rbf" begin
        X_dict = randn(2, 10)
        Y_dict = randn(2, 10)
        K, B, K_ZZ, K_ZY = kernel_edmd_rbf(X_dict, Y_dict, 1.0; alpha=1e-6)
        @test size(K) == (10, 10)
        @test size(B) == (2, 10)
        @test size(K_ZZ) == (10, 10)
        @test size(K_ZY) == (10, 10)
    end

    # --- edmd_predict_from_psi ---
    @testset "edmd_predict_from_psi" begin
        nPsi = 5
        n_snap = 20
        m_embed = 4
        ΨX = randn(nPsi, n_snap)
        K_edmd = randn(nPsi, nPsi)
        B_full = randn(m_embed, nPsi)
        S = randn(m_embed, n_snap + 5)
        start_indices = [1, 5]
        n_pred = 3
        X_true, X_pred = edmd_predict_from_psi(ΨX, K_edmd, B_full, S, start_indices, n_pred)
        @test size(X_true) == (1, n_pred + 1, 2)
        @test size(X_pred) == (1, n_pred + 1, 2)
    end
end

# ============================================================================
# Hankel
# ============================================================================
@testset "Hankel" begin
    # --- build_hankel ---
    @testset "build_hankel shape" begin
        v = collect(1.0:50.0)
        m_embed = 5
        tau_delay = 2
        S = build_hankel(v, m_embed, tau_delay)
        max_lag = (m_embed - 1) * tau_delay
        n_snap = length(v) - max_lag
        @test size(S, 1) == m_embed
        @test size(S, 2) == n_snap

        # Verify first column content
        expected_first_col = [v[1 + max_lag - d * tau_delay] for d in 0:(m_embed-1)]
        @test S[:, 1] ≈ expected_first_col
    end

    # --- build_hankel_multichannel ---
    @testset "build_hankel_multichannel" begin
        X = randn(2, 50)
        m_embed = 4
        tau_delay = 2
        S = build_hankel_multichannel(X, m_embed, tau_delay)
        @test size(S, 1) == 2 * m_embed
    end

    # --- hankel_dmd ---
    @testset "hankel_dmd" begin
        v = sin.(0:0.1:20)
        S = build_hankel(v, 5, 1)
        K, K_tilde, X_r, B_proj, U_r = hankel_dmd(S; r=3, dt=0.1)
        @test size(K, 1) == size(K, 2)
        @test size(U_r, 2) == 3
    end

    # --- hankel_edmd ---
    @testset "hankel_edmd" begin
        v = sin.(0:0.1:20)
        S = build_hankel(v, 5, 1)
        result = hankel_edmd(S;
            r=3, dt=0.1,
            dict_type=:hermite,
            dict_params=(max_deg=2,),
            edmd_method=:ridge,
            edmd_alpha=1e-3,
            return_ψ=true,
            use_svd=true
        )
        K_edmd = result[1]
        B_reduced = result[2]
        B_full = result[3]
        @test size(K_edmd, 1) == size(K_edmd, 2)
        @test size(B_full, 1) == 5  # m_embed
    end

    # --- hankel_kernel_edmd ---
    @testset "hankel_kernel_edmd" begin
        v = sin.(0:0.1:20)
        S = build_hankel(v, 5, 1)
        K, B, X, Y, U_r = hankel_kernel_edmd(S; r=3, sigma=1.0, alpha=1e-6)
        @test size(K, 1) == size(K, 2)
    end

    # --- select_svd_rank ---
    @testset "select_svd_rank" begin
        σ = [10.0, 5.0, 2.0, 1.0, 0.1, 0.01]
        @test select_svd_rank(σ; method=:energy, energy=0.99) >= 2
        @test select_svd_rank(σ; method=:hard_threshold) >= 2
        @test select_svd_rank(σ; method=:gap) >= 1
    end

    # --- havok_dmd + havok_predict ---
    @testset "havok_dmd and havok_predict" begin
        v = sin.(0:0.05:30)
        S = build_hankel(v, 8, 1)
        K, A, B, U_r, V_r, u = havok_dmd(S; r=4, dt=0.05)
        @test size(A, 1) == 3
        @test size(A, 2) == 3
        @test size(B, 1) == 3
        @test size(U_r, 2) == 4

        start_indices = [5, 10]
        n_pred = 5
        X_true, X_pred = havok_predict(A, B, U_r, S, start_indices, n_pred; dt=0.05)
        @test size(X_true) == (1, 6, 2)
        @test size(X_pred) == (1, 6, 2)
    end

    # --- delay_embed_training_data ---
    @testset "delay_embed_training_data" begin
        n = 2
        m_traj = 20
        n_traj = 2
        X_train = randn(n, m_traj * n_traj)
        Y_train = randn(n, m_traj * n_traj)
        X_delay, Y_delay = delay_embed_training_data(X_train, Y_train, 3, 1; n_trajectories=n_traj)
        @test size(X_delay, 1) == n * 3
        @test size(X_delay, 2) == size(Y_delay, 2)
    end

    # --- delay_space_edmd_prediction ---
    @testset "delay_space_edmd_prediction" begin
        m_embed = 5
        r = 3
        nPsi = get_dim_psi(r, 2)  # hermite max_deg=2 → 1 + 3 + 6 = 10
        S = randn(m_embed, 20)
        K_edmd = randn(nPsi, nPsi)
        B_full = randn(m_embed, nPsi)
        U_r = randn(m_embed, r)
        dict_info = (type=:hermite, max_deg=2, basis_type=:probabilist)
        start_indices = [1, 5]
        n_pred = 3
        X_true, X_pred = delay_space_edmd_prediction(S, K_edmd, B_full, U_r,
                                                      start_indices, n_pred, dict_info)
        @test size(X_true) == (1, 4, 2)
        @test size(X_pred) == (1, 4, 2)
    end
end

# ============================================================================
# DataGeneration
# ============================================================================
@testset "DataGeneration" begin
    # --- edmd_training_data ---
    @testset "edmd_training_data" begin
        Random.seed!(SEED)
        X_train, Y_train, cfg, rhs, X_fixed = edmd_training_data("FHN", "stable-node";
            m_train=20, n_trajectories=2, dt=0.01, window=0.5)
        @test size(X_train, 1) == 2
        @test size(Y_train, 1) == 2
        @test size(X_train, 2) == 40
        @test cfg isa RegimeConfig
    end

    # --- hankel_training_data ---
    @testset "hankel_training_data" begin
        Random.seed!(SEED)
        v_train, S, cfg, rhs = hankel_training_data("FHN", "stable-node";
            m_embed=5, tau_delay=1, m_train=50, dt=0.01, window=0.5)
        @test cfg isa RegimeConfig
        @test size(S, 1) == 5
    end

    # --- state_space_predictions ---
    @testset "state_space_predictions" begin
        p = fhn_parameters("stable-node")
        rhs = fhn_rhs(p)
        nPsi = get_dim_psi(2, 2)  # Psi_Hermite(Xm, 2) for 2D state returns 6 basis functions
        K = randn(nPsi, nPsi)
        x0s = randn(2, 2)
        Psi_func(Xm) = Psi_Hermite(Xm, 2)
        B_proj = randn(2, nPsi)
        X_true, X_pred = state_space_predictions(rhs, K, x0s, 0.01, 3;
            Psi_func=Psi_func, B_proj=B_proj)
        @test size(X_true) == (2, 4, 2)
        @test size(X_pred) == (2, 4, 2)
    end

    # --- state_space_predict_from_psi ---
    @testset "state_space_predict_from_psi" begin
        X_test = randn(2, 15)
        nPsi = get_dim_psi(2, 2)  # Psi_Hermite(Xm, 2) for 2D state returns 6 basis functions
        K = randn(nPsi, nPsi)
        Psi_func(Xm) = Psi_Hermite(Xm, 2)
        B_proj = randn(2, nPsi)
        start_indices = [1, 5]
        n_pred = 3
        X_true, X_pred = state_space_predict_from_psi(X_test, K, Psi_func, B_proj, start_indices, n_pred)
        @test size(X_true) == (2, 4, 2)
        @test size(X_pred) == (2, 4, 2)
    end

    # --- delay_space_predictions ---
    @testset "delay_space_predictions" begin
        S = randn(5, 20)
        K = randn(5, 5)
        start_indices = [1, 5]
        n_pred = 3
        X_true, X_pred = delay_space_predictions(S, K, start_indices, n_pred)
        @test size(X_true) == (1, 4, 2)
        @test size(X_pred) == (1, 4, 2)
    end
end

# ============================================================================
# Spectral
# ============================================================================
@testset "Spectral" begin
    # --- koopman_eigendecomposition ---
    @testset "koopman_eigendecomposition" begin
        K = randn(5, 5)
        λ, Ξ = koopman_eigendecomposition(K)
        @test length(λ) == 5
        @test size(Ξ, 1) == 5
        @test size(Ξ, 2) == 5
    end

    # --- detect_limit_cycle_modes ---
    @testset "detect_limit_cycle_modes" begin
        λ = [1.0, 0.9 + 0.4im, 0.9 - 0.4im, 0.5, 0.1]
        modes = detect_limit_cycle_modes(λ, 0.1)
        @test length(modes) == 2
    end

    # --- build_harmonic_branch ---
    @testset "build_harmonic_branch" begin
        λ = [1.0, exp(0.5im), exp(-0.5im), exp(1.0im), exp(-1.0im), 0.5]
        branch = build_harmonic_branch(λ, 2; TOL_HARMONIC=0.1)
        @test 2 ∈ branch
    end

    # --- find_all_harmonic_branches ---
    @testset "find_all_harmonic_branches" begin
        λ = [1.0, exp(0.5im), exp(-0.5im), exp(1.0im), exp(-1.0im), 0.5]
        branches = find_all_harmonic_branches(λ, 0.1)
        @test branches isa Dict{Int, Vector{Int}}
    end

    # --- select_phase_amplitude_modes ---
    @testset "select_phase_amplitude_modes" begin
        λ = [1.0, exp(0.5im), exp(-0.5im), 0.95, 0.5]
        j_phase, mu_phase, j_amp, mu_amp, branch = select_phase_amplitude_modes(λ, 0.1)
        @test j_phase > 0
        @test j_amp > 0
    end

    # --- evaluate_eigenfunction_grid ---
    @testset "evaluate_eigenfunction_grid" begin
        Ξ = randn(ComplexF64, 5, 3)
        Psi_func(Xm) = ones(5, size(Xm, 2))
        g1 = range(-1.0, 1.0, length=5)
        g2 = range(-1.0, 1.0, length=5)
        φ = evaluate_eigenfunction_grid(Ξ, Psi_func, g1, g2; n_modes=2)
        @test size(φ) == (5, 5, 2)
    end

    # --- evaluate_eigenfunction_slice ---
    @testset "evaluate_eigenfunction_slice" begin
        Ξ = randn(ComplexF64, 5, 3)
        Psi_func(Xm) = ones(5, size(Xm, 2))
        g1 = range(-1.0, 1.0, length=5)
        φ = evaluate_eigenfunction_slice(Ξ, Psi_func, g1;
                                          slice_dims=(1,), slice_values=Float64[], n_modes=2)
        @test size(φ) == (5, 2)
    end
end

# ============================================================================
# Plotting & Figures
# ============================================================================
@testset "Plotting & Figures" begin
    # --- Plotting smoke tests ---
    @testset "Plotting constants and functions" begin
        @test COLOR_PRIMARY isa Makie.RGB
        @test COLOR_ACCENT isa Makie.RGB
        @test COLOR_SECONDARY isa Makie.RGB
        @test COLOR_BACKGROUND isa Makie.RGB

        bt = blue_tones(5)
        @test length(bt) == 5
        @test all(c isa Makie.RGB for c in bt)

        ot = orange_tones(3)
        @test length(ot) == 3
        @test all(c isa Makie.RGB for c in ot)

        @test_nowarn set_koopman_theme!()
    end

    # --- Figure smoke tests ---
    @testset "fig_phase_portrait" begin
        X = randn(2, 50)
        fig = fig_phase_portrait([X]; fixed_points=[[0.0, 0.0]])
        @test fig isa Makie.Figure
    end

    @testset "fig_training_data" begin
        X = randn(2, 50)
        fig = fig_training_data(X; dim=2)
        @test fig isa Makie.Figure
    end

    @testset "fig_clusterized_data" begin
        X = randn(2, 50)
        centers = randn(2, 5)
        fig = fig_clusterized_data(X, centers; dim=2)
        @test fig isa Makie.Figure
    end

    @testset "fig_prediction" begin
        X_true = randn(1, 10, 2)
        X_pred = randn(1, 10, 2)
        fig = fig_prediction(X_true, X_pred; dt=0.01)
        @test fig isa Makie.Figure
    end

    @testset "fig_eigenvalues" begin
        λ = [1.0, 0.9 + 0.4im, 0.9 - 0.4im, 0.5]
        fig = fig_eigenvalues(λ)
        @test fig isa Makie.Figure
    end

    @testset "fig_eigenfunctions" begin
        Ξ = randn(ComplexF64, 5, 3)
        Psi_func(Xm) = randn(5, size(Xm, 2))
        grid_range = range(-1.0, 1.0, length=5)
        fig = fig_eigenfunctions(Ξ, Psi_func, (1, 2), grid_range, [1, 2];
                                  λ=nothing, fixed_points=nothing)
        @test fig isa Makie.Figure
    end

    @testset "fig_phase_amplitude" begin
        Q1 = randn(ComplexF64, 5, 5)
        Qr = randn(ComplexF64, 5, 5)
        fig = fig_phase_amplitude(Q1, Qr, (range(-1,1,5), range(-1,1,5)))
        @test fig isa Makie.Figure
    end

    @testset "fig_time_series" begin
        X = randn(2, 30)
        fig = fig_time_series(X, 0.01)
        @test fig isa Makie.Figure

        v = randn(30)
        fig2 = fig_time_series(v, 0.01)
        @test fig2 isa Makie.Figure
    end

    @testset "fig_phase_portrait_3d" begin
        X = randn(3, 50)
        fig = fig_phase_portrait_3d([X])
        @test fig isa Makie.Figure
    end

    @testset "fig_phase_portrait_all_views" begin
        X = randn(3, 50)
        fig = fig_phase_portrait_all_views([X])
        @test fig isa Makie.Figure
    end
end

# ============================================================================
# Regimes
# ============================================================================
@testset "Regimes" begin
    # --- regime_config for every supported system+regime ---
    @testset "regime_config all regimes" begin
        systems = Dict(
            "FHN" => ["stable-limit-cycle", "stable-node", "three-equilibrium-regime"],
            "Duffing" => ["stable-node", "three-equilibrium-attractors", "three-equilibrium-centers"],
            "Epileptor3D" => ["c2s-SN-SH", "c3s-SN-supH", "c10s-supH-SH", "c11s-supH-supH",
                               "c2b-SN-SH", "c4b-SN-FLC", "c14b-subH-SH", "c16b-subH-FLC"],
            "Lorenz" => ["chaotic"],
            "VanderPol" => ["stable-focus", "stable-limit-cycle", "chaotic"],
            "Rossler" => ["stable-limit-cycle", "chaotic"],
        )
        for (system, regimes) in systems
            for regime in regimes
                cfg = regime_config(system, regime)
                @test cfg isa RegimeConfig
                @test cfg.system == system
                @test cfg.regime == regime
                @test cfg.params isa NamedTuple
                @test !isempty(cfg.save_dir)
            end
        end
    end

    # --- list_regimes ---
    @testset "list_regimes" begin
        for system in ["FHN", "Duffing", "Epileptor3D", "Lorenz", "VanderPol", "Rossler"]
            regimes = list_regimes(system)
            @test regimes isa Vector{String}
            @test !isempty(regimes)
        end
    end

    # --- default_save_dir ---
    @testset "default_save_dir" begin
        dir = default_save_dir("FHN", "stable-limit-cycle")
        @test occursin("FHN", dir)
        @test occursin("limit-cycle-regime", dir)
    end
end

# ============================================================================
# Config
# ============================================================================
@testset "Config" begin
    # --- KoopmanConfig ---
    @testset "KoopmanConfig" begin
        cfg = KoopmanConfig()
        @test cfg isa KoopmanConfig
        @test cfg.m_embed == 5000
        @test cfg.dict_type == :RBF

        cfg2 = KoopmanConfig(m_embed=10, dict_type=:hermite)
        @test cfg2.m_embed == 10
        @test cfg2.dict_type == :hermite
    end

    # --- embed ---
    @testset "embed" begin
        v = sin.(0:0.1:20)
        cfg = KoopmanConfig(m_embed=5, tau_delay=1)
        S = embed(v, cfg)
        @test size(S, 1) == 5
    end

    # --- hankel_analysis ---
    @testset "hankel_analysis" begin
        v = sin.(0:0.1:20)
        cfg = KoopmanConfig(
            m_embed=5,
            tau_delay=1,
            r=3,
            dict_type=:hermite,
            dict_params=(max_deg=2,),
            edmd_method=:ridge,
            edmd_alpha=1e-3,
            dt=0.1,
            return_ψ=true,
            verbose=false
        )
        res = hankel_analysis(v, cfg)
        @test res isa AnalysisResult
        @test size(res.K, 1) == size(res.K, 2)
    end

    # --- state_analysis ---
    @testset "state_analysis" begin
        X = randn(2, 20)
        Y = randn(2, 20)
        cfg = KoopmanConfig(
            dict_type=:hermite,
            dict_params=(max_deg=2,),
            edmd_method=:ridge,
            edmd_alpha=1e-3,
            return_ψ=true,
            verbose=false
        )
        res = state_analysis(X, Y, cfg)
        @test res isa AnalysisResult
        @test size(res.K, 1) == size(res.K, 2)
    end

    # --- predict ---
    @testset "predict" begin
        v = sin.(0:0.1:20)
        cfg = KoopmanConfig(
            m_embed=5, tau_delay=1, r=3,
            dict_type=:hermite, dict_params=(max_deg=2,),
            edmd_method=:ridge, edmd_alpha=1e-3,
            dt=0.1, return_ψ=true, verbose=false
        )
        res = hankel_analysis(v, cfg)
        start_indices = [1, 5]
        n_pred = 3
        X_true, X_pred = predict(res, start_indices, n_pred)
        @test size(X_true) == (1, 4, 2)
        @test size(X_pred) == (1, 4, 2)
    end

    # --- spectrum ---
    @testset "spectrum" begin
        K = randn(5, 5)
        res = AnalysisResult(K, nothing, nothing, nothing, nothing,
                             nothing, nothing, nothing, nothing,
                             KoopmanConfig(verbose=false))
        λ, Ξ = spectrum(res)
        @test length(λ) == 5
        @test size(Ξ, 1) == 5
        @test size(Ξ, 2) == 5
    end

    # --- all_harmonic_branches ---
    @testset "all_harmonic_branches" begin
        # Construct a simple diagonal K with known eigenvalues
        λ_known = [1.0, exp(0.5im), exp(-0.5im), 0.5, 0.1]
        K = diagm(λ_known)
        res = AnalysisResult(K, nothing, nothing, nothing, nothing,
                             nothing, nothing, nothing, nothing,
                             KoopmanConfig(dt=0.1, verbose=false))
        branches = all_harmonic_branches(res)
        @test branches isa Dict{Int, Vector{Int}}
    end
end

# ============================================================================
# Integration tests
# ============================================================================
@testset "Integration" begin
    # --- FHN full pipeline ---
    @testset "FHN hankel pipeline" begin
        Random.seed!(SEED)
        v_train, S, cfg, rhs = hankel_training_data("FHN", "stable-node";
            m_embed=10, tau_delay=1, m_train=50, dt=0.01, window=0.5)

        analysis_cfg = KoopmanConfig(
            m_embed=10, tau_delay=1, r=3,
            dict_type=:hermite, dict_params=(max_deg=2,),
            edmd_method=:ridge, edmd_alpha=1e-3,
            dt=0.01, return_ψ=true, verbose=false
        )
        res = hankel_analysis(S, analysis_cfg)
        λ, Ξ = spectrum(res)
        @test length(λ) == size(res.K, 1)

        start_indices = [1, 5]
        n_pred = 3
        X_true, X_pred = predict(res, start_indices, n_pred)
        @test size(X_true) == (1, 4, 2)
        @test size(X_pred) == (1, 4, 2)
    end

    # --- Duffing state-space pipeline ---
    @testset "Duffing state-space pipeline" begin
        Random.seed!(SEED)
        X_train, Y_train, cfg, rhs, X_fixed = edmd_training_data("Duffing", "stable-node";
            m_train=20, n_trajectories=2, dt=0.01, window=0.5)

        analysis_cfg = KoopmanConfig(
            dict_type=:hermite, dict_params=(max_deg=2,),
            edmd_method=:ridge, edmd_alpha=1e-3,
            return_ψ=true, verbose=false
        )
        res = state_analysis(X_train, Y_train, analysis_cfg)
        λ, Ξ = spectrum(res)
        @test length(λ) == size(res.K, 1)
    end
end
