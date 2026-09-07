# prediction_example.jl
# ---------------------
# Minimal runnable example: train EDMD on FHN trajectories and plot predictions.
#
# Usage:
#   cd "/home/sofi/Documentos/Koopman Analysis modules"
#   julia --project=. examples/prediction_example.jl

using KoopmanAnalysis

# 1. Select regime and parameters
system = "FHN"
regime = "stable-limit-cycle"
cfg = regime_config(system, regime)

# 2. Build the vector field and generate a trajectory
rhs = fhn_rhs(cfg.params)
dt = 0.01
X = rk4(rhs, [0.1, 0.0], dt, 1000)

# 3. Split into current / next state matrices
Xc = X[:, 1:end-1]
Y = X[:, 2:end]

# 4. Build observable dictionary and Koopman operator
Psi_func(Xm) = Psi_Hermite(Xm, 4)
K, ΨX, ΨY = compute_koopman_operator(Xc, Y, Psi_func)

# 5. Generate test trajectories
X_test, X_init = generate_test_trajectories(rhs, 3, dt, 200;
                                              center=[0.0, 0.0], window=0.5)

# 6. Predict in lifted space
B_proj = construct_projection_operator(size(Xc, 1), ΨX, Xc)
X_true, X_pred = state_space_predictions(rhs, K, X_init, dt, 200;
                                          Psi_func=Psi_func, B_proj=B_proj)

# 7. Plot
fig = fig_prediction(X_true, X_pred; dt=dt, dim_labels=["v", "w"],
                     title="FHN prediction ($regime)",
                     subtitle="EDMD with Hermite observables (max_deg=4)")

# 8. Save
save_path = joinpath(cfg.save_dir, "prediction_example.png")
mkpath(cfg.save_dir)
CairoMakie.save(save_path, fig)
println("Saved figure to: $save_path")
