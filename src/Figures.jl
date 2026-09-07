module Figures

using LinearAlgebra
using Statistics
using CairoMakie
using LaTeXStrings
using Printf

using ..Plotting: COLOR_PRIMARY, COLOR_ACCENT, COLOR_SECONDARY, COLOR_BACKGROUND,
                  blue_tones, orange_tones
using ..Spectral: evaluate_eigenfunction_slice
using ..Utils: normalize_vector

export fig_phase_portrait, fig_phase_portrait_3d, fig_phase_portrait_all_views,
       fig_training_data, fig_clusterized_data,
       fig_prediction, fig_eigenvalues, fig_eigenfunctions, fig_phase_amplitude,
       fig_time_series

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

function _figsize(n_cols::Int, n_rows::Int;
                  ax_width::Int=550, ax_height::Int=260,
                  title_h::Int=30, pad::Int=50,
                  n_title_rows::Int=3)
    extra_w = 15 * max(0, n_cols - 1)
    extra_h = 10 * max(0, n_rows - 1)
    h_extra = n_title_rows * title_h
    w = max(500, n_cols * ax_width + pad + extra_w)
    h = max(280, n_rows * ax_height + pad + extra_h + h_extra)
    return (w, h)
end

"""
    _add_titles!(fig, title, subtitle, param_str)

Place up to three title rows directly in the figure's top-level layout so they
span the full width and center correctly.  Returns the next free row index.
"""
function _add_titles!(fig, title, subtitle, param_str)
    n_title_rows = 0
    !isempty(title) && (n_title_rows += 1)
    !isnothing(subtitle) && !isempty(subtitle) && (n_title_rows += 1)
    !isnothing(param_str) && !isempty(param_str) && (n_title_rows += 1)

    n_title_rows == 0 && return 0

    row = 0
    if !isempty(title)
        Label(fig[row, :], title, fontsize=24, font=:bold,
              tellwidth=false, halign=:center)
        row += 1
    end
    if !isnothing(subtitle) && !isempty(subtitle)
        Label(fig[row, :], subtitle, fontsize=18,
              tellwidth=false, halign=:center)
        row += 1
    end
    if !isnothing(param_str) && !isempty(param_str)
        Label(fig[row, :], param_str, fontsize=18,
              tellwidth=false, halign=:center)
        row += 1
    end

    return row
end

function _extract_projection(X, projection_dims)
    return ntuple(i -> X[projection_dims[i], :], length(projection_dims))
end

function _axis_labels(dim)
    return "x[$dim]"
end

function _limits_from_data(X, projection_dims; pad=0.1)
    x1 = X[projection_dims[1], :]
    x2 = X[projection_dims[2], :]
    lim1 = (minimum(x1) - pad * (maximum(x1) - minimum(x1)),
            maximum(x1) + pad * (maximum(x1) - minimum(x1)))
    lim2 = (minimum(x2) - pad * (maximum(x2) - minimum(x2)),
            maximum(x2) + pad * (maximum(x2) - minimum(x2)))
    return lim1, lim2
end

function _normalize_fixed_points(fixed_points)
    isnothing(fixed_points) && return nothing
    if fixed_points isa AbstractVector{<:Number} && !(fixed_points[1] isa AbstractVector)
        return [fixed_points]
    end
    return fixed_points isa AbstractVector ? fixed_points : [fixed_points]
end

# ---------------------------------------------------------------------------
# Phase portrait
# ---------------------------------------------------------------------------

function fig_phase_portrait(trajectories;
                            projection_dims::Tuple{Int,Int}=(1, 2),
                            fixed_points=nothing,
                            xlim=nothing, ylim=nothing,
                            title::String="Phase portrait",
                            subtitle::Union{Nothing,String}=nothing,
                            param_str::Union{Nothing,String}=nothing)
    fig = Figure(size=_figsize(1, 1; ax_width=600, ax_height=300))
    row_start = _add_titles!(fig, title, subtitle, param_str)

    ax = Axis(fig[row_start, 1],
        xlabel=_axis_labels(projection_dims[1]), xlabelsize=14,
        ylabel=_axis_labels(projection_dims[2]), ylabelsize=14,
        title=title, titlesize=18, titlefont=:bold)

    traj_vec = trajectories isa Vector ? trajectories : [trajectories]
    cols = blue_tones(length(traj_vec))
    for (i, X) in enumerate(traj_vec)
        x1, x2 = _extract_projection(X, projection_dims)
        lines!(ax, x1, x2, color=cols[i], linewidth=1.5, alpha=0.8)
    end

    if !isnothing(fixed_points)
        fp = _normalize_fixed_points(fixed_points)
        for p in fp
            x1, x2 = p[projection_dims[1]], p[projection_dims[2]]
            scatter!(ax, [x1], [x2], marker=:circle, markersize=10,
                     color=COLOR_ACCENT, strokecolor=:black, strokewidth=1.5)
        end
    end

    if isnothing(xlim) && !isempty(traj_vec)
        xlim, ylim = _limits_from_data(traj_vec[1], projection_dims)
    end
    !isnothing(xlim) && (ax.limits = (xlim[1], xlim[2], ylim[1], ylim[2]))

    return fig
end

# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

function fig_training_data(X_train::AbstractMatrix;
                           dim::Int=2,
                           projection_dims=(1, 2),
                           title::String="Training data",
                           subtitle::Union{Nothing,String}=nothing,
                           param_str::Union{Nothing,String}=nothing)
    @assert dim in (2, 3) "dim must be 2 or 3"
    @assert length(projection_dims) == dim "projection_dims must have length $dim"

    # Compute actual title rows so the figure is not over-allocated
    n_title = 0
    !isempty(title) && (n_title += 1)
    !isnothing(subtitle) && !isempty(subtitle) && (n_title += 1)
    !isnothing(param_str) && !isempty(param_str) && (n_title += 1)

    fig = Figure(size=_figsize(1, 1; ax_width=600, ax_height=300, n_title_rows=n_title))
    row_start = _add_titles!(fig, title, subtitle, param_str)

    coords = _extract_projection(X_train, projection_dims)

    if dim == 2
        ax = Axis(fig[row_start, 1],
            xlabel=_axis_labels(projection_dims[1]), xlabelsize=14,
            ylabel=_axis_labels(projection_dims[2]), ylabelsize=14
            #title=title, titlesize=18, titlefont=:bold
            )

        scatter!(ax, coords[1], coords[2], color=COLOR_PRIMARY, markersize=2,
                # alpha=0.4
                )

    else  # dim == 3
        ax = Axis3(fig[row_start, 1],
            xlabel=_axis_labels(projection_dims[1]),
            ylabel=_axis_labels(projection_dims[2]),
            zlabel=_axis_labels(projection_dims[3])
            # title=title, titlesize=18, titlefont=:bold
            )

        scatter!(ax, coords[1], coords[2], coords[3],
               color=COLOR_PRIMARY, markersize=2,
               #alpha=0.7
               )
    end

    return fig
end

# ---------------------------------------------------------------------------
# Clustered data for RBF
# ---------------------------------------------------------------------------

function fig_clusterized_data(X::AbstractMatrix, centers::AbstractMatrix;
                            dim::Int=2,
                            projection_dims=(1, 2),
                            title::String="RBF centres",
                            subtitle::Union{Nothing,String}=nothing,
                            param_str::Union{Nothing,String}=nothing)
    @assert dim in (2, 3) "dim must be 2 or 3"
    @assert length(projection_dims) == dim "projection_dims must have length $dim"

    n_title = 0
    !isempty(title) && (n_title += 1)
    !isnothing(subtitle) && !isempty(subtitle) && (n_title += 1)
    !isnothing(param_str) && !isempty(param_str) && (n_title += 1)

    fig = Figure(size=_figsize(1, 1; ax_width=600, ax_height=300, n_title_rows=n_title))
    row_start = _add_titles!(fig, title, subtitle, param_str)

    x_coords = _extract_projection(X, projection_dims)
    c_coords = _extract_projection(centers, projection_dims)

    if dim == 2
        ax = Axis(fig[row_start, 1],
            xlabel=_axis_labels(projection_dims[1]), xlabelsize=14,
            ylabel=_axis_labels(projection_dims[2]), ylabelsize=14
            # title=title, titlesize=18, titlefont=:bold
            )

        scatter!(ax, x_coords[1], x_coords[2], color=(COLOR_PRIMARY, 0.3), markersize=3,
                 label="Training data")

        scatter!(ax, c_coords[1], c_coords[2], color=COLOR_ACCENT, markersize=10, marker=:xcross,
                 label="RBF centres")

        axislegend(ax, position=:rt, framevisible=true)

    else  # dim == 3
        ax = Axis3(fig[row_start, 1],
            xlabel=_axis_labels(projection_dims[1]),
            ylabel=_axis_labels(projection_dims[2]),
            zlabel=_axis_labels(projection_dims[3])
            # title=title, titlesize=18, titlefont=:bold
            )

        scatter!(ax, x_coords[1], x_coords[2], x_coords[3],
                 color=(COLOR_PRIMARY, 0.3), markersize=3, label="Training data")

        scatter!(ax, c_coords[1], c_coords[2], c_coords[3],
                 color=COLOR_ACCENT, markersize=10, marker=:xcross, label="RBF centres")

        axislegend(ax, position=:rb, framevisible=true)
    end

    return fig
end

# ---------------------------------------------------------------------------
# Trajectory prediction
# ---------------------------------------------------------------------------

function fig_prediction(X_true, X_pred;
                        dt::Real=0.01,
                        dim_labels=nothing,
                        title::String="Prediction",
                        subtitle::Union{Nothing,String}=nothing,
                        param_str::Union{Nothing,String}=nothing,
                        max_trajectories::Int=10)
    function _to_array(X)
        if X isa AbstractVector
            return reshape(X, 1, length(X), 1)
        elseif ndims(X) == 2
            return reshape(X, size(X, 1), size(X, 2), 1)
        else
            return X
        end
    end

    Xt = _to_array(X_true)
    Xp = _to_array(X_pred)

    n_obs, Tt, nt = size(Xt)
    _, Tp, np = size(Xp)
    T = min(Tt, Tp)
    n_traj = min(nt, np, max_trajectories)

    labels = isnothing(dim_labels) ? ["x[$i]" for i in 1:n_obs] : dim_labels
    palette_funcs = [blue_tones, orange_tones]

    n_rows = 1 + n_obs  # error panel + per-observable rows
    ax_w = n_obs == 1 ? 520 : 420

    f = Figure(size=_figsize(2, n_rows; ax_width=ax_w, ax_height=240, title_h=30),
               backgroundcolor=:white)

    row_idx = _add_titles!(f, title, subtitle, param_str)

    # Error panel (spanning both columns)
    ax_err = Axis(f[row_idx, 1:2],
        xlabel="Step", ylabel="Error",
        title="Prediction error",
        titlesize=18, titlefont=:bold,
        xlabelsize=14, ylabelsize=14,
        backgroundcolor=:white)

    err_per_traj = zeros(T, n_traj)
    for i in 1:n_traj
        diff = Xt[:, 1:T, i] .- Xp[:, 1:T, i]
        err  = sqrt.(sum(diff.^2, dims=1))[:]
        err_per_traj[:, i] = err
        lines!(ax_err, 0:(T-1), err,
            linewidth=2,
            color=(COLOR_PRIMARY, 0.5))
    end

    if n_traj > 1
        ɛ_mean = vec(mean(err_per_traj, dims=2))
        lines!(ax_err, 0:(T-1), ɛ_mean,
            color=:black,
            linewidth=3,
            linestyle=:dash)
    end

    row_idx += 1

    # Per-observable panels: true (left) vs predicted (right)
    for obs in 1:n_obs
        palette = palette_funcs[mod1(obs, length(palette_funcs))](n_traj)

        ax_true = Axis(f[row_idx, 1],
            xlabel="Step", ylabel=labels[obs],
            title="Integrated $(labels[obs])",
            titlesize=18, titlefont=:bold,
            xlabelsize=14, ylabelsize=14,
            backgroundcolor=:white,
            tellheight=false,
            tellwidth=false)

        for i in 1:n_traj
            lines!(ax_true, 0:(T-1), Xt[obs, 1:T, i],
                linewidth=2.5,
                color=palette[i])
        end

        ax_pred = Axis(f[row_idx, 2],
            xlabel="Step", ylabel=labels[obs],
            title="Predicted $(labels[obs])",
            titlesize=18, titlefont=:bold,
            xlabelsize=14, ylabelsize=14,
            backgroundcolor=:white,
            tellheight=false,
            tellwidth=false)

        for i in 1:n_traj
            lines!(ax_pred, 0:(T-1), Xp[obs, 1:T, i],
                linewidth=2.5,
                color=palette[i])
        end

        linkxaxes!(ax_true, ax_pred)
        linkyaxes!(ax_true, ax_pred)

        row_idx += 1
    end

    for ax in f.content
        if ax isa Axis
            ax.xgridvisible = true
            ax.ygridvisible = true
            ax.xgridcolor = (COLOR_PRIMARY, 0.1)
            ax.ygridcolor = (COLOR_PRIMARY, 0.1)
        end
    end

    colsize!(f.layout, 1, Relative(0.5))
    colsize!(f.layout, 2, Relative(0.5))
    colgap!(f.layout, 15)
    rowgap!(f.layout, 10)

    return f
end

# ---------------------------------------------------------------------------
# Eigenvalues
# ---------------------------------------------------------------------------

function fig_eigenvalues(λ;
                         title::String="Koopman spectrum",
                         subtitle::Union{Nothing,String}=nothing,
                         param_str::Union{Nothing,String}=nothing,
                         highlight_indices=nothing)
    fig = Figure(size=_figsize(1, 1; ax_width=800, ax_height=300))
    row_start = _add_titles!(fig, title, subtitle, param_str)

    # Tight limits based on the data, not the unit circle
    xmin, xmax = extrema(real.(λ))
    ymin, ymax = extrema(imag.(λ))
    pad = 0.10
    xlim = (xmin - pad * (xmax - xmin), xmax + pad * (xmax - xmin))
    ylim = (ymin - pad * (ymax - ymin), ymax + pad * (ymax - ymin))

    ax = Axis(fig[row_start, 1],
        xlabel="Re(λ)", xlabelsize=14,
        ylabel="Im(λ)", ylabelsize=14,
        title="Koopman spectrum", titlesize=18, titlefont=:bold,
        limits=(xlim[1], xlim[2], ylim[1], ylim[2]))

    # Unit circle reference (plotted after limits are fixed so it does not expand them)
    θ = range(0.0, 2π, length=200)
    lines!(ax, cos.(θ), sin.(θ), color=(:black, 0.4), linewidth=1.5, linestyle=:dash)
    hlines!(ax, 0.0, color=(:black, 0.3), linewidth=1)
    vlines!(ax, 0.0, color=(:black, 0.3), linewidth=1)

    scatter!(ax, real.(λ), imag.(λ), color=COLOR_PRIMARY, markersize=6)

    if !isnothing(highlight_indices)
        scatter!(ax, real.(λ[highlight_indices]), imag.(λ[highlight_indices]),
                 color=COLOR_ACCENT, markersize=10, marker=:diamond)
    end

    return fig
end

# ---------------------------------------------------------------------------
# Eigenfunctions
# ---------------------------------------------------------------------------

function fig_eigenfunctions(Ξ::AbstractMatrix, Psi_func::Function,
                            plane_dims::Tuple{Int,Int},
                            grid_range::AbstractVector,
                            indices::AbstractVector{Int};
                            λ::Union{Nothing,AbstractVector}=nothing,
                            fixed_points=nothing,
                            normalize::Bool=true,
                            title::String="Eigenfunctions",
                            subtitle::Union{Nothing,String}=nothing,
                            param_str::Union{Nothing,String}=nothing,
                            slice_values::AbstractVector=Float64[])
    n_modes = length(indices)
    n_cols = min(3, n_modes)
    n_rows = ceil(Int, n_modes / n_cols)

    fig = Figure(size=_figsize(n_cols, n_rows + 1; ax_width=280, ax_height=280, title_h=30))
    row_start = _add_titles!(fig, title, subtitle, param_str)

    φ = evaluate_eigenfunction_slice(Ξ, Psi_func, grid_range, grid_range;
                                     slice_dims=plane_dims,
                                     slice_values=slice_values,
                                     n_modes=maximum(indices))

    for (k, idx) in enumerate(indices)
        row = (k - 1) ÷ n_cols + row_start
        col = (k - 1) % n_cols + 1
        λ_str = if isnothing(λ)
            @sprintf("\\varphi_{%d}", idx)
        else
            @sprintf("\\varphi_{%d}  (\\lambda = %.3f%+.3fj)",
                     idx, real(λ[idx]), imag(λ[idx]))
        end
        ax = Axis(fig[row, 2*col-1],
            xlabel=_axis_labels(plane_dims[1]), xlabelsize=14,
            ylabel=_axis_labels(plane_dims[2]), ylabelsize=14,
            title=λ_str,
            titlesize=18, titlefont=:bold,
            aspect=DataAspect())

        data = real.(φ[:, :, idx])
        if normalize
            data = normalize_vector(data; lb=0.0)
        end
        hm = heatmap!(ax, grid_range, grid_range, data, colormap=:thermal)
        contour!(ax, grid_range, grid_range, real.(φ[:, :, idx]),
                 levels=[0.0], color=:white, linewidth=2.0, linestyle=:dash)

        if !isnothing(fixed_points)
            fp = _normalize_fixed_points(fixed_points)
            for p in fp
                scatter!(ax, [p[plane_dims[1]]], [p[plane_dims[2]]],
                         marker=:circle, markersize=10, color=COLOR_ACCENT,
                         strokecolor=:black, strokewidth=1.0)
            end
        end

        Colorbar(fig[row, 2*col], hm, width=10)
    end

    for c in 1:n_cols
        colsize!(fig.layout, 2c - 1, Relative(0.9 / n_cols))
        colsize!(fig.layout, 2c,     Relative(0.1 / n_cols))
    end

    colgap!(fig.layout, 15)
    return fig
end

# ---------------------------------------------------------------------------
# Phase and amplitude
# ---------------------------------------------------------------------------

function fig_phase_amplitude(Q1_grid::AbstractMatrix{<:Complex},
                             Qr_grid::AbstractMatrix{<:Complex},
                             grid_ranges::Tuple{AbstractVector,AbstractVector};
                             title::String="Phase and amplitude",
                             subtitle::Union{Nothing,String}=nothing,
                             param_str::Union{Nothing,String}=nothing)
    fig = Figure(size=_figsize(2, 1; ax_width=520, ax_height=260, title_h=30))
    row_start = _add_titles!(fig, title, subtitle, param_str)

    g1, g2 = grid_ranges

    ax1 = Axis(fig[row_start, 1],
        xlabel=_axis_labels(1), xlabelsize=14,
        ylabel=_axis_labels(2), ylabelsize=14,
        title="Phase", titlesize=18, titlefont=:bold)
    hm1 = heatmap!(ax1, g1, g2, angle.(Q1_grid), colormap=:twilight)
    Colorbar(fig[row_start, 2], hm1, label="angle", width=12)

    ax2 = Axis(fig[row_start, 3],
        xlabel=_axis_labels(1), xlabelsize=14,
        ylabel=_axis_labels(2), ylabelsize=14,
        title="Amplitude", titlesize=18, titlefont=:bold)
    amp = log.(abs.(Qr_grid) .+ 1e-12)
    hm2 = heatmap!(ax2, g1, g2, amp, colormap=:inferno)
    Colorbar(fig[row_start, 4], hm2, label="log |Q_r|", width=12)

    linkyaxes!(ax1, ax2)

    colsize!(fig.layout, 1, Relative(0.45))
    colsize!(fig.layout, 2, Relative(0.05))
    colsize!(fig.layout, 3, Relative(0.45))
    colsize!(fig.layout, 4, Relative(0.05))

    colgap!(fig.layout, 15)
    return fig
end

# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

function fig_time_series(X::AbstractMatrix, dt::Real;
                         dim_labels=nothing,
                         title::String="Time series",
                         subtitle::Union{Nothing,String}=nothing,
                         param_str::Union{Nothing,String}=nothing,
                         highlight_region=nothing)
    n, T = size(X)
    t = (0:T-1) .* dt
    labels = isnothing(dim_labels) ? ["x[$i]" for i in 1:n] : dim_labels

    fig = Figure(size=_figsize(1, n; ax_width=800, ax_height=200, title_h=30))
    row_start = _add_titles!(fig, title, subtitle, param_str)

    for i in 1:n
        ax = Axis(fig[row_start + i - 1, 1],
            xlabel="t", xlabelsize=14,
            ylabel=labels[i], ylabelsize=14,
            titlesize=18, titlefont=:bold)
        lines!(ax, t, X[i, :], color=COLOR_PRIMARY, linewidth=1.2)
        if !isnothing(highlight_region)
            vspan!(ax, highlight_region[1], highlight_region[2],
                   color=(COLOR_ACCENT, 0.15))
        end
    end
    return fig
end

function fig_time_series(v::AbstractVector, dt::Real;
                         label::AbstractString="x(t)",
                         title::String="Time series",
                         subtitle::Union{Nothing,String}=nothing,
                         param_str::Union{Nothing,String}=nothing,
                         highlight_region=nothing)
    T = length(v)
    t = (0:T-1) .* dt
    fig = Figure(size=_figsize(1, 1; ax_width=800, ax_height=260, title_h=30))
    row_start = _add_titles!(fig, title, subtitle, param_str)

    ax = Axis(fig[row_start, 1],
        xlabel="t", xlabelsize=14,
        ylabel=label, ylabelsize=14,
        titlesize=18, titlefont=:bold)
    lines!(ax, t, v, color=COLOR_PRIMARY, linewidth=1.2)
    if !isnothing(highlight_region)
        vspan!(ax, highlight_region[1], highlight_region[2],
               color=(COLOR_ACCENT, 0.15))
    end
    return fig
end

# ---------------------------------------------------------------------------
# 3D phase portrait
# ---------------------------------------------------------------------------

function fig_phase_portrait_3d(trajectories;
                               fixed_points=nothing,
                               title::String="Phase portrait (3D)",
                               subtitle::Union{Nothing,String}=nothing,
                               param_str::Union{Nothing,String}=nothing)
    fig = Figure(size=_figsize(1, 1; ax_width=700, ax_height=300))
    row_start = _add_titles!(fig, title, subtitle, param_str)

    ax = Axis3(fig[row_start, 1],
        xlabel="x₁", ylabel="x₂", zlabel="z",
        title="Phase portrait (3D)", titlesize=18, titlefont=:bold)

    traj_vec = trajectories isa Vector ? trajectories : [trajectories]
    cols = blue_tones(length(traj_vec))
    for (i, X) in enumerate(traj_vec)
        lines!(ax, X[1, :], X[2, :], X[3, :],
               color=cols[i], linewidth=1.5, alpha=0.8)
    end

    if !isnothing(fixed_points)
        fp = _normalize_fixed_points(fixed_points)
        for p in fp
            scatter!(ax, [p[1]], [p[2]], [p[3]],
                     marker=:circle, markersize=10,
                     color=COLOR_ACCENT, strokecolor=:black, strokewidth=1.5)
        end
    end

    return fig
end

# ---------------------------------------------------------------------------
# All views: 3D + three coordinate planes
# ---------------------------------------------------------------------------

function fig_phase_portrait_all_views(trajectories;
                                      fixed_points=nothing,
                                      title::String="Phase portrait — all views",
                                      subtitle::Union{Nothing,String}=nothing,
                                      param_str::Union{Nothing,String}=nothing)
    fig = Figure(size=_figsize(2, 2; ax_width=500, ax_height=260, title_h=30))
    row_start = _add_titles!(fig, title, subtitle, param_str)

    traj_vec = trajectories isa Vector ? trajectories : [trajectories]
    cols = blue_tones(length(traj_vec))

    # 3D view (top-left)
    ax3d = Axis3(fig[row_start, 1],
        xlabel="x₁", ylabel="x₂", zlabel="z",
        title="3D view", titlesize=18, titlefont=:bold)
    for (i, X) in enumerate(traj_vec)
        lines!(ax3d, X[1, :], X[2, :], X[3, :],
               color=cols[i], linewidth=1.5, alpha=0.8)
    end
    if !isnothing(fixed_points)
        fp = _normalize_fixed_points(fixed_points)
        for p in fp
            scatter!(ax3d, [p[1]], [p[2]], [p[3]],
                     marker=:circle, markersize=10,
                     color=COLOR_ACCENT, strokecolor=:black, strokewidth=1.0)
        end
    end

    # x₁–x₂ plane (top-right)
    ax_xy = Axis(fig[row_start, 2],
        xlabel="x₁", xlabelsize=14,
        ylabel="x₂", ylabelsize=14,
        title="x₁–x₂ plane", titlesize=18, titlefont=:bold,
        aspect=DataAspect())
    for (i, X) in enumerate(traj_vec)
        lines!(ax_xy, X[1, :], X[2, :],
               color=cols[i], linewidth=1.5, alpha=0.8)
    end
    if !isnothing(fixed_points)
        fp = _normalize_fixed_points(fixed_points)
        for p in fp
            scatter!(ax_xy, [p[1]], [p[2]],
                     marker=:circle, markersize=10,
                     color=COLOR_ACCENT, strokecolor=:black, strokewidth=1.0)
        end
    end

    # x₁–z plane (bottom-left)
    ax_xz = Axis(fig[row_start + 1, 1],
        xlabel="x₁", xlabelsize=14,
        ylabel="z", ylabelsize=14,
        title="x₁–z plane", titlesize=18, titlefont=:bold,
        aspect=DataAspect())
    for (i, X) in enumerate(traj_vec)
        lines!(ax_xz, X[1, :], X[3, :],
               color=cols[i], linewidth=1.5, alpha=0.8)
    end
    if !isnothing(fixed_points)
        fp = _normalize_fixed_points(fixed_points)
        for p in fp
            scatter!(ax_xz, [p[1]], [p[3]],
                     marker=:circle, markersize=10,
                     color=COLOR_ACCENT, strokecolor=:black, strokewidth=1.0)
        end
    end

    # x₂–z plane (bottom-right)
    ax_yz = Axis(fig[row_start + 1, 2],
        xlabel="x₂", xlabelsize=14,
        ylabel="z", ylabelsize=14,
        title="x₂–z plane", titlesize=18, titlefont=:bold,
        aspect=DataAspect())
    for (i, X) in enumerate(traj_vec)
        lines!(ax_yz, X[2, :], X[3, :],
               color=cols[i], linewidth=1.5, alpha=0.8)
    end
    if !isnothing(fixed_points)
        fp = _normalize_fixed_points(fixed_points)
        for p in fp
            scatter!(ax_yz, [p[2]], [p[3]],
                     marker=:circle, markersize=10,
                     color=COLOR_ACCENT, strokecolor=:black, strokewidth=1.0)
        end
    end

    colsize!(fig.layout, 1, Relative(0.5))
    colsize!(fig.layout, 2, Relative(0.5))

    colgap!(fig.layout, 15)
    rowgap!(fig.layout, 10)
    return fig
end

end # module
