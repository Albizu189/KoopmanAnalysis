module Regimes

using ..Systems: fhn_parameters, duffing_parameters, epileptor3d_parameters, lorenz_parameters,
                 vanderpol_parameters, rossler_parameters

export RegimeConfig, regime_config, default_save_dir, list_regimes

"""
    RegimeConfig

Metadata for a single dynamical regime.

# Fields
- `system::String`: system key (`"FHN"`, `"Duffing"`, `"Epileptor3D"`, `"Lorenz"`, `"VanderPol"`).
- `regime::String`: regime identifier.
- `regime_name::String`: human-readable name.
- `regime_suffix::String`: suffix used in filenames.
- `params::NamedTuple`: model parameters.
- `save_dir::String`: directory where figures/results are written.
"""
struct RegimeConfig
    system::String
    regime::String
    regime_name::String
    regime_suffix::String
    params::NamedTuple
    save_dir::String
end

const _BASE_DIR = normpath(joinpath(dirname(@__DIR__), "..", ".."))

"""
    default_save_dir(system, regime; base_dir=_BASE_DIR, subdir="Results")

Return the default absolute directory used by the notebooks to save outputs.
"""
function default_save_dir(system::String, regime::String;
                          base_dir::String=_BASE_DIR,
                          subdir::String="Results")
    suffix = _filename_suffix(system, regime)
    return joinpath(base_dir, subdir, "$(system)-$(suffix)")
end

function _filename_suffix(system::String, regime::String)
    if system == "FHN"
        return regime == "stable-limit-cycle" ? "limit-cycle-regime" :
               regime == "stable-node" ? "stable-node-regime" :
               regime == "three-equilibrium-regime" ? "three-equilibrium-regime" : regime
    elseif system == "Duffing"
        return regime == "stable-node" ? "stable-node-regime" :
               regime == "three-equilibrium-attractors" ? "three-equilibrium-attractors-regime" :
               regime == "three-equilibrium-centers" ? "three-equilibrium-centers-regime" : regime
    elseif system == "Epileptor3D"
        return regime
    elseif system == "Lorenz"
        return regime == "chaotic" ? "chaotic-regime" : regime
    elseif system == "VanderPol"
        return regime == "stable-focus" ? "stable-focus-regime" :
               regime == "stable-limit-cycle" ? "limit-cycle-regime" :
               regime == "chaotic" ? "chaotic-regime" : regime
    elseif system == "Rossler"
        return regime == "stable-limit-cycle" ? "limit-cycle-regime" :
               regime == "chaotic" ? "chaotic-regime" : regime
    else
        return regime
    end
end

const _REGIME_NAMES = Dict(
    "FHN" => Dict(
        "stable-limit-cycle"      => "FHN — stable limit cycle",
        "stable-node"             => "FHN — stable node",
        "three-equilibrium-regime"=> "FHN — three-equilibrium regime",
    ),
    "Duffing" => Dict(
        "stable-node"               => "Duffing — stable node",
        "three-equilibrium-attractors" => "Duffing — three attractors",
        "three-equilibrium-centers"    => "Duffing — three centers",
    ),
    "Epileptor3D" => Dict(
        "c2s-SN-SH"   => "Epileptor 3D — c2s-SN-SH",
        "c3s-SN-supH" => "Epileptor 3D — c3s-SN-supH",
        "c10s-supH-SH"=> "Epileptor 3D — c10s-supH-SH",
        "c11s-supH-supH"=> "Epileptor 3D — c11s-supH-supH",
        "c2b-SN-SH"   => "Epileptor 3D — c2b-SN-SH",
        "c4b-SN-FLC"  => "Epileptor 3D — c4b-SN-FLC",
        "c14b-subH-SH"=> "Epileptor 3D — c14b-subH-SH",
        "c16b-subH-FLC"=> "Epileptor 3D — c16b-subH-FLC",
    ),
    "Lorenz" => Dict(
        "chaotic" => "Lorenz — chaotic",
    ),
    "VanderPol" => Dict(
        "stable-focus"       => "Van der Pol (3rd order) — stable focus",
        "stable-limit-cycle" => "Van der Pol (3rd order) — stable limit cycle",
        "chaotic"            => "Van der Pol (3rd order) — chaotic",
    ),
    "Rossler" => Dict(
    "stable-limit-cycle" => "Rössler — stable limit cycle",
    "chaotic"            => "Rössler — chaotic attractor",
    ),
)

"""
    regime_config(system::String, regime::String; save_dir=nothing)

Build a `RegimeConfig` for the requested `system` and `regime`.  If `save_dir`
is not provided, the default absolute path is used.
"""
function regime_config(system::String, regime::String; save_dir::Union{Nothing,String}=nothing)
    params = if system == "FHN"
        fhn_parameters(regime)
    elseif system == "Duffing"
        duffing_parameters(regime)
    elseif system == "Epileptor3D"
        epileptor3d_parameters(regime)
    elseif system == "Lorenz"
        lorenz_parameters(regime)
    elseif system == "VanderPol"
        vanderpol_parameters(regime)
    elseif system == "Rossler"
        rossler_parameters(regime)
    else
        error("Unknown system: $system")
    end

    name = get(_REGIME_NAMES, system) do
        error("Unknown system: $system")
    end
    regime_name = get(name, regime) do
        "$system — $regime"
    end
    suffix = _filename_suffix(system, regime)
    out_dir = isnothing(save_dir) ? default_save_dir(system, regime) : save_dir
    return RegimeConfig(system, regime, regime_name, suffix, params, out_dir)
end

"""
    list_regimes(system::String)

Return the list of supported regimes for `system`.
"""
function list_regimes(system::String)
    return collect(keys(get(_REGIME_NAMES, system, Dict())))
end

end # module