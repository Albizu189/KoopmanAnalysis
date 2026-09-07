module Plotting

using CairoMakie
using ColorSchemes

export COLOR_PRIMARY, COLOR_ACCENT, COLOR_SECONDARY, COLOR_BACKGROUND,
       blue_tones, orange_tones, set_koopman_theme!

# ---------------------------------------------------------------------------
# Shared colour palette
# ---------------------------------------------------------------------------

"""Deep teal used as the primary colour."""
const COLOR_PRIMARY   = Makie.RGB(45/255,  96/255, 126/255)

"""Warm accent colour (burnt orange)."""
const COLOR_ACCENT    = Makie.RGB(220/255, 110/255, 0/255)

"""Light blue used as the secondary colour."""
const COLOR_SECONDARY = Makie.RGB(143/255, 170/255, 220/255)

"""Near-white background colour."""
const COLOR_BACKGROUND = Makie.RGB(250/255, 250/255, 250/255)

"""
    set_koopman_theme!()

Apply a consistent CairoMakie theme for all Koopman analysis figures.
"""
function set_koopman_theme!()
    set_theme!(font="Arial Bold", fontsize=14)
end

"""
    blue_tones(n)

Return `n` distinct blue tones.
"""
function blue_tones(n)
    palette = [
        Makie.RGB(20/255,  40/255,  80/255),
        Makie.RGB(45/255,  96/255,  126/255),
        Makie.RGB(80/255,  140/255, 180/255),
        Makie.RGB(120/255, 170/255, 210/255),
        Makie.RGB(143/255, 170/255, 220/255),
        Makie.RGB(180/255, 200/255, 240/255),
        Makie.RGB(210/255, 225/255, 250/255),
        Makie.RGB(100/255, 150/255, 200/255),
    ]
    return [palette[mod1(i, length(palette))] for i in 1:n]
end

"""
    orange_tones(n)

Return `n` distinct orange tones.
"""
function orange_tones(n)
    palette = [
        Makie.RGB(120/255, 50/255,  0/255),
        Makie.RGB(180/255, 80/255,  0/255),
        Makie.RGB(220/255, 110/255, 0/255),
        Makie.RGB(255/255, 140/255, 0/255),
        Makie.RGB(255/255, 165/255, 0/255),
        Makie.RGB(255/255, 190/255, 60/255),
        Makie.RGB(255/255, 210/255, 120/255),
        Makie.RGB(200/255, 100/255, 20/255),
    ]
    return [palette[mod1(i, length(palette))] for i in 1:n]
end

end # module
