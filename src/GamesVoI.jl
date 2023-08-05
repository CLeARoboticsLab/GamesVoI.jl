module GamesVoI

using ParametricMCPs
using Symbolics
using BlockArrays

include("parametric_game.jl")
export ParametricGame, solve

end