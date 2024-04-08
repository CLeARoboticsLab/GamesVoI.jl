# GamesVoI.jl

[![GamesVoI](https://github.com/CLeARoboticsLab/GamesVoI.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/CLeARoboticsLab/GamesVoI.jl/actions/workflows/ci.yml)

A Julia package for computing the value of information in Bayesian games. Read the full paper [here](https://arxive.org)

## Paper Abstract 

We present a mathematical framework for modeling
two-player noncooperative games in which one player (the
defender) is uncertain of the costs of the game and the second
player’s (the attacker’s) intention, but can preemptively allocate
information-gathering resources to reduce this uncertainty. We
obtain the defender’s decisions by solving a two-stage problem. In
Stage 1, the defender allocates information-gathering resources,
and in Stage 2, the information-gathering resources output a
signal that informs the defender about the costs of the game and
the attacker’s intent, and then both players play a noncooperative
game. We provide a gradient-based algorithm to solve the two-
stage game and apply this framework to a tower-defense game
which can be interpreted as a variant of a Colonel Blotto game
with smooth payoff functions and uncertainty over battlefield
valuations. Finally, we analyze how optimal decisions shift with
changes in information-gathering allocations and perturbations

Read the full paper [here](https://arxiv.org/abs/2311.09439).

## Quickstart Guide

**Installation**

This package uses the proprietary PATH solver under the hood (via [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl)).
Therefore, you will need a license key to solve larger problems.
However, by courtesy of Steven Dirkse, Michael Ferris, and Tudd Munson,
[temporary licenses are available free of charge](https://pages.cs.wisc.edu/~ferris/path.html).
Please consult the documentation of [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl) to learn about loading the license key.

## Running Experiments
To run your own experiments, start Julia at the repository root as `julia --project`. Then, run the following code: 

```julia
julia> ] instantiate
julia> include("experiments/my_experiment.jl")
```

## Replicating paper visuals
If you would like to replicate the visualizations shown in the paper, run the following commands. 
Going through the source code for the visualizations is also a good way of understanding how the code is structured. 

First, instantiate the package and load the experiment scripts
```julia
julia> ] instantiate
julia> include("experiments/tower_defense.jl")
```

### Attacker/defense policies (Fig. 4, 5)

```julia
julia> run_stage_1_breakout(display_controls = 1) # 1 for defender, 2 for attacker
```

### Stage 1 cost landscape (Fig. 2)

```julia
julia> visualize_stage_1_cost()
```

### Stage 1 cost terms (Fig. 3)

```julia
julia> run_stage_1_breakout(display_controls = 0)
```

## Contact 

> Note: For any questions on how to use this code, do not hesitate to reach out to Fernando Palafox at [fernandopalafox@utexas.edu](mailto:fernandopalafox@utexa.edu) or open an issue.