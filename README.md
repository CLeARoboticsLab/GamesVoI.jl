# GamesVoI.jl

[![GamesVoI](https://github.com/CLeARoboticsLab/GamesVoI.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/CLeARoboticsLab/GamesVoI.jl/actions/workflows/ci.yml)

## Overview
A Julia package for computing the value of information in Bayesian games.

## Quickstart Guide

**Installation**

This package uses the proprietary PATH solver under the hood (via [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl)).
Therefore, you will need a license key to solve larger problems.
However, by courtesy of Steven Dirkse, Michael Ferris, and Tudd Munson,
[temporary licenses are available free of charge](https://pages.cs.wisc.edu/~ferris/path.html).
Please consult the documentation of [PATHSolver.jl](https://github.com/chkwon/PATHSolver.jl) to learn about loading the license key.

**Running Experiments**
To run your own experiments, start Julia at the repository root as `julia --project`. Then, run the following code: 

```julia
julia> ] instantiate
julia> include("experiments/my_experiment.jl")
```