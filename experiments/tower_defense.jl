using GamesVoI
using BlockArrays
using LinearAlgebra: norm_sqr
using Zygote  

""" Nomenclature
    n                         : Number of worlds (=3)
    pws = [P(w₁),..., P(wₙ)]  : prior distribution of k worlds for each signal, nx1 vector
    ws                        : vector containing P2's cost parameters for each world. vector of nx1 vectors
    x[Block(1)]               : u(0), P1's action given signal s¹=0 depends on r
    x[Block(2)]               : u(1), P1's action given signal s¹=1
    x[Block(3)]               : u(2), P1's action given signal s¹=2
    x[Block(4)]               : u(3), P1's action given signal s¹=3
    x[Block(5)] ~ x[Block(7)] : v(wₖ, 0), P2's action for each worlds given signal s¹=0 depends on r
    x[Block(8)]               : v(wₖ, 1), P2's action for world 1 given signal s¹=1
    x[Block(9)]               : v(wₖ, 2), P2's action for world 2 given signal s¹=2
    x[Block(10)]              : v(wₖ, 3), P2's action for world 3 given signal s¹=3
    θ = rₖ = [r₁, ... , rₙ]   : r, Scout allocation in each direction 
    J                         : Stage 1's objective function  
"""


"""
Solve Stage 1 to find optimal scout allocation r.

Inputs:
    pws: prior distribution of k worlds, nx1 vector
    r_init: initial guess scout allocation
Outputs:
    r: optimal scout allocation
"""
function solve_r(pws, ws; r_init = [1/3, 1/3, 1/3], iter_limit=50, target_error=.00001, α=1)
    cur_iter = 0
    n = length(pws)
    n_players = 1 + n^2
    var_dim = n # TODO: Change this to be more general
    game, _ = build_stage_2(pws, ws) 
    r = r_init
    println("0: r = $r")
    x = compute_stage_2(r, pws, ws, game)
    dJdr = zeros(Float64, n)
    while cur_iter < iter_limit # TODO: Break if change from last iteration is small
        dJdr = compute_dJdr(r, x, pws, ws, game)
        r_temp = r - α .* dJdr
        r_temp = max.(0, min.(1, r_temp)) # project onto [0,1] 
        r_temp = r_temp / sum(r_temp) # project onto (n-1) simplex
        r = r_temp
        x = compute_stage_2(
            r, pws, ws, game;
            initial_guess=vcat(x, zeros(total_dim(game) - n_players * var_dim))
        )
        cur_iter += 1
        println("$cur_iter: r = $r")
    end
    println("$cur_iter: r = $r")
    return r
end

"Defender cost function"
function J_1(u, v) 
    norm_sqr(u - v)
end

"""
Attacker cost function
ws: vector containing P2's (attacker) preference parameters for each world.
"""
function J_2(u, v, ws)
    δ = v - u
    -sum([activate(δ[j]) * ws[j] * δ[j]^2 for j in eachindex(ws)])
end 

"Approximate Heaviside step function"
function activate(δ; k=100000)
    return 1/(1 + exp(-2 * δ * k))
end

"""
Build parametric game for Stage 2.

Inputs: 
    pws: prior distribution of k worlds for each signal, nx1 vector
    ws: vector containing P1's cost parameters for each world. vector of nx1 vectors
Outputs: 
    parametric_game: ParametricGame object
    fs: vector of symbolic expressions for each player's objective function

"""
function build_stage_2(pws, ws)

    n = length(pws) # assume n_signals = n_worlds + 1
    n_players = 1 + n^2

    # Define Bayesian game player costs in Stage 2
    p_w_k_0(w_idx, θ) = (1 - θ[w_idx]) * pws[w_idx] / (1 - θ' * pws)
    fs = [
        (x, θ) ->  sum([J_1(x[Block(1)], x[Block(w_idx + n + 1)]) * p_w_k_0(w_idx, θ) for w_idx in 1:n]), # u|s¹=0 IPI
        [(x, θ) -> J_2(x[Block(1)], x[Block(w_idx + n + 1)], ws) for w_idx in 1:n]...,  # v|s¹=0 IPI
        [(x, θ) -> J_1(x[Block(w_idx + 1)], x[Block(w_idx + 2 * n + 1)]) for w_idx in 1:n]..., # u|s¹={1,2,3} PI
        [(x, θ) -> J_2(x[Block(w_idx + 1)], x[Block(w_idx + 2 * n + 1)], ws) for w_idx in 1:n]..., # v|s¹={1,2,3} PI
    ]

    # equality constraints   
    gs = [(x, θ) -> [sum(x[Block(i)]) - 1] for i in 1:n_players] # Everyone must attack/defend

    # inequality constraints 
    hs = [(x, θ) -> x[Block(i)] for i in 1:n_players] # All vars must be non-negative

    # shared constraints
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]

    ParametricGame(;
        objectives=fs,
        equality_constraints=gs,
        inequality_constraints=hs,
        shared_equality_constraint=g̃,
        shared_inequality_constraint=h̃,
        parameter_dimension=3,
        primal_dimensions=[3 for _ in 1:n_players],
        equality_dimensions=[1 for _ in 1:n_players],
        inequality_dimensions=[3 for _ in 1:n_players],
        shared_equality_dimension=1,
        shared_inequality_dimension=1
    ), fs
end

"""
Compute objective at Stage 1
"""
function compute_J(r, x, pws, ws)
    n = length(pws)
    1/(1 - r' * pws) * sum([(1 - r[j]) * pws[j] * J_1(x[Block(1)], x[Block(j + n + 1)]) for j in 1:n])
end

"""
Compute derivative of Stage 1's objective function w.r.t. x
"""
function compute_dJdx(r, x, pws, ws)
    gradient(x -> compute_J(r, x, pws, ws), x)[1] 
end

"""
Compute full derivative of Stage 1's objective function w.r.t. r

Inputs: 
    x: decision variables of Stage 2
    pws: prior distribution of k worlds, nx1 vector

Outputs: 
    djdq: Jacobian of Stage 1's objective function w.r.t. r
"""
function compute_dJdr(r, x, pws, ws, game)
    dJdx = compute_dJdx(r, x, pws, ws)
    dJdr = gradient(r -> compute_J(r, x, pws, ws), r)[1]
    dxdr = compute_dxdr(r, x, pws, ws, game)
    n = length(pws)
    for idx in 1:(1 + n^2)
        dJdr += (dJdx[Block(idx)]' * dxdr[Block(idx)])'
    end
    dJdr
end

"""
Solve stage 2 and return full derivative of objective function w.r.t. r 

Inputs: 
    r: scout allocation
    pws: prior distribution of k worlds, nx1 vector
    ws: vector containing P2's cost parameters for each world. vector of nx1 vectors

Outputs:
    dxdr: Blocked Jacobian of Stage 2's decision variables w.r.t. Stage 1's decision variable
"""
function compute_dxdr(r, x, pws, ws, game; verbose=false)
    n = length(pws)
    n_players = 1 + n^2
    var_dim = n # TODO: Change this to be more general

    # Return Jacobian
    dxdr = jacobian(r -> solve(
            game,
            r;
            initial_guess=vcat(x, zeros(total_dim(game) - n_players * var_dim)),
            verbose=false,
            return_primals=false
        ).variables[1:n_players*var_dim], r)[1]

    BlockArray(dxdr, [var_dim for _ in 1:n_players], [var_dim])
end

"""
Return Stage 2 decision variables given scout allocation r

Input: 
    r: scout allocation
    pws: prior distribution of k worlds, nx1 vector
    ws: vector containing P2's cost parameters for each world. Vector of nx1 vectors
Output: 
    x: decision variables of Stage 2 given r. BlockedArray with a block per player
"""
function compute_stage_2(r, pws, ws, game; initial_guess = nothing, verbose=false)
    n = length(pws) # assume n_signals = n_worlds + 1
    n_players = 1 + n^2
    var_dim = n # TODO: Change this to be more general

    solution = solve(
        game,
        r;
        initial_guess=isnothing(initial_guess) ? zeros(total_dim(game)) : initial_guess,
        verbose=verbose,
        return_primals=false
    )

    BlockArray(solution.variables[1:n_players * var_dim], [n for _ in 1:n_players])
end