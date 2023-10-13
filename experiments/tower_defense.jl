using GamesVoI
using BlockArrays
using LinearAlgebra
using Infiltrator
using Zygote  

""" Nomenclature
    n: Number of worlds (=3)
    pₖ = P(wₖ)                 : prior distribution of k worlds for each signal, 3x1 vector
    x[Block(1)]                : [x(s¹), y(s¹), z(s¹)]|₁ᵀ, P1's action given signal s¹=1
    x[Block(2)] ~ x[Block(4)]  : [a(wₖ), b(wₖ), c(wₖ)]|₁ᵀ, P2's action for world k given signal s¹=1
    x[Block(5)]                : [x(s¹), y(s¹P), z(s¹)]|₀ᵀ, P1's action given signal s¹=0
    x[Block(6)] ~ x[Block(8)]  : [a(wₖ), b(wₖ), c(wₖ)]|₀ᵀ, P2's action for world k given signal s¹=0
    θ = qₖ = P(1|wₖ)           : P1's signal structure (in Stage 2), 3x1 vector
    wₖ                         : vector containing P1's cost parameters for each world. length = n x number of decision vars per player = 3 x 3  

"""

function build_parametric_game(pₖ, wₖ)
    fs = [
        (x, θ) -> -(θ[1] * pₖ[1] * x[Block(1)]' * x[Block(2)] + θ[2] * pₖ[2] * x[Block(1)]' * x[Block(3)] + θ[3] * pₖ[3] * x[Block(1)]' * x[Block(4)]) / (θ' * pₖ),
        (x, θ) -> x[Block(1)]' * diagm(wₖ[1:3]) * x[Block(2)],
        (x, θ) -> x[Block(1)]' * diagm(wₖ[4:6]) * x[Block(3)],
        (x, θ) -> x[Block(1)]' * diagm(wₖ[7:9]) * x[Block(4)],
        (x, θ) -> -((1 - θ[1]) * pₖ[1] * x[Block(5)]' * x[Block(6)] + (1 - θ[2]) * pₖ[2] * x[Block(5)]' * x[Block(7)] + (1 - θ[3]) * pₖ[3] * x[Block(5)]' * x[Block(8)]) / ((1 .- θ)' * pₖ),
        (x, θ) -> x[Block(5)]' * diagm(wₖ[1:3]) * x[Block(6)],
        (x, θ) -> x[Block(5)]' * diagm(wₖ[4:6]) * x[Block(7)],
        (x, θ) -> x[Block(5)]' * diagm(wₖ[7:9]) * x[Block(8)],
    ]

    # equality constraints   
    gs = [(x, θ) -> [sum(x[Block(i)]) - 1] for i in 1:8]

    # inequality constraints 
    hs = [(x, θ) -> x[Block(i)] for i in 1:8]

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
        primal_dimensions=[3, 3, 3, 3, 3, 3, 3, 3],
        equality_dimensions=[1, 1, 1, 1, 1, 1, 1, 1],
        inequality_dimensions=[3, 3, 3, 3, 3, 3, 3, 3],
        shared_equality_dimension=1,
        shared_inequality_dimension=1
    ), fs
end

function_

"""Solve Stackelberg-like game with 2 stages. 

   Returns dz/dq: Jacobian of Stage 2's decision variable (z = P1 and P2's variables in a Bayesian game) w.r.t. Stage 1's decision variable (q = signal structure)"""
function dzdq()
    # Setup game
    wₖ = [0, 1, 1, 1, 0, 1, 1, 1, 0] # worlds
    pₖ = [1/3; 1/3; 1/3] # P1's prior distribution over worlds
    parametric_game, fs = build_parametric_game(pₖ, wₖ)

    # Solve Stage 1 
    q = [1/3; 1/3; 1/3] # TODO obtain q by solving Stage 1

    # Solve Stage 2 given q
    solution = solve(
            parametric_game,
            q;
            initial_guess = zeros(total_dim(parametric_game)),
            verbose=true,
        )
    z = BlockArray(solution.variables[1:24], [3,3,3,3,3,3,3,3])
 
    # Return Jacobian
    Zygote.jacobian(q -> solve(
            parametric_game,
            q;
            initial_guess=zeros(total_dim(parametric_game)),
            verbose=false,
            return_primals=false
        ).variables[1:24], q)[1]
end