using ParametricMCPs
using Symbolics 
using BlockArrays
using LinearAlgebra

include("parametric_game.jl")

# Define normal form game as a parametric game.
Ã = [-1 -1 -1;
     -4  0  0;]
B1 = [-2 0 -3;
      -4 0 -5;]
B2 = [-2 -3 0;
      -4 -5 0;]
θ = [0.5, 0.5]
A = [Ã Ã]
B = [B1 B2]

function run_partial_information_game()
    N = 2
    fs = [(x, θ) ->  x[Block(1)]'*A*diagm(vcat(θ[1]*ones(3), θ[2]*ones(3)))*x[Block(2)],
          (x, θ) ->  x[Block(1)]'*B*diagm(vcat(θ[1]*ones(3), θ[2]*ones(3)))*x[Block(2)] ]

    function g1(x, θ)
        [sum(x[Block(1)]) - 1]
    end
    function g2(x, θ)
        [
            sum(x[Block(2)][1:3]) - 1;
            sum(x[Block(2)][4:6]) - 1;
        ]
    end
    gs = [g1, g2]
    function h1(x, θ)
        x[Block(1)]
    end
    function h2(x, θ)
        x[Block(2)]
    end
    hs = [h1, h2]
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]

    problem = ParametricGame(;
        objectives = fs,
        equality_constraints = gs,
        inequality_constraints = hs,
        shared_equality_constraint = g̃,
        shared_inequality_constraint = h̃,
        parameter_dimension = 2,
        primal_dimensions = [2,6],
        equality_dimensions = [1,2],
        inequality_dimensions = [2,6],
        shared_equality_dimension = 1,
        shared_inequality_dimension = 1,
    )

    (;solution = solve(problem, parameter_value = θ), fs)
end

function solve_complete_information_game(B)  
    A = Ã

    fs = [(x, θ) ->  x[Block(1)]'*A*x[Block(2)],
          (x, θ) ->  x[Block(1)]'*B*x[Block(2)]]

    function g1(x, θ)
        [sum(x[Block(1)]) - 1]
    end
    function g2(x, θ)
        [sum(x[Block(2)]) - 1]
    end
    gs = [g1, g2]
    function h1(x, θ)
        x[Block(1)]
    end
    function h2(x, θ)
        x[Block(2)]
    end
    hs = [h1, h2]
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]

    problem = ParametricGame(;
        objectives = fs,
        equality_constraints = gs,
        inequality_constraints = hs,
        shared_equality_constraint = g̃,
        shared_inequality_constraint = h̃,
        parameter_dimension = 2,
        primal_dimensions = [2,3],
        equality_dimensions = [1,1],
        inequality_dimensions = [2,3],
        shared_equality_dimension = 1,
        shared_inequality_dimension = 1,
    )

    (;solution = solve(problem, parameter_value = θ), fs)
end

# VoI
function VoI()
    solution_partial = run_partial_information_game()
    solution_complete_1 = solve_complete_information_game(B1)
    solution_complete_2 = solve_complete_information_game(B2)

    cost_partial = solution_partial.fs[1](BlockArray(solution_partial.solution.z[1:8],[2,6]), θ)
    cost_complete_1 = solution_complete_1.fs[1](BlockArray(solution_complete_1.solution.z[1:5],[2,3]), θ)
    cost_complete_2 = solution_complete_2.fs[1](BlockArray(solution_complete_2.solution.z[1:5],[2,3]), θ)
    cost_complete = θ[1]*cost_complete_1 + θ[2]*cost_complete_2

    println("Cost of partial information game: ", cost_partial)
    println("Cost of complete information game: ", cost_complete)
    println("Value of information: ", cost_partial - cost_complete)

    cost_partial - cost_complete
end