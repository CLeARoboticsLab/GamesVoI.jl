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

function solve_tower_defense_game(θ, pₖ)  
    
    fs = [(x,θ) ->  -(pₖ[1]*x[Block(1)]'*x[Block(2)] + pₖ[2]*x[Block(1)]'*x[Block(3)] + pₖ[3]*x[Block(1)]'*x[Block(4)]),
          (x,θ) ->  x[Block(1)]'*diagm(θ[1:3])*x[Block(2)],
          (x,θ) ->  x[Block(1)]'*diagm(θ[4:6])*x[Block(3)],
          (x,θ) ->  x[Block(1)]'*diagm(θ[7:9])*x[Block(4)]]

    function g1(x, θ)
        [sum(x[Block(1)]) - 1]
    end
    function g2(x, θ)
        [sum(x[Block(2)]) - 1]
    end
    function g3(x, θ)
        [sum(x[Block(3)]) - 1]
    end
    function g4(x, θ)
        [sum(x[Block(4)]) - 1]
    end
    gs = [g1, g2, g3, g4]

    function h1(x, θ)
        x[Block(1)]
    end
    function h2(x, θ)
        x[Block(2)]
    end
    function h3(x, θ)
        x[Block(3)]
    end
    function h4(x, θ)
        x[Block(4)]
    end
    hs = [h1, h2, h3, h4]
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]
    
    #Main.@infiltrate

    problem = ParametricGame(;
    objectives = fs,
    equality_constraints = gs,
    inequality_constraints = hs,
    shared_equality_constraint = g̃,
    shared_inequality_constraint = h̃,
    parameter_dimension = 9,
    primal_dimensions = [3,3,3,3],
    equality_dimensions = [1,1,1,1],
    inequality_dimensions = [3,3,3,3],
    shared_equality_dimension = 1,
    shared_inequality_dimension = 1,
    )

    (;solution = solve(problem, parameter_value = θ), fs)
end

function solve_full_tower_defense_game(θ, pₖ)  

    fs = [# Stage 2
          (x,θ) ->  -(x[Block(9)][1]*pₖ[1]*x[Block(1)]'*x[Block(2)] + x[Block(9)][2]*pₖ[2]*x[Block(1)]'*x[Block(3)] + x[Block(9)][3]*pₖ[3]*x[Block(1)]'*x[Block(4)]) / (x[Block(9)]'*pₖ),
          (x,θ) ->  x[Block(1)]'*diagm(θ[1:3])*x[Block(2)], 
          (x,θ) ->  x[Block(1)]'*diagm(θ[4:6])*x[Block(3)],
          (x,θ) ->  x[Block(1)]'*diagm(θ[7:9])*x[Block(4)],
          (x,θ) ->  -((1-x[Block(9)][1])*pₖ[1]*x[Block(5)]'*x[Block(6)] + (1-x[Block(9)][2])*pₖ[2]*x[Block(5)]'*x[Block(7)] + (1-x[Block(9)][3])*pₖ[3]*x[Block(5)]'*x[Block(8)]) / (1 - x[Block(9)]'*pₖ),
          (x,θ) ->  x[Block(5)]'*diagm(θ[1:3])*x[Block(6)], 
          (x,θ) ->  x[Block(5)]'*diagm(θ[4:6])*x[Block(7)],
          (x,θ) ->  x[Block(5)]'*diagm(θ[7:9])*x[Block(8)],

           #Stage 1: Weighted sum of P1 costs, weighted by P(s¹=1) and P(s¹=0)
          (x,θ) -> -(pₖ[1]*(x[Block(9)][1]*(x[Block(1)]'*x[Block(2)] - x[Block(5)]'*x[Block(6)]) + x[Block(5)]'*x[Block(6)]) \ 
                   + pₖ[2]*(x[Block(9)][2]*(x[Block(1)]'*x[Block(3)] - x[Block(5)]'*x[Block(7)]) + x[Block(5)]'*x[Block(7)]) \
                   + pₖ[3]*(x[Block(9)][3]*(x[Block(1)]'*x[Block(4)] - x[Block(5)]'*x[Block(8)]) + x[Block(5)]'*x[Block(8)])) 
        ]

    # Equality Constraints
    function g1(x, θ)
        [sum(x[Block(1)]) - 1]
    end
    function g2(x, θ)
        [sum(x[Block(2)]) - 1]
    end
    function g3(x, θ)
        [sum(x[Block(3)]) - 1]
    end
    function g4(x, θ)
        [sum(x[Block(4)]) - 1]
    end
    function g5(x, θ)
        [sum(x[Block(5)]) - 1]
    end
    function g6(x, θ)
        [sum(x[Block(6)]) - 1]
    end
    function g7(x, θ)
        [sum(x[Block(7)]) - 1]
    end
    function g8(x, θ)
        [sum(x[Block(8)]) - 1]
    end
    function g9(x, θ) # test: qₖ = 1 ∀k or put placeholder?
        [
            #x[Block(9)][1] - 1;
            #x[Block(9)][2] - 1;
            #x[Block(9)][3] - 1;
            sum(x[Block(8)]) - 1
        ]
    end
    gs = [g1, g2, g3, g4, g5, g6, g7, g8, g9]

    # Inequality Constraints
    function h1(x, θ)
        x[Block(1)]
    end
    function h2(x, θ)
        x[Block(2)]
    end
    function h3(x, θ)
        x[Block(3)]
    end
    function h4(x, θ)
        x[Block(4)]
    end
    function h5(x, θ)
        x[Block(5)]
    end
    function h6(x, θ)
        x[Block(6)]
    end
    function h7(x, θ)
        x[Block(7)]
    end
    function h8(x, θ)
        x[Block(8)]
    end
    function h9(x, θ) 
        [            
            x[Block(9)];
            ones(size(x[Block(9)])) - x[Block(9)];
            #1.5 - sum(x[Block(9)])
        ]
    end
    hs = [h1, h2, h3, h4, h5, h6, h7, h8, h9]
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]
    
    #Main.@infiltrate

    problem = ParametricGame(;
    objectives = fs,
    equality_constraints = gs,
    inequality_constraints = hs,
    shared_equality_constraint = g̃,
    shared_inequality_constraint = h̃,
    parameter_dimension = 9, # θ
    primal_dimensions = [3,3,3,3,3,3,3,3,3], # x
    equality_dimensions = [1,1,1,1,1,1,1,1,1],
    inequality_dimensions = [3,3,3,3,3,3,3,3,6],
    shared_equality_dimension = 1,
    shared_inequality_dimension = 1,
    )

    (;solution = solve(problem, parameter_value = θ), fs)
end

function solve_full_tower_defense_game_stage2(θ, pₖ, wₖ)  

    p_1 = θ' * pₖ
    p_0 = (1 .- θ)' * pₖ

    # objectives
    fs = [
          (x,θ) ->  -(θ[1]*pₖ[1]*x[Block(1)]'*x[Block(2)] + θ[2]*pₖ[2]*x[Block(1)]'*x[Block(3)] + θ[3]*pₖ[3]*x[Block(1)]'*x[Block(4)])/p_1,
          (x,θ) ->  x[Block(1)]'*diagm(wₖ[1:3])*x[Block(2)],
          (x,θ) ->  x[Block(1)]'*diagm(wₖ[4:6])*x[Block(3)],
          (x,θ) ->  x[Block(1)]'*diagm(wₖ[7:9])*x[Block(4)],
          (x,θ) ->  -((1 - θ[1])*pₖ[1]*x[Block(5)]'*x[Block(6)] + (1 - θ[2])*pₖ[2]*x[Block(5)]'*x[Block(7)] + (1 - θ[3])*pₖ[3]*x[Block(5)]'*x[Block(8)])/p_0,
          (x,θ) ->  x[Block(5)]'*diagm(wₖ[1:3])*x[Block(6)], 
          (x,θ) ->  x[Block(5)]'*diagm(wₖ[4:6])*x[Block(7)],
          (x,θ) ->  x[Block(5)]'*diagm(wₖ[7:9])*x[Block(8)],
          ]

    # equality constraints   
    gs = [(x,θ) -> sum(x[Block(i)]) - 1 for i in 1:8]

    # inequality constraints 
    hs = [(x,θ) -> x[Block(i)] for i in 1:8]

    # shared constraints
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]
    
    problem = ParametricGame(;
    objectives = fs,
    equality_constraints = gs,
    inequality_constraints = hs,
    shared_equality_constraint = g̃,
    shared_inequality_constraint = h̃,
    parameter_dimension = 3, 
    primal_dimensions = [3,3,3,3,3,3,3,3], 
    equality_dimensions = [1,1,1,1,1,1,1,1],
    inequality_dimensions = [3,3,3,3,3,3,3,3],
    shared_equality_dimension = 1,
    shared_inequality_dimension = 1,
    )

    (;solution = solve(problem, parameter_value = θ), fs)
end

# VoI
function VoI_TD()

    # Worlds 
    wₖ = [0, 1, 1, 1, 0, 1, 1, 1, 0] # P2 cares about a single direction in each world

    # Prior distribution of k worlds
    pₖ = [1/3; 1/3; 1/3]

    # Stage 1 
    q = [1/3; 1/3; 1/3] # replace w/ solution from stage 1

    # Stage 2 
    results = solve_full_tower_defense_game_stage2(q, pₖ, wₖ)
    vars = BlockArray(results.solution.z[1:24], [3,3,3,3,3,3,3,3])

    # TODO Stage 1  

    # Stage 2
    P1_cost_stage2_s1 = results.fs[1](vars, q)
    P1_cost_stage2_s0 = results.fs[5](vars, q)

    # println("P1 Cost of tower defense game (stage 1): ", P1_cost_stage1)
    println("P1 Cost of tower defense game (stage 2, signal 1): ", P1_cost_stage2_s1)
    println("P1 Cost of tower defense game (stage 2, signal 0): ", P1_cost_stage2_s0)

    # TEMPORARY: 
    # Zygote.jacobian(q -> solve_full_tower_defense_game_stage2(q, pₖ, wₖ).solution.z[1:24], q)
end

"Jacobian of Stage 2 solution w.r.t. P1's signal structure"
function stage_2_Jacobian(q, pₖ, wₖ)
    # Solve Stage 2
    results = solve_full_tower_defense_game_stage2(q, pₖ, wₖ)

    # Compute Jacobian of solution

end