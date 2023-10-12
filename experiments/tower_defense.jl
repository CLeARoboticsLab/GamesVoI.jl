using GamesVoI
using BlockArrays
using LinearAlgebra
using Infiltrator

""" Nomenclature
    n: Number of worlds (=3)
    pₖ = P(wₖ)                  : prior distribution of k worlds for each signal, 3x1 vector
    x[Block(1)]                : [x(s¹), y(s¹), z(s¹)]|₁ᵀ, P1's action given signal s¹=1
    x[Block(2)] ~ x[Block(4)]  : [a(wₖ), b(wₖ), c(wₖ)]|₁ᵀ, P2's action for world k given signal s¹=1
    x[Block(5)]                : [x(s¹), y(s¹P), z(s¹)]|₀ᵀ, P1's action given signal s¹=0
    x[Block(6)] ~ x[Block(8)]  : [a(wₖ), b(wₖ), c(wₖ)]|₀ᵀ, P2's action for world k given signal s¹=0
    x[Block(9)]:= qₖ = P(1|wₖ)  : P1's signal structure (in Stage 1), 3x1 vector
    θ = [θ₁(wₖ), θ₂(wₖ), θ₃(wₖ)] : Diagonal elements of P2's cost matrix ([θ₁, 1-θ₁, θ₂]) for each world k

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

function solve_full_tower_defense_game_stage2(θ, pₖ)  

    fs = [# Stage 2
          (x,θ) ->  -(x[Block(4)][4]*pₖ[1]*x[Block(1)]'*x[Block(2)] + x[Block(4)][5]*pₖ[2]*x[Block(1)]'*x[Block(3)] + x[Block(4)][6]*pₖ[3]*x[Block(1)]'*x[Block(4)][1:3]) / (x[Block(4)][4:6]'*pₖ),
          (x,θ) ->  x[Block(1)]'*diagm(θ[1:3])*x[Block(2)], 
          (x,θ) ->  x[Block(1)]'*diagm(θ[4:6])*x[Block(3)],
          (x,θ) ->  x[Block(1)]'*diagm(θ[7:9])*x[Block(4)][1:3],
          #(x,θ) ->  -((1-x[Block(9)][1])*pₖ[1]*x[Block(5)]'*x[Block(6)] + (1-x[Block(9)][2])*pₖ[2]*x[Block(5)]'*x[Block(7)] + (1-x[Block(9)][3])*pₖ[3]*x[Block(5)]'*x[Block(8)]) / (1 - x[Block(9)]'*pₖ),
          #(x,θ) ->  x[Block(5)]'*diagm(θ[1:3])*x[Block(6)], 
          #(x,θ) ->  x[Block(5)]'*diagm(θ[4:6])*x[Block(7)],
          #(x,θ) ->  x[Block(5)]'*diagm(θ[7:9])*x[Block(8)],

           #Stage 1: Weighted sum of P1 costs, weighted by P(s¹=1) and P(s¹=0)
          #(x,θ) -> -(pₖ[1]*(x[Block(9)][1]*(x[Block(1)]'*x[Block(2)] - x[Block(5)]'*x[Block(6)]) + x[Block(5)]'*x[Block(6)]) \ 
          #         + pₖ[2]*(x[Block(9)][2]*(x[Block(1)]'*x[Block(3)] - x[Block(5)]'*x[Block(7)]) + x[Block(5)]'*x[Block(7)]) \
          #         + pₖ[3]*(x[Block(9)][3]*(x[Block(1)]'*x[Block(4)] - x[Block(5)]'*x[Block(8)]) + x[Block(5)]'*x[Block(8)])) 

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
        [
            sum(x[Block(4)][1:3]) - 1;
            x[Block(4)][4] - 1;
            x[Block(4)][5] - 1;
            x[Block(4)][6] - 1;
        ]
    end
    #=
    function g6(x, θ)
        [sum(x[Block(6)]) - 1]
    end
    function g7(x, θ)
        [sum(x[Block(7)]) - 1]
    end
    function g8(x, θ)
        [sum(x[Block(8)]) - 1]
    end
    function g9(x, θ) # test: qₖ = 1 ∀k
        [
            x[Block(9)][1] - 1;
            x[Block(9)][2] - 1;
            x[Block(9)][3] - 1;
        ]
    end
    =#
    gs = [g1, g2, g3, g4]#, g6, g7, g8, g9]

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
    #=
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
    =#
    hs = [h1, h2, h3, h4]#, h6, h7, h8, h9]
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
    primal_dimensions = [3,3,3,6], # x
    equality_dimensions = [1,1,1,4],
    inequality_dimensions = [3,3,3,6],
    shared_equality_dimension = 1,
    shared_inequality_dimension = 1,
    )

    (;solution = solve(problem, parameter_value = θ), fs)
end




# VoI
function VoI_TD()

    # Prior distribution of k worlds
    pₖ = [1/3; 1/3; 1/3]
    #pₖ = [1.0; 0.0; 0.0]    
    
    # Prev version
    #θ = [1/3;1/3;1/3;1/3;1/3;1/3;1/3;1/3;1/3]
    #solution_tower_defense = solve_tower_defense_game(θ, pₖ)
    #Main.@infiltrate

    # Solve []
    θ = [1;1;1;1;1;1;1;1;1] 
    solution_tower_defense = solve_full_tower_defense_game(θ, pₖ)
    Main.@infiltrate

    # Stage 1 
    P1_cost_stage1 = solution_tower_defense.fs[9](BlockArray(solution_tower_defense.solution.z[1:27], [3,3,3,3,3,3,3,3,3]))
    
    # Stage 2
    P1_cost_stage2_s1 = solution_tower_defense.fs[1](BlockArray(solution_tower_defense.solution.z[1:27], [3,3,3,3,3,3,3,3,3]))
    P1_cost_stage2_s0 = solution_tower_defense.fs[5](BlockArray(solution_tower_defense.solution.z[1:27], [3,3,3,3,3,3,3,3,3]))

    println("P1 Cost of tower defense game (stage 1): ", P1_cost_stage1)
    println("P1 Cost of tower defense game (stage 2, signal 1): ", P1_cost_stage2_s1)
    println("P1 Cost of tower defense game (stage 2, signal 0): ", P1_cost_stage2_s0)

    
    #cost_tower_defense = solution_tower_defense.fs[1](BlockArray(solution_tower_defense.solution.z[1:12], [3,3,3,3]), θ)

    #Main.@infiltrate


end