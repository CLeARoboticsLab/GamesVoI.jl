using Test: @testset, @test
using BlockArrays: Block
using GamesVoI: ParametricGame, solve
using LinearAlgebra: diagm

@testset "GameTests" begin
    N = 3
    fs = [(x, θ) -> sum(x[Block(ii)]) for ii in 1:N]
    gs = [(x, θ) -> [sum(x[Block(ii)] .^ 2) - 1] for ii in 1:N]
    hs = [(x, θ) -> -x[Block(ii)] for ii in 1:N]
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]

    problem = ParametricGame(;
        objectives = fs,
        equality_constraints = gs,
        inequality_constraints = hs,
        shared_equality_constraint = g̃,
        shared_inequality_constraint = h̃,
        parameter_dimension = 1,
        primal_dimensions = fill(2, N),
        equality_dimensions = fill(1, N),
        inequality_dimensions = fill(2, N),
        shared_equality_dimension = 1,
        shared_inequality_dimension = 1,
    )

    solution = solve(problem, [0])
    @test all(isapprox.(solution.variables[1:sum(problem.primal_dimensions)], -0.5sqrt(2), atol = 1e-6))
end

@testset "BayesianGame" begin
    wₖ = [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0]
    pₖ = [1/3; 1/3; 1/3]

    # Build parametric parametric game
    fs = [
        (x, θ) -> -pₖ[1] * x[Block(1)]' * x[Block(2)] - pₖ[2] * x[Block(1)]' * x[Block(3)] - pₖ[3] * x[Block(1)]' * x[Block(4)],
        (x, θ) -> x[Block(1)]' * diagm(wₖ[1:3]) * x[Block(2)],
        (x, θ) -> x[Block(1)]' * diagm(wₖ[4:6]) * x[Block(3)],
        (x, θ) -> x[Block(1)]' * diagm(wₖ[7:9]) * x[Block(4)],
    ]
    gs = [(x, θ) -> [sum(x[Block(i)]) - 1] for i in 1:4]
    hs = [(x, θ) -> x[Block(i)] for i in 1:4]
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]
    parametric_game = ParametricGame(;
        objectives = fs,
        equality_constraints = gs,
        inequality_constraints = hs,
        shared_equality_constraint = g̃,
        shared_inequality_constraint = h̃,
        parameter_dimension = 1,
        primal_dimensions = [3, 3, 3, 3],
        equality_dimensions = [1, 1, 1, 1],
        inequality_dimensions = [3, 3, 3, 3],
        shared_equality_dimension = 1,
        shared_inequality_dimension = 1
    )

    # Solve and check 
    solution = solve(parametric_game, [NaN])
    @test all(isapprox.(solution.primals, [[1/3, 1/3, 1/3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], atol = 1e-6))
end
