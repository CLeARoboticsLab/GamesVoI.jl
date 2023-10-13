using Test: @testset, @test
using BlockArrays: Block
using GamesVoI: ParametricGame, solve

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
