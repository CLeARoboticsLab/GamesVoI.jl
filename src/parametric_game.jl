""" Write a standard-form constrained static game as a MCP
i.e., if player i's problem is of the form
             min_{xᵢ} fᵢ(x, θ)
             s.t.     gᵢ(x, θ) = 0
                      hᵢ(x, θ) ≥ 0

and players share constraints
                      g̃(x, θ) = 0
                      h̃(x, θ) ≥ 0

Here, xᵢ is Pi's decision variable, x is the concatenation of all players'
variables, and θ is a vector of parameters. Express it in the following form:
             find   z
             s.t.   F(z, θ) ⟂ z̲ ≤ z ≤ z̅

where we interpret z = (x, λ, μ, λ̃, μ̃), with λ and μ the Lagrange multipliers
for the constraints g and h (with tildes as appropriate), respectively. The
expression F(z) ⟂ z̲ ≤ z ≤ z̅ should be read as the following three statements:
            - if z = z̲, then F(z, θ) ≥ 0
            - if z̲ < z < z̅, then F(z, θ) = 0
            - if z = z̅, then F(z, θ) ≤ 0

For more details, please consult the documentation for the package
`Complementarity.jl`, which may be found here:
https://github.com/chkwon/Complementarity.jl/tree/master
"""

"Generic description of a constrained parametric game problem."
Base.@kwdef struct ParametricGame{T1,T2,T3,T4,T5,T6,T7}
    "Objective functions for all players"
    objectives::T1
    "Equality constraints for all players"
    equality_constraints::T2 = nothing
    "Inequality constraints for all players"
    inequality_constraints::T3 = nothing
    "Shared equality constraint"
    shared_equality_constraint::T4 = nothing
    "Shared inequality constraint"
    shared_inequality_constraint::T5 = nothing

    "Dimension of parameter vector"
    parameter_dimension::T6 = 1
    "Dimension of primal variables for all players"
    primal_dimensions::T7
    "Dimension of equality constraints for all players"
    equality_dimensions::T7
    "Dimension of inequality constraints for all players"
    inequality_dimensions::T7
    "Dimension of shared equality constraint"
    shared_equality_dimension::T6
    "Dimension of shared inequality constraint"
    shared_inequality_dimension::T6
end

"Solve a constrained parametric game."
function solve(problem::ParametricGame; parameter_value = zeros(problem.parameter_dimension))
    @assert !isnothing(problem.equality_constraints)
    @assert !isnothing(problem.inequality_constraints)

    N = length(problem.objectives)
    @assert N ==
            length(problem.equality_constraints) ==
            length(problem.inequality_constraints) ==
            length(problem.primal_dimensions) ==
            length(problem.equality_dimensions) ==
            length(problem.inequality_dimensions)

    total_dimension =
        sum(problem.primal_dimensions) +
        sum(problem.equality_dimensions) +
        sum(problem.inequality_dimensions) +
        problem.shared_equality_dimension +
        problem.shared_inequality_dimension

    # Define symbolic variables for this MCP.
    @variables z̃[1:total_dimension]
    z = BlockArray(
        Symbolics.scalarize(z̃),
        [
            sum(problem.primal_dimensions),
            sum(problem.equality_dimensions),
            sum(problem.inequality_dimensions),
            problem.shared_equality_dimension,
            problem.shared_inequality_dimension
        ],
    )
    x = BlockArray(z[Block(1)], problem.primal_dimensions)
    λ = BlockArray(z[Block(2)], problem.equality_dimensions)
    μ = BlockArray(z[Block(3)], problem.inequality_dimensions)
    λ̃ = z[Block(4)]
    μ̃ = z[Block(5)]

    # Define a symbolic variable for the parameters.
    @variables θ̃[1:(problem.parameter_dimension)]
    θ = Symbolics.scalarize(θ̃)

    # Build symbolic expressions for objectives and constraints.
    fs = map(f -> f(x, θ), problem.objectives)
    gs = map(g -> g(x, θ), problem.equality_constraints)
    hs = map(h -> h(x, θ), problem.inequality_constraints)
    g̃ = problem.shared_equality_constraint(x, θ)
    h̃ = problem.shared_inequality_constraint(x, θ)
    
    # Build Lagrangians.
    Ls = map(zip(1:N, fs, gs, hs)) do (ii, f, g, h)
        f - λ[Block(ii)]'[1] * g - μ[Block(ii)]' * h - λ̃' * g̃ - μ̃' * h̃ # temp fix
    end

    # Build F = [∇ₓLs, gs, hs, g̃, h̃]'.
    ∇ₓLs = map(zip(Ls, blocks(x))) do (L, xᵢ)
        Symbolics.gradient(L, xᵢ)
    end

    # Set lower and upper bounds for z.
    z̲ = [
        fill(-Inf, sum(problem.primal_dimensions))
        fill(-Inf, sum(problem.equality_dimensions))
        fill(0, sum(problem.inequality_dimensions))
        fill(-Inf, problem.shared_equality_dimension)
        fill(0, problem.shared_inequality_dimension)
    ]
    z̅ = [
        fill(Inf, sum(problem.primal_dimensions))
        fill(Inf, sum(problem.equality_dimensions))
        fill(Inf, sum(problem.inequality_dimensions))
        fill(Inf, problem.shared_equality_dimension)
        fill(Inf, problem.shared_inequality_dimension)
    ]

    # Build parametric MCP.
    parametric_mcp = ParametricMCP(
        [reduce(vcat, ∇ₓLs); reduce(vcat, gs); reduce(vcat, hs); g̃; h̃],
        Symbolics.scalarize(z̃),
        θ,
        z̲,
        z̅;
        compute_sensitivities = false,
    )

    # Solve the problem.
    ParametricMCPs.solve(parametric_mcp, parameter_value; verbose = true)
end
