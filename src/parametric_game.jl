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
struct ParametricGame{T1,T2,T3,T4,T5,T6,T7}
    "Objective functions for all players"
    objectives::T1
    "Equality constraints for all players"
    equality_constraints::T2
    "Inequality constraints for all players"
    inequality_constraints::T3
    "Shared equality constraint"
    shared_equality_constraint::T4
    "Shared inequality constraint"
    shared_inequality_constraint::T5

    "Dimension of parameter vector"
    parameter_dimension::T6
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

    "Corresponding ParametricMCP."
    parametric_mcp::ParametricMCP
end

function ParametricGame(;
    objectives,
    equality_constraints,
    inequality_constraints,
    shared_equality_constraint,
    shared_inequality_constraint,
    parameter_dimension = 1,
    primal_dimensions,
    equality_dimensions,
    inequality_dimensions,
    shared_equality_dimension,
    shared_inequality_dimension,
)
    @assert !isnothing(equality_constraints)
    @assert !isnothing(inequality_constraints)

    N = length(objectives)
    @assert N ==
            length(equality_constraints) ==
            length(inequality_constraints) ==
            length(primal_dimensions) ==
            length(equality_dimensions) ==
            length(inequality_dimensions)

    total_dimension =
        sum(primal_dimensions) +
        sum(equality_dimensions) +
        sum(inequality_dimensions) +
        shared_equality_dimension +
        shared_inequality_dimension

    # Define symbolic variables for this MCP.
    @variables z̃[1:total_dimension]
    z = BlockArray(
        Symbolics.scalarize(z̃), 
        [
            sum(primal_dimensions),
            sum(equality_dimensions),
            sum(inequality_dimensions),
            shared_equality_dimension,
            shared_inequality_dimension,
        ]
    )
    x = BlockArray(z[Block(1)], primal_dimensions)
    λ = BlockArray(z[Block(2)], equality_dimensions)
    μ = BlockArray(z[Block(3)], inequality_dimensions)
    λ̃ = z[Block(4)]
    μ̃ = z[Block(5)]

    # Define a symbolic variable for the parameters.
    @variables θ̃[1:parameter_dimension]
    θ = Symbolics.scalarize(θ̃)

    # Build symbolic expressions for objectives and constraints for all players
    # (and shared constraints).
    fs = map(f -> f(x,θ), objectives)
    gs = map(g -> g(x,θ), equality_constraints)
    hs = map(h -> h(x,θ), inequality_constraints)
    g̃ = shared_equality_constraint(x,θ)
    h̃ = shared_inequality_constraint(x,θ)

    # Build Lagrangians for all players.
    Ls = map(zip(1:N, fs, gs, hs)) do (i, f, g, h)
        f - λ[Block(i)]' * g - μ[Block(i)]' * h - λ̃' * g̃ - μ̃' * h̃ 
    end

    # Build F = [∇ₓLs, gs, hs, g̃, h̃]'.
    ∇ₓLs = map(zip(Ls, blocks(x))) do (L, xᵢ)
        Symbolics.gradient(L, xᵢ)
    end

    F_symbolic = [reduce(vcat, ∇ₓLs); reduce(vcat, gs); reduce(vcat, hs); g̃; h̃]

    # Set lower and upper bounds for z.
    z̲ = [
        fill(-Inf, sum(primal_dimensions))
        fill(-Inf, sum(equality_dimensions))
        fill(0, sum(inequality_dimensions))
        fill(-Inf, shared_equality_dimension)
        fill(0, shared_inequality_dimension)
    ]
    z̅ = [
        fill(Inf, sum(primal_dimensions))
        fill(Inf, sum(equality_dimensions))
        fill(Inf, sum(inequality_dimensions))
        fill(Inf, shared_equality_dimension)
        fill(Inf, shared_inequality_dimension)
    ]

    # Build parametric MCP.
    # parametric_mcp = ParametricMCP(F, z̲, z̅, parameter_dimension)
    parametric_mcp = ParametricMCP(
        F_symbolic,
        Symbolics.scalarize(z̃),
        θ,
        z̲,
        z̅;
        compute_sensitivities = true,
    )

    ParametricGame(
        objectives,
        equality_constraints,
        inequality_constraints,
        shared_equality_constraint,
        shared_inequality_constraint,
        parameter_dimension,
        primal_dimensions,
        equality_dimensions,
        inequality_dimensions,
        shared_equality_dimension,
        shared_inequality_dimension,
        parametric_mcp,
    )
end

function total_dim(problem::ParametricGame)
    sum(problem.primal_dimensions) +
    sum(problem.equality_dimensions) +
    sum(problem.inequality_dimensions) +
    problem.shared_equality_dimension +
    problem.shared_inequality_dimension
end

"Solve a constrained parametric game."
function solve(
    problem::ParametricGame,
    parameter_value = zeros(problem.parameter_dimension);
    initial_guess = nothing,
    verbose = false,
    return_primals = true,
)
    z0 = if !isnothing(initial_guess)
        initial_guess
    else
        zeros(total_dim(problem))
    end

        z, status, info = ParametricMCPs.solve(
        problem.parametric_mcp,
        parameter_value;
        initial_guess = z0,
        verbose,
        cumulative_iteration_limit = 100000,
        proximal_perturbation = 1e-2,
        use_basics = true,
        use_start = true,
    )

    if return_primals
        primals = blocks(BlockArray(z[1:sum(problem.primal_dimensions)], problem.primal_dimensions))
        return (; primals, variables = z, status, info)
    else
        return (; variables = z, status, info)
    end
end
