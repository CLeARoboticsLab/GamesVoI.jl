using GamesVoI
using BlockArrays
using LinearAlgebra: norm_sqr, norm
using Zygote
using GLMakie: Figure, Axis, Colorbar, heatmap!, text!, surface!, Axis3

""" Nomenclature
    N                         : Number of worlds (=3)
    ps = [P(w₁),..., P(wₙ)]  : prior distribution of k worlds for each signal, nx1 vector
    βs                        : vector containing P2's cost parameters for each world. vector of nx1 vectors
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
    ps: prior distribution of k worlds, nx1 vector
    βs: attacker preference for each direction, nx1 vector in the simplex
    r_init: initial guess scout allocation
Outputs:
    r: optimal scout allocation
"""
function solve_r(
    ps,
    βs;
    r_init = [1 / 3, 1 / 3, 1 / 3],
    iter_limit = 50,
    target_error = 0.00001,
    α = 1,
    return_states = false,
)
    @assert sum(r_init) ≈ 1.0 "Initial guess r must be a probability distribution"
    cur_iter = 0
    n = length(ps)
    n_players = 1 + n^2
    var_dim = n # TODO: Change this to be more general
    if return_states
        x_list = []
        r_list = []
    end
    game, _ = build_stage_2(ps, βs)
    r = r_init
    println("0: r = $r")
    x = compute_stage_2(r, ps, βs, game)
    dKdr = zeros(Float64, n)
    while cur_iter < iter_limit # TODO: Break if change from last iteration is small
        dKdr = compute_dKdr(r, x, ps, βs, game)
        r_temp = r - α .* dKdr
        r = project_onto_simplex(r_temp)
        x = compute_stage_2(
            r,
            ps,
            βs,
            game;
            initial_guess = vcat(x, zeros(total_dim(game) - n_players * var_dim)),
        )
        if return_states
            push!(x_list, x)
            push!(r_list, r)
        end
        # compute stage 1 cost function for current r and x 
        K = compute_K(r, x, ps, βs)
        # println("r_temp = $(round.(r_temp, digits=3)), dKdr = $(round.(dKdr, digits=3)) r = $(round.(r, digits=3)) K = $(round(K, digits=3))")
        println("r = $(round.(r, digits=3))")
        cur_iter += 1
    end
    if return_states
        r_matrix = reduce(hcat, r_list)
        x_matrix = reduce(hcat, x_list)
        out = Dict("r" => r, "x" => x, "r_matrix" => r_matrix, "x_matrix" => x_matrix)
        return out
    end
    return r
end

"""
Temp. script to calculate and plot heatmap of Stage 1 cost function 
"""
function run_visualization()
    dr = 0.01
    ps = [1/3, 1 / 3, 1 / 3]
    βs = [[2, 1, 1], [1, 2, 1], [1, 1, 2]]
    Ks = calculate_stage_1_costs(ps, βs; dr)
    fig = display_surface(ps, Ks)
    fig
end

"""
Temp. script to calculate and plot surfaces for the terms in Stage 1's cost function 
"""
function run_stage_1_breakout()
    dr = 0.05
    ps = [1/3, 1/3, 1/3]
    βs = [
        [2.1, 2.0, 2.0], 
        [2.0, 2.1, 2.0], 
        [2.0, 2.0, 2.1]
    ] 
    world_1_misid_costs = calculate_misid_costs(ps, βs, 1; dr)
    world_2_misid_costs = calculate_misid_costs(ps, βs, 2; dr)
    world_3_misid_costs = calculate_misid_costs(ps, βs, 3; dr)
    world_1_id_costs = calculate_id_costs(ps, βs, 1; dr)
    world_2_id_costs = calculate_id_costs(ps, βs, 2; dr)
    world_3_id_costs = calculate_id_costs(ps, βs, 3; dr)

    # Normalize using maximum value across all worlds
    max_value =
        maximum(
            filter(
                !isnan,
                vcat(
                    world_1_misid_costs,
                    world_2_misid_costs,
                    world_3_misid_costs,
                    world_1_id_costs,
                    world_2_id_costs,
                    world_3_id_costs,
                ),
            ),
        )
    world_1_misid_costs = [isnan(c) ? NaN : c / max_value for c in world_1_misid_costs]
    world_2_misid_costs = [isnan(c) ? NaN : c / max_value for c in world_2_misid_costs]
    world_3_misid_costs = [isnan(c) ? NaN : c / max_value for c in world_3_misid_costs]
    world_1_id_costs = [isnan(c) ? NaN : c / max_value for c in world_1_id_costs]
    world_2_id_costs = [isnan(c) ? NaN : c / max_value for c in world_2_id_costs]
    world_3_id_costs = [isnan(c) ? NaN : c / max_value for c in world_3_id_costs]

    display_stage_1_costs(
        [
            world_1_id_costs,
            world_2_id_costs,
            world_3_id_costs,
            world_1_misid_costs,
            world_2_misid_costs,
            world_3_misid_costs,
        ],
        ps,
    )
end

function run_residuals()
    dr = 0.01
    ps = [1/3, 1/3, 1/3]
    βs = [
        [4.0, 2.0, 2.0], 
        [2.0, 4.0, 2.0], 
        [2.0, 2.0, 4.0]
    ] 
    world_1_residuals = calculate_residuals(ps, βs, 1; dr)
    world_2_residuals = calculate_residuals(ps, βs, 2; dr)
    world_3_residuals = calculate_residuals(ps, βs, 3; dr)

    display_residuals(
        [
            world_1_residuals,
            world_2_residuals,
            world_3_residuals,
        ],
        ps,
    )
end

function calculate_residuals(ps, βs, world_idx; dr = 0.05)
    @assert sum(ps) ≈ 1.0 "Prior distribution ps must be a probability distribution"
    game, _ = build_stage_2(ps, βs)
    rs = 0:dr:1
    num_worlds = length(ps)
    residuals = NaN * ones(Float64, Int(1 / dr + 1), Int(1 / dr + 1))
    for (i, r1) in enumerate(rs)
        for (j, r2) in enumerate(rs)
            if r1 + r2 > 1
                continue
            end
            r3 = 1 - r1 - r2
            r = [r1, r2, r3]
            _, residual = compute_stage_2(r, ps, βs, game; return_residual = true)
            residuals[i, j] = residual
        end
    end

    return residuals
end

function calculate_id_costs(ps, βs, world_idx; dr = 0.05)
    @assert sum(ps) ≈ 1.0 "Prior distribution ps must be a probability distribution"
    game, _ = build_stage_2(ps, βs)
    rs = 0:dr:1
    num_worlds = length(ps)
    id_costs = NaN * ones(Float64, Int(1 / dr + 1), Int(1 / dr + 1))
    for (i, r1) in enumerate(rs)
        for (j, r2) in enumerate(rs)
            if r1 + r2 > 1
                continue
            end
            r3 = 1 - r1 - r2
            r = [r1, r2, r3]
            x = compute_stage_2(r, ps, βs, game)
            id_cost =
                r[world_idx] *
                ps[world_idx] *
                J_1(
                    x[Block(world_idx + 1)],
                    x[Block(world_idx + 2 * num_worlds + 1)],
                    βs[world_idx],
                )
            id_costs[i, j] = id_cost
        end
    end

    return id_costs
end

function calculate_misid_costs(ps, βs, world_idx; dr = 0.05)
    @assert sum(ps) ≈ 1.0 "Prior distribution ps must be a probability distribution"
    game, _ = build_stage_2(ps, βs)
    rs = 0:dr:1
    num_worlds = length(ps)
    misid_costs = NaN * ones(Float64, Int(1 / dr + 1), Int(1 / dr + 1))
    for (i, r1) in enumerate(rs)
        for (j, r2) in enumerate(rs)
            if r1 + r2 > 1
                continue
            end
            r3 = 1 - r1 - r2
            r = [r1, r2, r3]
            x = compute_stage_2(r, ps, βs, game)
            defender_signal_0 = x[Block(1)]
            attacker_signal_0_world_idx = x[Block(world_idx + num_worlds + 1)]
            misid_cost = J_1(defender_signal_0, attacker_signal_0_world_idx, βs[world_idx])
            misid_cost = (1 - r[world_idx]) * ps[world_idx] * misid_cost  # weight by p(w_k|s¹=0)
            misid_costs[i, j] = misid_cost
        end
    end

    return misid_costs
end

"""
Display surface of Stage 1's objective function. Assumes number of worlds is 3.

Input: 
    Ks: 2D Matrix of stage 1's objective function values for each r in the simplex
Output: 
    fig: Figure with simplex heatmap
"""
function display_stage_1_costs(costs, ps)
    rs = 0:(1 / (size(costs[1])[1] - 1)):1
    num_worlds = length(ps)
    fig = Figure(size = (1500, 1000), title = "test")
    axs = [
        [
            Axis3(
                fig[1, world_idx],
                aspect = (1, 1, 1),
                perspectiveness = 0.5,
                elevation = pi / 5,
                azimuth = -π * (1 / 2 + 1 / 4),
                zgridcolor = :grey,
                ygridcolor = :grey,
                xgridcolor = :grey;
                xlabel = "r₁",
                ylabel = "r₂",
                zlabel = "Cost",
                title = "World $world_idx",
                limits = (nothing, nothing, (0.01, 1)),
            ) for world_idx in 1:num_worlds
        ],
        [
            Axis3(
                fig[2, world_idx],
                aspect = (1, 1, 1),
                perspectiveness = 0.5,
                elevation = pi / 5,
                azimuth = -π * (1 / 2 + 1 / 4),
                zgridcolor = :grey,
                ygridcolor = :grey,
                xgridcolor = :grey;
                xlabel = "r₁",
                ylabel = "r₂",
                zlabel = "Cost",
                title = "World $world_idx",
                limits = (nothing, nothing, (0.01, 1)),
            ) for world_idx in 1:num_worlds
        ],
    ]
    for world_idx in 1:num_worlds
        hmap = surface!(
            axs[1][world_idx],
            rs,
            rs,
            costs[world_idx],
            colormap = :viridis,
            colorrange = (0, 1),
        )
        # text!(axs[world_idx], "$(round(ps[1], digits=2))", position = (0.9, 0.4, cost_min), font = "Bold")
        # text!(axs[world_idx], "$(round(ps[2], digits=2))", position = (0.1, 0.95, cost_min), font = "Bold")
        # text!(axs[world_idx], "$(round(ps[3], digits=2))", position = (0.2, 0.1, cost_min), font = "Bold")

    end
    for world_idx in 1:num_worlds
        hmap = surface!(
            axs[2][world_idx],
            rs,
            rs,
            costs[world_idx + num_worlds],
            colormap = :viridis,
            colorrange = (0, 1),
        )
        # text!(axs[world_idx], "$(round(ps[1], digits=2))", position = (0.9, 0.4, cost_min), font = "Bold")
        # text!(axs[world_idx], "$(round(ps[2], digits=2))", position = (0.1, 0.95, cost_min), font = "Bold")
        # text!(axs[world_idx], "$(round(ps[3], digits=2))", position = (0.2, 0.1, cost_min), font = "Bold")

        if world_idx == num_worlds
            Colorbar(
                fig[1:2, num_worlds + 1],
                hmap;
                label = "Cost",
                width = 15,
                ticksize = 15,
                tickalign = 1,
            )
        end
    end
    fig
end


function display_residuals(costs, ps)
    rs = 0:(1 / (size(costs[1])[1] - 1)):1
    num_worlds = length(ps)
    fig = Figure(size = (1500, 500), title = "test")
    axs = [
        Axis3(
            fig[1, world_idx],
            aspect = (1, 1, 1),
            perspectiveness = 0.5,
            elevation = pi / 5,
            azimuth = -π * (1 / 2 + 1 / 4),
            zgridcolor = :grey,
            ygridcolor = :grey,
            xgridcolor = :grey;
            xlabel = "r₁",
            ylabel = "r₂",
            zlabel = "Residual",
            title = "World $world_idx",
            # limits = (nothing, nothing, (0.01, 1)),
        ) for world_idx in 1:num_worlds
    ]
    for world_idx in 1:num_worlds
        hmap = surface!(
            axs[world_idx],
            rs,
            rs,
            costs[world_idx],
            colormap = :viridis,
            # colorrange = (0, 1),
        )

    end
    fig
end

"""
Calculate Stage 1's objective function for all possible values of r.

Inputs: 
    ps: prior distribution of k worlds for each signal, nx1 vector
    βs: vector containing P2's cost parameters for each world. vector of nx1 vectors
    dr: step size for r
Outputs:
    Ks: 2D Matrix of stage 1's objective function values for each r in the simplex. Normalized, by default 
"""
function calculate_stage_1_costs(ps, βs; dr = 0.05, normalize = true)
    @assert sum(ps) ≈ 1.0 "Prior distribution ps must be a probability distribution"
    game, _ = build_stage_2(ps, βs)
    rs = 0:dr:1
    Ks = NaN * ones(Float64, Int(1 / dr + 1), Int(1 / dr + 1))
    for (i, r1) in enumerate(rs)
        for (j, r2) in enumerate(rs)
            if r1 + r2 > 1
                continue
            end
            r3 = 1 - r1 - r2
            r = [r1, r2, r3]
            x = compute_stage_2(r, ps, βs, game)
            K = compute_K(r, x, ps, βs)
            Ks[i, j] = K
        end
    end

    if !normalize
        return Ks
    end

    max_value = maximum(filter(!isnan, Ks))
    Ks = [isnan(K) ? NaN : K / max_value for K in Ks]

    return Ks
end

"""
Display surface of Stage 1's objective function. Assumes number of worlds is 3.

Input: 
    Ks: 2D Matrix of stage 1's objective function values for each r in the simplex
Output: 
    fig: Figure with simplex heatmap
"""
function display_surface(ps, Ks)
    rs = 0:(1 / (size(Ks)[1] - 1)):1
    fig = Figure(size = (600, 400))
    ax = Axis3(
        fig[1, 1],
        aspect = (1, 1, 1),
        perspectiveness = 0.5,
        elevation = pi / 4,
        azimuth = -π * (1 / 2 + 1 / 4),
        zgridcolor = :grey,
        ygridcolor = :grey,
        xgridcolor = :grey;
        xlabel = "r₁",
        ylabel = "r₂",
        zlabel = "K",
        title = "Normalized stage 1 cost\n priors = $(round.(ps, digits=2))",
        limits = (nothing, nothing, (0.01, 1)),
    )
    Ks_min = minimum(filter(!isnan, Ks))
    hmap = surface!(ax, rs, rs, Ks, colorrange = (0, 1))
    Colorbar(fig[1, 2], hmap; label = "K", width = 15, ticksize = 15, tickalign = 1)
    text!(ax, "$(round(ps[1], digits=2))", position = (0.9, 0.2, 0.01), font = "Bold")
    text!(ax, "$(round(ps[2], digits=2))", position = (0.1, 0.95, 0.01), font = "Bold")
    text!(ax, "$(round(ps[3], digits=2))", position = (0.1, 0.2, 0.01), font = "Bold")
    fig
end

"""
Project onto simplex using Fig. 1 Duchi 2008
"""
function project_onto_simplex(v; z = 1.0)
    μ = sort(v, rev = true)
    ρ = findfirst([μ[j] - 1 / j * (sum(μ[1:j]) - z) <= 0 for j in eachindex(v)])
    ρ = isnothing(ρ) ? length(v) : ρ - 1
    θ = 1 / ρ * (sum(μ[1:ρ]) - z)
    return [maximum([v[i] - θ, 0]) for i in eachindex(v)]
end

"Defender cost function"
function J_1(u, v, β)
    -J_2(u, v, β)
end

"""
Attacker cost function
β: vector containing P2's (attacker) preference parameters for each world.
"""
function J_2(u, v, β)
    -sum([β[ii]^(v[ii] - u[ii]) for ii in eachindex(β)])
end

"""
Build parametric game for Stage 2.

Inputs: 
    ps: prior distribution of k worlds for each signal, nx1 vector
    βs: vector containing P1's cost parameters for each world. vector of nx1 vectors
Outputs: 
    parametric_game: ParametricGame object
    fs: vector of symbolic expressions for each player's objective function

"""
function build_stage_2(ps, βs)
    n = length(ps) # assume n_signals = n_worlds + 1
    n_players = 1 + n^2

    # Define Bayesian game player costs in Stage 2
    p_w_k_0(w_idx, θ) = (1 - θ[w_idx]) * ps[w_idx] / (1 - θ' * ps)
    fs = [
        (x, θ) -> sum([
            J_1(x[Block(1)], x[Block(w_idx + n + 1)], βs[w_idx]) * p_w_k_0(w_idx, θ) for
            w_idx in 1:n
        ]), # u|s¹=0 IPI
        [
            (x, θ) -> J_1(x[Block(w_idx + 1)], x[Block(w_idx + 2 * n + 1)], βs[w_idx]) for
            w_idx in 1:n
        ]..., # u|s¹={1,2,3} PI
        [(x, θ) -> J_2(x[Block(1)], x[Block(w_idx + n + 1)], βs[w_idx]) for w_idx in 1:n]...,  # v|s¹=0 IPI
        [
            (x, θ) -> J_2(x[Block(w_idx + 1)], x[Block(w_idx + 2 * n + 1)], βs[w_idx]) for
            w_idx in 1:n
        ]..., # v|s¹={1,2,3} PI
    ]

    # equality constraints   
    gs = [(x, θ) -> [sum(x[Block(i)]) - 1] for i in 1:n_players] # Everyone must attack/defend

    # inequality constraints 
    hs = [(x, θ) -> x[Block(i)] for i in 1:n_players] # All vars must be non-negative

    # shared constraints
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]

    ParametricGame(;
        objectives = fs,
        equality_constraints = gs,
        inequality_constraints = hs,
        shared_equality_constraint = g̃,
        shared_inequality_constraint = h̃,
        parameter_dimension = 3,
        primal_dimensions = [3 for _ in 1:n_players],
        equality_dimensions = [1 for _ in 1:n_players],
        inequality_dimensions = [3 for _ in 1:n_players],
        shared_equality_dimension = 1,
        shared_inequality_dimension = 1,
    ),
    fs
end

"""
Compute objective at Stage 1
"""
function compute_K(r, x, ps, βs)
    n = length(ps)
    sum([(1 - r[j]) * ps[j] * J_1(x[Block(1)], x[Block(j + n + 1)], βs[j]) for j in 1:n]) + sum([r[j] * ps[j] * J_1(x[Block(j + 1)], x[Block(j + 2 * n + 1)], βs[j]) for j in 1:n])
end

"""
Compute derivative of Stage 1's objective function w.r.t. x
"""
function compute_dKdx(r, x, ps, βs)
    gradient(x -> compute_K(r, x, ps, βs), x)[1]
end

"""
Compute full derivative of Stage 1's objective function w.r.t. r

Inputs: 
    x: decision variables of Stage 2
    ps: prior distribution of k worlds, nx1 vector

Outputs: 
    djdq: Jacobian of Stage 1's objective function w.r.t. r
"""
function compute_dKdr(r, x, ps, βs, game)
    dKdx = compute_dKdx(r, x, ps, βs)
    dKdr = gradient(r -> compute_K(r, x, ps, βs), r)[1]
    dxdr = compute_dxdr(r, x, ps, βs, game)
    n = length(ps)
    for idx in 1:(1 + n^2)
        dKdr += (dKdx[Block(idx)]' * dxdr[Block(idx)])'
    end
    dKdr
end

"""
Solve stage 2 and return full derivative of objective function w.r.t. r 

Inputs: 
    r: scout allocation
    ps: prior distribution of k worlds, nx1 vector
    βs: vector containing P2's cost parameters for each world. vector of nx1 vectors

Outputs:
    dxdr: Blocked Jacobian of Stage 2's decision variables w.r.t. Stage 1's decision variable
"""
function compute_dxdr(r, x, ps, βs, game; verbose = false)
    n = length(ps)
    n_players = 1 + n^2
    var_dim = n # TODO: Change this to be more general

    # Return Jacobian
    dxdr = jacobian(
        r -> solve(
            game,
            r;
            initial_guess = vcat(x, zeros(total_dim(game) - n_players * var_dim)),
            verbose = false,
            return_primals = false,
        ).variables[1:(n_players * var_dim)],
        r,
    )[1]

    BlockArray(dxdr, [var_dim for _ in 1:n_players], [var_dim])
end

"""
Return Stage 2 decision variables given scout allocation r

Input: 
    r: scout allocation
    ps: prior distribution of k worlds, nx1 vector
    βs: vector containing P2's cost parameters for each world. Vector of nx1 vectors
Output: 
    x: decision variables of Stage 2 given r. BlockedArray with a block per player
"""
function compute_stage_2(r, ps, βs, game; initial_guess = nothing, verbose = false, return_residual = false)
    n = length(ps) # assume n_signals = n_worlds + 1
    n_players = 1 + n^2
    var_dim = n # TODO: Change this to be more general

    solution = solve(
        game,
        r;
        initial_guess = isnothing(initial_guess) ? zeros(total_dim(game)) : initial_guess,
        verbose = verbose,
        return_primals = false,
    )

    if return_residual
        return BlockArray(solution.variables[1:(n_players * var_dim)], [n for _ in 1:n_players]),
        solution.info.residual
    else
        return BlockArray(solution.variables[1:(n_players * var_dim)], [n for _ in 1:n_players])
    end
end