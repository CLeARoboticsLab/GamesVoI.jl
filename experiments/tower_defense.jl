using GamesVoI
using BlockArrays
using LinearAlgebra: norm_sqr, norm
using Zygote
using Colors
using GLMakie:
    Figure,
    Axis,
    Colorbar,
    heatmap!,
    text!,
    surface!,
    scatter!,
    Axis3,
    save,
    image,
    DataAspect,
    rotr90,
    hidedecorations!,
    hidezdecorations!
    record, 
    empty!, 
    resize_to_layout!
using FileIO
using LaTeXStrings

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
Attacker cost function. Sigmoidal cost function with force multipliers

Inputs: 
    u: vector containing P1's (defender) strategy for each world.
    v: vector containing P2's (attacker) strategy for each world.
    β: vector containing P2's (attacker) preference parameters for each world.
Outputs 
    J_2: Attacker's cost function
"""
function J_2(u, v, β)
    δ = [β[ii]*v[ii] - u[ii] for ii in eachindex(β)]
    -sum([activate(δ[j])*(β[j]*v[j]-u[j])^2 for j in eachindex(β)])
end

"Activation function for attacker cost function"
function activate(δ; k=10.0)
    return 1/(1 + exp(-2 * δ * k))
end

"""
Build parametric game for Stage 2. One single large game.

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
Build complete information parametric for Stage 2. Assumes 2 players, 3 signals, 3 worlds.
"""
function build_complete_info_game()
    fs = [
        (x, θ) -> J_1(x[Block(1)], x[Block(2)], θ)
        (x, θ) -> J_2(x[Block(1)], x[Block(2)], θ)
    ]
    gs = [(x, θ) -> [sum(x[Block(i)]) - 1] for i in 1:2]
    hs = [(x, θ) -> x[Block(i)] for i in 1:2]
    g̃ = (x, θ) -> [0]
    h̃ = (x, θ) -> [0]

    ParametricGame(;
        objectives = fs,
        equality_constraints = gs,
        inequality_constraints = hs,
        shared_equality_constraint = g̃,
        shared_inequality_constraint = h̃,
        parameter_dimension = 3,
        primal_dimensions = [3, 3],
        equality_dimensions = [1, 1],
        inequality_dimensions = [3, 3],
        shared_equality_dimension = 1,
        shared_inequality_dimension = 1,
    )
end

"""
Build incomplete information parametric for Stage 2. Assumes 2 players, 3 signals, 3 worlds.
"""
function build_incomplete_info_game(ps, βs)
    n = length(ps)# assume n_signals = n_worlds + 1
    n_players = 1 + n

    p_w_k_0(w_idx, θ) = (1 - θ[w_idx]) * ps[w_idx] / (1 - θ' * ps)
    fs = [
        (x, θ) -> sum([
            p_w_k_0(w_idx, θ) * J_1(x[Block(1)], x[Block(w_idx + 1)], βs[w_idx]) for w_idx in 1:n
        ]), # x^1(0, i)
        [(x, θ) -> J_2(x[Block(1)], x[Block(w_idx + 1)], βs[w_idx]) for w_idx in 1:n]...,
    ]
    gs = [(x, θ) -> [sum(x[Block(i)]) - 1] for i in 1:n_players]
    hs = [(x, θ) -> x[Block(i)] for i in 1:n_players]
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
    )
end

"""
Compute Stage 1 objective

Inputs: 
    r: scout allocation
    x: decision variables of Stage 2
    ps: prior distribution of k worlds, nx1 vector
    βs: vector containing P2's cost parameters for each world. vector of nx1 vectors
Output: 
    K: Stage 1's objective function value
"""
function compute_K(r, x, ps, βs)
    n = length(ps)
    sum([(1 - r[j]) * ps[j] * J_1(x[Block(1)], x[Block(j + n + 1)], βs[j]) for j in 1:n]) + sum([r[j] * ps[j] * J_1(x[Block(j + 1)], x[Block(j + 2 * n + 1)], βs[j]) for j in 1:n])
end

"""
Compute incomplete information cost term for a single world
Inputs: 
    r: scout allocation
    ps: prior distribution of k worlds, nx1 vector
    r_i: scout allocation for world i
    x_1_0: defender's decision for signal 0
    x_2_0_i: attacker's decision for signal 0 and world i
    ps_i: prior distribution of world i
    βs_i: P2's cost parameters for world i
Outputs: 
    cost_term: incomplete info cost term for world i
"""
function compute_incomplete_info_cost_term_i(r, ps, r_i, x_1_0, x_2_0_i, ps_i, βs_i)
    (1 - r_i) * ps_i / (1 - r' * ps) * J_1(x_1_0, x_2_0_i, βs_i)
end

function compute_P1_incomplete_info_cost(r, x_1_0, x_2_0s, ps, βs)
    n = length(ps)
    sum([compute_incomplete_info_cost_term_i(r, ps, r[i], x_1_0, x_2_0s[Block(i)], ps[i], βs[i]) for i in 1:n])
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
    var_dim = n 

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
Compute Stage 2 decision variables given r using PATH

Input: 
    r: scout allocation
    ps: prior distribution of k worlds, nx1 vector
Output: 
    x: decision variables of Stage 2 given r. BlockedArray with a block per player
"""
function compute_stage_2(
    r,
    ps,
    βs,
    complete_info_game,
    incomplete_info_game;
    initial_guess = nothing,
    verbose = false,
)
    num_worlds = length(ps) # assume n_signals = n_worlds + 1
    n_players = 1 + num_worlds^2
    var_dim = num_worlds # TODO: Change this to be more general

    solution_complete = [
        solve(
            complete_info_game,
            β;
            initial_guess = isnothing(initial_guess) ?
                            1 / 3 * ones(total_dim(complete_info_game)) : initial_guess,
            verbose,
            return_primals = true,
        ) for β in βs
    ]

    solution_incomplete = solve(
        incomplete_info_game,
        r;
        initial_guess = isnothing(initial_guess) ?
                        1 / 3 * ones(total_dim(incomplete_info_game)) : initial_guess,
        verbose,
        return_primals = true,
    )

    return BlockArray(
        vcat(
            solution_incomplete.variables[1:var_dim],
            [solution_complete[i].variables[1:var_dim] for i in 1:num_worlds]...,
            solution_incomplete.variables[(var_dim + 1):((num_worlds + 1) * var_dim)],
            [solution_complete[i].variables[(var_dim + 1):(2 * var_dim)] for i in 1:num_worlds]...,
        ),
        [var_dim for _ in 1:n_players],
    )
end

struct IBRGameSolver end

"""
Compute Stage 2 decision variables using Iterative Best Response

Inputs: 
    r: scout allocation
    ps: prior distribution of k worlds, nx1 vector
    βs: vector containing P2's cost parameters for each world. vector of nx1 vectors
    Js: vector containing P1 and P2's cost functions
Outputs:
    x: decision variables of Stage 2 given r. BlockedArray with a block per player
"""
function compute_stage_2(
    ::IBRGameSolver,
    r,
    ps,
    βs,
    Js;
    initial_guess = nothing,
    max_ibr_rounds = 100,
    ibr_convergence_tolerance = 1e-3,
    verbose = false,
)
    num_worlds = length(ps) # assume n_signals = n_worlds + 1
    total_num_vars = num_worlds + 1 + 2*num_worlds
    if isnothing(initial_guess)
        x = 1/3 * ones((num_worlds + 1) * num_worlds + 2 * num_worlds^2)
    else
        x = initial_guess
    end
    x = BlockArray(x, [num_worlds for _ in 1:total_num_vars])

    # Solve complete information games
    for world_idx in 1:num_worlds        
        β = βs[world_idx]
        x_1 = x[Block(1 + world_idx)]
        x_2 = x[Block(1 + 2 * num_worlds + world_idx)]
        for i_ibr in 1:max_ibr_rounds
            last_solution = [x_1, x_2]
            x_1 = gradient_play(x_1 -> Js[1](x_1, x_2, β), x_1; verbose)
            x_2 = gradient_play(x_2 -> Js[2](x_1, x_2, β), x_2; verbose)
            converged = norm([x_1, x_2] - last_solution) < ibr_convergence_tolerance
            if converged
                verbose && @info "World $world_idx complete info. game converged after $i_ibr IBR iterations"
                break
            end
        end
        x[Block(1 + world_idx)] = x_1
        x[Block(1 + 2 * num_worlds + world_idx)] = x_2
    end

    # Solve incomplete information game
    x_1_0  = x[Block(1)]
    x_2_0s = BlockArray(
        x[((1 + num_worlds) * num_worlds + 1):((1 + num_worlds) * num_worlds + num_worlds * num_worlds)], # P2, s = 0
        [num_worlds for _ in 1:num_worlds],
    )
    for i_ibr in 1:max_ibr_rounds
        last_solution = vcat(x_1_0, x_2_0s)
        x_1_0 = gradient_play(
            x_1_0 -> compute_P1_incomplete_info_cost(r, x_1_0, x_2_0s, ps, βs),
            x_1_0;
            verbose,
        )
        for world_idx in 1:num_worlds
            x_2_0s[Block(world_idx)] = gradient_play(
                x_2_0s -> Js[2](x_1_0, x_2_0s, βs[world_idx]),
                x_2_0s[Block(world_idx)];
                verbose,
            )
        end
        converged = norm(vcat(x_1_0, x_2_0s) - last_solution) < ibr_convergence_tolerance
        if converged
            verbose && @info "Incomplete complete info. game converged after $i_ibr IBR iterations"
            break
        end
    end
    x[Block(1)] = x_1_0
    for world_idx in 1:num_worlds
        x[Block(1 + num_worlds + world_idx)] = x_2_0s[Block(world_idx)]
    end
    return x
end

"""
Gradient descent while projecting onto the simplex

Input: 
    cost_function: function to minimize
    x: initial guess
Output: 
    x: minimizer of cost_function
"""
function gradient_play(cost_function, x; max_iter = 200, α = 0.05, tol = 1e-3, verbose = false, text = nothing)

    iter = 0
    x_prev = x
    while iter < max_iter
        x_prev = x
        dJdx = gradient(x -> cost_function(x), x)[1]
        x_temp = x - α .* dJdx
        x = project_onto_simplex(x_temp)
        iter += 1
        if (norm(x - x_prev) < tol)
            verbose && @info "  Gradient descent converged after $iter iterations"
            return x
        end
    end
    @warn "  Gradient descent did not converge after $max_iter iterations"
    return x
end

# --------------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------- VISUALIZATION ----------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------------

"""
Visualization. Calculate and plot Stage 1 cost as a function of r for a given prior distribution and attacker preference.
"""
function run_visualization()
    dr = 0.05
    ps = [1/3, 1 / 3, 1 / 3]
    βs = [
        [3.0, 2.0, 2.0], 
        [2.0, 3.0, 2.0], 
        [2.0, 2.0, 3.0]
    ]
    Ks = calculate_stage_1_costs(ps, βs; dr)
    fig = display_surface(ps, Ks)
    fig
end

"""
Visualization. Calculate and plot all terms in the Stage 1 cost as a function of r for a given prior distribution and attacker preference.
Assumes number of worlds and signals is 3.
"""
function run_stage_1_breakout(;
    display_controls = 0, 
    dr = 0.05,
    βs = [
        [3.0, 2.0, 2.0], 
        [2.0, 3.0 , 2.0], 
        [2.0, 2.0, 3.0]
    ],
    ps = [1/3, 1/3, 1/3],
)

    if (display_controls in [1,2])
        println("Calculating misid. costs")
        world_1_misid_costs, world_1_misid_controls = calculate_misid_costs(ps, βs, 1; dr, return_controls=display_controls)
        world_2_misid_costs, world_2_misid_controls = calculate_misid_costs(ps, βs, 2; dr, return_controls=display_controls)
        world_3_misid_costs, world_3_misid_controls = calculate_misid_costs(ps, βs, 3; dr, return_controls=display_controls)
        println("Calculating id costs")
        world_1_id_costs, world_1_id_controls = calculate_id_costs(ps, βs, 1; dr, return_controls=display_controls)
        world_2_id_costs, world_2_id_controls = calculate_id_costs(ps, βs, 2; dr, return_controls=display_controls)
        world_3_id_costs, world_3_id_controls = calculate_id_costs(ps, βs, 3; dr, return_controls=display_controls)
    else
        world_1_misid_costs = calculate_misid_costs(ps, βs, 1; dr)
        world_2_misid_costs = calculate_misid_costs(ps, βs, 2; dr)
        world_3_misid_costs = calculate_misid_costs(ps, βs, 3; dr)
        world_1_id_costs = calculate_id_costs(ps, βs, 1; dr)
        world_2_id_costs = calculate_id_costs(ps, βs, 2; dr)
        world_3_id_costs = calculate_id_costs(ps, βs, 3; dr)
    end
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

    fig = nothing
    if (display_controls in [1,2])
        fig = display_stage_1_costs_controls(
            [
                world_1_id_costs,
                world_2_id_costs,
                world_3_id_costs,
                world_1_misid_costs,
                world_2_misid_costs,
                world_3_misid_costs,
            ],
            [
                world_1_id_controls,
                world_2_id_controls,
                world_3_id_controls,
                world_1_misid_controls,
                world_2_misid_controls,
                world_3_misid_controls,
            ],
            ps;
        )
    else
        fig = display_stage_1_costs(
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
    return fig
end

"""
Visualization. Run sweep over a set of perturbations for the attacker's cost functions. 
"""
function run_sweep(perturbations, k, perturbation_type; dr = 0.05)
    ps = [1/3, 1/3, 1/3]
    fig = Figure(size = (1300, 800))
    for perturbation in perturbations
        βs = [
                [3.0 + perturbation, 2.0, 2.0], 
                [2.0, 3.0, 2.0], 
                [2.0, 2.0, 3.0]
            ]
        # βs = [
        #     [2.0 + perturbation, 2.0, 2.0], 
        #     [2.0, 2.0 + perturbation, 2.0], 
        #     [2.0, 2.0, 2.0 + perturbation]
        # ]
        Ks = calculate_stage_1_costs(ps, βs; dr)
        
        # Nasty but gets the job done
        fig = Figure(size = (1300, 800))
        
        run_stage_1_breakout(display_controls = 1, dr = dr, βs = βs, ps = ps)
        defender_controls = load("figures/stage_1_controls.png")
        image(fig[1, 1], rotr90(defender_controls), axis = (aspect = DataAspect(), title = "defender"))
        hidedecorations!(fig.content[1])

        run_stage_1_breakout(display_controls = 2, dr = dr, βs = βs, ps = ps)
        attacker_controls = load("figures/stage_1_controls.png")
        image(fig[1, 2], rotr90(attacker_controls), axis = (aspect = DataAspect(), title = "attacker"))
        hidedecorations!(fig.content[2])

        display_surface(ps, Ks)
        stage_1_surface = load("figures/stage_1_surface.png")
        image(fig[2, 2], rotr90(stage_1_surface), axis = (aspect = DataAspect(), title = "stage 1"))
        hidedecorations!(fig.content[3])

        Axis(fig[2, 1], aspect = DataAspect(), title = perturbation_type * " \n perturbation: $perturbation \n k = $k", backgroundcolor = :gray50)
        # hidedecorations!(fig.content[4])

        save("figures/sweep/sweep_$(perturbation_type)_s$(perturbation)_k$(k).png", fig)

        # Show the figure
        fig
    end
end

"""
Calculate costs corresponding to a perfect info term in the Stage 1 cost function with index world_idx.

Input: 
    ps: prior distribution of k worlds for each signal, nx1 vector
    βs: vector containing P2's cost parameters for each world. vector of nx1 vectors
    world_idx: index of the world for which the cost is calculated
Output: 
    id_costs: 2D Matrix of costs for each r in the simplex
"""
function calculate_id_costs(ps, βs, world_idx; dr = 0.05, return_controls=0)
    @assert sum(ps) ≈ 1.0 "Prior distribution ps must be a probability distribution"
    # complete_info_game = build_complete_info_game()
    # incomplete_info_game = build_incomplete_info_game(ps, βs)
    rs = 0:dr:1
    num_worlds = length(ps)
    id_costs = NaN * ones(Float64, Int(1 / dr + 1), Int(1 / dr + 1))
    if(return_controls>0) ## ideally, it should be 1 or 2 for P1 or P2
        if(return_controls <= 2)
            controls = NaN * ones(Float64, Int(1 / dr + 1), Int(1 / dr + 1), 3)
        else 
            println("Invalid return_controls option.")
            return_controls = 0
        end
    end

    for (i, r1) in enumerate(rs)
        for (j, r2) in enumerate(rs)
            if r1 + r2 > 1
                continue
            end
            r3 = 1 - r1 - r2
            r = [r1, r2, r3]
            # x = compute_stage_2(r, ps, βs, complete_info_game, incomplete_info_game)
            x = compute_stage_2(IBRGameSolver(), r, ps, βs, [J_1, J_2])
            id_cost =
                r[world_idx] *
                ps[world_idx] *
                J_1(
                    x[Block(world_idx + 1)],
                    x[Block(world_idx + 2 * num_worlds + 1)],
                    βs[world_idx],
                )
            id_costs[i, j] = id_cost
            if (return_controls > 0)
                ind = return_controls == 2 ? world_idx + 2 * num_worlds + 1 : 1 + world_idx ## select the right block
                controls[i, j, :] = x[Block(ind)]
            end
        end
    end

    if(return_controls>0)
        return id_costs, controls
    else
        return id_costs
    end
end

"""
Calculate costs corresponding to a imperfect info term in the Stage 1 cost function with index world_idx.

Input: 
    ps: prior distribution of k worlds for each signal, nx1 vector
    βs: vector containing P2's cost parameters for each world. vector of nx1 vectors
    world_idx: index of the world for which the cost is calculated
Output: 
    id_costs: 2D Matrix of costs for each r in the simplex
"""
function calculate_misid_costs(ps, βs, world_idx; dr = 0.05, return_controls = 0)
    @assert sum(ps) ≈ 1.0 "Prior distribution ps must be a probability distribution"
    # complete_info_game = build_complete_info_game()
    # incomplete_info_game = build_incomplete_info_game(ps, βs)
    rs = 0:dr:1
    num_worlds = length(ps)
    misid_costs = NaN * ones(Float64, Int(1 / dr + 1), Int(1 / dr + 1))
    if(return_controls>0) ## ideally, it should be 1 or 2 for P1 or P2
        if(return_controls <= 2)
            controls = NaN * ones(Float64, Int(1 / dr + 1), Int(1 / dr + 1), 3)
        else 
            println("Invalid return_controls option.")
            return_controls = 0
        end
    end

    for (i, r1) in enumerate(rs)
        for (j, r2) in enumerate(rs)
            if r1 + r2 > 1
                continue
            end
            r3 = 1 - r1 - r2
            r = [r1, r2, r3]
            # x = compute_stage_2(r, ps, βs, complete_info_game, incomplete_info_game)
            x = compute_stage_2(IBRGameSolver(), r, ps, βs, [J_1, J_2])
            defender_signal_0 = x[Block(1)]
            attacker_signal_0_world_idx = x[Block(world_idx + num_worlds + 1)]
            misid_cost = J_1(defender_signal_0, attacker_signal_0_world_idx, βs[world_idx])
            misid_cost = (1 - r[world_idx]) * ps[world_idx] * misid_cost  # weight by p(w_k|s¹=0)
            misid_costs[i, j] = misid_cost
            if (return_controls > 0)
                ind = return_controls == 2 ? world_idx + num_worlds + 1 : 1 ## select the right block
                controls[i, j, :] = x[Block(ind)]
            end
        end
    end

    if(return_controls>0)
        return misid_costs, controls
    else
        return misid_costs
    end

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
    fig = Figure(size = (900, 700), title = "test", fontsize = 22)
    max_value = 1.0
    axs = [
        [
            Axis3(
                fig[1, world_idx],
                aspect = (1, 1, 1),
                perspectiveness = 0.5,
                elevation = pi / 9,
                azimuth = -π * (1 / 2 + 1 / 4),
                zgridcolor = :grey,
                ygridcolor = :grey,
                xgridcolor = :grey;
                xlabel = "r₁",
                ylabel = "r₂",
                title = L"\mathbf{r}_{%$world_idx} p(\omega_{%$world_idx})J^1(...)",
                limits = (nothing, nothing, (0.01, max_value)),
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
                title = L"(1 - \mathbf{r}_{%$world_idx}) p(\omega_{%$world_idx})J^1(...)",
                limits = (nothing, nothing, (0.01, max_value)),
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
            colorrange = (0, max_value),
        )
        hidezdecorations!(axs[1][world_idx]; ticklabels = false, ticks = false, grid = false)
    end
    for world_idx in 1:num_worlds
        hmap = surface!(
            axs[2][world_idx],
            rs,
            rs,
            costs[world_idx + num_worlds],
            colormap = :viridis,
            colorrange = (0, max_value),
        )
        hidezdecorations!(axs[2][world_idx]; ticklabels = false, ticks = false, grid = false)
    end

    save("figures/stage_1_costs.png", fig)
    fig
end

"""
Display surface of Stage 1's objective function, colored according to a player's action. Assumes number of worlds is 3.

Input: 
    Ks: 2D Matrix of stage 1's objective function values for each r in the simplex
Output: 
    fig: Figure with simplex heatmap
"""
function display_stage_1_costs_controls(costs, controls, ps)
    rs = 0:(1 / (size(costs[1])[1] - 1)):1
    num_worlds = length(ps)
    fig = Figure(size = (600, 400), title = "test")
    axs = [
        [
            Axis(
                fig[1, world_idx],
                aspect = 1,
                ygridcolor = :grey,
                xgridcolor = :grey;
                xlabel = "r₁",
                ylabel = "r₂",
                title = "World $world_idx, Signal $world_idx",
                limits = (nothing, nothing),
            ) for world_idx in 1:num_worlds
        ],
        [
            Axis(
                fig[2, world_idx],
                aspect = 1,
                ygridcolor = :grey,
                xgridcolor = :grey;
                xlabel = "r₁",
                ylabel = "r₂",
                title = "World $world_idx, Signal 0",
                limits = (nothing, nothing),
            ) for world_idx in 1:num_worlds
        ],
    ]

    for world_idx in 1:num_worlds
        colors = get_RGB_vect(controls[world_idx])
        for ii in 1:size(costs[world_idx])[1]
            for jj in 1:size(costs[world_idx])[2]
                if rs[ii] + rs[jj] > 1
                    continue
                end
                hmap = scatter!(axs[1][world_idx], rs[ii], rs[jj], color = colors[ii, jj])
            end
        end
    end
    for world_idx in 1:num_worlds
        colors = get_RGB_vect(controls[world_idx + num_worlds])
        for ii in 1:size(costs[world_idx + num_worlds])[1]
            for jj in 1:size(costs[world_idx + num_worlds])[2]
                if rs[ii] + rs[jj] > 1
                    continue
                end
                hmap =
                    scatter!(axs[2][world_idx], rs[ii], rs[jj], color = colors[ii, jj])
            end
        end
    end
    save("figures/stage_2_controls.png", fig)
    fig
end

"""
This is quick function to turn my controls into RGB vectors
"""
function get_RGB_vect(controls)
    R = controls[:,:,1]
    G = controls[:,:,2]
    B = controls[:,:,3]
    if size(R) == size(G) == size(B)
        n = size(R)[1]
        m = size(R)[2]
        RGB_values = Matrix{RGB}(undef, n, n)
        for i in 1:n
            for j in 1:m
                RGB_values[i, j] = RGB(R[i, j], G[i, j], B[i, j])
            end
        end
        return RGB_values
    else
        return(RGB(1,0,0))
    end

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
    rs = 0:dr:1
    Ks = NaN * ones(Float64, Int(1 / dr + 1), Int(1 / dr + 1))
    for (i, r1) in enumerate(rs)
        for (j, r2) in enumerate(rs)
            if r1 + r2 > 1
                continue
            end
            r3 = 1 - r1 - r2
            r = [r1, r2, r3]
            x = compute_stage_2(IBRGameSolver(), r, ps, βs, [J_1, J_2])
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
    fig = Figure(size = (450, 375))
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
        zlabel = "|K|",
        limits = (nothing, nothing, (0.01, 1)),
    )
    Ks_min = minimum(filter(!isnan, Ks))
    hmap = surface!(ax, rs, rs, Ks, colorrange = (0, 1))
    Colorbar(fig[1, 2], hmap; label = "|K|", width = 15, ticksize = 15, tickalign = 1)

    save("figures/stage_1_surface.png", fig)
    fig
end
