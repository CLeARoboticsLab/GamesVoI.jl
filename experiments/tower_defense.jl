using GamesVoI
using BlockArrays
using LinearAlgebra
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

"""Solve for q (stage 1)

   Returns dj/dq: optimal value of q"""
function djdqaa(iter_limit=50, target_error=.00001, α=1, pₖ = [1/3; 1/3; 1/3], q=[1/2; 1/2; 1/2])
    # iter_limit = 50
    cur_iter = 0

    z = BlockArray(zeros(24), [3,3,3,3,3,3,3,3])

    error = 3.0
    error_v = [1, 1, 1]
    
    while cur_iter < iter_limit || error > target_error
        println("At iteration $cur_iter we have p = $q")
        dz = dzdq(q, false, z, pₖ)
        dj = djdq_partial(z, pₖ)'
        dv = [0.0 0.0 0.0]
        for s = 0:1
            for w = 1:3
                i = s * 12 + 3 * w  + 1
                dv += djdv(s, w, z, q, pₖ)' * dz[i:i + 2, 1:3]
            end
        end
        # dv = djdv(0, 1, z, q, pₖ) * + djdv(1, 0, z, q, pₖ)
        du = djdu(0, z, q, pₖ)' * dz[1:3, 1:3] + djdu(1, z, q, pₖ)' * dz[13:15, 1:3]
        dJdq = dj + dv + du

        q_temp = q - α .* dJdq'
        for i in 1:3
            q_temp[i] = max(0, min(1, q_temp[i]))
        end
        error_v = q_temp - q
        error = 0
        for i in 1:3
            error += error_v[i] * error_v[i]
        end
        q = q_temp


        cur_iter += 1
    end
    println("At the final iteration $cur_iter we have p = $q")
    return q
end

function djdq_partial(z, pₖ)
    u_1 = z[1:3]
    v_1_w1 = z[4:6]
    v_1_w2 = z[7:9]
    v_1_w3 = z[10:12]
    u_0 = z[13:15]
    v_0_w1 = z[16:18]
    v_0_w2 = z[19:21]
    v_0_w3 = z[22:24]

    v = [0.0, 0.0, 0.0]
    v[1] = pₖ[1] * (dot(u_0, v_0_w1) - dot(u_1, v_1_w1))
    v[2] = pₖ[2] * (dot(u_0,  v_0_w2) - dot(u_1, v_1_w2))
    v[3] = pₖ[3] * (dot(u_0, v_0_w3) - dot(u_1, v_1_w3))
    return v
end

function djdv(s, w, z, qₖ, pₖ)
    #TODO: WRONG
    # state vars
    u_1 = z[1:3]
    v_1_w1 = z[4:6]
    v_1_w2 = z[7:9]
    v_1_w3 = z[10:12]
    u_0 = z[13:15]
    v_0_w1 = z[16:18]
    v_0_w2 = z[19:21]
    v_0_w3 = z[22:24]

    # return vector
    v = [0.0, 0.0, 0.0]

    k = w
    if s == 0 # Signal is 0
        
        v[1] += qₖ[k] * pₖ[k] * u_0[1] - pₖ[k] * u_0[1]
        v[2] += qₖ[k] * pₖ[k] * u_0[2] - pₖ[k] * u_0[2]
        v[3] += qₖ[k] * pₖ[k] * u_0[3] - pₖ[k] * u_0[3] 
    else
        v[1] -= qₖ[k] * pₖ[k] * u_1[1]
        v[2] -= qₖ[k] * pₖ[k] * u_1[2]
        v[3] -= qₖ[k] * pₖ[k] * u_1[3]
    end
    return v
end


function djdu(s, z, qₖ, pₖ)
    # state vars
    u_1 = z[1:3]
    v_1_w1 = z[4:6]
    v_1_w2 = z[7:9]
    v_1_w3 = z[10:12]
    u_0 = z[13:15]
    v_0_w1 = z[16:18]
    v_0_w2 = z[19:21]
    v_0_w3 = z[22:24]
    
    # vector to return
    v = [0.0, 0.0, 0.0]
    if s == 0
        for k in 1:3
            v[1] += qₖ[k] * pₖ[k] * v_0_w1[k] - pₖ[k] * v_0_w1[1]
            v[2] += qₖ[k] * pₖ[k] * v_0_w2[k] - pₖ[k] * v_0_w2[2]
            v[3] += qₖ[k] * pₖ[k] * v_0_w3[k] - pₖ[k] * v_0_w3[3]
        end
    else
        for k in 1:3
            v[1] -= qₖ[k] * pₖ[k] * v_1_w1[1]
            v[2] -= qₖ[k] * pₖ[k] * v_1_w2[2]
            v[3] -= qₖ[k] * pₖ[k] * v_1_w3[3]
        end
    end
    return v
end


"""Solve Stackelberg-like game with 2 stages. 

   Returns dz/dq: Jacobian of Stage 2's decision variable (z = P1 and P2's variables in a Bayesian game) w.r.t. Stage 1's decision variable (q = signal structure)"""
function dzdq(q, verbose, z1, pₖ)
    # Setup game
    wₖ = [0, 1, 1, 1, 0, 1, 1, 1, 0] # worlds
    # pₖ = [1/3; 1/3; 1/3] # P1's prior distribution over worlds
    parametric_game, fs = build_parametric_game(pₖ, wₖ)

    # Solve Stage 1 
    # q = [1/3; 1/3; 1/3] # TODO obtain q by solving Stage 1

    # Solve Stage 2 given q
    solution = solve(
            parametric_game,
            q;
            initial_guess = zeros(total_dim(parametric_game)),
            verbose=verbose,
        )
    z = BlockArray(solution.variables[1:24], [3,3,3,3,3,3,3,3])
    z1 .= z
 
    # Return Jacobian
    Zygote.jacobian(q -> solve(
            parametric_game,
            q;
            initial_guess=zeros(total_dim(parametric_game)),
            verbose=false,
            return_primals=false
        ).variables[1:24], q)[1]
end