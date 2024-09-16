using LinearAlgebra: rank
include("tower_defense.jl")

# Sim parameteres
ps = [1/3, 1/3, 1/3]
βs = [[3.0, 2.0, 2.0], [2.0, 3.0, 2.0], [2.0, 2.0, 3.0]]
dr = 0.2
n = 3

# Function to compute player 1 cost in imperfect information game
function E_J_1(x, r, βs)
    p_w_k_0(w_idx, r) = (1 - r[w_idx]) * ps[w_idx] / (1 - r' * ps)
    return sum([J_1(x[Block(1)], x[Block(w_idx + n + 1)], βs[w_idx]) * p_w_k_0(w_idx, r) for w_idx in 1:n])
end


# Function to check proposition
function check_proposition(r, x)
    proposition_true = true
    
    # J_2_Hessians
    J_2_hessians = []
    for (i, β) in enumerate(βs)
        x_1_0 = x[Block(1)]
        x_2_i = x[Block(i + n + 1)]
        J_2_hessian = hessian(v -> J_2(x_1_0, v, β), x_2_i)

        # Check Prop 1, condition 1
        hessian_rank = rank(J_2_hessian)
        if hessian_rank < n
            proposition_true = false
            println("   Condition 1 failed for player $i")
            break
        end        
        push!(J_2_hessians, J_2_hessian)
    end

    # J_1_hessian
    E_J_1_jac = hessian(w -> E_J_1(w, r, βs), x)
    E_J_1_hessian = E_J_1_jac[1:n, 1:n]
    dx2x1_J_1 = []
    dx1x2_J_1 = []
    # Indexing: 
    # dx2ix1_J_1 is the n + 1 + i-th block column, and first n rows
    # dx1x2i_J_1 is the n + 1 + i-th block row, and first n columns
    # Each block is n x n
    for i in 1:n
        dx2ix1_J_1 = E_J_1_jac[1:n, n * (n + i):n * (n + i) + n - 1]
        dx1x2i_J_1 = E_J_1_jac[n * (n + i):n * (n + i) + n - 1, 1:n]
        push!(dx2x1_J_1, dx2ix1_J_1)
        push!(dx1x2_J_1, dx1x2i_J_1)
    end

    # Compute C 
    C = sum([dx2x1_J_1[i] * (J_2_hessians[i] \ dx1x2_J_1[i]) for i in 1:n]) - E_J_1_hessian

    # Check Prop 1, condition 2
    if rank(C) < n
        proposition_true = false
        println("   Condition 2 failed")
    end

    return proposition_true
end

# For every possible value r, solve stage 2 game
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

        println("r: $r")
        proposition_check = check_proposition(r, x)
    end
end

# Use solution of stage 2 and compute relevant derivatives  

# Check if proposition conditions hold