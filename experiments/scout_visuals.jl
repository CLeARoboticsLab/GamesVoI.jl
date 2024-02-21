module scout_visuals

using GamesVoI
using GLMakie
using LinearAlgebra
using JSON3
include("tower_defense.jl")

Makie.inline!(false)

# Visual parameters
    # Axis parameters
        # borders
        ax_aspect = 1 
        ax_limits = (0, 2, 0, 2)
        # title
        ax_titlegap = 1
        ax_titlesize = 30
        # x-axis
        ax_xautolimitmargin = (0, 0)
        ax_xgridwidth = 2
        ax_xticklabelsize = 0
        ax_xticks = -10:10
        ax_xticksize = 18
        # y-axis
        ax_yautolimitmargin = (0, 0)
        ax_ygridwidth = 2
        ax_yticklabelpad = 14
        ax_yticklabelsize = 0
        ax_yticks = -10:10
        ax_yticksize = 18

""" TODO: 
1. Do visualiation for each given world/signal received by Player 1 
2. Solve for all different combinations and store them in a "look-up dictionary" so that you dont have to solve the game all the time
"""

function demo(; attacker_preference = [[0.9; 0.05; 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]], use_file=false)

    save_file = nothing
    if use_file
        save_file = JSON3.read(open("data2.tmp", "r"), Dict{String, Vector{Float64}})
        println("read file")
    end
    
    # Game Parameters
        attacker_preference = [[0.9; 0.05; 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        num_worlds = 3
        prior_range_step = 0.01
        prior_range_step_precision = 1
        prior_range = 0.01:prior_range_step:1
        save_file_name = "precomputed_r.txt"
        save_precision = 4
        K = 100
        num_unit_scaling_factor = 20

    # Axis parameters
        # borders
        ax_aspect = 1 
        ax_limits = (0, 1, 0, 1)
        # title
        ax_titlegap = 1
        ax_titlesize = 30
        # x-axis
        ax_xautolimitmargin = (0, 0)
        ax_xgridwidth = 2
        ax_xticklabelsize = 0
        ax_xticks = -10:10
        ax_xticksize = 18
        # y-axis
        ax_yautolimitmargin = (0, 0)
        ax_ygridwidth = 2
        ax_yticklabelpad = 14
        ax_yticklabelsize = 0
        ax_yticks = -10:10
        ax_yticksize = 18

        opacity = 0.5

    # Initialize plot
    fig = Figure()

    # Add axis for each direction
    ax_north = Axis(fig[1,2],
        # borders
        aspect = ax_aspect, limits = ax_limits,
        # title
        title = "North",
        titlegap = ax_titlegap, titlesize = ax_titlesize,
        # x-axis
        xautolimitmargin = ax_xautolimitmargin, xgridwidth = ax_xgridwidth, 
        xticklabelsize = ax_xticklabelsize,
        xticks = ax_xticks, xticksize = ax_xticksize,
        # y-axis
        yautolimitmargin = ax_yautolimitmargin, ygridwidth = ax_ygridwidth,
        yticklabelpad = ax_yticklabelpad,
        yticklabelsize = ax_yticklabelsize, yticks = ax_yticks, yticksize = ax_yticksize
    )
    ax_west = Axis(fig[2,1],
        # borders
        aspect = ax_aspect, limits = ax_limits,
        # title
        title = "West",
        titlegap = ax_titlegap, titlesize = ax_titlesize,
        # x-axis
        xautolimitmargin = ax_xautolimitmargin, xgridwidth = ax_xgridwidth, 
        xticklabelsize = ax_xticklabelsize,
        xticks = ax_xticks, xticksize = ax_xticksize,
        # y-axis
        yautolimitmargin = ax_yautolimitmargin, ygridwidth = ax_ygridwidth,
        yticklabelpad = ax_yticklabelpad,
        yticklabelsize = ax_yticklabelsize, yticks = ax_yticks, yticksize = ax_yticksize
    )
    ax_east = Axis(fig[2,3],
        # borders
        aspect = ax_aspect, limits = ax_limits,
        # title
        title = "East",
        titlegap = ax_titlegap, titlesize = ax_titlesize,
        # x-axis
        xautolimitmargin = ax_xautolimitmargin, xgridwidth = ax_xgridwidth, 
        xticklabelsize = ax_xticklabelsize,
        xticks = ax_xticks, xticksize = ax_xticksize,
        # y-axis
        yautolimitmargin = ax_yautolimitmargin, ygridwidth = ax_ygridwidth,
        yticklabelpad = ax_yticklabelpad,
        yticklabelsize = ax_yticklabelsize, yticks = ax_yticks, yticksize = ax_yticksize
    )

    ax_simplex = Axis3(fig[2,2], aspect = (1,1,1), 
        limits = ((0.0, 1.0, 0.0, 1.0, 0.0, 1.1)),
        xreversed = true, 
        yreversed = true, 
        xlabel = "",
        ylabel = "",
        zlabel = "",
    )

    hidedecorations!(ax_north)
    hidedecorations!(ax_east)
    hidedecorations!(ax_west)

    # Create sliders
    sg = SliderGrid(
        fig[3, 2],
        (label = "prior_north", range = prior_range, format = x-> "", startvalue = 1.0), # z
        (label = "prior_east", range = prior_range, format = x-> "", startvalue = 1.0), # y
        (label = "prior_west", range = prior_range, format = x-> "", startvalue = 1.0) # x
    )
    observable_prior_sliders = [s.value for s in sg.sliders]
 
    # Plot priors on the Simplex
    scatterlines!(ax_simplex, [1;0;0;1], [0;1;0;0], [0;0;1;0], markersize = 15)

    # Normalize priors
    normalized_observable_p = lift(observable_prior_sliders...) do a, b, c
        round.(normalize([a,b,c], 1), digits = 2)
    end
    @lift println("priors: ", $normalized_observable_p)

    # p₁ : west, p₂ : east, p₃ : north
    p1, p2, p3 = [lift((x,i)->x[i], normalized_observable_p, idx) for idx in 1:num_worlds]
    scatter!(ax_simplex, p3, p2, p1 ; markersize = 15, color = :red)

    # Solve for scout_allocation, r 
    observable_r = on(normalized_observable_p) do x
        if use_file
            save_file[string(round.(x))]
        else
            solve_r(x, attacker_preference)
        end
    end
    scout_north, scout_east, scout_west = [lift((x,i)->x[i], observable_r.observable, idx) for idx in 1:num_worlds]

    function get_random_point_within_ball(; radius = 0.3, center = [ax_limits[2]/2, ax_limits[2]/2], num_points = 1)
        # Check center is Tuple
        @assert length(center) == 2 "Center must be a 2-element vector [x, y]"
        x_coord, y_coord = center

        # Generate random angle in radians
        angle = [2π * rand() for _ in 1:num_points]

        # Generate random distance within the specificed radius
        r = [radius * sqrt(rand()) for _ in 1:num_points]

        # Calculate new x and y coordinates and create an array of Point2f objects

        [Point2f(x_coord + r[i] * cos(angle[i]), y_coord + r[i] * sin(angle[i])) for i in 1:num_points]
    end
    
    # Check if scout_allocation results are normalized
    @lift println("Scout allocation: ", [$scout_north, $scout_east, $scout_west])

    # Display scout allocation as a text on the Figure
    text_directions = [lift((x) -> "$(round(Int, x*K))%", scout) for scout in [scout_north, scout_east, scout_west]]
    Label(fig[1,2], text_directions[1], fontsize = 20, tellwidth = false, tellheight = false)
    Label(fig[2,3], text_directions[2], fontsize = 20, tellwidth = false, tellheight = false)
    Label(fig[2,1], text_directions[3], fontsize = 20, tellwidth = false, tellheight = false)
    
    # Plot scout allocation 
    north_points = lift(x->get_random_point_within_ball(; radius = x*0.5, num_points = round(Int, 100*x)), scout_north)
    east_points = lift(x->get_random_point_within_ball(; radius = x*0.5, num_points = round(Int, 100*x)), scout_east)
    west_points = lift(x->get_random_point_within_ball(; radius = x*0.5, num_points = round(Int, 100*x)), scout_west)
    scatter!(ax_north, north_points, markersize = 15, color = (:orange, opacity))
    scatter!(ax_east, east_points, markersize = 15, color = (:pink, opacity+0.2))
    scatter!(ax_west, west_points, markersize = 15, color = (:green, opacity))

   
    # Plot Enemy
    scatter!(ax_north, rand(10), rand(10), color = :red)

    display(fig, fullscreen = true)
end

function demo_stage2(;use_file=true)
    save_file = nothing
    if use_file
        save_file = JSON3.read(open("data2.tmp", "r"), Dict{String, Vector{Float64}})
        println("read file")
    end
    # Game Parameters
        attacker_preference = [[0.9; 0.05; 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
        num_worlds = 3
        prior_range_step = 0.01
        prior_range_step_precision = 1
        prior_range = 0.01:prior_range_step:1
        save_file_name = "precomputed_r.txt"
        save_precision = 4
        K = 100
        num_unit_scaling_factor = 20

# Axis parameters
    # borders
    ax_aspect = 1 
    ax_limits = (0, 2, 0, 2)
    # title
    ax_titlegap = 1
    ax_titlesize = 30
    # x-axis
    ax_xautolimitmargin = (0, 0)
    ax_xgridwidth = 2
    ax_xticklabelsize = 0
    ax_xticks = -10:10
    ax_xticksize = 18
    # y-axis
    ax_yautolimitmargin = (0, 0)
    ax_ygridwidth = 2
    ax_yticklabelpad = 14
    ax_yticklabelsize = 0
    ax_yticks = -10:10
    ax_yticksize = 18

    opacity = 0.5

# Initialize plot
fig = Figure()

# Add axis for each direction
ax_north = Axis(fig[1,2],
    # borders
    aspect = ax_aspect, limits = ax_limits,
    # title
    title = "North",
    titlegap = ax_titlegap, titlesize = ax_titlesize,
    # x-axis
    xautolimitmargin = ax_xautolimitmargin, xgridwidth = ax_xgridwidth, 
    xticklabelsize = ax_xticklabelsize,
    xticks = ax_xticks, xticksize = ax_xticksize,
    # y-axis
    yautolimitmargin = ax_yautolimitmargin, ygridwidth = ax_ygridwidth,
    yticklabelpad = ax_yticklabelpad,
    yticklabelsize = ax_yticklabelsize, yticks = ax_yticks, yticksize = ax_yticksize
)
ax_west = Axis(fig[2,1],
    # borders
    aspect = ax_aspect, limits = ax_limits,
    # title
    title = "West",
    titlegap = ax_titlegap, titlesize = ax_titlesize,
    # x-axis
    xautolimitmargin = ax_xautolimitmargin, xgridwidth = ax_xgridwidth, 
    xticklabelsize = ax_xticklabelsize,
    xticks = ax_xticks, xticksize = ax_xticksize,
    # y-axis
    yautolimitmargin = ax_yautolimitmargin, ygridwidth = ax_ygridwidth,
    yticklabelpad = ax_yticklabelpad,
    yticklabelsize = ax_yticklabelsize, yticks = ax_yticks, yticksize = ax_yticksize
)
ax_east = Axis(fig[2,3],
    # borders
    aspect = ax_aspect, limits = ax_limits,
    # title
    title = "East",
    titlegap = ax_titlegap, titlesize = ax_titlesize,
    # x-axis
    xautolimitmargin = ax_xautolimitmargin, xgridwidth = ax_xgridwidth, 
    xticklabelsize = ax_xticklabelsize,
    xticks = ax_xticks, xticksize = ax_xticksize,
    # y-axis
    yautolimitmargin = ax_yautolimitmargin, ygridwidth = ax_ygridwidth,
    yticklabelpad = ax_yticklabelpad,
    yticklabelsize = ax_yticklabelsize, yticks = ax_yticks, yticksize = ax_yticksize
)

ax_simplex = Axis3(fig[2,2], aspect = (1,1,1), 
    limits = ((0.0, 1.0, 0.0, 1.0, 0.0, 1.1)),
    xreversed = true, 
    yreversed = true, 
    xlabel = "",
    ylabel = "",
    zlabel = "",
)

hidedecorations!(ax_north)
hidedecorations!(ax_east)
hidedecorations!(ax_west)

# Create sliders
sg = SliderGrid(
    fig[3, 2],
    (label = "prior_north", range = prior_range, format = x-> "", startvalue = 1.0), # z
    (label = "prior_east", range = prior_range, format = x-> "", startvalue = 1.0), # y
    (label = "prior_west", range = prior_range, format = x-> "", startvalue = 1.0) # x
)
observable_prior_sliders = [s.value for s in sg.sliders]

# Plot priors on the Simplex
scatterlines!(ax_simplex, [1;0;0;1], [0;1;0;0], [0;0;1;0], markersize = 15)

# Normalize priors
normalized_observable_p = lift(observable_prior_sliders...) do a, b, c
    round.(normalize([a,b,c], 1), digits = 2)
end
@lift println("priors: ", $normalized_observable_p)

# p₁ : west, p₂ : east, p₃ : north
p1, p2, p3 = [lift((x,i)->x[i], normalized_observable_p, idx) for idx in 1:num_worlds]
scatter!(ax_simplex, p3, p2, p1 ; markersize = 15, color = :red)

# Solve for scout_allocation, r 
observable_r = on(normalized_observable_p) do x
    if use_file
        save_file[string(round.(x))]
    else
        solve_r(x, attacker_preference)
    end
end
scout_north, scout_east, scout_west = [lift((x,i)->x[i], observable_r.observable, idx) for idx in 1:num_worlds]

function get_random_point_within_ball(; radius = 0.3, center = [ax_limits[2]/2, ax_limits[2]/2], num_points = 1)
    # Check center is Tuple
    @assert length(center) == 2 "Center must be a 2-element vector [x, y]"
    x_coord, y_coord = center

    # Generate random angle in radians
    angle = [2π * rand() for _ in 1:num_points]

    # Generate random distance within the specificed radius
    r = [radius * sqrt(rand()) for _ in 1:num_points]

    # Calculate new x and y coordinates and create an array of Point2f objects

    [Point2f(x_coord + r[i] * cos(angle[i]), y_coord + r[i] * sin(angle[i])) for i in 1:num_points]
end

# Check if scout_allocation results are normalized
@lift println("Scout allocation: ", [$scout_north, $scout_east, $scout_west])

# Display scout allocation as a text on the Figure
text_directions = [lift((x) -> "$(round(Int, x*K))%", scout) for scout in [scout_north, scout_east, scout_west]]
Label(fig[1,2], text_directions[1], fontsize = 20, tellwidth = false, tellheight = false)
Label(fig[2,3], text_directions[2], fontsize = 20, tellwidth = false, tellheight = false)
Label(fig[2,1], text_directions[3], fontsize = 20, tellwidth = false, tellheight = false)

# Main.@infiltrate
menu = Menu(fig[0,1], options = zip(["Signal 0 World 1", "Signal 0 World 2", "Signal 0 World 3", "Signal 1 World 1", "Signal 2 World 2", "Signal 3 World 3"],
[0, 1, 2, 3, 4 ,5]))

game = lift((x) -> build_stage_2(x, attacker_preference), normalized_observable_p)
b_array = lift((r, p, game) -> compute_stage_2(r, p, attacker_preference, game[1]), 
    observable_r.observable, normalized_observable_p, game)

# Create an array of observables from b_array
# b_array_obs = [lift((x,i,j) -> x[Block(i)][j], b_array, i,j) for j in 1:3 for i in 1:10]

b_array_obs_f = (i, x) -> x[][Block(i)]
b_array_obs_f_block = (i) -> b_array[][Block(i)]

# Main.@infiltrate

u_north = Observable(0.0)
u_east = Observable(0.0)
u_west = Observable(0.0)
v_north = Observable(0.0)
v_east = Observable(0.0)
v_west = Observable(0.0)

on(b_array) do y
    x = menu.selection[]
    if x == 0 #s 0 w 1
        # draw_world(;u = b_array_obs_f_block(1), v = b_array_obs_f_block(5), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        u_north[] = round.(b_array_obs_f_block(1), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(1), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(1), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(5), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(5), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(5), digits=2)[3]
        # v[] = round.(b_array_obs_f_block(5), digits=2)
    elseif x == 1
        # draw_world(;u = b_array_obs_f_block(1), v = b_array_obs_f_block(6), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        # u[] = round.(b_array_obs_f_block(1), digits=2)
        # v[] = round.(b_array_obs_f_block(6), digits=2)

        u_north[] = round.(b_array_obs_f_block(1), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(1), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(1), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(6), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(6), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(6), digits=2)[3]
    elseif x == 2
        # draw_world(;u = b_array_obs_f_block(1), v = b_array_obs_f_block(7), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        # u[] = round.(b_array_obs_f_block(1), digits=2)
        # v[] = round.(b_array_obs_f_block(7), digits=2)

        u_north[] = round.(b_array_obs_f_block(1), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(1), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(1), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(7), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(7), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(7), digits=2)[3]
    elseif x == 3
        # draw_world(;u = b_array_obs_f_block(2), v = b_array_obs_f_block(8), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        # u[] = round.(b_array_obs_f_block(1), digits=2)
        # v[] = round.(b_array_obs_f_block(8), digits=2)

        u_north[] = round.(b_array_obs_f_block(1), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(1), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(1), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(8), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(8), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(8), digits=2)[3]
    elseif x == 4
        # draw_world(;u = b_array_obs_f_block(3), v = b_array_obs_f_block(9), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        # u[] = round.(b_array_obs_f_block(3), digits=2)
        # v[] = round.(b_array_obs_f_block(9), digits=2)
        u_north[] = round.(b_array_obs_f_block(3), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(3), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(3), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(9), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(9), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(9), digits=2)[3]
    elseif x == 5
        # draw_world(;u = b_array_obs_f_block(4), v = b_array_obs_f_block(10), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        # u[] = round.(b_array_obs_f_block(4), digits=2)
        # v[] = round.(b_array_obs_f_block(10), digits=2)
        u_north[] = round.(b_array_obs_f_block(4), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(4), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(4), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(10), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(10), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(10), digits=2)[3]
    end
end

on(menu.selection) do x
    if x == 0 #s 0 w 1
        # draw_world(;u = b_array_obs_f_block(1), v = b_array_obs_f_block(5), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        u_north[] = round.(b_array_obs_f_block(1), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(1), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(1), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(5), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(5), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(5), digits=2)[3]
        # v[] = round.(b_array_obs_f_block(5), digits=2)
    elseif x == 1
        # draw_world(;u = b_array_obs_f_block(1), v = b_array_obs_f_block(6), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        # u[] = round.(b_array_obs_f_block(1), digits=2)
        # v[] = round.(b_array_obs_f_block(6), digits=2)

        u_north[] = round.(b_array_obs_f_block(1), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(1), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(1), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(6), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(6), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(6), digits=2)[3]
    elseif x == 2
        # draw_world(;u = b_array_obs_f_block(1), v = b_array_obs_f_block(7), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        # u[] = round.(b_array_obs_f_block(1), digits=2)
        # v[] = round.(b_array_obs_f_block(7), digits=2)

        u_north[] = round.(b_array_obs_f_block(1), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(1), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(1), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(7), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(7), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(7), digits=2)[3]
    elseif x == 3
        # draw_world(;u = b_array_obs_f_block(2), v = b_array_obs_f_block(8), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        # u[] = round.(b_array_obs_f_block(1), digits=2)
        # v[] = round.(b_array_obs_f_block(8), digits=2)

        u_north[] = round.(b_array_obs_f_block(1), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(1), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(1), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(8), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(8), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(8), digits=2)[3]
    elseif x == 4
        # draw_world(;u = b_array_obs_f_block(3), v = b_array_obs_f_block(9), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        # u[] = round.(b_array_obs_f_block(3), digits=2)
        # v[] = round.(b_array_obs_f_block(9), digits=2)
        u_north[] = round.(b_array_obs_f_block(3), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(3), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(3), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(9), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(9), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(9), digits=2)[3]
    elseif x == 5
        # draw_world(;u = b_array_obs_f_block(4), v = b_array_obs_f_block(10), fig=fig, ax_n=ax_north, ax_e=ax_east, ax_w=ax_west)
        # u[] = round.(b_array_obs_f_block(4), digits=2)
        # v[] = round.(b_array_obs_f_block(10), digits=2)
        u_north[] = round.(b_array_obs_f_block(4), digits=2)[1]
        u_east[] = round.(b_array_obs_f_block(4), digits=2)[2]
        u_west[] = round.(b_array_obs_f_block(4), digits=2)[3]

        v_north[] = round.(b_array_obs_f_block(10), digits=2)[1]
        v_east[] = round.(b_array_obs_f_block(10), digits=2)[2]
        v_west[] = round.(b_array_obs_f_block(10), digits=2)[3]
    end
end

defender_triangle_north = @lift Point2f[(1 - $u_north, 0), (1 + $u_north, 0), (1, 1 * $u_north)]
enemy_triangle_north = @lift Point2f[(1 - $v_north, 2), (1 + $v_north, 2), (1, 2 - (1 * $v_north))]
poly!(ax_north, defender_triangle_north, color = (:orange, opacity))
poly!(ax_north, enemy_triangle_north, color = (:red, opacity))

defender_triangle_east = @lift Point2f[(0, 1 - $u_east * 1), (0, 1 + 1 * $u_east), (1 * $u_east, 1)]
enemy_triangle_east = @lift Point2f[(2, 1 - $v_east * 1), (2, 1 + 1 * $v_east), (2 - (1 * $v_east), 1)]
poly!(ax_east, defender_triangle_east, color = (:orange, opacity))
poly!(ax_east, enemy_triangle_east, color = (:red, opacity))

defender_triangle_west = @lift Point2f[(2, 1 - $u_west * 1), (2, 1 + 1 * $u_west), (2 - 1 * $u_west, 1)]
enemy_triangle_west = @lift Point2f[(0, 1 - $v_west * 1), (0, 1 + 1 * $v_west), (1 - (1 * $v_west), 1)]
poly!(ax_west, defender_triangle_west, color = (:orange, opacity))
poly!(ax_west, enemy_triangle_west, color = (:red, opacity))

# scatter!(ax_east, u_east, u_east, markersize = 15, color = (:pink, opacity + .2))
# scatter!(ax_west, u_west, u_west, markersize = 15, color = (:green, opacity))

# # scatter!(ax_north, v_north, v_north, markersize = 15, color = (:red, opacity))
# scatter!(ax_east, v_east, v_east, markersize = 15, color = (:red, opacity))
# scatter!(ax_west, v_west, v_west, markersize = 15, color = (:red, opacity))


display(fig, fullscreen = true)
end

function draw_world(;u, v, fig, ax_n, ax_e, ax_w)
    # u = round.(u, digits = 2)
    # v = round.(v, digits = 2)
    opacity = 0.5

    # print("\nU:\n")
    # print(u)
    # print("\nV:\n")
    # print(v)
    
    scatter!(ax_n, u, u, markersize = 15, color = (:orange, opacity))
    scatter!(ax_e, u, u, markersize = 15, color = (:pink, opacity + .2))
    scatter!(ax_w, u, u, markersize = 15, color = (:green, opacity))

    scatter!(ax_n, v, v, markersize = 15, color = (:red, opacity))
    scatter!(ax_e, v, v, markersize = 15, color = (:red, opacity))
    scatter!(ax_w, v, v, markersize = 15, color = (:red, opacity))
    display(fig)
end


function compute_all_r_save_to_file(;_prior_range = 0.01:.1:1.1,
     attacker_preference = [[0.9; 0.05; 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
     save_file_name = "data2.tmp")
    prior_range = round.(_prior_range, digits=1)
    # save_file = open(save_file_name, "w+")
    hashmap = Dict{Vector{Float64}, Vector{Float64}}()
    for prior_north in prior_range
        for prior_east in prior_range
            for prior_west in prior_range
                # print("calculating: ", [prior_north, prior_east, prior_west])
                current_prior = [prior_north, prior_east, prior_west]
                if norm(current_prior, 1) <= 1.11 && norm(current_prior) >= .89
                    r = solve_r(current_prior, attacker_preference, verbose = false)
                    hashmap[current_prior] = r
                end
            end
        end
    end
    hashmap[[0.0, 0.0, 0.0]] = [0.0, 0.0, 0.0]
    JSON3.write(save_file_name, hashmap)
    # write(save_file, hashmap)
    # close(save_file)
end

function draw_given()
    # saved_computations = open(save_file_name, "r")
    # hashmap = read(save_file, Dict{Tuple{Float64, Float64, Float64}, Tuple{Float64, Float64, Float64}})
    saved_computations = read(save_file_name, String)
    hashmap = JSON3.read(saved_computations, Dict{Tuple{Float64, Float64, Float64}, Tuple{Float64, Float64, Float64}})


    fig = Figure(; size = (3840, 2160))

    # Add axis for each direction
    ax_north = Axis(fig[1,2],
        # borders
        aspect = ax_aspect, limits = ax_limits,
        # title
        title = "North",
        titlegap = ax_titlegap, titlesize = ax_titlesize,
        # x-axis
        xautolimitmargin = ax_xautolimitmargin, xgridwidth = ax_xgridwidth, 
        xticklabelsize = ax_xticklabelsize,
        xticks = ax_xticks, xticksize = ax_xticksize,
        # y-axis
        yautolimitmargin = ax_yautolimitmargin, ygridwidth = ax_ygridwidth,
        yticklabelpad = ax_yticklabelpad,
        yticklabelsize = ax_yticklabelsize, yticks = ax_yticks, yticksize = ax_yticksize
    )
    ax_east = Axis(fig[2,1],
        # borders
        aspect = ax_aspect, limits = ax_limits,
        # title
        title = "East",
        titlegap = ax_titlegap, titlesize = ax_titlesize,
        # x-axis
        xautolimitmargin = ax_xautolimitmargin, xgridwidth = ax_xgridwidth, 
        xticklabelsize = ax_xticklabelsize,
        xticks = ax_xticks, xticksize = ax_xticksize,
        # y-axis
        yautolimitmargin = ax_yautolimitmargin, ygridwidth = ax_ygridwidth,
        yticklabelpad = ax_yticklabelpad,
        yticklabelsize = ax_yticklabelsize, yticks = ax_yticks, yticksize = ax_yticksize
    )
    ax_west = Axis(fig[2,3],
        # borders
        aspect = ax_aspect, limits = ax_limits,
        # title
        title = "West",
        titlegap = ax_titlegap, titlesize = ax_titlesize,
        # x-axis
        xautolimitmargin = ax_xautolimitmargin, xgridwidth = ax_xgridwidth, 
        xticklabelsize = ax_xticklabelsize,
        xticks = ax_xticks, xticksize = ax_xticksize,
        # y-axis
        yautolimitmargin = ax_yautolimitmargin, ygridwidth = ax_ygridwidth,
        yticklabelpad = ax_yticklabelpad,
        yticklabelsize = ax_yticklabelsize, yticks = ax_yticks, yticksize = ax_yticksize
    )

    # Create sliders
    sg = SliderGrid(
        fig[3, 2],
        (label = "prior_north", range = prior_range, format = "{:.2f}", startvalue = 0),
        (label = "prior_east", range = prior_range, format = "{:.2f}", startvalue = 0),
        (label = "prior_west", range = prior_range, format = "{:.2f}", startvalue = 0)
    )

    prior_north_listener = sg.sliders[1].value
    prior_east_listener = sg.sliders[2].value
    prior_west_listener = sg.sliders[3].value

    # TODO: We want normalized priors to appear on GUI (next to slider)
    priors = @lift(round.(normalize([$prior_north_listener; $prior_east_listener; $prior_west_listener]),
    digits = prior_range_step_precision))

    get_r(x) = hashmap[x[]]

    r = lift(get_r, priors)

    # print(r)
    scat1 = scatter!(ax_north, r, r, markersize = 10, color = :red)
    display(fig)
end

#module end
end