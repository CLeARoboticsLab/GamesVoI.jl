module scout_visuals

using GamesVoI
using GLMakie
using LinearAlgebra
using JSON3, FileIO
include("tower_defense.jl")

Makie.inline!(false)
# TODO: USE [[2, 1, 1], [1, 2, 1], [1, 1, 2]] .+ 1 = [[3, 2, 2], [2, 3, 2], [2, 2, 3]] for recomputation of hashmap

function demo(; attacker_preference = [[3, 2, 2], [2, 3, 2], [2, 2, 3]], use_file=false)

    save_file = nothing
    if use_file
        save_file = JSON3.read(open("data2.tmp", "r"), Dict{String, Vector{Float64}})
        println("read file")
    end
    
    # Game Parameters
        attacker_preference = [[3, 2, 2], [2, 3, 2], [2, 2, 3]]
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

function demo_stage2(;use_file=true, attacker_preference = [[3, 2, 2], [2, 3, 2], [2, 2, 3]])
    save_file = nothing
    
    if use_file
        save_file = JSON3.read(open("./experiments/data.tmp", "r"), Dict{String, Vector{Float64}})
        println("read file")
    else
        println("warning: save file not found, please run scout_visuals.compute_all_r_save_to_file()")
    end
    # Game Parameters
        num_worlds = 3
        prior_range_step = 0.01
        prior_range = 0.01:prior_range_step:1

# Initialize plot
fig = Figure(figure_padding = 2)
rowgap!(fig.layout, 0)
world_signal_pairs = [(1, 0), (2, 0), (3, 0), (1, 1), (2, 2), (3, 3)]
axes = [Axis(fig[x[2] != 0 ? 2 : 1, x[1] + 1], aspect = DataAspect(), 
    title= L"\omega_{%$(x[1])} \enspace \sigma_{%$(x[2])}", titlesize = 30, titlegap = 3)
    for x in world_signal_pairs]
for ax in axes
    hidedecorations!(ax)
end
Label(fig[1,5], "", fontsize = 1)
rowsize!(fig.layout, 1, Aspect(1, 0.7))
rowsize!(fig.layout, 2, Aspect(1, 0.7))

# 0304: Add scout_allocation figure instead of simplex figure 
ax_scout = Axis(fig[1,1], aspect = DataAspect(),
    title=L"\mathbf{r}", titlesize = 30, titlegap = 3)
hidedecorations!(ax_scout)

# 0304: Remove Simplex figure for now 
# ax_simplex = Axis3(fig[1,1], aspect = (1,1,1), 
#     limits = ((0.0, 1.0, 0.0, 1.0, 0.0, 1.1)),
#     xreversed = true, 
#     yreversed = true, 
#     xlabel = "",
#     ylabel = "",
#     zlabel = "",
#     tellheight = false
# )

# Plot priors on the Simplex
# scatterlines!(ax_simplex, [1;0;0;1], [0;1;0;0], [0;0;1;0], markersize = 15)

# Create sliders
sg = SliderGrid(
    fig[2, 1],
    (label = L"p(\omega_1)", range = prior_range, format = x-> "", startvalue = 1.0), # z
    (label = L"p(\omega_2)", range = prior_range, format = x-> "", startvalue = 1.0), # y
    (label = L"p(\omega_3)", range = prior_range, format = x-> "", startvalue = 1.0), # x
    tellheight = false,
)
observable_prior_sliders = [s.value for s in sg.sliders]

# p₁ : west, p₂ : east, p₃ : north
# Normalize priors
normalized_observable_p = lift(observable_prior_sliders...) do a, b, c
    round.(normalize([a,b,c], 1), digits = 2)
end
p1, p2, p3 = [lift((x,i)->x[i], normalized_observable_p, idx) for idx in 1:num_worlds]
# scatter!(ax_simplex, p3, p2, p1 ; markersize = 15, color = (:red, .75), label=L"\mathbf{p}(\omega)")

# Solve for scout_allocation, r 
observable_r = on(normalized_observable_p) do x
    if use_file
        save_file[string(round.(x))]
    else
        solve_r(x, attacker_preference)
    end
end
scout_north, scout_east, scout_west = [lift((x,i)->x[i], observable_r.observable, idx) for idx in 1:num_worlds]
# scatter!(ax_simplex, scout_north, scout_east, scout_west ; markersize = 15, color = (:green, .75), label=L"\mathbf{r}")
# axislegend()

game = lift((x) -> build_stage_2(x, attacker_preference), normalized_observable_p)
b_array = lift((r, p, game) -> compute_stage_2(r, p, attacker_preference, game[1]), 
    observable_r.observable, normalized_observable_p, game)

b_array_obs_f = (i, x) -> x[][Block(i)]
b_array_obs_f_block = (i) -> b_array[][Block(i)]

function bval2int(defender, attacker)
    defender = round(defender, digits = 2)
    attacker = round(attacker, digits = 2)
    num_d = round(defender * 10, digits = 0)
    num_a = round(attacker * 10, digits = 0)
    if num_a == num_d && num_a > 0
        if attacker > defender
            if num_a == 10
                num_d -= 1
            else
                num_a += 1
            end
        else #attacker <= defender
            if num_d == 10
                num_a -= 1
            else
                num_d += 1
            end
        end
    end

    return (Int(num_d), Int(num_a))
end

# Begin plotting on the map
stage2_map = load("./experiments/stage2_map.jpg") #832x1132

# Set stage for defender and attacker
for i in axes
    image!(i, rotr90(stage2_map))
end
a_north = 565
b_north = 600
increment = 60
top_increment = 80
point_positions_north = [(a_north, b_north - increment), (a_north + increment,b_north - increment), (a_north - increment,b_north - increment), (a_north + 2increment, b_north - increment), (a_north - 2increment, b_north - increment),
    (a_north, b_north), (a_north + top_increment,b_north), (a_north - top_increment,b_north), (a_north + 2top_increment, b_north), (a_north - 2top_increment, b_north)]

a_east = 350
b_east = 250
point_positions_east = [(a_east, b_east),               (a_east, increment + b_east),        (a_east, -increment + b_east),         (a_east, 2increment + b_east),        (a_east, -2increment + b_east),
                        (a_east - increment, b_east), (a_east - increment, b_east + top_increment), (a_east - increment, b_east - top_increment), (a_east - increment, b_east + 2top_increment), (a_east - increment, b_east - 2top_increment)]

defender_size = (25, 10)
atker_size = 15

# Plot scout allocation (Stage 1)
image!(ax_scout, rotr90(stage2_map))
north_scout = @lift round(Int, $scout_north * 10)
east_scout = @lift round(Int, $scout_east * 10)
west_scout = @lift round(Int, $scout_west * 10)
scout_size = 15

# North
north_scout_points = @lift [Point2f(x[1], x[2]) for x in point_positions_north[1:$north_scout]]
scatter!(ax_scout, north_scout_points, markersize = scout_size, color = :green)
@lift println("n scout: ", $north_scout, " n scout_dec: ", $scout_north)

# East
east_scout_points = @lift [Point2f(x[1], x[2]) for x in point_positions_east[1:$east_scout]]
scatter!(ax_scout, east_scout_points, markersize = scout_size, color = :green)
@lift println("e scout: ", $east_scout, " e scout_dec: ", $scout_east)

# West
west_scout_points = @lift [Point2f(1130 - x[1], x[2]) for x in point_positions_east[1:$west_scout]]
scatter!(ax_scout, west_scout_points, markersize = scout_size, color = :green)
@lift println("w scout: ", $west_scout, " w scout_dec: ", $scout_west)


# Plot defender and attack allocation (Stage 2)
for (idx, world_signal) in enumerate(world_signal_pairs)
    # North
    defender_index = world_signal[2] + 1
    atker_index = (world_signal[2] == 0 ? 4 : 7) + world_signal[1] 
    north_defenders  = @lift bval2int($b_array[Block(defender_index)][1], $b_array[Block(atker_index)][1])[1]
    north_atker = @lift bval2int($b_array[Block(defender_index)][1], $b_array[Block(atker_index)][1])[2]
    # @lift println("n def: ", $north_defenders, " atk: ", $north_atker)
    north_defenders_points = @lift [Point2f(x[1], x[2]) for x in point_positions_north[1:$north_defenders]]
    north_atker_points = @lift [Point2f(x[1], x[2] + 150) for x in point_positions_north[1:$north_atker]]

    scatter!(axes[idx], north_defenders_points, marker = :rect, markersize = defender_size, color = :blue)
    scatter!(axes[idx], north_atker_points, marker = :dtriangle, markersize = atker_size, color = :red)

    # East
    east_defenders = @lift bval2int($b_array[Block(defender_index)][2], $b_array[Block(atker_index)][2])[1]
    east_atker = @lift bval2int($b_array[Block(defender_index)][2], $b_array[Block(atker_index)][2])[2]
    # @lift println("e def: ", $east_defenders, " atk: ", $east_atker)
    east_defenders_points = @lift [Point2f(x[1], x[2]) for x in point_positions_east[1:$east_defenders]]
    east_atker_points = @lift [Point2f(x[1] - 150, x[2]) for x in point_positions_east[1:$east_atker]]

    scatter!(axes[idx], east_defenders_points, marker = :rect, markersize = reverse(defender_size), color = :blue)
    scatter!(axes[idx], east_atker_points, marker = :rtriangle, markersize = atker_size, color = :red)

    # West
    west_defenders = @lift bval2int($b_array[Block(defender_index)][3], $b_array[Block(atker_index)][3])[1]
    west_atker = @lift bval2int($b_array[Block(defender_index)][3], $b_array[Block(atker_index)][3])[2]
    # @lift println("w def: ", $west_defenders, " atk: ", $west_atker)
    west_defenders_points = @lift [Point2f(1130 - x[1], x[2]) for x in point_positions_east[1:$west_defenders]]
    west_atker_points = @lift [Point2f(1130 - x[1] + 150, x[2]) for x in point_positions_east[1:$west_atker]]

    scatter!(axes[idx], west_defenders_points, marker = :rect, markersize = reverse(defender_size), color = :blue)
    scatter!(axes[idx], west_atker_points, marker = :ltriangle, markersize = atker_size, color = :red)
end
trim!(fig.layout)
display(fig, fullscreen = true)
end

function compute_all_r_save_to_file(;_prior_range = 0.01:.1:1.1,
     attacker_preference = [[3, 2, 2], [2, 3, 2], [2, 2, 3]],
     save_file_name = "./experiments/data.tmp")
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

#module end
end