module scout_visuals

using GamesVoI
using GLMakie
using LinearAlgebra
using JSON3, FileIO
include("tower_defense.jl")
include("scout_visuals_parameters.jl")

Makie.inline!(false)

function demo_stage_1(;use_file=false)

    # If Stage 1 prior -> r has been precomputed, use the precomputed values
    save_file = nothing
    if use_file
        save_file = JSON3.read(open(save_file_name, "r"), Dict{String, Vector{Float64}})
    end
    
    # Initialize plot
    fig = Figure()

    # Axis for 3 directions
    axis_placement = [(1,2,"North"), (2,1,"West"), (2,3,"East")]
    axis_directions = []
    for idx in axis_placement
        push!(axis_directions, Axis(
        # placement
        fig[idx[1], idx[2]],
        # aspect
        aspect = ax_aspect, limits = ax_limits,
        # title
        title = idx[3],
        titlegap = ax_titlegap, titlesize = ax_titlesize,
        # x-axis
        xautolimitmargin = ax_xautolimitmargin, xgridwidth = ax_xgridwidth, 
        xticklabelsize = ax_xticklabelsize,
        xticks = ax_xticks, xticksize = ax_xticksize,
        # y-axis
        yautolimitmargin = ax_yautolimitmargin, ygridwidth = ax_ygridwidth,
        yticklabelpad = ax_yticklabelpad,
        yticklabelsize = ax_yticklabelsize, yticks = ax_yticks, yticksize = ax_yticksize
        ))
        hidedecorations!(axis_directions[length(axis_directions)])
    end

    # Axis for prior simplex
    ax_simplex = Axis3(fig[2,2], aspect = (1,1,1), 
        limits = ((0.0, 1.0, 0.0, 1.0, 0.0, 1.1)),
        xreversed = true, 
        yreversed = true, 
        xlabel = "",
        ylabel = "",
        zlabel = "",
    )

    # Create sliders
    sg = SliderGrid(
        fig[3, 2],
        (label = "prior_north", range = prior_range, format = x-> "", startvalue = 1.0),
        (label = "prior_west", range = prior_range, format = x-> "", startvalue = 1.0),
        (label = "prior_east", range = prior_range, format = x-> "", startvalue = 1.0)
    )

    # Create slider observable
    observable_prior_sliders = [s.value for s in sg.sliders]
 
    # Plot priors on the Simplex
    scatterlines!(ax_simplex, [1;0;0;1], [0;1;0;0], [0;0;1;0], markersize = 15)

    # Normalize priors
    normalized_observable_p = lift(observable_prior_sliders...) do a, b, c
        round.(normalize([a,b,c], 1), digits = 2)
    end

    # Plot prior on simplex
    p1, p2, p3 = [lift((x,i)->x[i], normalized_observable_p, idx) for idx in 1:num_worlds]
    scatter!(ax_simplex, p3, p2, p1 ; markersize = 15, color = :red)

    # Solve or retrieve scout_allocation, r 
    observable_r = on(normalized_observable_p) do x
        if use_file
            save_file[string(round.(x))]
        else
            solve_r(x, attacker_preference)
        end
    end
    scouts = [lift((x,i)->x[i], observable_r.observable, idx) for idx in 1:num_worlds]

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

    # Make plot labels
    text_directions = [lift((x) -> "$(round(Int, x*K))%", scout) for scout in scouts]

    # Plot scout allocation and labels
    for idx in 1:3
        Label(fig[axis_placement[idx][1], axis_placement[idx][2]], text_directions[idx], 
                fontsize = 20, tellwidth = false, tellheight = false)
        
        points = lift(x->get_random_point_within_ball(; radius = x*0.5, num_points = round(Int, 100*x)), scouts[idx])
        scatter!(axis_directions[idx], points, markersize = 15, color = (ab[idx], opacity))
    end

    display(fig, fullscreen = true)
end

function demo_full(;use_file=true)
    # Use save file if it exists
    save_file = nothing
    if use_file
        save_file = JSON3.read(open("./experiments/" * save_file_name , "r"), Dict{String, Vector{Float64}})
    end

# Initialize plot
fig = Figure(figure_padding = 2)
# Fix spacing
rowgap!(fig.layout, 0)

axes = [Axis(fig[x[2] != 0 ? 2 : 1, x[1] + 1], aspect = DataAspect(), 
    title= L"\omega_{%$(x[1])} \enspace \sigma_{%$(x[2])}", titlesize = 30, titlegap = 3)
    for x in world_signal_pairs]
for ax in axes
    hidedecorations!(ax)
end

# Fix spacing v2
rowsize!(fig.layout, 1, Aspect(1, 0.7))
rowsize!(fig.layout, 2, Aspect(1, 0.7))

# Create simplex
ax_simplex = Axis3(fig[1,1], aspect = (1,1,1), 
    limits = ((0.0, 1.0, 0.0, 1.0, 0.0, 1.1)),
    xreversed = true, 
    yreversed = true, 
    xlabel = "",
    ylabel = "",
    zlabel = "",
    tellheight = false
)

# Create sliders
sg = SliderGrid(
    fig[2, 1],
    (label = L"p(\omega_1)", range = prior_range, format = x-> "", startvalue = 1.0),
    (label = L"p(\omega_2)", range = prior_range, format = x-> "", startvalue = 1.0),
    (label = L"p(\omega_3)", range = prior_range, format = x-> "", startvalue = 1.0),
    tellheight = false
)
# Create slider observable
observable_prior_sliders = [s.value for s in sg.sliders]

# Plot priors on the Simplex
scatterlines!(ax_simplex, [1;0;0;1], [0;1;0;0], [0;0;1;0], markersize = 15)

# Normalize priors
normalized_observable_p = lift(observable_prior_sliders...) do a, b, c
    round.(normalize([a,b,c], 1), digits = 2)
end

# p₁ : west, p₂ : east, p₃ : north
p1, p2, p3 = [lift((x,i)->x[i], normalized_observable_p, idx) for idx in 1:num_worlds]
scatter!(ax_simplex, p3, p2, p1 ; markersize = 15, color = (:red, .75), label=L"\mathbf{p}(\omega)")

# Solve for scout_allocation, r 
observable_r = on(normalized_observable_p) do x
    if use_file
        save_file[string(round.(x))]
    else
        solve_r(x, attacker_preference)
    end
end
scout_north, scout_east, scout_west = [lift((x,i)->x[i], observable_r.observable, idx) for idx in 1:num_worlds]
scatter!(ax_simplex, scout_north, scout_east, scout_west ; markersize = 15, color = (:green, .75), label=L"\mathbf{r}")
axislegend()

# Build stage 2 game
game = lift((x) -> build_stage_2(x, attacker_preference), normalized_observable_p)
game_results = lift((r, p, game) -> compute_stage_2(r, p, attacker_preference, game[1]), 
    observable_r.observable, normalized_observable_p, game)

# Find the number of icons to draw for attacker/defender allocation
function convert_to_int(defender, attacker)
    defender = round(defender, digits = 2)
    attacker = round(attacker, digits = 2)
    num_defenders = round(defender * 10, digits = 0)
    num_attackers = round(attacker * 10, digits = 0)
    if num_attackers == num_defenders && num_attackers > 0
        if attacker > defender
            if num_attackers == 10
                num_defenders -= 1
            else
                num_attackers += 1
            end
        else #attacker <= defender
            if num_defenders == 10
                num_attackers -= 1
            else
                num_defenders += 1
            end
        end
    end

    return (Int(num_defenders), Int(num_attackers))
end

# Get world map
stage2_map = load("./experiments/stage2_map.jpg") # resolution: 832x1132
for i in axes
    image!(i, rotr90(stage2_map))
end

# Get positions of attacker/defender icons
point_positions_north = [(x_north_center, y_north_center - increment), (x_north_center + increment,y_north_center - increment), (x_north_center - increment,y_north_center - increment), (x_north_center + 2increment, y_north_center - increment), (x_north_center - 2increment, y_north_center - increment),
    (x_north_center, y_north_center), (x_north_center + top_increment,y_north_center), (x_north_center - top_increment,y_north_center), (x_north_center + 2top_increment, y_north_center), (x_north_center - 2top_increment, y_north_center)]

point_positions_east = [(x_east_center, y_east_center),               (x_east_center, increment + y_east_center),        (x_east_center, -increment + y_east_center),         (x_east_center, 2increment + y_east_center),        (x_east_center, -2increment + y_east_center),
                        (x_east_center - increment, y_east_center), (x_east_center - increment, y_east_center + top_increment), (x_east_center - increment, y_east_center - top_increment), (x_east_center - increment, y_east_center + 2top_increment), (x_east_center - increment, y_east_center - 2top_increment)]

# West positions are mirrored east positions

for (idx, world_signal) in enumerate(world_signal_pairs)

    defender_index = world_signal[2] + 1
    attacker_index = (world_signal[2] == 0 ? 4 : 7) + world_signal[1] 

    # North
    north_defenders = @lift convert_to_int($game_results[Block(defender_index)][1], $game_results[Block(attacker_index)][1])[1]
    north_attackers = @lift convert_to_int($game_results[Block(defender_index)][1], $game_results[Block(attacker_index)][1])[2]
    north_defenders_points = @lift [Point2f(x[1], x[2]) for x in point_positions_north[1:$north_defenders]]
    north_attackers_points = @lift [Point2f(x[1], x[2] + 150) for x in point_positions_north[1:$north_attackers]]

    scatter!(axes[idx], north_defenders_points, marker = :rect, markersize = defender_size, color = :blue)
    scatter!(axes[idx], north_attackers_points, marker = :dtriangle, markersize = attacker_size, color = :red)

    # East
    east_defenders = @lift convert_to_int($game_results[Block(defender_index)][2], $game_results[Block(attacker_index)][2])[1]
    east_attackers = @lift convert_to_int($game_results[Block(defender_index)][2], $game_results[Block(attacker_index)][2])[2]
    east_defenders_points = @lift [Point2f(x[1], x[2]) for x in point_positions_east[1:$east_defenders]]
    east_attackers_points = @lift [Point2f(x[1] - 150, x[2]) for x in point_positions_east[1:$east_attackers]]

    scatter!(axes[idx], east_defenders_points, marker = :rect, markersize = reverse(defender_size), color = :blue)
    scatter!(axes[idx], east_attackers_points, marker = :rtriangle, markersize = attacker_size, color = :red)

    # West
    west_defenders = @lift convert_to_int($game_results[Block(defender_index)][3], $game_results[Block(attacker_index)][3])[1]
    west_attackers = @lift convert_to_int($game_results[Block(defender_index)][3], $game_results[Block(attacker_index)][3])[2]
    west_defenders_points = @lift [Point2f(1130 - x[1], x[2]) for x in point_positions_east[1:$west_defenders]]
    west_attackers_points = @lift [Point2f(1130 - x[1] + 150, x[2]) for x in point_positions_east[1:$west_attackers]]

    scatter!(axes[idx], west_defenders_points, marker = :rect, markersize = reverse(defender_size), color = :blue)
    scatter!(axes[idx], west_attackers_points, marker = :ltriangle, markersize = attacker_size, color = :red)
end

trim!(fig.layout)
display(fig, fullscreen = true)
end

# Pre-compute all r values for all priors for a given attacker preference
function compute_all_r_save_to_file()
    hashmap = Dict{Vector{Float64}, Vector{Float64}}()
    for prior_north in prior_range
        for prior_east in prior_range
            for prior_west in prior_range
                current_prior = [prior_north, prior_east, prior_west]
                if norm(current_prior, 1) <= 1.0 + margin && norm(current_prior, 1) >= 1.0 - margin
                    r = solve_r(current_prior, attacker_preference)
                    hashmap[current_prior] = r
                end
            end
        end
    end
    hashmap[[0.0, 0.0, 0.0]] = [0.0, 0.0, 0.0]
    JSON3.write(save_file_name, hashmap)
end

#module end
end