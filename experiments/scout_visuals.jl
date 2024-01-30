module scout_visuals

using GamesVoI
using GLMakie
using LinearAlgebra
using JSON3
include("tower_defense.jl")

Makie.inline!(false)

# Game Parameters
attacker_preference = [[0.9; 0.05; 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]

# Visual parameters
    # Axis parameters
        # borders
        ax_aspect = 1 
        ax_limits = (0, 1, 0, 1)
        # title
        ax_titlegap = 48
        ax_titlesize = 60
        # x-axis
        ax_xautolimitmargin = (0, 0)
        ax_xgridwidth = 2
        ax_xticklabelsize = 36
        ax_xticks = -10:10
        ax_xticksize = 18
        # y-axis
        ax_yautolimitmargin = (0, 0)
        ax_ygridwidth = 2
        ax_yticklabelpad = 14
        ax_yticklabelsize = 36
        ax_yticks = -10:10
        ax_yticksize = 18

""" TODO: 
1. Do visualiation for each given world/signal received by Player 1 
2. Solve for all different combinations and store them in a "look-up dictionary" so that you dont have to solve the game all the time
"""

function demo(; attacker_preference = [[0.9; 0.05; 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    
    num_worlds = 3
    prior_range_step = 0.01
    prior_range_step_precision = 1
    prior_range = 0.01:prior_range_step:1
    save_file_name = "precomputed_r.txt"
    save_precision = 4

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

    ax_simplex = Axis3(fig[2,2], aspect = (1,1,1), 
        limits = ((0.0, 1.0, 0.0, 1.0, 0.0, 1.1)),
        xreversed = true, 
        yreversed = true
    )

    # Create sliders
    sg = SliderGrid(
        fig[3, 2],
        (label = "prior_north", range = prior_range, format = "{:.2f}", startvalue = 1.0),
        (label = "prior_east", range = prior_range, format = "{:.2f}", startvalue = 1.0),
        (label = "prior_west", range = prior_range, format = "{:.2f}", startvalue = 1.0)
    )
    observable_prior_sliders = [s.value for s in sg.sliders]
 
    # Plot priors on the Simplex
    scatterlines!(ax_simplex, [1;0;0;1], [0;1;0;0], [0;0;1;0], markersize = 15)

    # Normalize priors
    normalized_observable_p = lift(observable_prior_sliders...) do a, b, c
        round.(normalize([a,b,c], 1), digits = 2)
    end
    @lift println("priors: ", $normalized_observable_p)

    p1, p2, p3 = [lift((x,i)->x[i], normalized_observable_p, idx) for idx in 1:num_worlds]
    scatter!(ax_simplex, p1, p2, p3 ; markersize = 15, color = :red)

    # Solve for scout_allocation, r 
    observable_r = on(normalized_observable_p) do x
        solve_r(x, attacker_preference)
    end
    scout_north, scout_east, scout_west = [lift((x,i)->x[i], observable_r.observable, idx) for idx in 1:num_worlds]

    # TODO: Be creative with scout_allocation
    scatter!(ax_north, scout_north, scout_north, markersize = 15, color = :red)
    scatter!(ax_east, scout_east, scout_east, markersize = 15, color = :purple)
    scatter!(ax_west, scout_west, scout_west, markersize = 15, color = :green)
   
    display(fig)
end

function compute_all_r_save_to_file()
    # save_file = open(save_file_name, "w+")
    hashmap = Dict{Tuple{Float64, Float64, Float64}, Tuple{Float64, Float64, Float64}}()
    for prior_north in prior_range
        for prior_east in prior_range
            for prior_west in prior_range
                current_prior = (prior_north, prior_east, prior_west)
                r = solve_r(current_prior, attacker_preference)
                hashmap[current_prior] = round.(r, digits = save_precision)
            end
        end
    end
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