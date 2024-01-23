module scout_visuals

using GamesVoI
using GLMakie
using LinearAlgebra
using JSON3
include("tower_defense.jl")

Makie.inline!(false)

# Globals:
num_worlds = 3
prior_range_step = 0.01
prior_range_step_precision = 2
prior_range = 0:prior_range_step:1
save_file_name = "precomputed_r.txt"
save_precision = 4

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

function demo()
    ## 1. Sliders
    # Initialize plot
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

    #temporary
    observable_priors = @lift([$prior_north_listener; $prior_east_listener; $prior_west_listener])
    priors = normalize(observable_priors[])

    r = solve_r(priors, attacker_preference)
    # print(r)
    scat1 = scatter!(ax_north, r, r, markersize = 10, color = :red)

    # Plot line
    # line1 = lines!(ax1, x, y, linewidth = 2, color = :blue)

    ## Another way ##
    # sliderob = [s.value for s in sg.sliders]

    # y = lift(sliderob...) do slope, intercept
    #     slope .* x .+ intercept
    # end
    # y = lift((slope, intercept) -> slope .* x .+ intercept, sliderob...) # also same 

    # Plot line
    # line1 = lines!(ax1, x, y, linewidth = 2, color = :blue)

    # # 2. Button
    # fig[3,1] = buttongrid = GridLayout(tellwidth = false)
    # buttonlabels = ["Red", "Green", "Blue"]

    # buttons = buttongrid[1, 1:3] = [
    #     Button(fig, label = l, height = 60, width = 250, fontsize = 30) for l in buttonlabels]

    # bt_sublayout = GridLayout(height = 150)
    # fig[3,1] = bt_sublayout

    # # Random dataset we want to see

    # x = -10:0.01:10
    # data = []

    # for i in 1:3
    #     d = rand(-10:0.01:10, length(x))
    #     push!(data, d)
    # end

    # # Set y_data as observable
    # y = Observable(data[1])

    # # Set color as observable
    # colors = [:red, :green, :blue]
    # c = Observable(colors[1])

    # # Set markersize as observable
    # markersizes = [8, 12, 16]
    # ms = Observable(markersizes[1])

    # # Add scatter plot
    # scat1 = scatter!(ax1, x, y, markersize = ms, color = c)

    # # Button instructions using on...do...end syntax

    # for i in 1:3
    #     on(buttons[i].clicks) do _
    #         y[] = data[i]
    #         c[] = colors[i]
    #         ms[] = markersizes[i]
    #     end 
    # end
    display(fig)

#demo function end
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

    observable_priors = @lift([$prior_north_listener; $prior_east_listener; $prior_west_listener])
    priors = normalize(observable_priors[])
    priors .= round.(priors, digits = prior_range_step_precision)
    # print(r)
    scat1 = scatter!(ax_north, r, r, markersize = 10, color = :red)
    display(fig)
end
#module end
end