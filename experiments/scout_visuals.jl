module scout_visuals

using GamesVoI
using GLMakie
include("tower_defense.jl")

Makie.inline!(false)

# export demo

function demo()

    # Game parameters
    attacker_preference = [.1; .2; .7]

    ## 1. Sliders
    # Initialize plot
    fig = Figure(;size = (3840, 2160))

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
        (label = "prior_north", range = 0:0.01:1, format = "{:.2f}", startvalue = 01),
        (label = "prior_east", range = 0:0.01:1, format = "{:.2f}", startvalue = 0),
        (label = "prior_west", range = 0:0.01:1, format = "{:.2f}", startvalue = 0)
    )

    # Create listener
    prior_north_listener = sg.sliders[1].value
    prior_east_listener = sg.sliders[2].value
    prior_west_listener = sg.sliders[3].value

    # TODO
    #r = @lift(solve_r([$prior_north_listener; $prior_east_listener; $prior_west_listener], attacker_preference))

    #temporary
    r = @lift([$prior_north_listener; $prior_east_listener; $prior_west_listener])
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

#module end
end