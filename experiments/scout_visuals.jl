module scout_visuals

using GamesVoI
using GLMakie
include("tower_defense.jl")

Makie.inline!(false)

# export demo

# function demo()

# Game parameters
attacker_preference = [.1; .2; .7]

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



# ## Observables
# # Observables are a way to create a reactive programming environment in Julia.
# # Listener is an Observable that is dependent on another Observable
# x = Observable(1.0) # no listeners
# y = Observable(2.0) # no listensers
# z = @lift($x .+ $y) # set up listener using the lift macro

# x[] = 2.0 # change x using empty brackets
# println(z) # updates automatically

# # ObserverFunction. on(...) adds function f as a listener to observable
# w = on(x) do val
#     println("x changed to $val")
# end

# # Simple Example (REPL에서 Makie.inline!(false) 해주기)
# ox = 1:4
# oy = Observable(rand(4)) # Y-values are Observables
# lw = Observable(2.0) # line width is an Observable

# fig = Figure()
# ax = Axis(fig[1,1])
# lines!(ox, oy, linewidth = lw) # plot initial values
# ylims!(ax, 0, 1) # set y limits

# lw[] = 50.0 # change line width
# oy[] = rand(4) # change y values


## 1. Sliders
# initialize plot
fig = Figure(;size = (3840, 2160))
# add axis

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


# darken axes

# vlines!(ax1, [0], linewidth = 2, color = :black)
# hlines!(ax1, [0], linewidth = 2, color = :black)

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

# print("{$prior_north_listener}, {$prior_east_listener}, {$prior_west_listener}\n")

x = -10:0.01:10
# r = @lift(solve_r([$prior_north_listener; $prior_east_listener; $prior_west_listener], attacker_preference))

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
# end
#module end
end