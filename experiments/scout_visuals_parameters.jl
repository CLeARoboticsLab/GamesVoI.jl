# Game Parameters
attacker_preference = [[ 3.0; 2.0; 2.0], [2.0, 3.0, 2.0], [2.0, 2.0, 3.0]]
num_worlds = 3
prior_range_step = 0.01
prior_range = 0.01:prior_range_step:1
K = 100

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

#Plotting utilities
opacity = 0.5
ab = [:orange, :green, :pink]

#Save file
save_file_name = "./experiments/Stage1Allocation.data"

# Stage 2 demo utilities
world_signal_pairs = [(1, 0), (2, 0), (3, 0), (1, 1), (2, 2), (3, 3)]

x_north_center, y_north_center = 565 , 600
x_east_center, y_east_center = 350, 250

increment, top_increment = 60, 80

defender_size = (25, 10)
attacker_size = 15

# Precompute utilities
margin = 0.11