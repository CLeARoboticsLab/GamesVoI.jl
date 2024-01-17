using Plots

function f(δ, k)
    return 1/(1 + exp(-2 * δ * k))
end

k = 1:1:10
δ = -2:0.05:2

p = plot()
for i in 1:length(k)    
    plot!(p, δ, f.(δ, k[i]), label="k = $(k[i])", linewidth=7)
end
plot!(p, size=(800, 600), title="Logistic function", textfontsize=20, legendfontsize=20, tickfontsize=10, legend=:topleft, titlefontsize=20, labelfontsize = 20, xlabel="δ", ylabel="f(δ, k)")
display(p)