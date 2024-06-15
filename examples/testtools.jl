using Plots, Distributions

function ci_bounds(y; ci=0.95, dim=argmin(size(y)))
    n = size(y, dim)
    t = quantile(TDist(n-1), 1-(1-ci)/2)
    mu = mean(y; dims=dim)
    diff = t * std(y; dims=dim) / sqrt(n)
    bounds = mu .+ cat(-diff, diff; dims=dim)
    return mu, bounds
end

error_plot(args...; kwargs...) = error_plot!(plot(), args...; kwargs...)
error_plot!(p, y, bounds; kwargs...) = error_plot!(p, 1:length(y), y, bounds; kwargs...)
function error_plot!(p, x, y, bounds; c=1+p.n÷2, fillcolor=c, fillalpha=0.2, linealpha=0, bounds_label=false, label=false, linewidth=1.5, kwargs...)
    plot!(p, x, bounds[:,1]; fillrange = bounds[:,2], fillalpha, linealpha, fillcolor, label=bounds_label, kwargs...)
    plot!(p, x, y; label, linewidth, c, kwargs...)
end

plot_smoothing(y; kwargs...) = plot_smoothing(1:length(y), y; kwargs...)
plot_smoothing(x, y; kwargs...) = plot_smoothing!(plot(), x, y; kwargs...)
plot_smoothing!(p, y; kwargs...) = plot_smoothing!(p, 1:length(y), y; kwargs...) 
function plot_smoothing!(p, x, y; nx=250, k=1, kwargs...)
    x_new_maybe = range(minimum(x), stop=maximum(x), length=nx)
    x_new = Float64[]
    y_new = Float64[]
    for i in eachindex(x_new_maybe)
        x1 = x_new_maybe[max(1, i-k)]
        x2 = x_new_maybe[min(nx, i+k)]
        idxs = findall(x1 .<= x .<= x2)
        if !isempty(idxs)
            push!(x_new, mean(x[idxs]))
            push!(y_new, mean(y[idxs]))
        end
    end
    plot!(p, x_new, y_new; kwargs...)
end