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

function f(buff_cap, train_intensity, warmup_steps)
    sum(1 - ((i-1)/i)^train_intensity for i in warmup_steps:buff_cap)
end